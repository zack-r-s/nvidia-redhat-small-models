"""Evaluate a local LFM checkpoint on held-out prompts against the checked-in PyAT ring.

The evaluator supplies a verified pre-read, as the live goal harness does. It forces
only the channel_write tool name for write tasks; the model chooses PVs and values.
No model proposal is written until its channel, plane, value, and bounds validate.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from hackring.machine import BPMS, LIMIT, SETPOINTS, LocalMachine

ROOT = Path(__file__).resolve().parents[1]
READBACKS = [name.replace(":SP", ":RB") for name in SETPOINTS]
TOOL_SCHEMA = [
    {"type": "function", "function": {"name": "channel_read", "description": "Read current PV values.",
     "parameters": {"type": "object", "properties": {"channels": {"type": "array", "items": {"type": "string"}}},
                    "required": ["channels"]}}},
    {"type": "function", "function": {"name": "channel_write", "description": "Write corrector PV setpoints.",
     "parameters": {"type": "object", "properties": {"operations": {"type": "array", "items": {"type": "object"}},
                                                 "confirm": {"type": "boolean"}}, "required": ["operations"]}}},
]
SYSTEM = (
    "You operate a simulated FODO storage ring. Use channel_read before a write and read BPMs "
    "and corrector readbacks after it. BPM positions are mm; absolute corrector kicks are mrad "
    "and limited to ±0.35. Correct only the plane requested by the operator. Report only measured "
    "results. List of tools: " + json.dumps(TOOL_SCHEMA, separators=(",", ":"))
)
CALL_PREFIX = "<|tool_call_start|>[channel_write("
CALL_END = "<|tool_call_end|>"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(pvs: dict[str, float], plane: str) -> float:
    channels = [pv for pv in BPMS if plane == "both" or pv.endswith(":X" if plane == "x" else ":Y")]
    return math.sqrt(sum(pvs[pv] ** 2 for pv in channels) / len(channels))


def selected(case: dict) -> tuple[list[str], list[str]]:
    if case["kind"] == "exact_write":
        return [case["channel"]], [case["channel"]]
    plane = case["plane"]
    bpm = [pv for pv in BPMS if plane == "both" or pv.endswith(":X" if plane == "x" else ":Y")]
    correctors = [pv for pv in SETPOINTS if plane == "both" or (":MAG:H" in pv if plane == "x" else ":MAG:V" in pv)]
    return bpm + correctors, correctors


def parse_write(text: str) -> list[dict]:
    """Decode only literal LFM bracket calls, never execute generated Python."""
    if not text.startswith(CALL_PREFIX) or CALL_END not in text:
        raise ValueError("no complete channel_write tool call")
    body = text[len("<|tool_call_start|>["):text.index(CALL_END)].strip()
    if not body.endswith("]"):
        raise ValueError("tool call has no closing bracket")
    try:
        call = ast.parse(body[:-1], mode="eval").body
    except SyntaxError:
        raise ValueError("malformed tool arguments") from None
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "channel_write" or call.args:
        raise ValueError("expected channel_write keyword arguments")
    try:
        arguments = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords if keyword.arg}
    except (ValueError, TypeError, RecursionError):
        raise ValueError("nonliteral tool arguments") from None
    operations = arguments.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("missing operations")
    return operations


def validate_write(operations: list[dict], allowed: list[str], case: dict) -> list[dict]:
    if len(operations) > len(allowed):
        raise ValueError("too many correctors")
    clean = []
    seen = set()
    for op in operations:
        if not isinstance(op, dict):
            raise ValueError("operation is not an object")
        channel = op.get("channel", op.get("name"))
        if channel not in allowed or channel in seen:
            raise ValueError(f"disallowed or duplicate corrector: {channel}")
        value = op.get("value")
        if isinstance(value, bool):
            raise ValueError("kick is not numeric")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("kick is not numeric") from None
        if not math.isfinite(value) or abs(value) > LIMIT:
            raise ValueError("kick exceeds finite bounds")
        clean.append({"channel": channel, "value": value})
        seen.add(channel)
    if case["kind"] == "exact_write" and (len(clean) != 1 or
            clean[0]["channel"] != case["channel"] or
            abs(clean[0]["value"] - case["value_mrad"]) > 1e-6):
        raise ValueError("exact write differs from operator request")
    return clean


def tool_read(pvs: dict[str, float], channels: list[str]) -> str:
    return json.dumps({"status": "success", "summary": {"readings": {
        pv: {"value": round(pvs[pv], 6)} for pv in channels}}}, separators=(",", ":"))


def tool_call(name: str, **kwargs: object) -> str:
    arguments = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
    return f"<|tool_call_start|>[{name}({arguments})]<|tool_call_end|>"


def trim(text: str) -> str:
    return text.split("<|im_end|>", 1)[0].strip()


class CpuLfm:
    def __init__(self, name: str, threads: int, device: str = "cpu"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable on this host")
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, local_files_only=True, dtype=torch.bfloat16 if device == "mps" else torch.float32,
        ).to(device).eval()

    def generate(self, messages: list[dict], *, prefix: str = "", max_new_tokens: int = 192) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix
        tokens = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        if tokens.shape[-1] + max_new_tokens > 4096:
            raise ValueError("model context exceeds 4096 tokens")
        with self.torch.inference_mode():
            output = self.model.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=self.tokenizer.eos_token_id)
        return prefix + self.tokenizer.decode(output[0, tokens.shape[-1]:].cpu(), skip_special_tokens=False)


def score_read(case: dict, text: str, pvs: dict[str, float]) -> bool:
    answer = trim(text)
    if case["kind"] == "compare_planes":
        larger = "X" if rms(pvs, "x") > rms(pvs, "y") else "Y"
        return bool(re.search(rf"\b({larger}|{'horizontal' if larger == 'X' else 'vertical'})\b.{0,35}\b(larger|farther|greater|higher)\b", answer, re.I)
                    or re.search(rf"\b(larger|farther|greater|higher)\b.{0,35}\b({larger}|{'horizontal' if larger == 'X' else 'vertical'})\b", answer, re.I))
    numbers = [float(x) for x in re.findall(r"(?<![\w.])[+-]?(?:\d+\.\d+|\.\d+)", answer)]
    expected = [pvs[pv] for pv in BPMS]
    return len(numbers) >= 8 and all(abs(a - b) <= 0.005 for a, b in zip(numbers[:8], expected))


def evaluate_case(case: dict, model: CpuLfm, disturbance: float) -> dict:
    machine = LocalMachine()
    before = machine.reset(case["seed"], disturbance)["pvs"]
    result = {"id": case["id"], "seed": case["seed"], "kind": case["kind"], "prompt": case["prompt"],
              "before": before, "before_x_rms_mm": rms(before, "x"), "before_y_rms_mm": rms(before, "y")}
    read_channels = BPMS if case["kind"] in ("read_bpms", "compare_planes") else selected(case)[0]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": case["prompt"]},
                {"role": "assistant", "content": tool_call("channel_read", channels=read_channels)},
                {"role": "tool", "content": tool_read(before, read_channels)}]
    if case["kind"] in ("read_bpms", "compare_planes"):
        answer = model.generate(messages, max_new_tokens=192)
        result.update({"model_answer": trim(answer), "passed": score_read(case, answer, before),
                       "write_attempted": False, "after": before})
        return result
    _, allowed = selected(case)
    proposal = model.generate(messages, prefix=CALL_PREFIX, max_new_tokens=256)
    result["model_proposal"] = trim(proposal)
    try:
        operations = validate_write(parse_write(proposal), allowed, case)
    except (SyntaxError, ValueError, TypeError) as exc:
        result.update({"passed": False, "write_attempted": False, "error": str(exc), "after": before})
        return result
    settings = {op["channel"]: op["value"] for op in operations}
    after = machine.write(settings)["state"]["pvs"]
    result.update({"operations": operations, "write_attempted": True, "after": after,
                   "after_x_rms_mm": rms(after, "x"), "after_y_rms_mm": rms(after, "y")})
    if case["kind"] == "exact_write":
        result["passed"] = (abs(after[case["channel"]] - case["value_mrad"]) <= 1e-6 and
                            case["follow_up"] in after)
    else:
        plane = case["plane"]
        result["before_target_rms_mm"] = rms(before, plane)
        result["after_target_rms_mm"] = rms(after, plane)
        result["passed"] = (result["after_x_rms_mm"] < result["before_x_rms_mm"] and
                            result["after_y_rms_mm"] < result["before_y_rms_mm"] if plane == "both"
                            else result["after_target_rms_mm"] < result["before_target_rms_mm"])
    write_result = {"status": "success", "summary": {"results": [
        {"channel": op["channel"], "outcome": "confirmed", "observed_value": after[op["channel"]]}
        for op in operations]}}
    follow_up = ([case["follow_up"]] if case["kind"] == "exact_write" else
                 [pv for pv in BPMS if case["plane"] == "both" or pv.endswith(":X" if case["plane"] == "x" else ":Y")])
    follow_up += [op["channel"].replace(":SP", ":RB") for op in operations]
    messages.extend([{"role": "assistant", "content": proposal.split(CALL_END)[0] + CALL_END},
                     {"role": "tool", "content": json.dumps(write_result, separators=(",", ":"))},
                     {"role": "assistant", "content": tool_call("channel_read", channels=follow_up)},
                     {"role": "tool", "content": tool_read(after, follow_up)}])
    result["model_answer"] = trim(model.generate(messages, max_new_tokens=192))
    return result


def run(args: argparse.Namespace) -> dict:
    cases_path = args.cases.resolve()
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    if args.limit:
        cases = cases[:args.limit]
    if len({case["id"] for case in cases}) != len(cases) or len({case["seed"] for case in cases}) != len(cases):
        raise ValueError("evaluation case IDs and seeds must be unique")
    if not 0 < args.disturbance_mrad <= 0.2:
        raise ValueError("disturbance must be in (0, 0.2] mrad")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    model = CpuLfm(args.model, args.threads, args.device)
    traces = []
    for case in cases:
        trace = evaluate_case(case, model, args.disturbance_mrad)
        traces.append(trace)
        print(f"{case['id']}: {'PASS' if trace['passed'] else 'FAIL'}"
              + (f" ({trace['error']})" if trace.get("error") else ""), flush=True)
    counts = Counter(case["kind"] for case in cases)
    passed = Counter(trace["kind"] for trace in traces if trace["passed"])
    report = {"schema": "hackring.pyat_eval/1", "evaluation_series": args.series,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "model": args.model,
              "inference": f"Transformers {args.device.upper()} greedy ({'bfloat16' if args.device == 'mps' else 'float32'}), verified harness pre-read; write tool name forced",
              "plant": "LocalMachine on checked-in PyAT ring; seeded ERR kicks, not Osprey calibration offsets",
              "disturbance_mrad": args.disturbance_mrad, "cases_sha256": digest(cases_path),
              "evaluator_sha256": digest(Path(__file__)),
              "deck_sha256": digest(ROOT / "data/facility/decks/RING.json"),
              "pv_contract_sha256": digest(ROOT / "data/pv_contract.json"),
              "software": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "accelerator-toolbox")},
              "counts": dict(counts), "passed_by_kind": dict(passed), "passed": sum(passed.values()),
              "total": len(traces), "case_results": [{"id": trace["id"], "passed": trace["passed"],
                                                  "error": trace.get("error"),
                                                  "before_target_rms_mm": trace.get("before_target_rms_mm"),
                                                  "after_target_rms_mm": trace.get("after_target_rms_mm")}
                                                 for trace in traces]}
    output.mkdir(parents=True)
    (output / "traces.jsonl").write_text("".join(json.dumps(trace, allow_nan=False) + "\n" for trace in traces))
    report["traces_sha256"] = digest(output / "traces.jsonl")
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="LiquidAI/LFM2.5-230M")
    parser.add_argument("--cases", type=Path, default=ROOT / "evals/mark-i/pyat-v001/cases.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "evals/mark-ii/pyat-v001/base-230m")
    parser.add_argument("--disturbance-mrad", type=float, default=0.08)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--series", choices=("mark-ii", "mark-iii"), default="mark-ii")
    parser.add_argument("--limit", type=int, help="Run the first N cases into a separate output directory")
    args = parser.parse_args()
    if args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("threads and limit must be positive")
    report = run(args)
    print(json.dumps({"passed": report["passed"], "total": report["total"],
                      "report": str(args.output / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
