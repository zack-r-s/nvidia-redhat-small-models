"""Evaluate v004 skill routing and measured bounded corrections on frozen PyAT cases."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

from eval_local_pyat import CpuLfm
from hackring.control import EFFORT_PENALTY, SKILL_SYSTEM, requested_effort, rms, solve
from hackring.machine import ROOT, LocalMachine


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_skill(output: str) -> tuple[str, str]:
    start = "<|tool_call_start|>["
    end = "]<|tool_call_end|>"
    if not output.startswith(start) or end not in output:
        raise ValueError("no complete tool call")
    body = output[len(start):output.index(end)]
    node = ast.parse(body, mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "orbit_correct" or node.args:
        raise ValueError("expected orbit_correct")
    arguments = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
    plane, effort = arguments.get("plane"), arguments.get("effort")
    if plane not in ("x", "y", "both") or effort not in EFFORT_PENALTY:
        raise ValueError("invalid plane or effort")
    return plane, effort


def evaluate(case: dict, model: CpuLfm) -> dict:
    machine = LocalMachine()
    before = machine.reset(case["seed"], 0.08)["pvs"]
    result = {"id": case["id"], "seed": case["seed"], "prompt": case["prompt"],
              "expected_plane": case["plane"], "before": before}
    proposal = model.generate([{"role": "system", "content": SKILL_SYSTEM},
                               {"role": "user", "content": case["prompt"]}], max_new_tokens=96)
    result["model_proposal"] = proposal
    try:
        plane, effort = parse_skill(proposal)
        result.update({"selected_plane": plane, "selected_effort": effort})
        if plane != case["plane"]:
            raise ValueError("wrong correction plane")
        if effort != requested_effort(case["prompt"]):
            raise ValueError("wrong correction effort")
        settings = solve(before, plane, effort)
        after = machine.write(settings)["state"]["pvs"]
        result.update({"settings": settings, "after": after,
                       "before_target_rms_mm": rms(before, plane),
                       "after_target_rms_mm": rms(after, plane)})
        planes = ("x", "y") if plane == "both" else (plane,)
        result["passed"] = all(rms(after, item) < rms(before, item) for item in planes)
    except (ValueError, SyntaxError, TypeError, KeyError) as exc:
        result.update({"passed": False, "error": str(exc), "after": before})
    return result


def run(args: argparse.Namespace) -> dict:
    cases_path = args.cases.resolve()
    cases = [json.loads(line) for line in cases_path.read_text().splitlines()]
    cases = [case for case in cases if case.get("kind", "correct") == "correct"
             and (args.split is None or case.get("split") == args.split)]
    if args.limit:
        cases = cases[:args.limit]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    model = CpuLfm(args.model, args.threads)
    traces = []
    for case in cases:
        trace = evaluate(case, model)
        traces.append(trace)
        print(f"{case['id']}: {'PASS' if trace['passed'] else 'FAIL'}"
              + (f" ({trace['error']})" if trace.get("error") else ""), flush=True)
    output.mkdir(parents=True)
    trace_path = output / "traces.jsonl"
    trace_path.write_text("".join(json.dumps(t, allow_nan=False) + "\n" for t in traces))
    report = {"schema": "hackring.skill_eval/1", "evaluation_series": args.series, "model": args.model,
              "plant": "LocalMachine on checked-in PyAT ring with seeded error kicks",
              "inference": "Transformers CPU greedy; no forced tool prefix; numeric solve by checked-in response teacher",
              "cases_sha256": digest(cases_path), "evaluator_sha256": digest(Path(__file__)),
              "deck_sha256": digest(ROOT / "data/facility/decks/RING.json"),
              "response_teacher_sha256": digest(ROOT / "datasets/v002/response_teacher.json"),
              "total": len(traces), "passed": sum(t["passed"] for t in traces),
              "route_correct": sum(t.get("selected_plane") == t["expected_plane"] for t in traces),
              "intent_correct": sum(t.get("selected_plane") == t["expected_plane"] and
                                    t.get("selected_effort") == requested_effort(t["prompt"]) for t in traces),
              "split": args.split,
              "by_plane": {plane: {"passed": sum(t["passed"] for t in traces if t["expected_plane"] == plane),
                                   "total": sum(t["expected_plane"] == plane for t in traces)}
                           for plane in ("x", "y", "both")},
              "errors": dict(Counter(t.get("error", "none") for t in traces)),
              "traces_sha256": digest(trace_path)}
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / "evals/mark-i/pyat-v002/cases.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "evals/mark-ii/pyat-v004/skill-v004")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--series", choices=("mark-ii", "mark-iii"), default="mark-ii")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=("train", "validation", "test"))
    print(json.dumps(run(parser.parse_args()), indent=2))
