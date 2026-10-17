"""Add one checked, pre-action-only GPT paraphrase per v002 measured task."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

from export_control_conversations import canonical_prompt
from generate_control_dataset import digest
from generate_dataset import BPMS, ROOT

MODEL = "gpt-6-luna"
SCHEMA = "hackring.control_paraphrases/1"
STYLES = (
    "a concise direct request",
    "a polite request from an operator",
    "an informal but precise request",
    "a brief shift-operator instruction",
    "a measured-outcome-focused request",
    "a straightforward diagnostic request",
    "a short task-first request",
    "a natural conversational request",
)


def style_for(task_id: str) -> str:
    index = int.from_bytes(hashlib.sha256(task_id.encode()).digest()[:4], "big") % len(STYLES)
    return STYLES[index]


def validate_prompt(prompt: str, row: dict) -> None:
    if not isinstance(prompt, str) or "\n" in prompt or not 6 <= len(prompt.split()) <= 55:
        raise ValueError("prompt must be one or two short sentences")
    if len(re.split(r"(?<=[.!?])\s+", prompt.strip())) > 2:
        raise ValueError("prompt has more than two sentences")
    if re.search(r"\b(improved|worsened|confirmed|already corrected|was reduced)\b", prompt, re.I):
        raise ValueError("prompt claims an outcome")
    if "cal_offset" in prompt or "hidden" in prompt.lower():
        raise ValueError("prompt mentions a hidden fault")
    if re.search(r"\b(do not|don't|never|without)\b", prompt, re.I):
        raise ValueError("prompt negates the requested action")
    goal = row["goal"]
    if goal == "exact_write":
        original = canonical_prompt(row)
        sp = row["write"]["pvs_written"][0]
        bpm = next(pv for pv in BPMS if pv in original)
        if prompt.count(sp) != 1 or prompt.count(bpm) != 1:
            raise ValueError("exact write must preserve its setpoint and follow-up BPM")
        prose = prompt.replace(sp, "").replace(bpm, "")
        numbers = re.findall(r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])", prose)
        expected = row["write"]["setpoints_mrad"][sp]
        if len(numbers) != 1 or abs(float(numbers[0]) - expected) > 5e-7 or "mrad" not in prose.lower():
            raise ValueError("exact write changed the requested kick or unit")
        if not re.search(r"\b(set|write|change|adjust)\b", prompt, re.I):
            raise ValueError("exact write omitted its action")
    else:
        if not re.search(r"\b(beam|orbit|bpm)\b", prompt, re.I):
            raise ValueError("correction prompt omitted the beam or orbit target")
        if re.search(r"RING:|\bBPM\d+\b|(?<!\w)\d+(?:\.\d+)?", prompt, re.I):
            raise ValueError("correction prompt invented a PV or numeric claim")
        if not re.search(r"\b(correct|center|centre|reduce|minimize|minimise|flatten|restore)\b", prompt, re.I):
            raise ValueError("correction prompt omitted its goal")
        x = bool(re.search(r"\b(horizontal|x-plane|x orbit)\b", prompt, re.I))
        y = bool(re.search(r"\b(vertical|y-plane|y orbit)\b", prompt, re.I))
        if goal == "x" and (not x or y):
            raise ValueError("prompt changed the X-only goal")
        if goal == "y" and (not y or x):
            raise ValueError("prompt changed the Y-only goal")
        if goal == "both" and not (x and y or re.search(r"\b(both|whole|entire)\b", prompt, re.I)):
            raise ValueError("prompt omitted the two-plane goal")


def request(row: dict, model: str) -> dict:
    canonical = canonical_prompt(row)
    instructions = (
        "Paraphrase the simulated storage-ring operator request in one or two short natural sentences. "
        "Preserve the exact task intent. This is PRE-ACTION language: do not claim any result, "
        "infer a cause, introduce a new device, invent a measurement, or specify an unprovided kick. "
        "For an exact write, copy both PV identifiers, the exact numeric kick, and mrad unit unchanged. "
        "For a correction goal, preserve the requested plane(s) explicitly and include no PV names or numbers. "
        "Keep the request to read or report the measured follow-up. Vary phrasing naturally according to the "
        "given style cue, without adding facts or changing the task. Return only the requested JSON field."
    )
    params = {"model": model, "instructions": instructions,
              "input": json.dumps({"task": row["goal"], "canonical_operator_request": canonical,
                                   "style_cue": style_for(row["task_id"])}),
              "text": {"format": {"type": "json_schema", "name": "control_paraphrase", "strict": True,
                                  "schema": {"type": "object", "properties": {"operator_prompt": {"type": "string"}},
                                             "required": ["operator_prompt"], "additionalProperties": False}}},
              "max_output_tokens": 120, "store": False}
    if model == MODEL:
        params["reasoning"] = {"effort": "none"}
    return params


def exact_write_fallback(row: dict) -> str:
    canonical = canonical_prompt(row)
    prompt = "Please " + canonical[0].lower() + canonical[1:]
    validate_prompt(prompt, row)
    return prompt


async def generate(api, row: dict, model: str) -> tuple[str, str]:
    params = request(row, model)
    last_error = None
    for attempt in range(3):
        response = await api.responses.create(**params)
        try:
            if response.status != "completed":
                raise ValueError(f"response status {response.status}")
            prompt = json.loads(response.output_text)["operator_prompt"].strip()
            validate_prompt(prompt, row)
            if prompt == canonical_prompt(row) and row["goal"] != "exact_write":
                raise ValueError("model repeated the canonical request")
            return prompt, "gpt"
        except (ValueError, KeyError, TypeError) as exc:
            last_error = exc
            params["instructions"] = request(row, model)["instructions"] + f" Previous candidate was rejected: {exc}."
    if row["goal"] == "exact_write":
        return exact_write_fallback(row), "deterministic_fallback"
    raise ValueError(f"three invalid paraphrases for {row['task_id']}: {last_error}")


async def collect(rows: list[dict], completed: dict[str, str], sources: dict[str, str],
                  progress: Path, model: str, workers: int) -> None:
    from openai import AsyncOpenAI

    queue = asyncio.Queue()
    for row in rows:
        if row["task_id"] not in completed:
            queue.put_nowait(row)
    failures = []
    async with AsyncOpenAI(max_retries=4, timeout=60) as api:
        async def worker() -> None:
            while not queue.empty():
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    prompt, source = await generate(api, row, model)
                    with progress.open("a") as stream:
                        stream.write(json.dumps({"task_id": row["task_id"], "operator_prompt": prompt,
                                                 "source": source}) + "\n")
                        stream.flush()
                    completed[row["task_id"]] = prompt
                    sources[row["task_id"]] = source
                    if len(completed) % 50 == 0 or len(completed) == len(rows):
                        print(f"Paraphrased {len(completed)}/{len(rows)}", flush=True)
                except Exception as exc:
                    failures.append(f"{row['task_id']}: {exc}")
                finally:
                    queue.task_done()
        await asyncio.gather(*(worker() for _ in range(min(workers, queue.qsize()))))
    if failures:
        raise RuntimeError("Paraphrase pass incomplete; rerun the same command to resume. " + "; ".join(failures[:5]))


def run(args: argparse.Namespace) -> None:
    raw = args.raw.resolve()
    manifest = json.loads((raw / "manifest.json").read_text())
    actions = raw / "actions.jsonl"
    if digest(actions) != manifest["files_sha256"]["actions.jsonl"]:
        raise ValueError("raw v002 actions differ from manifest")
    rows = [json.loads(line) for line in actions.read_text().splitlines()]
    if args.limit:
        rows = rows[:args.limit]
    expected = {"schema": SCHEMA, "source_actions_sha256": digest(actions), "model": args.model,
                "script_sha256": digest(Path(__file__)), "selected_ids": [row["task_id"] for row in rows]}
    output = args.output.resolve()
    settings = output / "run.json"
    prior_script_hashes = []
    if output.exists():
        stored = json.loads(settings.read_text())
        for key in ("schema", "source_actions_sha256", "model", "selected_ids"):
            if stored.get(key) != expected[key]:
                raise ValueError("paraphrase run settings changed; select a new output")
        prior_script_hashes = stored.get("script_sha256_history", [])
        if stored.get("script_sha256") != expected["script_sha256"]:
            if not args.resume_after_edit:
                raise ValueError("paraphrase script changed; review it, then rerun with --resume-after-edit")
            prior_script_hashes = [*prior_script_hashes, stored["script_sha256"]]
    else:
        output.mkdir(parents=True)
        settings.write_text(json.dumps(expected, indent=2) + "\n")
    progress = output / "progress.jsonl"
    completed = {}
    sources = {}
    if progress.exists():
        lookup = {row["task_id"]: row for row in rows}
        for line in progress.read_text().splitlines():
            note = json.loads(line)
            task_id = note["task_id"]
            if task_id not in lookup or task_id in completed:
                raise ValueError("unexpected or duplicate progress row")
            validate_prompt(note["operator_prompt"], lookup[task_id])
            completed[task_id] = note["operator_prompt"]
            source = note.get("source", "gpt")
            if source not in ("gpt", "deterministic_fallback"):
                raise ValueError("unknown paraphrase source")
            sources[task_id] = source
    if output.exists() and settings.exists() and json.loads(settings.read_text()).get("script_sha256") != expected["script_sha256"]:
        settings.write_text(json.dumps({**expected, "script_sha256_history": prior_script_hashes}, indent=2) + "\n")
    if len(completed) < len(rows):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before the direct API paraphrase pass")
        asyncio.run(collect(rows, completed, sources, progress, args.model, args.workers))
    final = output / "paraphrases.jsonl"
    expected_text = "".join(json.dumps({"task_id": row["task_id"],
                                        "operator_prompt": completed[row["task_id"]],
                                        "source": sources[row["task_id"]]}) + "\n" for row in rows)
    if final.exists():
        if final.read_text() != expected_text:
            raise ValueError("existing completed paraphrases differ from validated progress")
    else:
        temporary = output / "paraphrases.jsonl.tmp"
        temporary.write_text(expected_text)
        temporary.replace(final)
    manifest_text = json.dumps({**expected, "script_sha256_history": prior_script_hashes,
                                "deterministic_fallback_task_ids": [row["task_id"] for row in rows
                                                                     if sources[row["task_id"]] == "deterministic_fallback"],
                                "paraphrases_sha256": digest(final)}, indent=2) + "\n"
    existing_manifest = output / "manifest.json"
    if existing_manifest.exists() and existing_manifest.read_text() != manifest_text:
        raise ValueError("existing paraphrase manifest differs from completed output")
    existing_manifest.write_text(manifest_text)
    print(f"Saved {len(rows)} checked paraphrases to {final}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "datasets/v002/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/v002/annotations/gpt-v001")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume-after-edit", action="store_true",
                        help="After reviewing a script change, validate saved prompts and preserve prior script hash")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
