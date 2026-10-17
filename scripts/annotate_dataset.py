"""Annotate immutable Xopt attempts through direct OpenAI Responses calls."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/v001"
DEFAULT_MODEL = "gpt-6-luna"
SCHEMA = "hackring.operator_annotation/1"

PROMPT_INSTRUCTIONS = (
    "Write one short, natural operator request for a simulated storage ring. "
    "The operator wants to bring the measured orbit toward center using modest corrector kicks. "
    "Refer to one real feature of the supplied PRE-ACTION BPM readings so the request is specific "
    "to this state. Use only those pre-action public PV readings. Do not prescribe a specific knob value, "
    "claim to know a hidden fault, predict an outcome, or mention a later measurement. "
    "Vary the wording naturally. BPM positions are mm; corrector settings are mrad."
)
OBSERVATION_INSTRUCTIONS = (
    "Write one short factual sentence describing this simulated read-write-read transition. "
    "Use only the supplied public PVs and objective values. State whether the measured orbit "
    "cost improved, worsened, or did not change. Do not infer causes, hidden faults, or intent. "
    "BPM positions are mm; corrector settings are mrad."
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def records(path: Path):
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                yield json.loads(line), number


def key(row: dict) -> str:
    return f"{row['episode_id']}-step-{row['step']:04d}"


def source(dataset: Path) -> tuple[Path, dict]:
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    attempts = dataset / "attempts.jsonl"
    if digest(attempts) != manifest["files_sha256"]["attempts.jsonl"]:
        raise ValueError("attempts.jsonl differs from its immutable source manifest")
    return attempts, manifest


def selected_rows(attempts: Path, limit: int | None) -> list[dict]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    rows = []
    seen = set()
    for row, _ in records(attempts):
        request_id = key(row)
        if request_id in seen:
            raise ValueError(f"duplicate attempt ID: {request_id}")
        seen.add(request_id)
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def rounded_pvs(pvs: dict[str, float]) -> dict[str, float]:
    return {name: round(value, 6) for name, value in pvs.items()}


def prompt_context(row: dict) -> dict:
    # This API call cannot leak future actions or outcomes: it receives only the first read.
    return {"read_before": rounded_pvs(row["read_before"]),
            "goal": "Center the BPM orbit while keeping corrector kicks modest."}


def observation_context(row: dict) -> dict:
    before = row["before_metrics"]["orbit_cost"]
    after = row["after_metrics"]["orbit_cost"]
    trend = "improved" if after < before - 1e-12 else "worsened" if after > before + 1e-12 else "unchanged"
    return {
        "read_before": rounded_pvs(row["read_before"]),
        "write": {name: round(row["write"]["setpoints_mrad"][name], 6)
                  for name in row["write"]["pvs_written"]},
        "read_after": rounded_pvs(row["read_after"]),
        "orbit_cost_before": round(before, 8),
        "orbit_cost_after": round(after, 8),
        "orbit_cost_trend": trend,
    }


def response_format(field: str) -> dict:
    return {"type": "json_schema", "name": field, "strict": True,
            "schema": {"type": "object", "properties": {field: {"type": "string"}},
                       "required": [field], "additionalProperties": False}}


def request(model: str, field: str, row: dict) -> dict:
    if field == "operator_prompt":
        instructions, context = PROMPT_INSTRUCTIONS, prompt_context(row)
    elif field == "observation":
        instructions, context = OBSERVATION_INSTRUCTIONS, observation_context(row)
    else:
        raise ValueError(field)
    params = {"model": model, "instructions": instructions,
              "input": json.dumps(context, separators=(",", ":")),
              "text": {"format": response_format(field)},
              "max_output_tokens": 120, "store": False}
    if model == DEFAULT_MODEL:
        params["reasoning"] = {"effort": "none"}
    return params


def parse_response(response, field: str) -> str:
    if response.status != "completed":
        raise ValueError(f"{field} response status: {response.status}")
    value = json.loads(response.output_text)
    if set(value) != {field}:
        raise ValueError(f"{field} response has unexpected fields")
    text = value[field]
    if not isinstance(text, str) or not text.strip() or len(text) > 500:
        raise ValueError(f"{field} response is empty or too long")
    return text.strip()


def run_settings(dataset: Path, attempts: Path, model: str, rows: list[dict]) -> dict:
    return {"schema": SCHEMA, "dataset": dataset.name,
            "source_attempts_sha256": digest(attempts),
            "source_manifest_sha256": digest(dataset / "manifest.json"),
            "annotator_sha256": digest(Path(__file__)),
            "model": model, "selected_count": len(rows),
            "first_id": key(rows[0]), "last_id": key(rows[-1]),
            "created_utc": datetime.now(timezone.utc).isoformat()}


def existing_progress(path: Path, expected: set[str]) -> dict[str, dict]:
    progress = {}
    if path.exists():
        for record, _ in records(path):
            request_id = record["custom_id"]
            if request_id not in expected or request_id in progress:
                raise ValueError(f"unexpected or duplicate progress ID: {request_id}")
            if set(record) != {"custom_id", "operator_prompt", "observation"}:
                raise ValueError(f"malformed progress record: {request_id}")
            progress[request_id] = record
    return progress


async def annotate_row(api, row: dict, model: str) -> dict:
    # The operator request is generated first from pre-action PVs only.
    prompt_response = await api.responses.create(**request(model, "operator_prompt", row))
    operator_prompt = parse_response(prompt_response, "operator_prompt")
    observation_response = await api.responses.create(**request(model, "observation", row))
    observation = parse_response(observation_response, "observation")
    return {"custom_id": key(row), "operator_prompt": operator_prompt,
            "observation": observation}


async def collect(args: argparse.Namespace, rows: list[dict], completed: dict[str, dict]) -> None:
    from openai import AsyncOpenAI

    pending = [row for row in rows if key(row) not in completed]
    if not pending:
        return
    queue = asyncio.Queue()
    for row in pending:
        queue.put_nowait(row)
    failures = []
    progress_path = args.output / "progress.jsonl"
    async with AsyncOpenAI(max_retries=4, timeout=60) as api:
        async def worker() -> None:
            while not queue.empty():
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    annotation = await annotate_row(api, row, args.model)
                    with progress_path.open("a") as stream:
                        stream.write(json.dumps(annotation, allow_nan=False) + "\n")
                        stream.flush()
                    completed[annotation["custom_id"]] = annotation
                    if len(completed) % 50 == 0 or len(completed) == len(rows):
                        print(f"Annotated {len(completed)}/{len(rows)}", flush=True)
                except Exception as exc:
                    failures.append((key(row), str(exc)))
                finally:
                    queue.task_done()

        await asyncio.gather(*(worker() for _ in range(min(args.workers, len(pending)))))
    if failures:
        print(f"{len(failures)} attempts failed; rerun the same command to resume.")
        for request_id, error in failures[:5]:
            print(f"  {request_id}: {error}")
        raise RuntimeError("annotation incomplete")


def finalize(args: argparse.Namespace, rows: list[dict], completed: dict[str, dict], run: dict) -> None:
    if len(completed) != len(rows):
        raise ValueError(f"only {len(completed)}/{len(rows)} attempts have both annotations")
    output = args.output / "annotated_attempts.jsonl"
    if output.exists():
        raise FileExistsError(output)
    with output.open("w") as stream:
        for row in rows:
            note = completed[key(row)]
            annotated = {**row, "annotation": {"schema": SCHEMA,
                                                "custom_id": note["custom_id"],
                                                "model": run["model"],
                                                "operator_prompt": note["operator_prompt"],
                                                "observation": note["observation"]}}
            stream.write(json.dumps(annotated, allow_nan=False) + "\n")
    write_json(args.output / "annotation_manifest.json", {
        **run, "annotations": len(rows), "annotated_attempts_sha256": digest(output),
        "progress_sha256": digest(args.output / "progress.jsonl"),
    })
    print(f"Saved {len(rows)} complete annotated rows to {output}")


def run(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    attempts, _ = source(args.dataset)
    rows = selected_rows(attempts, args.limit)
    if not rows:
        raise ValueError("source has no attempts")
    settings_path = args.output / "run.json"
    if args.output.exists():
        if not settings_path.exists():
            raise FileExistsError(f"output exists without run.json: {args.output}")
        settings = json.loads(settings_path.read_text())
        expected = run_settings(args.dataset, attempts, args.model, rows)
        for name in ("schema", "dataset", "source_attempts_sha256", "source_manifest_sha256",
                     "annotator_sha256", "model", "selected_count", "first_id", "last_id"):
            if settings[name] != expected[name]:
                raise ValueError(f"cannot resume: run setting {name} changed")
    else:
        args.output.mkdir(parents=True)
        settings = run_settings(args.dataset, attempts, args.model, rows)
        write_json(settings_path, settings)
    if (args.output / "annotated_attempts.jsonl").exists():
        raise FileExistsError("annotated_attempts.jsonl already exists")
    completed = existing_progress(args.output / "progress.jsonl", {key(row) for row in rows})
    if len(completed) < len(rows):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before running paid direct API annotation")
        asyncio.run(collect(args, rows, completed))
    finalize(args, rows, completed, settings)


def preview(args: argparse.Namespace) -> None:
    attempts, _ = source(args.dataset)
    row = selected_rows(attempts, 1)[0]
    for field in ("operator_prompt", "observation"):
        print(json.dumps({"field": field, **request(args.model, field, row)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "run"))
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=DATASET / "annotations/direct-v001")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, help="annotate the first N attempts for a pilot")
    parser.add_argument("--workers", type=int, default=2, help="maximum concurrent attempts (default: 2)")
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
