"""Prepare, run, and join GPT Batch annotations for immutable Xopt attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/v001"
SCHEMA = "hackring.operator_annotation/1"
INSTRUCTIONS = (
    "You write short, plausible accelerator-operator language for synthetic orbit correction. "
    "Return exactly the requested JSON. The operator_prompt is a request an operator could have "
    "made BEFORE the action: base it only on the pre-action BPM readings and the standing goal "
    "of centering the orbit with modest corrector kicks. Vary natural phrasing, but do not name "
    "a specific corrector command or predict its outcome. The observation is a retrospective "
    "sentence about the supplied read/write/read transition. Describe worsening or no change "
    "honestly. Do not invent causes, hidden faults, unseen PVs, or physics measurements. "
    "BPM positions are in mm and corrector settings are in mrad."
)
FORMAT = {
    "type": "json_schema",
    "name": "operator_annotation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "operator_prompt": {"type": "string"},
            "observation": {"type": "string"},
        },
        "required": ["operator_prompt", "observation"],
        "additionalProperties": False,
    },
}


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
    manifest = json.loads((dataset / "manifest.json").read_text())
    attempts = dataset / "attempts.jsonl"
    if digest(attempts) != manifest["files_sha256"]["attempts.jsonl"]:
        raise ValueError("attempts.jsonl differs from the immutable source manifest")
    return attempts, manifest


def compact_pvs(pvs: dict) -> dict[str, float]:
    return {name.split("RING:DIAG:", 1)[1]: round(value, 5)
            for name, value in pvs.items() if name.startswith("RING:DIAG:")}


def prepare(args: argparse.Namespace) -> None:
    attempts, source_manifest = source(args.dataset)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.output.exists():
        raise FileExistsError(f"annotation run already exists: {args.output}")
    args.output.mkdir(parents=True)
    seen: set[str] = set()
    count = 0
    with (args.output / "batch_input.jsonl").open("w") as out:
        for row, _ in records(attempts):
            request_id = key(row)
            if request_id in seen:
                raise ValueError(f"duplicate attempt ID: {request_id}")
            seen.add(request_id)
            # Hidden episode faults and other model-only metadata never enter the request.
            context = {
                "before_bpm_mm": compact_pvs(row["read_before"]),
                "written_setpoints_mrad": {
                    name: round(row["write"]["setpoints_mrad"][name], 5)
                    for name in row["write"]["pvs_written"]
                },
                "after_bpm_mm": compact_pvs(row["read_after"]),
                "before_orbit_cost": round(row["before_metrics"]["orbit_cost"], 7),
                "after_orbit_cost": round(row["after_metrics"]["orbit_cost"], 7),
            }
            request = {
                "custom_id": request_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": args.model,
                    "instructions": INSTRUCTIONS,
                    "input": json.dumps(context, separators=(",", ":")),
                    "text": {"format": FORMAT},
                    "max_output_tokens": 180,
                },
            }
            out.write(json.dumps(request, separators=(",", ":"), allow_nan=False) + "\n")
            count += 1
            if args.limit is not None and count >= args.limit:
                break
    write_json(args.output / "run.json", {
        "schema": SCHEMA, "dataset": args.dataset.name,
        "source_attempts_sha256": digest(attempts),
        "source_manifest_sha256": digest(args.dataset / "manifest.json"),
        "source_attempt_count": source_manifest["attempts"],
        "request_count": count, "model": args.model,
        "batch_input_sha256": digest(args.output / "batch_input.jsonl"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Prepared {count} Batch requests in {args.output / 'batch_input.jsonl'}")


def client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the optional OpenAI SDK: uv pip install --python .venv/bin/python openai") from exc
    return OpenAI()


def submit(args: argparse.Namespace) -> None:
    run = json.loads((args.output / "run.json").read_text())
    if (args.output / "batch.json").exists():
        raise FileExistsError("batch.json already exists; refusing to submit twice")
    path = args.output / "batch_input.jsonl"
    if digest(path) != run["batch_input_sha256"]:
        raise ValueError("batch input changed after preparation")
    api = client()
    with path.open("rb") as stream:
        uploaded = api.files.create(file=stream, purpose="batch")
    batch = api.batches.create(
        input_file_id=uploaded.id, endpoint="/v1/responses", completion_window="24h",
        metadata={"dataset": run["dataset"], "purpose": "operator-annotation"},
    )
    write_json(args.output / "batch.json", {
        "id": batch.id, "input_file_id": uploaded.id, "status": batch.status,
    })
    print(f"Submitted {run['request_count']} requests as {batch.id} ({batch.status})")


def status(args: argparse.Namespace) -> None:
    batch_id = json.loads((args.output / "batch.json").read_text())["id"]
    batch = client().batches.retrieve(batch_id)
    print(json.dumps({"id": batch.id, "status": batch.status,
                      "request_counts": batch.request_counts.model_dump() if batch.request_counts else None,
                      "output_file_id": batch.output_file_id,
                      "error_file_id": batch.error_file_id}, indent=2))


def fetch(args: argparse.Namespace) -> None:
    batch_id = json.loads((args.output / "batch.json").read_text())["id"]
    api = client()
    batch = api.batches.retrieve(batch_id)
    if batch.status != "completed" or not batch.output_file_id:
        raise RuntimeError(f"batch is {batch.status}; wait until completed")
    output = args.output / "batch_output.jsonl"
    if output.exists():
        raise FileExistsError(output)
    output.write_bytes(api.files.content(batch.output_file_id).content)
    if batch.error_file_id:
        (args.output / "batch_errors.jsonl").write_bytes(api.files.content(batch.error_file_id).content)
    print(f"Saved Batch result to {output}")


def response_text(body: dict) -> str:
    if body.get("status") != "completed":
        raise ValueError(f"incomplete response: {body.get('status')}")
    parts = [part["text"] for item in body.get("output", [])
             if item.get("type") == "message" for part in item.get("content", [])
             if part.get("type") == "output_text"]
    if len(parts) != 1:
        raise ValueError(f"expected one output_text part, got {len(parts)}")
    return parts[0]


def join(args: argparse.Namespace) -> None:
    attempts, _ = source(args.dataset)
    run = json.loads((args.output / "run.json").read_text())
    if digest(attempts) != run["source_attempts_sha256"] or digest(args.dataset / "manifest.json") != run["source_manifest_sha256"]:
        raise ValueError("source dataset changed after annotation preparation")
    if (args.output / "annotated_attempts.jsonl").exists():
        raise FileExistsError("annotated_attempts.jsonl already exists")
    annotations = {}
    for result, _ in records(args.output / "batch_output.jsonl"):
        request_id = result["custom_id"]
        if request_id in annotations:
            raise ValueError(f"duplicate Batch result: {request_id}")
        reply = result.get("response")
        if result.get("error") or not reply or reply["status_code"] != 200:
            raise ValueError(f"failed Batch result: {request_id}")
        annotation = json.loads(response_text(reply["body"]))
        if set(annotation) != {"operator_prompt", "observation"} or any(
            not isinstance(value, str) or not value.strip() or len(value) > 500
            for value in annotation.values()
        ):
            raise ValueError(f"invalid annotation: {request_id}")
        annotations[request_id] = annotation
    expected = {item["custom_id"] for item, _ in records(args.output / "batch_input.jsonl")}
    if set(annotations) != expected or len(expected) != run["request_count"]:
        raise ValueError(f"Batch coverage mismatch: {len(annotations)} results for {len(expected)} requests")
    count = 0
    out_path = args.output / "annotated_attempts.jsonl"
    with out_path.open("w") as out:
        for row, _ in records(attempts):
            request_id = key(row)
            if request_id in annotations:
                out.write(json.dumps({"schema": SCHEMA, "custom_id": request_id,
                                      "episode_id": row["episode_id"], "step": row["step"],
                                      "split": row["split"], **annotations[request_id]},
                                     allow_nan=False) + "\n")
                count += 1
    write_json(args.output / "annotation_manifest.json", {
        **run, "annotations": count, "batch_output_sha256": digest(args.output / "batch_output.jsonl"),
        "annotated_attempts_sha256": digest(out_path),
    })
    print(f"Joined {count} annotations to {out_path}; raw attempts unchanged")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "submit", "status", "fetch", "join"))
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=DATASET / "annotations/gpt-v001")
    parser.add_argument("--model", help="API model for prepare; select a structured-output capable model")
    parser.add_argument("--limit", type=int, help="Prepare only this many rows for a pilot")
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.model:
            parser.error("prepare requires --model")
        prepare(args)
    else:
        globals()[args.command](args)


if __name__ == "__main__":
    main()
