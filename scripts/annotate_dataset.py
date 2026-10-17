"""Annotate immutable Xopt attempts through direct OpenAI Responses calls."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import math
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/v001"
DEFAULT_MODEL = "gpt-6-luna"
SCHEMA = "hackring.operator_annotation/2"
PILOT_SEED = 20261017
STYLE_CUES = (
    "a terse control-room directive",
    "a conversational request to the assistant",
    "open with the supplied BPM feature, then ask for correction",
    "open with the desired orbit state, then mention the supplied BPM feature",
    "compact operator shorthand without politeness filler",
    "ask for a careful incremental adjustment",
    "ask for the orbit to be corrected and the result reported",
    "a technical request using the relevant BPM or plane name",
)

PROMPT_INSTRUCTIONS = (
    "Write a natural operator request for one simulated storage-ring correction episode. "
    "The assistant will use public PV reads, bounded corrector writes, and follow-up reads "
    "to work toward the supplied goal. Use only the INITIAL public PVs and the supplied "
    "factual focus; no later action or result is available. Follow the style cue loosely, "
    "and mention the focused BPM and its plane, or the focused plane alone. Assert only "
    "the supplied factual focus. Let wording vary naturally: request an inspection and "
    "correction, a small adjustment, or a correction with before/after feedback. Do not "
    "force every request to mention kick size or a follow-up check; these are available "
    "through the tool workflow. Avoid the repeated template 'cautious correction ... "
    "corrector kicks modest ... check the result'. Do not invent a "
    "new objective, hidden fault, cause, outcome, knob setting, or measurement. Do not put "
    "numeric measurement values in prose; the exact values remain in the PV record. "
    "Use one or two short sentences, about 8–40 words total. Avoid stock phrasing. "
    "BPM positions are mm; corrector settings are mrad."
)
OBSERVATION_INSTRUCTIONS = (
    "Write one or two short factual sentences about this simulated read-write-read "
    "transition. Use only supplied public PVs and objective values. Include the exact "
    "supplied orbit_cost_trend word: improved, worsened, or unchanged. Describe the "
    "measured result, even when it worsened; do not praise a worsening write. Do not "
    "infer causes, hidden faults, intent, or later results. A setpoint does not move "
    "between correctors; describe each written device separately if naming devices. "
    "Use operator-facing prose, never JSON field names or underscores. Do not put numeric "
    "measurement values in prose; exact values remain in the PV record. Keep to about 8–40 words."
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


def episode_starts(attempts: Path, rows: list[dict]) -> dict[str, dict]:
    selected = {row["episode_id"] for row in rows}
    starts = {}
    for row, _ in records(attempts):
        episode_id = row["episode_id"]
        if episode_id in selected and episode_id not in starts:
            if row["step"] != 0:
                raise ValueError(f"episode {episode_id} has no initial step")
            starts[episode_id] = row
    if starts.keys() != selected:
        raise ValueError("selected episode is missing its initial attempt")
    return starts


def source(dataset: Path) -> tuple[Path, dict]:
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    attempts = dataset / "attempts.jsonl"
    if digest(attempts) != manifest["files_sha256"]["attempts.jsonl"]:
        raise ValueError("attempts.jsonl differs from its immutable source manifest")
    return attempts, manifest


def selected_rows(attempts: Path, limit: int | None, sample: int | None = None) -> list[dict]:
    if (limit is not None and limit < 1) or (sample is not None and sample < 1):
        raise ValueError("--limit and --sample must be positive")
    if limit is not None and sample is not None:
        raise ValueError("choose either --limit or --sample")
    rows = []
    seen = set()
    for row, _ in records(attempts):
        request_id = key(row)
        if request_id in seen:
            raise ValueError(f"duplicate attempt ID: {request_id}")
        seen.add(request_id)
        rows.append(row)
    if sample is not None:
        if sample > len(rows):
            raise ValueError(f"--sample exceeds {len(rows)} source attempts")
        chosen = set(random.Random(PILOT_SEED).sample(range(len(rows)), sample))
        return [row for index, row in enumerate(rows) if index in chosen]
    if limit is not None:
        return rows[:limit]
    return rows


def rounded_pvs(pvs: dict[str, float]) -> dict[str, float]:
    return {name: round(value, 6) for name, value in pvs.items()}


def prompt_attributes(row: dict) -> dict[str, str]:
    readings = {name: value for name, value in row["read_before"].items()
                if name.startswith("RING:DIAG:BPM")}
    worst_name, worst_value = max(readings.items(), key=lambda item: abs(item[1]))
    plane_values = {axis: [value for name, value in readings.items() if name.endswith(f":{axis}")]
                    for axis in ("X", "Y")}
    rms = {axis: math.sqrt(sum(value * value for value in values) / len(values))
           for axis, values in plane_values.items()}
    dominant = max(rms, key=rms.get)
    focus_cues = (
        f"The largest absolute BPM reading is {worst_name}: {worst_value:+.3f} mm.",
        f"The {dominant} plane has the larger BPM RMS ({rms[dominant]:.3f} mm).",
        f"The {dominant} readings span {min(plane_values[dominant]):+.3f} to "
        f"{max(plane_values[dominant]):+.3f} mm across the BPMs.",
    )
    index = int(row["episode_id"].split("-")[-1])
    return {"style_cue": STYLE_CUES[index % len(STYLE_CUES)],
            "factual_focus": focus_cues[(index // len(STYLE_CUES)) % len(focus_cues)],
            "focus_kind": ("largest_bpm", "dominant_plane", "plane_span")
                          [(index // len(STYLE_CUES)) % len(focus_cues)],
            "focus_pv": worst_name,
            "focus_plane": dominant}


def prompt_context(row: dict) -> dict:
    # Called only with the first attempt of an episode; no future actions or outcomes.
    return {"read_before": rounded_pvs(row["read_before"]),
            **prompt_attributes(row),
            "goal": "Move measured BPM positions toward zero within the available corrector bounds."}


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
    if not isinstance(text, str) or not text.strip() or len(text) > 320:
        raise ValueError(f"{field} response is empty or too long")
    return text.strip()


def validate_text(text: str, field: str, row: dict) -> None:
    if "\n" in text or not 1 <= len(re.split(r"(?<=[.!?])\s+", text)) <= 2:
        raise ValueError(f"{field} must be one or two sentences")
    if not 6 <= len(text.split()) <= 55:
        raise ValueError(f"{field} has implausible length")
    if "_" in text:
        raise ValueError(f"{field} exposes a schema field name")
    if field == "observation" and re.search(
        r"\b(?:moved|transferred|shifted)\b.{0,60}\b[HV]\d{2}\b.{0,30}\bto\b.{0,20}\b[HV]\d{2}\b",
        text, re.I,
    ):
        raise ValueError("observation describes a setpoint as moving between correctors")
    # PV identifiers may contain digits, but free-standing numbers invite false precision.
    prose = re.sub(r"RING:[A-Z0-9:]+|\bBPM\d+\b", "", text, flags=re.I)
    if re.search(r"(?<!\w)[+-]?\d+(?:\.\d+)?\b", prose):
        raise ValueError(f"{field} contains a numeric claim")
    bpm_names = {re.search(r"BPM\d+", name).group().upper()
                 for name in row["read_before"] if "BPM" in name}
    if any(name.upper() not in bpm_names for name in re.findall(r"\bBPM\d+\b", text, re.I)):
        raise ValueError(f"{field} names an unknown BPM")
    if field == "operator_prompt":
        focus = prompt_attributes(row)
        if focus["focus_kind"] == "largest_bpm":
            target = re.search(r"BPM\d+", focus["focus_pv"]).group()
            if not re.search(rf"\b{target}\b", text, re.I):
                raise ValueError("operator_prompt omits its factual BPM focus")
            plane = focus["focus_pv"].split(":")[-1]
            words = (r"\bX\b|horizontal" if plane == "X" else r"\bY\b|vertical")
            if not re.search(words, text, re.I):
                raise ValueError("operator_prompt omits the focused BPM plane")
        else:
            plane = focus["focus_plane"]
            words = (r"\bX\b|horizontal" if plane == "X" else r"\bY\b|vertical")
            if not re.search(words, text, re.I):
                raise ValueError("operator_prompt omits its factual plane focus")
    elif field == "observation":
        before = row["before_metrics"]["orbit_cost"]
        after = row["after_metrics"]["orbit_cost"]
        trend = "improved" if after < before - 1e-12 else "worsened" if after > before + 1e-12 else "unchanged"
        trend_words = set(re.findall(r"\b(?:improved|worsened|unchanged)\b", text, re.I))
        if {word.lower() for word in trend_words} != {trend}:
            raise ValueError(f"observation does not state only the measured trend: {trend}")
        if not re.search(r"\b(?:orbit )?cost\b", text, re.I):
            raise ValueError("observation must identify the measured orbit cost")


def run_settings(dataset: Path, attempts: Path, model: str, rows: list[dict]) -> dict:
    selected_ids = "\n".join(key(row) for row in rows).encode()
    return {"schema": SCHEMA, "dataset": dataset.name,
            "source_attempts_sha256": digest(attempts),
            "source_manifest_sha256": digest(dataset / "manifest.json"),
            "annotator_sha256": digest(Path(__file__)),
            "model": model, "selected_count": len(rows),
            "episode_count": len({row["episode_id"] for row in rows}),
            "selected_ids_sha256": hashlib.sha256(selected_ids).hexdigest(),
            "first_id": key(rows[0]), "last_id": key(rows[-1]),
            "created_utc": datetime.now(timezone.utc).isoformat()}


def existing_progress(path: Path, expected: set[str], fields: set[str]) -> dict[str, dict]:
    progress = {}
    if path.exists():
        for record, _ in records(path):
            request_id = record.get("custom_id")
            if request_id not in expected or request_id in progress:
                raise ValueError(f"unexpected or duplicate progress ID: {request_id}")
            allowed = {"custom_id", *fields}
            if not set(record) in (allowed, allowed | {"source"}):
                raise ValueError(f"malformed progress record: {request_id}")
            if "source" in record and (fields != {"observation"} or
                                       record["source"] not in {"model", "deterministic_fallback"}):
                raise ValueError(f"invalid progress source: {request_id}")
            progress[request_id] = record
    return progress


def deterministic_observation(row: dict) -> str:
    before = row["before_metrics"]["orbit_cost"]
    after = row["after_metrics"]["orbit_cost"]
    trend = "improved" if after < before - 1e-12 else "worsened" if after > before + 1e-12 else "unchanged"
    verb = f"was {trend}" if trend == "unchanged" else trend
    result = f"The measured orbit cost {verb} after the corrector write and follow-up read."
    validate_text(result, "observation", row)
    return result


async def generate(api, row: dict, model: str, field: str) -> tuple[str, str]:
    params = request(model, field, row)
    for attempt in range(3):
        response = await api.responses.create(**params)
        try:
            result = parse_response(response, field)
            validate_text(result, field, row)
            return result, "model"
        except ValueError as exc:
            if attempt == 2:
                if field == "observation":
                    return deterministic_observation(row), "deterministic_fallback"
                raise
            params["instructions"] = (request(model, field, row)["instructions"]
                                       + f" A previous candidate was rejected: {exc}. Try again.")
    raise AssertionError("unreachable")


async def collect(args: argparse.Namespace, items: list[dict], completed: dict[str, dict],
                  field: str, progress_path: Path) -> None:
    from openai import AsyncOpenAI

    item_id = (lambda row: row["episode_id"]) if field == "operator_prompt" else key
    pending = [row for row in items if item_id(row) not in completed]
    if not pending:
        return
    queue = asyncio.Queue()
    for row in pending:
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
                    result, source = await generate(api, row, args.model, field)
                    annotation = {"custom_id": item_id(row), field: result}
                    if field == "observation":
                        annotation["source"] = source
                    with progress_path.open("a") as stream:
                        stream.write(json.dumps(annotation, allow_nan=False) + "\n")
                        stream.flush()
                    completed[annotation["custom_id"]] = annotation
                    if len(completed) % 50 == 0 or len(completed) == len(items):
                        print(f"{field}: {len(completed)}/{len(items)}", flush=True)
                except Exception as exc:
                    failures.append((item_id(row), str(exc)))
                finally:
                    queue.task_done()

        await asyncio.gather(*(worker() for _ in range(min(args.workers, len(pending)))))
    if failures:
        print(f"{len(failures)} {field} calls failed; rerun the same command to resume.")
        for request_id, error in failures[:5]:
            print(f"  {request_id}: {error}")
        raise RuntimeError("annotation incomplete")


def finalize(args: argparse.Namespace, rows: list[dict], prompts: dict[str, dict],
             observations: dict[str, dict], run: dict) -> None:
    if len(observations) != len(rows) or len(prompts) != run["episode_count"]:
        raise ValueError("annotation is incomplete")
    output = args.output / "annotated_attempts.jsonl"
    if output.exists():
        raise FileExistsError(output)
    with output.open("w") as stream:
        for row in rows:
            episode_id = row["episode_id"]
            note = observations[key(row)]
            annotated = {**row, "annotation": {"schema": SCHEMA,
                                                "custom_id": note["custom_id"],
                                                "model": run["model"],
                                                "episode_prompt_id": episode_id,
                                                "operator_prompt": prompts[episode_id]["operator_prompt"],
                                                "observation": note["observation"],
                                                "observation_source": note.get("source", "model")}}
            stream.write(json.dumps(annotated, allow_nan=False) + "\n")
    write_json(args.output / "annotation_manifest.json", {
        **run, "annotations": len(rows), "annotated_attempts_sha256": digest(output),
        "episode_prompts_sha256": digest(args.output / "episode_prompts.jsonl"),
        "observations_sha256": digest(args.output / "observations.jsonl"),
    })
    print(f"Saved {len(rows)} complete annotated rows to {output}")


def run(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    attempts, _ = source(args.dataset)
    rows = selected_rows(attempts, args.limit, args.sample)
    if not rows:
        raise ValueError("source has no attempts")
    starts = episode_starts(attempts, rows)
    settings_path = args.output / "run.json"
    if args.output.exists():
        if not settings_path.exists():
            raise FileExistsError(f"output exists without run.json: {args.output}")
        settings = json.loads(settings_path.read_text())
        expected = run_settings(args.dataset, attempts, args.model, rows)
        for name in ("schema", "dataset", "source_attempts_sha256", "source_manifest_sha256",
                     "model", "selected_count", "episode_count", "selected_ids_sha256",
                     "first_id", "last_id"):
            if settings[name] != expected[name]:
                raise ValueError(f"cannot resume: run setting {name} changed")
        if settings["annotator_sha256"] != expected["annotator_sha256"]:
            if not args.resume_after_edit:
                raise ValueError("annotator changed; pass --resume-after-edit after reviewing the new script")
            settings["resume_annotator_sha256"] = expected["annotator_sha256"]
            write_json(settings_path, settings)
    else:
        args.output.mkdir(parents=True)
        settings = run_settings(args.dataset, attempts, args.model, rows)
        write_json(settings_path, settings)
    if (args.output / "annotated_attempts.jsonl").exists():
        raise FileExistsError("annotated_attempts.jsonl already exists")
    prompt_path = args.output / "episode_prompts.jsonl"
    observation_path = args.output / "observations.jsonl"
    prompts = existing_progress(prompt_path, set(starts), {"operator_prompt"})
    observations = existing_progress(observation_path, {key(row) for row in rows}, {"observation"})
    for episode_id, note in prompts.items():
        validate_text(note["operator_prompt"], "operator_prompt", starts[episode_id])
    for row in rows:
        if key(row) in observations:
            validate_text(observations[key(row)]["observation"], "observation", row)
    if len(prompts) < len(starts) or len(observations) < len(rows):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before running paid direct API annotation")
        asyncio.run(collect(args, list(starts.values()), prompts, "operator_prompt", prompt_path))
        asyncio.run(collect(args, rows, observations, "observation", observation_path))
    finalize(args, rows, prompts, observations, settings)


def preview(args: argparse.Namespace) -> None:
    attempts, _ = source(args.dataset)
    rows = selected_rows(attempts, args.limit, args.sample)
    starts = episode_starts(attempts, rows)
    for row in list(starts.values())[:8]:
        print(json.dumps({"episode_id": row["episode_id"], **prompt_attributes(row)}, indent=2))
    print(f"{len(rows)} attempts across {len(starts)} episodes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "run"))
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=DATASET / "annotations/direct-v003")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, help="annotate the first N attempts for a pilot")
    parser.add_argument("--sample", type=int, help="deterministically sample N attempts across episodes")
    parser.add_argument("--workers", type=int, default=2, help="maximum concurrent attempts (default: 2)")
    parser.add_argument("--resume-after-edit", action="store_true",
                        help="resume same-schema checkpoints after a reviewed annotator edit")
    args = parser.parse_args()
    globals()[args.command](args)


if __name__ == "__main__":
    main()
