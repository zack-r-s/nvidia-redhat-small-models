"""Export measured v002 actions as Liquid LFM2.5 tool-call SFT conversations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from generate_control_dataset import digest
from generate_dataset import BPMS, ROOT, SETPOINTS
from validate_control_dataset import validate

TOOLS = [
    {"type": "function", "function": {"name": "channel_read", "description": "Read current PV values.",
     "parameters": {"type": "object", "properties": {"channels": {"type": "array", "items": {"type": "string"}},
                                                 "include_metadata": {"type": "boolean"}}, "required": ["channels"]}}},
    {"type": "function", "function": {"name": "channel_write", "description": "Write corrector PV setpoints.",
     "parameters": {"type": "object", "properties": {"operations": {"type": "array", "items": {"type": "object"}},
                                                 "confirm": {"type": "boolean"}}, "required": ["operations"]}}},
]
SYSTEM = (
    "You operate a simulated FODO storage ring. Use channel_read before a write and read BPMs "
    "and corrector readbacks after it. BPM positions are mm; absolute corrector kicks are mrad "
    "and limited to ±0.35. Correct only the plane requested by the operator. Report only measured "
    "results. List of tools: " + json.dumps(TOOLS, separators=(",", ":"))
)


def canonical_prompt(row: dict) -> str:
    goal = row["goal"]
    if goal == "x":
        return "Center the horizontal orbit and report the measured result."
    if goal == "y":
        return "Center the vertical orbit and report the measured result."
    if goal == "both":
        return "Correct the orbit in both planes and report the measured result."
    if goal == "exact_write":
        sp = row["write"]["pvs_written"][0]
        value = row["write"]["setpoints_mrad"][sp]
        bpm = next(pv for pv in BPMS if pv.endswith(":X" if ":MAG:H" in sp else ":Y"))
        return f"Set {sp} to {value:.6f} mrad, then read {bpm}."
    raise ValueError(goal)


def tool_call(name: str, **kwargs) -> str:
    arguments = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
    return f"<|tool_call_start|>[{name}({arguments})]<|tool_call_end|>"


def read_response(row: dict, channels: list[str], field: str) -> str:
    return json.dumps({"status": "success", "summary": {"readings": {
        pv: {"value": round(row[field][pv], 6)} for pv in channels}}}, separators=(",", ":"))


def write_response(row: dict, operations: list[dict]) -> str:
    return json.dumps({"status": "success", "summary": {"results": [
        {"channel": op["channel"], "outcome": "confirmed", "observed_value": row["read_after"][op["channel"]]}
        for op in operations]}}, separators=(",", ":"))


def correction_report(row: dict, bpm: list[str], operations: list[dict]) -> str:
    lines = []
    for op in operations:
        rb = op["channel"].replace(":SP", ":RB")
        lines.append(f"{op['channel']}: requested {op['value']:g} mrad; Osprey confirmed; "
                     f"readback {row['read_after'][rb]:.6f} mrad.")
    for plane in ("X", "Y"):
        channels = [pv for pv in bpm if pv.endswith(f":{plane}")]
        if channels:
            before = math.sqrt(sum(row["read_before"][pv] ** 2 for pv in channels) / len(channels))
            after = math.sqrt(sum(row["read_after"][pv] ** 2 for pv in channels) / len(channels))
            lines.append(f"{plane} RMS: {before:.4f} → {after:.4f} mm (measured).")
    return "\n".join(lines)


def conversation(row: dict, prompt: str) -> dict:
    goal = row["goal"]
    if goal != "exact_write":
        bpm = [pv for pv in BPMS if goal == "both" or pv.endswith(":X" if goal == "x" else ":Y")]
        correctors = [sp for sp in SETPOINTS if goal == "both" or
                      (":MAG:H" in sp if goal == "x" else ":MAG:V" in sp)]
        before = bpm + correctors
        operations = [{"channel": sp, "value": round(row["write"]["setpoints_mrad"][sp], 6)}
                      for sp in row["write"]["pvs_written"]]
        after = bpm + [sp.replace(":SP", ":RB") for sp in row["write"]["pvs_written"]]
        final = correction_report(row, bpm, operations)
    else:
        sp = row["write"]["pvs_written"][0]
        bpm = next(pv for pv in BPMS if pv in prompt)
        before = [sp]
        operations = [{"channel": sp, "value": round(row["write"]["setpoints_mrad"][sp], 6)}]
        after = [sp.replace(":SP", ":RB"), bpm]
        final = (f"{sp}: Osprey reported confirmed for {operations[0]['value']:g} mrad; "
                 f"observed setpoint {row['read_after'][sp]:.6f} mrad.\n"
                 f"{bpm}: {row['read_after'][bpm]:.6f} mm (read from Osprey).\n"
                 f"{after[0]}: {row['read_after'][after[0]]:.6f} mrad (post-write readback).")
    messages = [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt},
        {"role": "assistant", "content": tool_call("channel_read", channels=before)},
        {"role": "tool", "content": read_response(row, before, "read_before")},
        {"role": "assistant", "content": tool_call("channel_write", operations=operations, confirm=True)},
        {"role": "tool", "content": write_response(row, operations)},
        {"role": "assistant", "content": tool_call("channel_read", channels=after)},
        {"role": "tool", "content": read_response(row, after, "read_after")},
        {"role": "assistant", "content": final},
    ]
    return {"messages": messages}


def read_only(row: dict) -> dict:
    channels = BPMS
    prompt = "Report the beam position at all four BPMs." if row["seed"] % 2 == 0 else "Which orbit plane has the larger measured RMS?"
    if row["seed"] % 2 == 0:
        final = "Measured beam position (mm):\n" + "\n".join(
            f"BPM{number:02d}: X={row['read_before'][BPMS[2 * (number - 1)]]:+.4f}, "
            f"Y={row['read_before'][BPMS[2 * (number - 1) + 1]]:+.4f}"
            for number in range(1, 5))
    else:
        x = row["before_metrics"]["x_rms_mm"]
        y = row["before_metrics"]["y_rms_mm"]
        comparison = f"The {'X' if x > y else 'Y'} plane is larger." if x != y else "The planes are equal."
        final = f"Measured RMS: X={x:.4f} mm, Y={y:.4f} mm. {comparison}"
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt},
                         {"role": "assistant", "content": tool_call("channel_read", channels=channels)},
                         {"role": "tool", "content": read_response(row, channels, "read_before")},
                         {"role": "assistant", "content": final}]}


def export(raw: Path, output: Path, paraphrases: Path | None) -> dict:
    quality = validate(raw)
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in (raw / "actions.jsonl").read_text().splitlines()]
    notes = {}
    if paraphrases:
        from paraphrase_control_prompts import validate_prompt
        lookup = {row["task_id"]: row for row in rows}
        for line in paraphrases.read_text().splitlines():
            note = json.loads(line)
            if note["task_id"] not in lookup or note["task_id"] in notes:
                raise ValueError("duplicate paraphrase")
            validate_prompt(note["operator_prompt"], lookup[note["task_id"]])
            notes[note["task_id"]] = note["operator_prompt"]
        if set(notes) != {row["task_id"] for row in rows}:
            raise ValueError("paraphrases must cover exactly the 512 measured tasks")
    output.mkdir(parents=True)
    counts = {}
    index_file = output / "index.jsonl"
    with index_file.open("w") as index_stream:
        streams = {split: (output / f"{split}.jsonl").open("w") for split in ("train", "validation", "test")}
        try:
            for row in rows:
                if row["goal"] != "exact_write" and row["task_id"] in quality["non_improving_corrections"]:
                    continue
                for variant, prompt in (("canonical", canonical_prompt(row)),
                                        *(([("gpt", notes[row["task_id"]])] if notes else []))):
                    encoded = conversation(row, prompt)
                    streams[row["split"]].write(json.dumps(encoded, allow_nan=False) + "\n")
                    index_stream.write(json.dumps({"task_id": row["task_id"], "split": row["split"],
                                                   "variant": variant}) + "\n")
                    counts[row["split"]] = counts.get(row["split"], 0) + 1
                if row["goal"] == "x":
                    streams[row["split"]].write(json.dumps(read_only(row), allow_nan=False) + "\n")
                    index_stream.write(json.dumps({"task_id": row["task_id"], "split": row["split"],
                                                   "variant": "read_only"}) + "\n")
                    counts[row["split"]] += 1
        finally:
            for stream in streams.values():
                stream.close()
    manifest = {"schema": "hackring.lfm_conversations/1", "source_actions_sha256": digest(raw / "actions.jsonl"),
                "source_manifest_sha256": digest(raw / "manifest.json"),
                "paraphrases_sha256": digest(paraphrases) if paraphrases else None,
                "language_complete": bool(paraphrases), "counts": counts,
                "files_sha256": {path.name: digest(path) for path in output.iterdir() if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "datasets/v002/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/v002/sft")
    parser.add_argument("--paraphrases", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.raw.resolve(), args.output.resolve(),
                            args.paraphrases.resolve() if args.paraphrases else None), indent=2))


if __name__ == "__main__":
    main()
