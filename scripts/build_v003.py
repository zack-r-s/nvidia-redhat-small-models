"""Build a frozen mixed-backend SFT pass from measured local PyAT and v002 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

from export_control_conversations import BPMS, SETPOINTS, SYSTEM, conversation, tool_call
from generate_control_dataset import target
from hackring.machine import LocalMachine

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"train": (20264000, 64), "validation": (20265000, 8), "test": (20266000, 8)}
PROMPTS = {
    "x": ["Reduce the horizontal BPM offsets, then report the measured X orbit.",
          "Steer the X orbit toward center and tell me what the BPMs show."],
    "y": ["Reduce the vertical BPM offsets, then report the measured Y orbit.",
          "Steer the Y orbit toward center and tell me what the BPMs show."],
    "both": ["Center the beam in both planes and report the measured residuals.",
             "Adjust horizontal and vertical orbit together, then check the BPMs."],
}
READ_PROMPTS = ["List the measured X and Y position at BPM01 through BPM04.",
                "Give me the beam position in millimeters at every BPM."]
COMPARE_PROMPTS = ["Using all four BPMs, which plane has the higher RMS position?",
                   "Compare the measured X and Y orbit RMS and identify the larger one."]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def rms(pvs: dict[str, float], plane: str) -> float:
    names = [pv for pv in BPMS if pv.endswith(":" + plane.upper())]
    return math.sqrt(sum(pvs[pv] ** 2 for pv in names) / len(names))


def tool_read(pvs: dict[str, float], channels: list[str]) -> str:
    return json.dumps({"status": "success", "summary": {"readings": {
        pv: {"value": round(pvs[pv], 6)} for pv in channels}}}, separators=(",", ":"))


def read_conversation(before: dict[str, float], prompt: str, compare: bool) -> dict:
    if compare:
        x, y = rms(before, "x"), rms(before, "y")
        answer = f"Measured RMS: X={x:.4f} mm, Y={y:.4f} mm. The {'X' if x > y else 'Y'} plane has the larger RMS."
    else:
        answer = "Measured beam position (mm):\n" + "\n".join(
            f"BPM{i:02d}: X={before[BPMS[2*(i-1)]]:+.4f}, Y={before[BPMS[2*(i-1)+1]]:+.4f}"
            for i in range(1, 5))
    return {"messages": [
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt},
        {"role": "assistant", "content": tool_call("channel_read", channels=BPMS)},
        {"role": "tool", "content": tool_read(before, BPMS)},
        {"role": "assistant", "content": answer},
    ]}


def measured_row(seed: int, split: str, goal: str, matrix: np.ndarray) -> dict:
    machine = LocalMachine()
    before = machine.reset(seed, 0.08)["pvs"]
    rng = np.random.default_rng(seed + {"x": 1, "y": 2, "both": 3, "exact_write": 4}[goal])
    settings = target(goal, before, matrix, rng, seed)
    written = ([sp for sp in SETPOINTS if goal == "both" or
                (":MAG:H" in sp if goal == "x" else ":MAG:V" in sp)]
               if goal != "exact_write" else
               [next(sp for sp in SETPOINTS if settings[sp] != before[sp])])
    after = machine.write({sp: settings[sp] for sp in written})["state"]["pvs"]
    if goal in ("x", "y", "both"):
        planes = ("x", "y") if goal == "both" else (goal,)
        if not all(rms(after, plane) < rms(before, plane) for plane in planes):
            raise ValueError(f"non-improving measured action: {seed} {goal}")
    return {"task_id": f"local-{seed}-{goal}", "seed": seed, "split": split, "goal": goal,
            "backend": "local-pyat", "read_before": before, "read_after": after,
            "write": {"setpoints_mrad": settings, "pvs_written": written},
            "before_metrics": {"x_rms_mm": rms(before, "x"), "y_rms_mm": rms(before, "y")},
            "after_metrics": {"x_rms_mm": rms(after, "x"), "y_rms_mm": rms(after, "y")}}


def build(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    v002 = ROOT / "datasets/v002"
    teacher_path = v002 / "response_teacher.json"
    teacher = json.loads(teacher_path.read_text())
    matrix = np.asarray(teacher["matrix_mm_per_mrad"], dtype=float)
    if matrix.shape != (len(BPMS), len(SETPOINTS)):
        raise ValueError("response matrix differs from fixed PV contract")
    output.mkdir(parents=True)
    raw = output / "local_actions.jsonl"
    sources = []
    with raw.open("w") as stream:
        for split, (first, count) in SPLITS.items():
            for seed in range(first, first + count):
                for goal in ("x", "y", "both", "exact_write"):
                    row = measured_row(seed, split, goal, matrix)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    sources.append(row)
    by_split: dict[str, list[tuple[dict, dict]]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        prior_rows = [json.loads(line) for line in (v002 / "sft-canonical" / f"{split}.jsonl").read_text().splitlines()]
        prior_index = [json.loads(line) for line in (v002 / "sft-canonical/index.jsonl").read_text().splitlines()
                       if json.loads(line)["split"] == split]
        if len(prior_rows) != len(prior_index):
            raise ValueError("v002 index disagrees with conversations")
        by_split[split].extend(zip(prior_index, prior_rows))
    for row in sources:
        split, seed, goal = row["split"], row["seed"], row["goal"]
        if goal == "exact_write":
            sp = row["write"]["pvs_written"][0]
            value = row["write"]["setpoints_mrad"][sp]
            bpm = next(pv for pv in BPMS if pv.endswith(":X" if ":MAG:H" in sp else ":Y"))
            prompt = f"Set {sp} to {value:.6f} mrad, then read {bpm}."
        else:
            prompt = PROMPTS[goal][seed % 2]
        item = conversation(row, prompt)
        item["messages"][-1]["content"] = item["messages"][-1]["content"].replace("Osprey", "PyAT")
        by_split[split].append(({"source": "local-pyat", "task_id": row["task_id"], "kind": goal}, item))
        if goal == "x":
            before = row["read_before"]
            for kind, prompts in (("read_bpms", READ_PROMPTS), ("compare_planes", COMPARE_PROMPTS)):
                by_split[split].append(({"source": "local-pyat", "task_id": row["task_id"], "kind": kind},
                                        read_conversation(before, prompts[seed % 2], kind == "compare_planes")))
    index = []
    for split, rows in by_split.items():
        random.Random(3000 + len(rows)).shuffle(rows)
        with (output / f"{split}.jsonl").open("w") as stream:
            for i, (source, sample) in enumerate(rows):
                stream.write(json.dumps(sample, allow_nan=False) + "\n")
                index.append({"split": split, "row": i, **source})
    with (output / "index.jsonl").open("w") as stream:
        for entry in index:
            stream.write(json.dumps(entry) + "\n")
    manifest = {"schema": "hackring.lfm_conversations/1", "version": "v003",
                "source": "v002 canonical Osprey conversations plus new measured local PyAT seed-distinct conversations",
                "response_teacher_sha256": sha(teacher_path),
                "parent_manifest_sha256": sha(v002 / "sft-canonical/manifest.json"),
                "deck_sha256": sha(ROOT / "data/facility/decks/RING.json"),
                "pv_contract_sha256": sha(ROOT / "data/pv_contract.json"),
                "counts": {split: len(rows) for split, rows in by_split.items()},
                "local_seeds": {split: {"first": first, "count": count} for split, (first, count) in SPLITS.items()},
                "local_actions": len(sources),
                "files_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/v003")
    print(json.dumps(build(parser.parse_args().output.resolve()), indent=2))
