"""Generate measured skill-routing SFT conversations on the fixed local PyAT ring."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from hackring.control import EFFORT_PENALTY, SKILL_SYSTEM, response_matrix, rms, solve
from hackring.machine import BPMS, ROOT, SETPOINTS, LocalMachine

SPLITS = {"train": (20268000, 144), "validation": (20269000, 24), "test": (20270000, 24)}
GOALS = ("x", "y", "both")
PROMPTS = {
    "x": ("Center the horizontal orbit", "Reduce the X-plane BPM offsets", "Bring the X orbit closer to zero"),
    "y": ("Center the vertical orbit", "Reduce the Y-plane BPM offsets", "Bring the Y orbit closer to zero"),
    "both": ("Center the beam in both planes", "Reduce both X and Y orbit offsets", "Bring both orbits closer to zero"),
}
SUFFIX = {"gentle": ("with small corrector kicks", "gently"),
          "balanced": ("with a balanced correction", "using the usual correction"),
          "aggressive": ("as far as the bounded correctors allow", "prioritizing the lowest residual")}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(plane: str, effort: str) -> str:
    return f"<|tool_call_start|>[orbit_correct(plane={plane!r}, effort={effort!r})]<|tool_call_end|>"


def example(seed: int, split: str, plane: str, effort: str, matrix) -> tuple[dict, dict]:
    machine = LocalMachine()
    before = machine.reset(seed, 0.08)["pvs"]
    settings = solve(before, plane, effort, matrix)
    after = machine.write(settings)["state"]["pvs"]
    planes = ("x", "y") if plane == "both" else (plane,)
    if not all(rms(after, item) < rms(before, item) for item in planes):
        raise ValueError(f"non-improving correction: {seed} {plane} {effort}")
    wording = random.Random(f"{seed}:{plane}:wording")
    opening = PROMPTS[plane][wording.randrange(len(PROMPTS[plane]))]
    if effort == "balanced" and seed % 2 == 0:
        prompt = f"{opening}, then report the measured result."
    else:
        prompt = f"{opening} {SUFFIX[effort][wording.randrange(len(SUFFIX[effort]))]}, then report the measured result."
    measured = {item: {"before_rms_mm": round(rms(before, item), 6),
                       "after_rms_mm": round(rms(after, item), 6)} for item in planes}
    tool_result = {"status": "completed", "plane": plane, "effort": effort,
                   "measured": measured,
                   "setpoints_mrad": {pv: round(after[pv], 6) for pv in settings},
                   "readbacks_mrad": {pv.replace(":SP", ":RB"): round(after[pv.replace(":SP", ":RB")], 6)
                                       for pv in settings}}
    answer = " ".join(f"{item.upper()} RMS: {value['before_rms_mm']:.4f} → "
                      f"{value['after_rms_mm']:.4f} mm (measured)." for item, value in measured.items())
    conversation = {"messages": [
        {"role": "system", "content": SKILL_SYSTEM},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": call(plane, effort)},
        {"role": "tool", "content": json.dumps(tool_result, separators=(",", ":"))},
        {"role": "assistant", "content": answer},
    ]}
    source = {"id": f"seed-{seed}-{plane}", "seed": seed, "split": split, "plane": plane,
              "effort": effort, "prompt": prompt, "before": before, "after": after,
              "tool_result": tool_result}
    return source, conversation


def build(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    matrix = response_matrix()
    output.mkdir(parents=True)
    counts = Counter()
    with (output / "sources.jsonl").open("w") as source_file, \
            (output / "index.jsonl").open("w") as index_file:
        for split, (first, count) in SPLITS.items():
            rows = []
            for seed in range(first, first + count):
                for plane_idx, plane in enumerate(GOALS):
                    effort = tuple(EFFORT_PENALTY)[(seed + plane_idx) % len(EFFORT_PENALTY)]
                    source, item = example(seed, split, plane, effort, matrix)
                    source_file.write(json.dumps(source, allow_nan=False) + "\n")
                    rows.append((source["id"], item))
            random.Random(4004 + first).shuffle(rows)
            with (output / f"{split}.jsonl").open("w") as stream:
                for row, (source_id, item) in enumerate(rows):
                    stream.write(json.dumps(item, allow_nan=False) + "\n")
                    index_file.write(json.dumps({"split": split, "row": row, "source_id": source_id}) + "\n")
            counts[split] = len(rows)
    manifest = {"schema": "hackring.lfm_conversations/1", "version": "v004",
                "architecture": "natural-language skill selection; deterministic bounded response solve; measured PyAT result",
                "source": "fresh seed-disjoint local PyAT disturbances; no GPT annotation",
                "seed_ranges": {split: {"first": first, "count": count} for split, (first, count) in SPLITS.items()},
                "counts": dict(counts),
                "response_teacher_sha256": digest(ROOT / "datasets/v002/response_teacher.json"),
                "deck_sha256": digest(ROOT / "data/facility/decks/RING.json"),
                "pv_contract_sha256": digest(ROOT / "data/pv_contract.json"),
                "files_sha256": {p.name: digest(p) for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/v004")
    print(json.dumps(build(parser.parse_args().output.resolve()), indent=2))
