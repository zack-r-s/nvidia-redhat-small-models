"""Freeze a larger seed-disjoint PyAT evaluation before training v002."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evals/pyat-v002/cases.jsonl"

PROMPTS = {
    "read_bpms": ["Report X and Y positions for BPM01, BPM02, BPM03, and BPM04.",
                  "Please read every BPM position in millimeters."],
    "compare_planes": ["From the four BPMs, is horizontal or vertical RMS larger?",
                       "Which transverse orbit has the greater measured RMS?"],
    "x": ["Steer the horizontal beam nearer the axis and measure the residual X orbit.",
          "Reduce the X-plane BPM RMS and show the resulting readings."],
    "y": ["Steer the vertical beam nearer the axis and measure the residual Y orbit.",
          "Reduce the Y-plane BPM RMS and show the resulting readings."],
    "both": ["Reduce the measured orbit in X and Y, then report both residuals.",
             "Use the correctors to center both transverse planes and check the BPMs."],
}


def build() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    prior = [json.loads(line) for line in (ROOT / "evals/pyat-v001/cases.jsonl").read_text().splitlines()]
    cases = list(prior)
    modes = ("read_bpms", "compare_planes", "exact_write", "x", "y", "both")
    for i in range(48):
        seed = 20267000 + i
        mode = modes[i % len(modes)]
        if mode == "exact_write":
            axis = "H" if (i // len(modes)) % 2 == 0 else "V"
            device = 1 + ((i // len(modes)) % 4)
            channel = f"RING:MAG:{axis}{device:02d}:KICK:SP"
            value = round((-1 if i % 2 else 1) * (0.035 + 0.012 * (i // len(modes))), 3)
            follow_up = f"RING:DIAG:BPM02:POSITION:{'X' if axis == 'H' else 'Y'}"
            case = {"id": f"new-{i:02d}-exact", "seed": seed, "kind": "exact_write",
                    "channel": channel, "value_mrad": value, "follow_up": follow_up,
                    "prompt": f"Set {channel} to {value:+.3f} mrad, then read {follow_up}."}
        elif mode in ("read_bpms", "compare_planes"):
            case = {"id": f"new-{i:02d}-{mode}", "seed": seed, "kind": mode,
                    "prompt": PROMPTS[mode][(i // len(modes)) % 2]}
        else:
            case = {"id": f"new-{i:02d}-{mode}", "seed": seed, "kind": "correct", "plane": mode,
                    "prompt": PROMPTS[mode][(i // len(modes)) % 2]}
        cases.append(case)
    assert len(cases) == 56 and len({c["seed"] for c in cases}) == 56
    OUTPUT.parent.mkdir(parents=True)
    OUTPUT.write_text("".join(json.dumps(case) + "\n" for case in cases))


if __name__ == "__main__":
    build()
