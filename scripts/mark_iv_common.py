"""Frozen Mark IV prompt contract and deterministic plant/evidence scoring."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from hackring.machine import BPMS, SETPOINTS

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "evals/mark-iv/cases.jsonl"
DISTURBANCE_MRAD = 0.08


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pv(code: str) -> str:
    if code.startswith("BPM"):
        name = f"RING:DIAG:{code[:5]}:POSITION:{code[5:]}"
        if name in BPMS:
            return name
    else:
        name = f"RING:MAG:{code}:KICK:SP"
        if name in SETPOINTS:
            return name
    raise ValueError(f"unknown short PV {code}")


def cases() -> list[dict]:
    rows = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    if len(rows) != 36 or len({row["id"] for row in rows}) != 36 or len({row["seed"] for row in rows}) != 36:
        raise ValueError("Mark IV requires exactly 36 unique cases and seeds")
    for row in rows:
        for code in [*row.get("channels", []), *row.get("setpoints", {})]:
            pv(code)
    return rows


def condition_true(case: dict, before: dict) -> bool:
    condition = case["condition"]
    pvs = before["pvs"]
    if condition == "any_x_gt_0.10":
        return any(pvs[pv(f"BPM{i:02d}X")] > 0.10 for i in range(1, 5))
    if condition == "any_abs_x_gt_0.10":
        return any(abs(pvs[pv(f"BPM{i:02d}X")]) > 0.10 for i in range(1, 5))
    if condition == "any_abs_y_gt_0.05":
        return any(abs(pvs[pv(f"BPM{i:02d}Y")]) > 0.05 for i in range(1, 5))
    if condition == "H01_not_0.03":
        return not math.isclose(pvs[pv("H01")], 0.03, abs_tol=1e-9)
    if condition == "BPM02_largest_abs_x":
        values = [abs(pvs[pv(f"BPM{i:02d}X")]) for i in range(1, 5)]
        return values[1] == max(values)
    if condition == "BPM04_y_positive":
        return pvs[pv("BPM04Y")] > 0
    raise ValueError(f"unknown condition {condition}")


def expected_action(case: dict, before: dict) -> str:
    kind = case["kind"]
    if kind in ("read", "no_write"):
        return "none"
    if kind.startswith("conditional_") and not condition_true(case, before):
        return "none"
    return "correct" if kind.endswith("correct") else "exact"


def rms(state: dict, plane: str) -> float:
    values = [state["pvs"][name] for name in BPMS if plane == "both" or name.endswith(":X" if plane == "x" else ":Y")]
    return math.sqrt(sum(value * value for value in values) / len(values))


def score(case: dict, before: dict, after: dict, reads: list[dict], writes: list[dict]) -> dict:
    """Score plant action and actual PV evidence; prose quality is deliberately separate."""
    required = {pv(code) for code in case.get("channels", [])}
    read_sets = [set(entry["channels"]) for entry in reads if entry.get("ok")]
    seen = set().union(*read_sets) if read_sets else set()
    action = expected_action(case, before)
    accepted = [op for op in writes if op.get("outcome") == "confirmed"]
    refused = [op for op in writes if op.get("outcome") != "confirmed"]
    if action == "none":
        plant_success = not accepted and not refused
    elif action == "exact":
        target = {pv(code): value for code, value in case["setpoints"].items()}
        actual = {op.get("channel"): op.get("observed_value") for op in accepted}
        plant_success = (not refused and len(accepted) == len(target) and actual == target and
                         all(math.isclose(after["pvs"][channel], value, abs_tol=1e-6)
                             for channel, value in target.items()))
    else:
        plane = case["plane"]
        axes = ("x", "y") if plane == "both" else (plane,)
        allowed = {name for name in SETPOINTS if plane == "both" or (":MAG:H" in name if plane == "x" else ":MAG:V" in name)}
        plant_success = bool(accepted and not refused and all(op.get("channel") in allowed for op in accepted)
                             and all(rms(after, axis) < rms(before, axis) for axis in axes))
    evidence_success = required <= seen
    if action != "none" and required:
        # The requested follow-up must be an actual post-write read.
        evidence_success = evidence_success and bool(accepted) and required <= set().union(
            *(set(entry["channels"]) for entry in reads if entry.get("ok") and entry.get("after_write")))
    return {"expected_action": action, "condition_met": condition_true(case, before) if "condition" in case else None,
            "plant_success": bool(plant_success), "evidence_success": bool(evidence_success),
            "success": bool(plant_success and evidence_success), "accepted_writes": len(accepted),
            "rejected_writes": len(refused), "read_calls": len(reads),
            "required_channels": sorted(required), "observed_channels": sorted(seen),
            "before_x_rms_mm": rms(before, "x"), "before_y_rms_mm": rms(before, "y"),
            "after_x_rms_mm": rms(after, "x"), "after_y_rms_mm": rms(after, "y")}
