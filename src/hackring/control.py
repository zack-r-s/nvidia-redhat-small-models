"""Bounded response-matrix orbit correction for the fixed 24-PV ring."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from hackring.machine import BPMS, LIMIT, ROOT, SETPOINTS

TEACHER = ROOT / "datasets/v002/response_teacher.json"
EFFORT_PENALTY = {"gentle": 0.1, "balanced": 0.01, "aggressive": 0.001}
SKILL_TOOL = {"type": "function", "function": {
    "name": "orbit_correct", "description": "Read BPM and corrector PVs, solve bounded orbit correction, write setpoints, and read measured result.",
    "parameters": {"type": "object", "properties": {
        "plane": {"type": "string", "enum": ["x", "y", "both"]},
        "effort": {"type": "string", "enum": list(EFFORT_PENALTY)},
    }, "required": ["plane", "effort"]},
}}
SKILL_SYSTEM = (
    "You operate a simulated FODO storage ring. For orbit-centering requests call orbit_correct "
    "with the requested plane and effort: gentle for small corrector kicks, aggressive for "
    "maximum residual reduction, balanced otherwise. Report only measured tool results. "
    "BPM positions are mm and corrector kicks are mrad. List of tools: "
    + json.dumps([SKILL_TOOL], separators=(",", ":"))
)


def requested_effort(prompt: str) -> str:
    """Resolve explicit operator tradeoffs; an unstated tradeoff is balanced."""
    if re.search(r"\b(gentl\w*|small (?:corrector )?kicks?|minimal kicks?|conservative)\b", prompt, re.I):
        return "gentle"
    if re.search(r"\b(aggressive\w*|lowest residual|maximum correction)\b|as far as .* allow",
                 prompt, re.I):
        return "aggressive"
    return "balanced"


def response_matrix() -> np.ndarray:
    data = json.loads(TEACHER.read_text())
    if data["schema"] != "hackring.response_teacher/1":
        raise ValueError("unexpected response teacher schema")
    if data["input_readbacks_mrad"] != [sp.replace(":SP", ":RB") for sp in SETPOINTS] or data["output_bpms_mm"] != BPMS:
        raise ValueError("response teacher does not match the fixed PV contract")
    if data["source_attempts_sha256"] != hashlib.sha256((ROOT / "datasets/v001/attempts.jsonl").read_bytes()).hexdigest():
        raise ValueError("response teacher source has changed")
    matrix = np.asarray(data["matrix_mm_per_mrad"], dtype=float)
    if matrix.shape != (len(BPMS), len(SETPOINTS)) or not np.all(np.isfinite(matrix)):
        raise ValueError("invalid response matrix")
    return matrix


def selected(plane: str) -> tuple[list[int], list[int]]:
    if plane not in ("x", "y", "both"):
        raise ValueError("plane must be x, y, or both")
    bpm = [i for i, pv in enumerate(BPMS) if plane == "both" or pv.endswith(":" + plane.upper())]
    kick = [i for i, pv in enumerate(SETPOINTS)
            if plane == "both" or (":MAG:H" in pv if plane == "x" else ":MAG:V" in pv)]
    return bpm, kick


def rms(pvs: dict[str, float], plane: str) -> float:
    indices, _ = selected(plane)
    return math.sqrt(sum(float(pvs[BPMS[i]]) ** 2 for i in indices) / len(indices))


def solve(pvs: dict[str, float], plane: str, effort: str = "balanced",
          matrix: np.ndarray | None = None) -> dict[str, float]:
    """Return absolute setpoints; this function never writes to the plant."""
    if effort not in EFFORT_PENALTY:
        raise ValueError("effort must be gentle, balanced, or aggressive")
    if matrix is None:
        matrix = response_matrix()
    bpm, kick = selected(plane)
    start = np.asarray([pvs[BPMS[i]] for i in bpm], dtype=float)
    current = np.asarray([pvs[SETPOINTS[i]] for i in kick], dtype=float)
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(current)):
        raise ValueError("non-finite measured PV")
    response = matrix[np.ix_(bpm, kick)]
    penalty = EFFORT_PENALTY[effort]
    a = np.vstack((response / math.sqrt(len(bpm)), math.sqrt(penalty) * np.eye(len(kick))))
    rhs = np.concatenate(((response @ current - start) / math.sqrt(len(bpm)), np.zeros(len(kick))))
    result = lsq_linear(a, rhs, bounds=(-LIMIT, LIMIT), tol=1e-12)
    if not result.success:
        raise RuntimeError(f"bounded orbit solve failed: {result.message}")
    return {SETPOINTS[i]: float(value) for i, value in zip(kick, result.x)}
