"""Collect v002 measured correction and exact-write tasks on Osprey's served PyAT ring."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from generate_dataset import BPMS, LIMIT, READBACKS, ROOT, SETPOINTS, ServedRing, objective

GOALS = ("x", "y", "both", "exact_write")
SCHEMA = "hackring.control_action/1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_map(count: int, seed: int) -> dict[int, str]:
    if count != 128:
        raise ValueError("v002 fixes the collection at 128 seeds")
    indices = list(range(count))
    random.Random(seed).shuffle(indices)
    return {index: "train" if order < 96 else "validation" if order < 112 else "test"
            for order, index in enumerate(indices)}


def selected(goal: str) -> tuple[list[int], list[int]]:
    bpm_indices = [i for i, name in enumerate(BPMS)
                   if goal == "both" or name.endswith(":X" if goal == "x" else ":Y")]
    sp_indices = [i for i, name in enumerate(SETPOINTS)
                  if goal == "both" or (":MAG:H" in name if goal == "x" else ":MAG:V" in name)]
    return bpm_indices, sp_indices


def target(goal: str, before: dict[str, float], matrix: np.ndarray,
           rng: np.random.Generator, seed: int) -> dict[str, float]:
    current = np.array([before[pv] for pv in SETPOINTS])
    if goal == "exact_write":
        index = seed % len(SETPOINTS)
        value = float(rng.uniform(0.03, 0.2)) * (1 if seed % 2 else -1)
        current[index] = value
    else:
        bpm_indices, sp_indices = selected(goal)
        response = matrix[np.ix_(bpm_indices, sp_indices)]
        start = np.array([before[BPMS[i]] for i in bpm_indices])
        initial = current[sp_indices]
        # Minimize mean squared BPM position plus the v001 kick penalty.
        A = np.vstack((response / math.sqrt(len(bpm_indices)),
                       math.sqrt(0.01) * np.eye(len(sp_indices))))
        rhs = np.concatenate(((response @ initial - start) / math.sqrt(len(bpm_indices)),
                              np.zeros(len(sp_indices))))
        solution = lsq_linear(A, rhs, bounds=(-LIMIT, LIMIT), tol=1e-12)
        if not solution.success:
            raise RuntimeError(f"bounded teacher solve failed: {solution.message}")
        current[sp_indices] = solution.x
    return {pv: float(value) for pv, value in zip(SETPOINTS, current)}


def metrics(goal: str, pvs: dict[str, float]) -> dict:
    values = objective(pvs)
    if goal == "x":
        score = values["x_rms_mm"] ** 2 + 0.01 * sum(pvs[sp] ** 2 for sp in SETPOINTS if ":MAG:H" in sp)
    elif goal == "y":
        score = values["y_rms_mm"] ** 2 + 0.01 * sum(pvs[sp] ** 2 for sp in SETPOINTS if ":MAG:V" in sp)
    else:
        score = values["orbit_cost"]
    return {**values, "goal_cost": score}


def check_inputs(deployment: Path, teacher_path: Path) -> tuple[dict, np.ndarray]:
    provenance = json.loads((deployment / "dataset_deployment.json").read_text())
    source_deck = ROOT / "data/facility/decks/RING.json"
    if provenance["source_deck_sha256"] != digest(source_deck) or provenance["staged_deck_sha256"] != digest(deployment / "data/facility/decks/RING.json"):
        raise ValueError("source or staged lattice changed")
    if provenance["source_pv_contract_sha256"] != digest(ROOT / "data/pv_contract.json"):
        raise ValueError("PV contract changed")
    if provenance["temporary_bpm_class_conversion"]:
        raise ValueError("v002 must use the checked-in Monitor deck without conversion")
    teacher = json.loads(teacher_path.read_text())
    if teacher["schema"] != "hackring.response_teacher/1" or teacher["fit_split"] != "train":
        raise ValueError("unexpected response teacher")
    if teacher["source_attempts_sha256"] != digest(ROOT / "datasets/v001/attempts.jsonl"):
        raise ValueError("teacher is not fitted to current v001")
    if teacher["input_readbacks_mrad"] != READBACKS or teacher["output_bpms_mm"] != BPMS:
        raise ValueError("teacher PV order differs from contract")
    matrix = np.asarray(teacher["matrix_mm_per_mrad"], dtype=float)
    if matrix.shape != (len(BPMS), len(SETPOINTS)) or not np.all(np.isfinite(matrix)):
        raise ValueError("invalid teacher response matrix")
    return provenance, matrix


def run(args: argparse.Namespace) -> None:
    deployment, output, teacher_path = args.deployment.resolve(), args.output.resolve(), args.teacher.resolve()
    provenance, matrix = check_inputs(deployment, teacher_path)
    if (args.ca_port, args.pva_port) != (provenance["ca_port"], provenance["pva_port"]):
        raise ValueError("EPICS ports differ from the staged deployment")
    if (output / "manifest.json").exists():
        raise FileExistsError("v002 collection is already complete; use a new version")
    config = {"schema": "hackring.control_run/1", "seeds": 128, "seed_base": args.seed_base,
              "split_seed": args.split_seed, "disturbance_mrad": args.disturbance_mrad,
              "deployment_provenance_sha256": digest(deployment / "dataset_deployment.json"),
              "teacher_sha256": digest(teacher_path), "generator_sha256": digest(Path(__file__)),
              "ca_port": args.ca_port, "pva_port": args.pva_port, "tick_s": args.tick_s}
    if not 0 < args.disturbance_mrad <= 0.2:
        raise ValueError("disturbance must be in (0, 0.2] mrad")
    config_path = output / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("partial v002 run has different inputs; resume without changing code or parameters")
    else:
        output.mkdir(parents=True, exist_ok=False)
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    actions_path = output / "actions.jsonl"
    completed: dict[str, dict] = {}
    if actions_path.exists():
        for line in actions_path.read_text().splitlines():
            row = json.loads(line)
            if row["task_id"] in completed:
                raise ValueError("duplicate completed task in partial v002 run")
            completed[row["task_id"]] = row
    splits = split_map(128, args.split_seed)
    machine = ServedRing(deployment, args.ca_port, args.pva_port, args.tick_s)
    try:
        preflight = machine.preflight()
        with actions_path.open("a") as actions_file:
            for index in range(128):
                seed = args.seed_base + index
                rng = np.random.default_rng(seed)
                offsets = {sp: float(rng.uniform(-args.disturbance_mrad, args.disturbance_mrad))
                           for sp in SETPOINTS}
                for goal in GOALS:
                    task_id = f"seed-{seed}-{goal}"
                    if task_id in completed:
                        continue
                    before = machine.reset_episode(offsets)
                    settings = target(goal, before, matrix, rng, seed)
                    after, written = machine.write(settings, before=before)
                    bpm_before = np.array([before[pv] for pv in BPMS])
                    delta = np.array([after[pv] - before[pv] for pv in READBACKS])
                    prediction = bpm_before + matrix @ delta
                    row = {"schema": SCHEMA, "task_id": task_id, "seed": seed, "split": splits[index],
                           "goal": goal, "hidden_calibration_offsets_mrad": offsets,
                           "read_before": before, "write": {"setpoints_mrad": settings, "pvs_written": written},
                           "read_after": after, "before_metrics": metrics(goal, before),
                           "after_metrics": metrics(goal, after),
                           "prediction_max_abs_error_mm": float(np.max(np.abs(prediction - np.array([after[pv] for pv in BPMS])))),
                           "status": "completed", "measured_utc": timestamp()}
                    actions_file.write(json.dumps(row, allow_nan=False) + "\n")
                    actions_file.flush()
                    print(f"{task_id} [{splits[index]}] {row['before_metrics']['goal_cost']:.6g} -> {row['after_metrics']['goal_cost']:.6g}", flush=True)
        counts = Counter(row["split"] for row in map(json.loads, actions_path.read_text().splitlines()))
        if len(actions_path.read_text().splitlines()) != 128 * len(GOALS):
            raise RuntimeError("incomplete v002 collection")
        manifest = {"schema": "hackring.control_manifest/1", "version": output.name,
                    "created_utc": timestamp(), "source_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "parent_version": "v001", "parent_manifest_sha256": digest(ROOT / "datasets/v001/manifest.json"),
                    "response_teacher_sha256": digest(teacher_path), "source_deck_sha256": digest(ROOT / "data/facility/decks/RING.json"),
                    "pv_contract_sha256": digest(ROOT / "data/pv_contract.json"),
                    "deployment_provenance": provenance, "model_rpc_instance": preflight["instance"],
                    "backend": "Osprey served PyAT via EPICS Channel Access",
                    "run": config, "actions": 512, "actions_by_split": dict(counts),
                    "software": {name: importlib.metadata.version(name) for name in
                                 ("numpy", "scipy", "pyepics", "accelerator-toolbox", "osprey-framework")},
                    "files_sha256": {name: digest(output / name) for name in ("run_config.json", "actions.jsonl")}}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    finally:
        machine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", type=Path, default=Path("/private/tmp/hackring-data-v002-lite"))
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/v002/raw")
    parser.add_argument("--teacher", type=Path, default=ROOT / "datasets/v002/response_teacher.json")
    parser.add_argument("--ca-port", type=int, default=6264)
    parser.add_argument("--pva-port", type=int, default=6275)
    parser.add_argument("--tick-s", type=float, default=0.1)
    parser.add_argument("--seed-base", type=int, default=20262000)
    parser.add_argument("--split-seed", type=int, default=20262000)
    parser.add_argument("--disturbance-mrad", type=float, default=0.08)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
