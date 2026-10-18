"""Explain hackring_orbit eval runs: what each model wrote, and reference policies on the same tasks.

For each run folder (from scripts/run_gym_eval.sh) it reports how many written BPM values equal the
read value, the corrector values written, and the reward if only the corrector operations had been
applied. It then scores reference policies on the validation tasks with the server's physics and
objective: no write, random kicks, a naive per-BPM rule, a one-shot regularized response-matrix solve
from the pre-write read, and the best cost Xopt reached. Rewards here are the raw fractional cost
reduction; the server clips them to [-1, 1].

Run with this server's venv from the repo root:
  gym/resources_servers/hackring_orbit/.venv/bin/python gym/resources_servers/hackring_orbit/analyze_runs.py \
      var/gym/mark-i/<model>/<run> [var/gym/mark-i/<model>/<run> ...]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import numpy as np

from app import BPMS, LIMIT, REPO, SETPOINTS, Ring, objective

TASKS = Path(__file__).resolve().parent / "data" / "validation.jsonl"
ZERO = {sp: 0.0 for sp in SETPOINTS}


def readings(task: dict) -> dict[str, float]:
    output = json.loads(task["responses_create_params"]["input"][3]["output"])
    return {name: reading["value"] for name, reading in output["summary"]["readings"].items()}


def reduction(ring: Ring, task: dict, setpoints: dict[str, float]) -> float:
    offsets = task["episode"]["hidden_calibration_offsets_mrad"]
    initial = objective(ring.pvs(ZERO, offsets))["orbit_cost"]
    return (initial - objective(ring.pvs(setpoints, offsets))["orbit_cost"]) / initial


def describe_run(ring: Ring, run: Path, tasks: dict[str, dict]) -> None:
    meta = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
    rows = [json.loads(line) for line in (run / "rollouts.jsonl").open()]
    bpm_ops = copied = 0
    corrector_values: list[float] = []
    counterfactual: list[float] = []
    for row in rows:
        task = tasks[row["episode_id"]]
        read = readings(task)
        for item in row["response"]["output"]:
            if item.get("type") != "function_call":
                continue
            try:
                operations = json.loads(item["arguments"]).get("operations") or []
            except (json.JSONDecodeError, AttributeError):
                continue
            setpoints = dict(ZERO)
            has_corrector = False
            for op in operations:
                if not isinstance(op, dict):
                    continue
                channel = op.get("channel", op.get("name"))
                try:
                    value = float(op.get("value"))
                except (TypeError, ValueError):
                    continue
                if channel in BPMS:
                    bpm_ops += 1
                    copied += abs(value - read[channel]) < 1e-3
                elif channel in SETPOINTS:
                    corrector_values.append(value)
                    setpoints[channel] = max(-LIMIT, min(LIMIT, value))
                    has_corrector = True
            if has_corrector:
                counterfactual.append(reduction(ring, task, setpoints))
    print(f"== {meta.get('served_model_name', run.parent.name)} ({run.name}): {len(rows)} rollouts, "
          f"mean reward {statistics.mean(row['reward'] for row in rows):+.3f}")
    print(f"  BPM values written: {bpm_ops}, equal to the read value (±1e-3): {copied}")
    if corrector_values:
        print(f"  corrector values written: {len(corrector_values)}, zero: {sum(v == 0 for v in corrector_values)}, "
              f"range [{min(corrector_values):.3f}, {max(corrector_values):.3f}]")
        print(f"  reward if only corrector operations applied: {len(counterfactual)} batches, "
              f"mean {statistics.mean(counterfactual):+.3f}, improved {sum(c > 0 for c in counterfactual)}")
    else:
        print("  corrector values written: 0")


def references(ring: Ring, tasks: list[dict]) -> None:
    base = ring.pvs(ZERO, {})
    response = np.zeros((len(BPMS), len(SETPOINTS)))
    for j, sp in enumerate(SETPOINTS):
        pvs = ring.pvs(ZERO | {sp: 0.01}, {})
        response[:, j] = [(pvs[b] - base[b]) / 0.01 for b in BPMS]
    rng = random.Random(0)
    results: dict[str, list[float]] = {name: [] for name in (
        "no write", "random kicks ±0.35 mrad", "random kicks ±0.05 mrad", "naive: kick = -same-index BPM",
        "one-shot response-matrix solve from the read", "Xopt best of 64 evaluations")}
    for task in tasks:
        read = readings(task)
        orbit = np.array([read[b] for b in BPMS])
        # Minimize |orbit + R d|^2 + 0.01 |d|^2, the objective's own penalty, then clip to the limit.
        solve = np.linalg.solve(response.T @ response + 0.01 * np.eye(len(SETPOINTS)), -response.T @ orbit)
        naive = {sp: float(np.clip(-read[f"RING:DIAG:BPM{sp.split(':')[2][1:]}:POSITION:{'X' if ':MAG:H' in sp else 'Y'}"],
                                   -LIMIT, LIMIT)) for sp in SETPOINTS}
        offsets = task["episode"]["hidden_calibration_offsets_mrad"]
        initial = objective(ring.pvs(ZERO, offsets))["orbit_cost"]
        results["no write"].append(0.0)
        for name, width in (("random kicks ±0.35 mrad", 0.35), ("random kicks ±0.05 mrad", 0.05)):
            results[name].append(statistics.mean(
                reduction(ring, task, {sp: rng.uniform(-width, width) for sp in SETPOINTS}) for _ in range(50)))
        results["naive: kick = -same-index BPM"].append(reduction(ring, task, naive))
        results["one-shot response-matrix solve from the read"].append(
            reduction(ring, task, {sp: float(np.clip(v, -LIMIT, LIMIT)) for sp, v in zip(SETPOINTS, solve)}))
        results["Xopt best of 64 evaluations"].append((initial - task["episode"]["xopt_best_orbit_cost"]) / initial)
    print(f"== Reference policies on {len(tasks)} validation tasks "
          f"(response matrix condition number {np.linalg.cond(response):.1f})")
    for name, values in results.items():
        print(f"  {name:46s} mean {statistics.mean(values):+8.3f}  min {min(values):+8.3f}  max {max(values):+8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path, help="Run folders under var/gym/mark-i/<model>/")
    args = parser.parse_args()
    ring = Ring(REPO / "data/facility/decks/RING.json")
    tasks = [json.loads(line) for line in TASKS.open()]
    by_episode = {task["episode"]["episode_id"]: task for task in tasks}
    for run in args.runs:
        describe_run(ring, run, by_episode)
    references(ring, tasks)


if __name__ == "__main__":
    main()
