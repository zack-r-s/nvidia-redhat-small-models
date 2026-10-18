"""Validate the immutable v002 measured actions and their v001 lineage."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from generate_control_dataset import GOALS, SCHEMA, digest, metrics, split_map
from generate_dataset import BPMS, LIMIT, PVS, READBACKS, ROOT, SETPOINTS


def validate(folder: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["schema"] != "hackring.control_manifest/1" or manifest["actions"] != 512:
        raise ValueError("invalid v002 manifest")
    if manifest["parent_manifest_sha256"] != digest(ROOT / "datasets/v001/manifest.json"):
        raise ValueError("v001 parent manifest mismatch")
    if manifest["source_deck_sha256"] != digest(ROOT / "data/facility/decks/RING.json"):
        raise ValueError("ring lattice changed")
    if manifest["pv_contract_sha256"] != digest(ROOT / "data/pv_contract.json"):
        raise ValueError("PV contract changed")
    if manifest["response_teacher_sha256"] != digest(folder.parent / "response_teacher.json"):
        raise ValueError("response teacher changed")
    for name, expected in manifest["files_sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"checksum mismatch: {name}")
    config = json.loads((folder / "run_config.json").read_text())
    splits = split_map(128, config["split_seed"])
    rows = [json.loads(line) for line in (folder / "actions.jsonl").read_text().splitlines()]
    if len(rows) != 512:
        raise ValueError("v002 must have 512 actions")
    seen = set()
    counts = Counter()
    non_improving = []
    for row in rows:
        if row["schema"] != SCHEMA or row["status"] != "completed":
            raise ValueError("invalid or incomplete action")
        task_id = row["task_id"]
        if task_id in seen:
            raise ValueError(f"duplicate task: {task_id}")
        seen.add(task_id)
        goal, seed = row["goal"], row["seed"]
        if goal not in GOALS or not config["seed_base"] <= seed < config["seed_base"] + 128:
            raise ValueError(f"invalid task key: {task_id}")
        if task_id != f"seed-{seed}-{goal}" or row["split"] != splits[seed - config["seed_base"]]:
            raise ValueError(f"task ID or split mismatch: {task_id}")
        before, after = row["read_before"], row["read_after"]
        if set(before) != set(PVS) or set(after) != set(PVS):
            raise ValueError(f"missing or extra PVs: {task_id}")
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   for value in [*before.values(), *after.values()]):
            raise ValueError(f"non-finite PV value: {task_id}")
        settings = row["write"]["setpoints_mrad"]
        written = row["write"]["pvs_written"]
        if set(settings) != set(SETPOINTS) or not written or not set(written) <= set(SETPOINTS):
            raise ValueError(f"invalid write: {task_id}")
        if any(not math.isfinite(value) or abs(value) > LIMIT + 1e-12 for value in settings.values()):
            raise ValueError(f"out-of-bounds write: {task_id}")
        for sp in SETPOINTS:
            rb = sp.replace(":SP", ":RB")
            if abs(after[sp] - settings[sp]) > 1e-6 or abs(after[rb] - settings[sp]) > 1e-6:
                raise ValueError(f"setpoint/readback mismatch: {task_id} {sp}")
        for field, snapshot in (("before_metrics", before), ("after_metrics", after)):
            expected = metrics(goal, snapshot)
            if any(abs(row[field][name] - value) > 1e-8 for name, value in expected.items()):
                raise ValueError(f"incorrect {field}: {task_id}")
        if goal != "exact_write" and row["after_metrics"]["goal_cost"] >= row["before_metrics"]["goal_cost"]:
            non_improving.append(task_id)
        if row["prediction_max_abs_error_mm"] < 0 or not math.isfinite(row["prediction_max_abs_error_mm"]):
            raise ValueError(f"invalid response residual: {task_id}")
        counts[row["split"]] += 1
    expected_ids = {f"seed-{config['seed_base'] + index}-{goal}"
                    for index in range(128) for goal in GOALS}
    if seen != expected_ids or dict(counts) != manifest["actions_by_split"]:
        raise ValueError("missing task or split counts differ")
    return {"actions": len(rows), "by_split": dict(counts),
            "non_improving_corrections": non_improving,
            "max_response_error_mm": max(row["prediction_max_abs_error_mm"] for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path, default=ROOT / "datasets/v002/raw")
    print(json.dumps(validate(parser.parse_args().folder.resolve()), indent=2))


if __name__ == "__main__":
    main()
