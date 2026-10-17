"""Validate a complete physical Xopt dataset and print a compact quality summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from hackring.machine import BPMS, LIMIT, SETPOINTS
from generate_dataset import READBACKS, SCHEMA, objective

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PVS = set(BPMS + SETPOINTS + READBACKS)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-5) -> bool:
    return abs(left - right) <= tolerance


def validate(folder: Path) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["schema"] != "hackring.dataset_manifest/1":
        raise ValueError("wrong manifest schema")
    for name, expected in manifest["files_sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"checksum mismatch: {name}")
    if digest(ROOT / "data/facility/decks/RING.json") != manifest["source_deck_sha256"]:
        raise ValueError("dataset lattice does not match current checked-in lattice")
    if digest(ROOT / "data/pv_contract.json") != manifest["pv_contract_sha256"]:
        raise ValueError("dataset PV contract does not match current checked-in contract")
    episodes = load_jsonl(folder / "episodes.jsonl")
    attempts = load_jsonl(folder / "attempts.jsonl")
    expected_count = manifest["run"]["episodes"] * manifest["run"]["steps_per_episode"]
    if len(episodes) != manifest["run"]["episodes"] or len(attempts) != expected_count:
        raise ValueError("episode or attempt count differs from manifest")
    if len(attempts) != manifest["attempts"]:
        raise ValueError("attempt count differs from manifest")

    by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in attempts:
        if row["schema"] != SCHEMA or row["status"] != "completed":
            raise ValueError("invalid or incomplete attempt")
        before, after = row["read_before"], row["read_after"]
        if set(before) != EXPECTED_PVS or set(after) != EXPECTED_PVS:
            raise ValueError("attempt does not contain the exact 24-PV contract")
        if any(not math.isfinite(value) for value in [*before.values(), *after.values()]):
            raise ValueError("non-finite PV reading")
        settings = row["write"]["setpoints_mrad"]
        written = row["write"]["pvs_written"]
        if set(settings) != set(SETPOINTS) or not written or not set(written) <= set(SETPOINTS):
            raise ValueError("attempt has invalid corrector commands or writes")
        if any(not math.isfinite(value) or abs(value) > LIMIT + 1e-12 for value in settings.values()):
            raise ValueError("out-of-bounds command")
        for sp, value in settings.items():
            if not close(after[sp], value, 1e-6) or not close(after[sp.replace(":SP", ":RB")], value, 1e-6):
                raise ValueError("setpoint or readback differs from command")
        for field, pvs in (("before_metrics", before), ("after_metrics", after)):
            for name, expected in objective(pvs).items():
                if not close(row[field][name], expected, 1e-8):
                    raise ValueError(f"incorrect {field}.{name}")
        by_episode[row["episode_id"]].append(row)

    episode_ids = {episode["episode_id"] for episode in episodes}
    if len(episode_ids) != len(episodes) or set(by_episode) != episode_ids:
        raise ValueError("episode metadata does not match attempts")
    counts = Counter()
    improved = 0
    for episode in episodes:
        episode_id, split = episode["episode_id"], episode["split"]
        if split not in {"train", "validation", "test"}:
            raise ValueError("unknown split")
        rows = by_episode[episode_id]
        if len(rows) != manifest["run"]["steps_per_episode"]:
            raise ValueError(f"incomplete episode: {episode_id}")
        if [row["step"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"non-sequential steps: {episode_id}")
        if any(row["split"] != split for row in rows):
            raise ValueError(f"split leakage: {episode_id}")
        if any(not close(rows[0]["read_before"][name], episode["initial_pvs"][name]) for name in EXPECTED_PVS):
            raise ValueError(f"first attempt does not follow episode reset: {episode_id}")
        for previous, current in zip(rows, rows[1:]):
            if any(not close(previous["read_after"][name], current["read_before"][name]) for name in EXPECTED_PVS):
                raise ValueError(f"PV sequence is discontinuous: {episode_id}")
        counts[split] += len(rows)
        if rows[-1]["after_metrics"]["orbit_cost"] < episode["initial_metrics"]["orbit_cost"]:
            improved += 1
    if {split: counts[split] for split in ("train", "validation", "test")} != manifest["attempts_by_split"]:
        raise ValueError("split counts differ from manifest")
    return {"episodes": len(episodes), "attempts": len(attempts), "by_split": dict(counts),
            "episodes_with_final_improvement": improved}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=ROOT / "datasets/v001")
    print(json.dumps(validate(parser.parse_args().path.resolve()), indent=2))


if __name__ == "__main__":
    main()
