"""Split a physical dataset by episode order: the first fraction for fine-tuning, the rest for evaluation.

The split is chronological at the episode level, so no episode contributes rows to both portions.
It ignores the generator's shuffled `split` field. The raw version directory is never edited;
`split.json` (episode IDs, counts, and source checksums) is the committed record, and the
materialized JSONL copies beside it are ignored build output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def split(folder: Path, finetune_fraction: float) -> dict:
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if digest(folder / name) != expected:
            raise ValueError(f"checksum mismatch: {name}")
    episodes = load_jsonl(folder / "episodes.jsonl")
    ids = [episode["episode_id"] for episode in episodes]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        raise ValueError("episodes.jsonl must list unique episode IDs in order")
    cut = round(len(ids) * finetune_fraction)
    if not 0 < cut < len(ids):
        raise ValueError(f"fraction {finetune_fraction} leaves an empty portion")
    portions = {"finetune": ids[:cut], "eval": ids[cut:]}
    attempts = load_jsonl(folder / "attempts.jsonl")
    steps = manifest["run"]["steps_per_episode"]
    counts = {}
    for portion, members in portions.items():
        rows = [row for row in attempts if row["episode_id"] in set(members)]
        if len(rows) != len(members) * steps:
            raise ValueError(f"{portion}: expected {len(members) * steps} attempts, got {len(rows)}")
        counts[portion] = {"episodes": len(members), "attempts": len(rows)}
    return {
        "schema": "hackring.dataset_split/1",
        "source_version": manifest["version"],
        "source_files_sha256": manifest["files_sha256"],
        "method": "episode order; first fraction fine-tunes, remainder evaluates; generator split field ignored",
        "finetune_fraction": finetune_fraction,
        "counts": counts,
        "episodes": portions,
    }


def materialize(folder: Path, record: dict, out: Path) -> None:
    for portion, members in record["episodes"].items():
        keep = set(members)
        for name in ("episodes.jsonl", "attempts.jsonl"):
            lines = [line for line in (folder / name).read_text().splitlines()
                     if line.strip() and json.loads(line)["episode_id"] in keep]
            (out / f"{portion}_{name}").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Version directory, e.g. datasets/v001")
    parser.add_argument("--finetune-fraction", type=float, default=0.75)
    parser.add_argument("--name", default="ft75-eval25", help="Output folder under <dataset>/splits/")
    args = parser.parse_args()
    record = split(args.dataset, args.finetune_fraction)
    out = args.dataset / "splits" / args.name
    out.mkdir(parents=True, exist_ok=True)
    existing = out / "split.json"
    if existing.exists() and json.loads(existing.read_text()) != record:
        raise SystemExit(f"{existing} exists with a different split; choose another --name")
    existing.write_text(json.dumps(record, indent=2) + "\n")
    materialize(args.dataset, record, out)
    for portion, members in record["episodes"].items():
        print(f"{portion}: {members[0]}..{members[-1]} {record['counts'][portion]}")


if __name__ == "__main__":
    main()
