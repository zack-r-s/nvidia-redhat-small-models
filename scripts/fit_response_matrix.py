"""Fit the v002 linear orbit-response teacher from v001 train episodes only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_dataset import BPMS, READBACKS, SETPOINTS

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(source: Path) -> dict:
    manifest = json.loads((source / "manifest.json").read_text())
    if digest(source / "attempts.jsonl") != manifest["files_sha256"]["attempts.jsonl"]:
        raise ValueError("v001 attempts checksum mismatch")
    by_split: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        "train": [], "validation": [], "test": []}
    for line in (source / "attempts.jsonl").open():
        row = json.loads(line)
        before, after = row["read_before"], row["read_after"]
        du = np.array([after[pv] - before[pv] for pv in READBACKS])
        db = np.array([after[pv] - before[pv] for pv in BPMS])
        if not np.all(np.isfinite(du)) or not np.all(np.isfinite(db)):
            raise ValueError("non-finite v001 observation")
        by_split[row["split"]].append((du, db))
    train_u = np.stack([pair[0] for pair in by_split["train"]])
    train_b = np.stack([pair[1] for pair in by_split["train"]])
    coefficients, _, rank, singular_values = np.linalg.lstsq(train_u, train_b, rcond=None)
    if rank != len(SETPOINTS):
        raise ValueError(f"response fit is rank {rank}, expected {len(SETPOINTS)}")
    metrics = {}
    for split, pairs in by_split.items():
        x = np.stack([pair[0] for pair in pairs])
        y = np.stack([pair[1] for pair in pairs])
        residual = x @ coefficients - y
        metrics[split] = {"rows": len(pairs),
                          "rmse_mm": float(np.sqrt(np.mean(residual ** 2))),
                          "max_abs_error_mm": float(np.max(np.abs(residual)))}
    return {
        "schema": "hackring.response_teacher/1",
        "source_version": "v001",
        "source_manifest_sha256": digest(source / "manifest.json"),
        "source_attempts_sha256": digest(source / "attempts.jsonl"),
        "fit_split": "train",
        "input_readbacks_mrad": READBACKS,
        "output_bpms_mm": BPMS,
        "matrix_mm_per_mrad": coefficients.T.tolist(),
        "rank": int(rank),
        "singular_values": singular_values.tolist(),
        "heldout_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/v001")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/v002/response_teacher.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = fit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["heldout_metrics"], indent=2))


if __name__ == "__main__":
    main()
