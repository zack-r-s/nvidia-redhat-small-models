"""Run a versioned Liquid Finetune SFT job on one CUDA or Apple Metal device.

Use the Python interpreter from an installed Liquid Finetune checkout. The
Metal route calls Liquid's single-process trainer directly because its CLI
currently rejects non-CUDA devices; it does not patch upstream files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=ROOT / "training/v001/job.yaml")
    parser.add_argument("--backend", choices=("mps", "cuda"), default="mps")
    parser.add_argument("--output-root", type=Path, default=ROOT / "training/v001/output")
    parser.add_argument("--max-steps", type=int, help="One-off pipeline smoke; omit for full epochs")
    args = parser.parse_args()

    import torch

    available = torch.backends.mps.is_available() if args.backend == "mps" else torch.cuda.is_available()
    if not available:
        raise RuntimeError(f"{args.backend} is unavailable to this process")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")

    job = args.job.resolve()
    output_root = args.output_root.resolve()
    os.environ["OUTPUT_DIR"] = str(output_root)
    if args.backend == "mps":
        os.environ["ACCELERATE_TORCH_DEVICE"] = "mps"

    from liquid_finetune.config.parser import materialize_job_config, parse_job_config
    from liquid_finetune.distribution.local_trainer import local_trainer
    from liquid_finetune.training.utils.logging import setup_training_environment

    parsed = parse_job_config(job)
    resolved = materialize_job_config(parsed)
    resolved.dataset.quick_validate()
    config = resolved.to_dict()
    train_config = config["training_config"]
    if args.max_steps is not None:
        train_config["max_steps"] = args.max_steps
        train_config["eval_strategy"] = "no"
        train_config["save_strategy"] = "no"
    run_dir = Path(train_config["output_dir"])
    provenance = {
        "job_sha256": sha256(job),
        "train_sha256": sha256(Path(resolved.dataset.dataset_path)),
        "validation_sha256": sha256(Path(resolved.dataset.val_dataset_path)),
        "backend": args.backend,
        "torch_version": torch.__version__,
        "max_steps_override": args.max_steps,
        "run_dir": str(run_dir),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = run_dir / "hackring_run.json"
    if provenance_path.exists():
        prior = json.loads(provenance_path.read_text())
        if prior != provenance:
            raise RuntimeError(f"Refusing to reuse run directory with different provenance: {run_dir}")
    else:
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True), flush=True)
    setup_training_environment()
    local_trainer(config)


if __name__ == "__main__":
    main()
