# PyAT baseline evaluation v001

**Historical pre-reconciliation result.** Rerun this checkpoint in a separately named `mark-ii` output under the reconciled PR #8 code before using it in a new cross-version comparison; preserve these traces and hashes.

Eight fixed operator requests use seeds `20263000`–`20263007`, outside v002's `20262000`–`20262127` training-data seeds. Each case resets a fresh `LocalMachine` instance built from the checked-in PyAT ring. The local disturbance is a seeded kick on the deck's error elements; it is different from the served Osprey calibration-offset disturbance used for v002 data. Neither the lattice nor the Osprey configuration is changed.

The evaluator gives the model a measured pre-read in the same read/tool-result format as the training conversations. For write cases it forces the `channel_write` tool name, as the live goal harness does, while the model chooses operations. It accepts only literal, bounded, correctly targeted setpoints. Rejected proposals never reach PyAT. After an accepted write, it reads the plant again and gives the model the observed result. Inference here uses cached `LiquidAI/LFM2.5-230M` through Transformers on CPU with greedy decoding. This is a small model/plant check, not the full Osprey MCP or vLLM UI path.

Scoring is deliberately narrow: BPM report requires all eight measured values within 0.005 mm in BPM order; plane comparison requires identifying the plane with larger four-BPM RMS; exact writes require the requested PV and kick within 1e-6 mrad; orbit correction requires an allowed, bounded write that reduces RMS in the requested plane (both X and Y for a two-plane request). Final prose after a write is recorded for review but does not determine the write score. The eight cases are a smoke baseline, not a statistical performance estimate.

```sh
uv pip install --python .venv/bin/python -e '.[eval]'
MPLCONFIGDIR=/private/tmp/mpl-hackring-eval PYTHONPATH=src .venv/bin/python scripts/eval_local_pyat.py
```

The first base-model run is in `base-230m/`: 1/8 passed (BPM report 1/1, plane comparison 0/1, exact writes 0/2, corrections 0/4). All proposed writes were rejected before plant execution. A bounded response-teacher check confirms all four correction cases are physically correctable on this PyAT plant. `report.json` records case scores and provenance; `traces.jsonl` keeps the pre-read, raw model proposal, final text where available, and measured after-state. Use a new `--output` directory for another checkpoint; the script refuses to overwrite an evaluation.

The Liquid Finetune v001 merged checkpoint is evaluated in `liquid-v001/`: 2/8 passed, both exact writes. Its four syntactically valid correction writes all increased the requested orbit RMS. The read-only report and plane comparison also failed. See `training/v001/results.json` for the training artifact identity and hashes.
