# Liquid skill-routing pass, dataset/model v004

This chronological training run starts from `LiquidAI/LFM2.5-230M` and fine-tunes on `datasets/v004/train.jsonl`; it does not continue either earlier checkpoint. The task is natural-language selection of `orbit_correct(plane, effort)`. A checked-in deterministic solver computes bounded PV writes from measured BPMs and the versioned response matrix. The conversation format is Liquid SFT with assistant-only loss, two epochs, rank-8 default LoRA, and a 1536-token maximum length.

On Apple Metal with the pinned Liquid Finetune checkout:

```sh
HF_DATASETS_CACHE=/private/tmp/hackring-hf-datasets-v004 \
  LIQUID_FINETUNE_DIR=/private/tmp/liquid-finetune-smoke \
  /private/tmp/liquid-finetune-smoke/.venv/bin/python scripts/run_liquid_training.py \
  --job training/v003/job.yaml --backend mps --output-root training/v003/output
```

`output/` is ignored by Git. The frozen local PyAT correction evaluation uses `scripts/eval_v004_skill.py` and the correction subset of `evals/pyat-v002/cases.jsonl`. It checks an unforced skill call, the selected plane, bounded solver output, and measured post-write BPM RMS. This tests the model-plus-skill architecture; it is not a fair claim that the model alone learned inverse optics. Read/report and exact-write regressions are outside this skill-specific evaluation.

The completed MPS run took 224.7 seconds for 108 optimizer steps; validation loss was 0.0381 after epoch 1 and 0.0315 after epoch 2. [`results.json`](results.json) records the input, job, adapter, and merged-model hashes. A first local candidate exposed a wording/effort correlation, so the v004 dataset was regenerated with independent wording and balanced-default examples before this release run. The earlier local candidate is not the published v004 checkpoint.

On the 28 frozen correction requests, the base model selected a usable plane and effort on 0/28 and this checkpoint on 12/28; v004 chose the correct plane on all 28 but sometimes chose aggressive when effort was unstated. With the live harness's explicit rule that an unstated effort means balanced, replay of those **same saved model calls** produced measured improvement on 14/28 base and 28/28 v004 cases. That normalized result includes 16 corrected effort choices, so it measures model routing plus a deterministic default and bounded solver. On the separate 72-case v004 test split, raw plane-and-effort selection passed 67/72 and plane selection passed 72/72. Three failures requested aggressive but received balanced; the live guard rejects those without a write. Read/report and exact-write behavior were not re-evaluated for this checkpoint.
