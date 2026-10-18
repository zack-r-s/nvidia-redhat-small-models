# Liquid Finetune pass v002

The published checkpoint is [model v003](https://huggingface.co/zack-r-s/hackring-LFM2.5-230M/tree/v003), matching its [dataset v003](https://huggingface.co/datasets/zack-r-s/hackring-pyat-control/tree/v003) input. `training/v002` is the chronological local run directory; see the [release chart](../README.md).

This pass starts from the original `LiquidAI/LFM2.5-230M` instruction checkpoint and trains on `datasets/v003/train.jsonl`, not on the v001 fine-tune. The 864-row train split mixes canonical Osprey v002 conversations with new measured local-PyAT conversations. The 128-row validation and 128-row test splits use disjoint seeds. The fixed expanded PyAT evaluation at `evals/pyat-v002/cases.jsonl` was frozen before this training run and includes the original eight cases plus 48 new ones.

The configuration in `job.yaml` uses the pinned Liquid Finetune checkout at commit `d9cc23dae5a16498ab7dc11958b80ca803e0440a`, two epochs, rank-8 default LoRA, assistant-only loss, max length 3072, and 216 planned optimizer steps. On Apple Metal, use the repository wrapper and an installed Liquid environment:

```sh
HF_DATASETS_CACHE=/private/tmp/hackring-hf-datasets-v003 \
  LIQUID_FINETUNE_DIR=/private/tmp/liquid-finetune-smoke \
  /private/tmp/liquid-finetune-smoke/.venv/bin/python scripts/run_liquid_training.py \
  --job training/v002/job.yaml --backend mps --output-root training/v002/output
```

`output/` is ignored by Git. The completed run took 955 seconds for 216 optimizer steps; validation loss was 0.0828 after epoch 1 and 0.0783 after epoch 2. `results.json` identifies both adapter checkpoints, the merged weights, hashes, and held-out PyAT reports.

On the identical frozen 56-case PyAT set, v001 passed 21 and v002 passed 20. V002 improved BPM reporting from 5/9 to 7/9, held exact writes at 10/10, failed all nine plane comparisons, and reduced successful orbit corrections from 6/28 to 3/28. All 28 candidate correction writes were accepted by the harness, so the remaining control issue is numerical choice, not call syntax. The local evaluator forces the write tool name and is not a live Osprey deployment; the results do not establish autonomous plant control.

The public [dataset repository](https://huggingface.co/datasets/zack-r-s/hackring-pyat-control) preserves this input under tag `v003`. The public [shared model repository](https://huggingface.co/zack-r-s/hackring-LFM2.5-230M) preserves this merged checkpoint under tag `v003`; load `revision="v003"` for this pass. The original `v001` model tag and `legacy-training-v002` tag preserve the former chronological labels. The old per-run v001 model repository remains a private archive.
