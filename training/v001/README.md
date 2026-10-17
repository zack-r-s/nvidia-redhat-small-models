# Liquid Finetune pass v001

This run trains `LiquidAI/LFM2.5-230M` with Liquid Finetune SFT and rank-8 default LoRA on the GPT-augmented v002 train split (864 conversations). It evaluates against the fixed 144-row validation split. The 144-row test split and eight-case PyAT evaluation remain held out. The 2-epoch plan uses assistant-only loss, a 3072-token sequence limit, batch size 1 with 8 gradient-accumulation steps, and saves at each epoch. The longest checked conversation is 2436 tokens.

Use Liquid Finetune commit `d9cc23dae5a16498ab7dc11958b80ca803e0440a` in a separate environment. Its supported CLI requires CUDA. On a CUDA machine, run its CLI from this repository root with `OUTPUT_DIR` set to an ignored output location. On the tested Apple Metal machine, the wrapper calls Liquid's one-process trainer directly without editing Liquid source:

```sh
LIQUID_FINETUNE_DIR=/path/to/liquid-finetune \
  /path/to/liquid-finetune/.venv/bin/python scripts/run_liquid_training.py \
  --backend mps --max-steps 1 --output-root /private/tmp/hackring-liquid-smoke

LIQUID_FINETUNE_DIR=/path/to/liquid-finetune \
  /path/to/liquid-finetune/.venv/bin/python scripts/run_liquid_training.py \
  --backend mps
```

The completed pass wrote ignored checkpoints and a merged model under `training/v001/output/`. The wrapper records source-file checksums in `hackring_run.json`; tracked `results.json` identifies the exact artifacts and hashes. The two-epoch run completed 216 optimizer steps in 1095 seconds. Validation loss fell from 0.0837 to 0.0806.

The frozen eight-case PyAT evaluation is in `evals/pyat-v001/liquid-v001/`. The trained model passed 2/8 versus the base model's 1/8: both exact writes succeeded, while the read and comparison tasks failed and all four correction attempts increased measured target RMS. The evaluator supplied measured pre-reads, forced the write tool name, and ran against local PyAT with seeded error kicks, so this result does not establish live Osprey behavior. vLLM Metal loaded the merged checkpoint on a separate temporary port and parsed an exact `channel_write` request; that server was then stopped. The weights are ignored by Git and remain local to this checkout.
