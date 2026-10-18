# Mark II NeMo Gym evaluations

The Gym wrapper writes new runs here as `<model>/<split>-<timestamp>/` and labels `run.json` with `evaluation_series: mark-ii`. The historical runs are under [`../mark-i/`](../mark-i/README.md).

## Results

All runs use repo commit `e83200c`, the `hackring_orbit` validation split (16 v001 episodes × 4 repeats = 64 rollouts), `evaluation_interface: channel_write`, temperature 0.2, 512 output tokens, NeMo Gym 0.4.0, and vLLM 0.31.0 (vllm-metal, Apple M2 Max). Each model was served pinned to the listed revision. No run had infrastructure failures.

| Model | Source | Revision | Mean reward | One-call rule | Improved / worsened | Accepted writes | No tool call | Run |
|---|---|---|---|---|---|---|---|---|
| LFM2.5-230M base | `LiquidAI/LFM2.5-230M` | `40cb2ad` | 0.000 | 0.000 | 0 / 0 | 0 / 64 | 23 | [`LFM2.5-230M/validation-20261017-215446`](LFM2.5-230M/validation-20261017-215446/) |
| LFM2.5-1.2B-Instruct | `LiquidAI/LFM2.5-1.2B-Instruct` | `0f604ad` | 0.000 | 0.000 | 0 / 0 | 0 / 64 | 6 | [`LFM2.5-1.2B-Instruct/validation-20261017-222924`](LFM2.5-1.2B-Instruct/validation-20261017-222924/) |
| hackring v002 | `zack-r-s/hackring-LFM2.5-230M@v002` | `d957415` | −0.034 | −0.034 | 21 / 43 | 64 / 64 | 0 | [`hackring-LFM2.5-230M-v002/validation-20261017-223151`](hackring-LFM2.5-230M-v002/validation-20261017-223151/) |
| hackring v003 | `zack-r-s/hackring-LFM2.5-230M@v003` | `8ad5bc1` | −0.019 | −0.006 | 29 / 29 | 57 / 64 | 5 | [`hackring-LFM2.5-230M-v003/validation-20261017-223357`](hackring-LFM2.5-230M-v003/validation-20261017-223357/) |
| hackring v004 ⚠️ | `zack-r-s/hackring-LFM2.5-230M@v004` | `14419bf` | 0.000 | n/a | 0 / 0 | 0 / 64 | 28 | [`hackring-LFM2.5-230M-v004/validation-20261017-223609`](hackring-LFM2.5-230M-v004/validation-20261017-223609/) |

Reward is the fractional reduction of the v001 objective (`x_rms_mm^2 + y_rms_mm^2 + 0.01*sum(sp_mrad^2)`), clipped to [-1, 1]. The deployed `scripts/small_agent.py` accepts exactly one `channel_write` call; "one-call rule" rescores rollouts with several write calls as no write (v003 split its write in 16 rollouts). The server does not yet enforce this.

**Checkpoint identity.** By `model.safetensors` SHA-256 there are three fine-tuned checkpoints: `v002` (`f38487…`, also the `v001` alias, the `release-dataset-v002` branch, and `zack-r-s/LFM2.5-230M-hackring-v001`), `v003` (`8d6c62…`, also `legacy-training-v002`), and `v004` (`e35185…`, also `main`). Aliases were not evaluated separately.

**Reading the results.** Neither base model produces an accepted write. v002 and v003 write valid corrector batches, but their kicks do not track the correction: rewards scatter around zero, and both match their Mark I scores within sampling noise (v002 −0.041, v003 −0.027). v004 is trained for `orbit_correct(plane, effort)` and is not scorable by this task; see its `NEEDS_RETEST.md`.

**Disclosures.** The episode-order evaluation split includes 11 v001 `train`-labelled episodes used to fit the v002 response teacher, so it is not a strict holdout for checkpoints trained on teacher-derived data. These runs score the both-plane v001 objective, not the requested-plane RMS of the local PyAT reports under `evals/`; compare only runs with the same objective.
