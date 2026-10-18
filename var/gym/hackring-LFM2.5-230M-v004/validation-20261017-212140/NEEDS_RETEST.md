# Needs retest: interface mismatch

This run does not measure v004's intended behavior. Do not compare its reward with other checkpoints.

- **Model:** `zack-r-s/hackring-LFM2.5-230M` at revision `14419bfc11d03da0fb92902a7cd15e7d2a9d0fa4` (tag `v004`), served as `hackring-LFM2.5-230M-v004`.
- **What v004 is trained for:** calling `orbit_correct(plane, effort)`, where a deterministic bounded least-squares skill computes and writes the corrector kicks. The model never emits kick values.
- **What this eval offered:** only `channel_write` with `tool_choice: required`, the interface used by `scripts/small_agent.py` on this branch. v004 was not trained on that tool.
- **Result:** mean reward 0.000; 0 accepted and 46 rejected writes over 64 rollouts. Rejections: 39 malformed batches (truncated channel names such as `RING:DIAG` and `RING:MAG`, string values), 5 BPM channels, 2 `RING:DIAG`. 18 rollouts had no tool call because they hit the 512-token limit.

## Retest when

`hackring_orbit` has an `orbit_correct` task variant:

1. Offer `orbit_correct(plane, effort)` as the task tool, matching the harness on `zrs-lfm-training`.
2. Run the same bounded solver in the resources server, against each episode's hidden calibration offsets.
3. Score the measured orbit-cost reduction, plus whether the plane and effort are correct.

Then rerun this revision with `scripts/run_gym_eval.sh --num-repeats 4` and replace this note with a link to the new run.
