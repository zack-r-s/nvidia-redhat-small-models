# Needs retest: interface mismatch (Mark II)

This Mark II run does not measure v004's intended behavior. Do not compare its reward with other checkpoints.

- **Model:** `zack-r-s/hackring-LFM2.5-230M` tag `v004`, revision `14419bfc11d03da0fb92902a7cd15e7d2a9d0fa4`, served as `hackring-LFM2.5-230M-v004`.
- **Interface:** v004 is trained to call `orbit_correct(plane, effort)`, with a deterministic bounded solver computing the kicks. This task offers only `channel_write` (`evaluation_interface: channel_write` in `run.json`).
- **Result:** mean reward 0.000; 0 accepted writes over 64 rollouts. 31 oversized or malformed batches, 5 disallowed channels, 28 rollouts with no tool call.

Retest after `hackring_orbit` gains a frozen `orbit_correct` task variant (same tool and solver as the v004 harness, scored on measured cost reduction plus plane and effort), then replace this note with a link to that run. The Mark I diagnostic is `var/gym/mark-i/hackring-LFM2.5-230M-v004/`.
