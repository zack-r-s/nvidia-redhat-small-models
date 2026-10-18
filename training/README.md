# Training data and model releases

**Evaluation status:** [Mark III](../evals/mark-iii/README.md) is the new PR #8 shared-fixture rerun. It validates the immutable v001–v004 data, evaluates direct `channel_write` and bounded `orbit_correct` as separate interfaces, and records local PyAT, NeMo Gym, and live-UI checks under frozen task hashes. Mark I and the partial Mark II runs remain historical; compare checkpoints only within the same Mark III task and scoring rule.

The [Mark II Gym runs](../var/gym/mark-ii/README.md) were a partial rerun: v004 used the wrong `channel_write` interface and some v002/v003 rollouts made multiple write calls, so their one-call figures required post-hoc rescoring. Mark III replaces that comparison with separate frozen direct and skill tasks, one-attempt enforcement in the server, and fresh local PyAT and dataset checks.

Dataset and published model tags use the same version. The local `training/vNNN/` directories retain the chronological training-run numbers; they are one behind the dataset/model tags because the first raw Xopt collection is `datasets/v001/` and has no trained checkpoint.

| Data version | Training record | Published model | Historical Mark I local PyAT evaluation |
| --- | --- | --- | --- |
| [`v001`](../datasets/v001/manifest.json): raw Xopt traces | — | — | — |
| [`v002`](https://huggingface.co/datasets/zack-r-s/hackring-pyat-control/tree/v002): Osprey/PyAT control conversations, including the GPT-augmented SFT export | [`training/v001`](v001/README.md) · [results](v001/results.json) | [`model v002`](https://huggingface.co/zack-r-s/hackring-LFM2.5-230M/tree/v002) | [8-case report](../evals/mark-i/pyat-v001/liquid-v001/report.json) · [56-case report](../evals/mark-i/pyat-v002/liquid-v001/report.json) |
| [`v003`](https://huggingface.co/datasets/zack-r-s/hackring-pyat-control/tree/v003): canonical served examples plus measured local PyAT examples | [`training/v002`](v002/README.md) · [results](v002/results.json) | [`model v003`](https://huggingface.co/zack-r-s/hackring-LFM2.5-230M/tree/v003) | [56-case report](../evals/mark-i/pyat-v002/liquid-v002/report.json) · [paired comparison](../evals/mark-i/pyat-v002/comparison.json) |
| [`v004`](https://huggingface.co/datasets/zack-r-s/hackring-pyat-control/tree/v004): measured PyAT skill-routing conversations | [`training/v003`](v003/README.md) · [results](v003/results.json) | [`model v004`](https://huggingface.co/zack-r-s/hackring-LFM2.5-230M/tree/v004) | [strict frozen report](../evals/mark-i/pyat-v004/liquid-v004/report.json) · [balanced-default replay](../evals/mark-i/pyat-v004/default-replay.json) · [v004 test report](../evals/mark-i/pyat-v004/test-v004/report.json) |

**Material change in v004:** v002 and v003 train the model to emit numerical `channel_write` setpoints. V004 trains the model to choose `orbit_correct(plane, effort)` from the operator's wording. A deterministic bounded response-matrix solver then chooses the PV values, and PyAT supplies the measured after-state. A v004 correction score therefore measures model routing plus the solver, not the model's ability to calculate kicks. The v004 pass also uses fresh local PyAT examples only, without a GPT language pass; it is scoped to orbit correction rather than BPM reporting and exact writes.

The complete [Mark III results index](../evals/mark-iii/results.json) and [protocol](../evals/mark-iii/README.md) are the current comparison. It revalidated all four immutable data versions and evaluated pinned base and released checkpoints through separate direct-write and skill interfaces:

| Interface and measure | Base 230M | Base 1.2B | v002 | v003 | v004 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Local direct, greedy MPS (56 mixed tasks) | 1 | 0 | 24 | 26 | — |
| Gym direct, vLLM (112 correction rollouts) | 0 | 0 | 12 | 6 | — |
| Local skill, greedy CPU strict (28 corrections) | 0 | 0 | — | — | 12 |
| Gym skill, vLLM (112 correction rollouts) | 73 | 28 | — | — | 72 |

The local direct evaluator forces the write tool name, while Gym scores model tool selection and permits only one attempt; their counts are not interchangeable. V002/v003 beat base on the local direct task, but the v002-v003 ordering reverses in Gym. V004's Gym skill success is statistically indistinguishable from base 230M on this fixed task set (72 versus 73); its raw plane-and-effort selection is 42/112 versus 31/112, with a paired task-bootstrap interval that includes no advantage. The [live fixture](../evals/mark-iii/live-fixture-v004.json) and [browser UI](../evals/mark-iii/ui-smoke-v004.json) pass their shared-state smoke checks. Explicit numeric writes are executed and labeled by a bounded harness action, not credited to model tool selection.

The later live chat bridge retries a mismatched `orbit_correct` plane once before any plant write and falls back to a measured BPM read when a beam-report turn produces no verified tool result. Those runtime assists are labeled in chat traces and are **not** part of the frozen Mark III model scores; a new evaluation would need to declare and test that changed interface explicitly.

In the earlier local greedy test on 28 frozen correction prompts, raw v004 plane-and-effort choice passed 12/28 (base 0/28), and plane choice alone passed 28/28. The live policy normalizes unspecified effort to balanced; replaying the saved calls with that rule passed 28/28 for v004 and 14/28 for base. On 72 seed-distinct v004 test prompts, raw intent passed 67/72. These are different measurements and should not be compared as if they were one model-only control score.

All three model checkpoints start from `LiquidAI/LFM2.5-230M`; neither model v003 nor v004 continues an earlier fine-tune. The model tags name their **input dataset version**, not the chronological local training directory. The original model `v001` tag remains as a legacy alias for the dataset-v002 checkpoint, and `legacy-training-v002` preserves the former model-`v002` reference to the dataset-v003 checkpoint. Use the matching `v002`, `v003`, and `v004` tags for new work.

The [dataset repository](https://huggingface.co/datasets/zack-r-s/hackring-pyat-control) and [model repository](https://huggingface.co/zack-r-s/hackring-LFM2.5-230M) are public. Generated weights stay outside Git; the tracked results files contain their SHA-256 hashes and training configuration. The 56-case local PyAT evaluation passed 21/56 for model v002 and 20/56 for model v003, so the newer checkpoint is not established as a better controller.
