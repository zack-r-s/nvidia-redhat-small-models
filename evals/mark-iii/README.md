# Mark III evaluation protocol

Mark III evaluates the PR #8 shared local PyAT fixture after the demo wiring and Gym interface correction. Mark I and Mark II remain historical. The v001–v004 manifests are revalidated in [`dataset-validation.json`](dataset-validation.json); no training data or weights are changed.

The frozen correction source is `evals/mark-i/pyat-v002/cases.jsonl` (28 correction cases, unique seeds outside the training splits). [`prepare_gym_mark_iii.py`](../../scripts/prepare_gym_mark_iii.py) checks two committed Gym task files against those cases and the checked-in deck. The direct task provides the harness's pre-write PV read and offers `channel_write`; the skill task offers `orbit_correct(plane, effort)` and lets the bounded solver read the plant. Every Gym session gets a fresh `LocalMachine.reset(seed, 0.08)` and permits one tool attempt. An extra or rejected attempt earns zero reward.

Gym reward is clipped fractional reduction of requested-plane BPM RMS squared. The separate `goal_success` field requires a single confirmed attempt and a lower measured RMS in **each** requested plane. For an unspecified effort, the skill path uses the deployed balanced-default normalization; raw plane-and-effort selection remains a separate metric. Direct-write and skill results are distinct comparisons. Gym samples at temperature 0.2 with four repeats. The 56-case local direct evaluator uses greedy Transformers inference on Apple MPS (`bfloat16`) for all four checkpoints; the 28-case local skill evaluator uses greedy Transformers CPU (`float32`). Both reset the same seeded local PyAT plant. Differences in inference backend, scoring, and tool interface must stay visible in comparisons.

The intended model matrix is base 230M, base 1.2B, v002, and v003 on direct writes; base 230M, base 1.2B, and v004 on skill selection. The local mixed-task report also covers BPM reporting and exact writes. The UI smoke check exercises read, write, correction, rejection, and reset on the shared fixture.

## Completed results

[`results.json`](results.json) validates and indexes every full run, its pinned Hugging Face revision, source/task hash, raw rollout file, and local report. All seven Gym runs contain 28 frozen tasks × four repeats and zero rollout failures. The 56-case local direct task includes 28 corrections, 10 exact writes, nine BPM reports, and nine plane comparisons; the local skill task contains the 28 corrections only.

| Direct-write checkpoint | Local greedy MPS /56 | Gym measured success /112 | Gym confirmed writes /112 |
| --- | ---: | ---: | ---: |
| Base 230M | 1 | 0 | 0 |
| Base 1.2B Instruct | 0 | 0 | 0 |
| Model v002 | 24 | 12 | 69 |
| Model v003 | 26 | 6 | 73 |

The local direct task forces the `channel_write` tool name, then checks the model's PV arguments and measures any valid write. Gym instead asks the model to use the supplied tool through vLLM at temperature 0.2 and enforces one attempt. The 24-versus-26 local difference and 12-versus-6 Gym difference point in opposite directions; neither establishes v003 as the better direct controller.

| Skill checkpoint | Local greedy CPU strict /28 | Gym measured success /112 | Gym raw plane+effort /112 |
| --- | ---: | ---: | ---: |
| Base 230M | 0 | 73 | 31 |
| Base 1.2B Instruct | 0 | 28 | 0 |
| Model v004 | 12 | 72 | 42 |

The local [balanced-default replay](skill/default-replay.json) reaches 14/28 for base 230M and 28/28 for v004 from their saved greedy calls; that is a separate harness-plus-solver measure. In the corrected Gym skill task, v004 and base 230M are effectively tied on measured success (72/112 and 73/112). A paired bootstrap over the 28 tasks estimates v004 minus base success at −0.9 percentage points with a 95% interval of −22.3 to +20.5 points; raw intent is +9.8 points with a −10.7 to +29.5 point interval. These intervals are descriptive resampling intervals for this fixed task set, not proof of wider generalization. The 1.2B model often attempted multiple tools, which the one-attempt rule rejects even if an earlier write was accepted.

The [live API smoke](live-fixture-v004.json) passed four cases, and the [browser smoke](ui-smoke-v004.json) verified that a chat write, Agent & goal panel, displayed PV, and reset use the same fixture. Exact numeric writes and over-limit refusals are bounded harness behavior; orbit-goal tool selection is the model's behavior. The [pre-guard smoke](live-fixture-v004-pre-guard.json) records the exact-write failure that led to the explicit harness path and is not a final pass result. No moving 3D beam or extrapolated trajectory was introduced.

To reproduce the direct local reports, run `PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=src:scripts .venv/bin/python scripts/eval_local_pyat.py --model <pinned snapshot> --cases evals/mark-i/pyat-v002/cases.jsonl --output <new directory> --series mark-iii --device mps`. For Gym, serve each pinned revision with vLLM and use `scripts/run_gym_mark_iii.sh direct|skill --num-repeats 4`; the wrapper checks the prepared interface. Run `PYTHONPATH=src:scripts .venv/bin/python scripts/summarize_mark_iii.py` after all seven runs to verify and regenerate the index. Mark II's shared `channel_write` task for v004 and early Mark III skill diagnostics loaded the wrong dataset; they are not included in this table.
