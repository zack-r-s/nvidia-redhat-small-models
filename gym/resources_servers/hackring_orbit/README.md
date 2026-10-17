# hackring_orbit

NeMo Gym resources server for orbit correction on the hackring FODO ring. Each task is one recorded episode from `datasets/v001`. The model gets the deployed small agent's goal turn: the goal system prompt, `Correct the beam orbit.`, the harness's pre-write `channel_read` of all eight BPMs and eight corrector setpoints, and one `channel_write` tool (Osprey's schema, snapshotted in `osprey_tools.json`) with `tool_choice: required`. The agent's `max_steps: 1` allows one write batch, as in `scripts/small_agent.py`.

The server replays the episode offline in PyAT. It applies the episode's hidden corrector calibration offsets to the checked-in deck, so each kick is setpoint + offset; the tests reproduce v001's recorded `orbit_cost` before and after sampled attempts. Offsets go to the server only; the model sees public PVs. A write batch is accepted or rejected whole, using the small agent's rules: distinct correctors in the allowed list, finite values within ±0.35 mrad, numeric strings accepted.

`verify` scores the final state with the v001 objective `x_rms_mm^2 + y_rms_mm^2 + 0.01*sum(sp_mrad^2)`:

- `reward`: fractional cost reduction `(initial - final) / initial`, clipped to [-1, 1]. No write or a rejected write scores 0; a worsening write is negative.
- `orbit_cost_initial`, `orbit_cost_final`, `x_rms_final_mm`, `y_rms_final_mm`, `accepted_writes`, `rejected_writes`.
- `xopt_best_orbit_cost` and `beats_xopt_best`: the lowest cost Xopt reached in that episode over 64 evaluations. This is a reference for one write batch, not a target.

## Data

`train.jsonl` holds the first 75% of v001 episodes (fine-tuning portion, usable for RL); `validation.jsonl` holds the last 25% (evaluation). Both are generated and ignored; `example.jsonl` (three evaluation tasks) is committed. Regenerate from the repo root:

```sh
PYTHONPATH=src:scripts .venv/bin/python scripts/prepare_gym_data.py datasets/v001/splits/ft75-eval25/split.json
```

## Run

From the repo root, with the model served by `scripts/bootstrap_demo.sh`:

```sh
scripts/run_gym_eval.sh                          # all 16 evaluation tasks
scripts/run_gym_eval.sh --limit 2 --num-repeats 1
```

It reads `VLLM_BASE_URL`, `VLLM_API_KEY`, and `VLLM_MODEL_NAME` from `.env` and writes each run to `var/gym/<served model name>/<split>-<timestamp>/`: `run.json` (served name, model source, endpoint, split, repo commit, extra arguments), `rollouts.jsonl`, `rollouts_aggregate_metrics.json`, `rollouts_materialized_inputs.jsonl`, and `rollouts_failures.jsonl`. NeMo Gym 0.4.0 finds out-of-tree servers from the working directory, so the wrapper runs from `gym/`. Server tests: `cd gym && gym env test --resources-server hackring_orbit`.

## Baselines

Committed runs are under `var/gym/<model>/<run>/` (16 validation tasks × 4 repeats, temperature 0.2, 512 output tokens):

| Model | mean reward | accepted writes | rollouts |
|---|---|---|---|
| LFM2.5-230M | 0.000 | 0 / 64 | `var/gym/LFM2.5-230M/validation-20261017-181953` |
| LFM2.5-1.2B-Instruct | 0.000 | 0 / 64 | `var/gym/LFM2.5-1.2B-Instruct/validation-20261017-183220` |

Both models copy the pre-write read into `channel_write`: every written BPM value matched the reading for 230M (352/352) and most for 1.2B (388/418), and every corrector value was its current 0.0, so even applying only the corrector operations changes nothing. Reference policies on the same tasks (raw fractional reduction): one-shot regularized response-matrix solve from the read +0.999, Xopt best of 64 +0.943, random kicks within ±0.35 mrad −26. Reproduce with:

```sh
gym/resources_servers/hackring_orbit/.venv/bin/python gym/resources_servers/hackring_orbit/analyze_runs.py \
  var/gym/LFM2.5-230M/validation-20261017-181953 var/gym/LFM2.5-1.2B-Instruct/validation-20261017-183220
```

# Licensing information
Code: same as this repository.
Data: generated from this repository's `datasets/v001`.

Dependencies
- nemo_gym: Apache 2.0
- accelerator-toolbox: BSD-3-Clause
