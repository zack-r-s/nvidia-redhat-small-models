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

## Setup

Tested with NeMo Gym 0.4.0, vLLM 0.31.0 (vllm-metal on an Apple M2 Max), and Python 3.12 for the project. From a fresh clone:

```sh
uv venv --python 3.12 && uv pip install --python .venv/bin/python -e '.[dev,gym]'
pip install nemo-gym==0.4.0      # any Python >= 3.12 environment that puts `gym` on PATH
gym --version                    # NeMo Gym v0.4.0
```

`uv` must be on `PATH`; Gym builds this server's own venv on first use. The `gym` extra supplies the `mcp` and `python-dotenv` imports that task generation needs from `scripts/small_agent.py`; Osprey and Docker are not required for the evaluation. `run_gym_eval.sh` refuses other Gym versions because it relies on 0.4.0 behavior (`--split`, out-of-tree server lookup from the working directory, `++` overrides); set `NEMO_GYM_ALLOW_UNTESTED=1` to try one anyway. Check the server with `cd gym && gym env test --resources-server hackring_orbit`.

## Serve a model

The evaluation talks to any OpenAI-compatible vLLM endpoint that uses the `lfm2` tool parser. `scripts/bootstrap_demo.sh --model ...` serves one on port 8000 and records its name in `.env`, but it also resets the Osprey demo. To evaluate without Osprey, serve the model directly. Pin `--revision` for a Hugging Face model so the run is reproducible:

```sh
vllm serve zack-r-s/LFM2.5-230M-hackring-v001 --revision bbb7d45470cf55d3ddf0a78c7c971431ae9d200f \
  --served-model-name LFM2.5-230M-hackring-v001 --enable-auto-tool-choice --tool-call-parser lfm2 \
  --host 127.0.0.1 --port 8012 --max-model-len 4096 --gpu-memory-utilization 0.05
```

`--max-model-len` and `--gpu-memory-utilization` are the values used on the tested Mac; adjust them for your host. A local checkpoint directory works in place of the Hugging Face ID.

## Run

```sh
scripts/run_gym_eval.sh --num-repeats 4                       # model named in .env, all 16 evaluation tasks
VLLM_BASE_URL=http://127.0.0.1:8012/v1 VLLM_MODEL_NAME=LFM2.5-230M-hackring-v001 \
  scripts/run_gym_eval.sh --num-repeats 4                     # another endpoint; shell values win over .env
scripts/run_gym_eval.sh --limit 2                             # smoke test
```

The wrapper checks that the endpoint serves the named model, regenerates task rows from `split.json`, and runs `gym eval run` from `gym/`, where NeMo Gym 0.4.0 finds out-of-tree servers. Sampling defaults match the deployed agent (temperature 0.2, 512 output tokens); extra options pass through and override them. Each run writes `var/gym/<served model name>/<split>-<timestamp>/`, which is committed:

- `run.json`: served name, model source, model revision and how it was determined, vLLM version, backend, and server command, NeMo Gym version, evaluation host, split, repo commit, and extra arguments. For a local server, the revision and backend are read from the listening vLLM process; otherwise set `VLLM_MODEL_REVISION`. `scripts/gym_run_metadata.py backfill <run>` adds these fields to an older run while its server is still running.
- `rollouts.jsonl`, `rollouts_aggregate_metrics.json`, `rollouts_materialized_inputs.jsonl`, `rollouts_failures.jsonl`, and Gym's `preprocessed_datasets/`.

Runs are not bit-for-bit reproducible. Sampling at temperature 0.2 has no fixed seed, so repeated runs of one model differ by a few rollouts; compare models with `--num-repeats 4` or more rather than single samples. Different vLLM backends or hardware, such as vllm-metal and CUDA builds, can also change sampled tokens, so record and compare runs with the same `vllm_version` and `vllm_backend` in `run.json`. Physics, tasks, and scoring are deterministic: the same rollout text always receives the same reward.

## Baselines

Committed runs are under `var/gym/<model>/<run>/` (16 validation tasks × 4 repeats = 64 rollouts, temperature 0.2, 512 output tokens; a rollout without a complete tool call has no write):

| Model | mean reward | accepted / rejected writes | rollouts |
|---|---|---|---|
| LFM2.5-230M | 0.000 | 0 / 51 | `var/gym/LFM2.5-230M/validation-20261017-181953` |
| LFM2.5-1.2B-Instruct | 0.000 | 0 / 54 | `var/gym/LFM2.5-1.2B-Instruct/validation-20261017-183220` |
| LFM2.5-230M-hackring-v001 (v002 SFT) | −0.041 | 67 / 0 | `var/gym/LFM2.5-230M-hackring-v001/validation-20261017-191459` |

The two base models copy the pre-write read into `channel_write`: every written BPM value matched the reading for 230M (352/352) and most for 1.2B (388/418), and every corrector value was its current 0.0, so even applying only the corrector operations changes nothing. The v002 fine-tune writes only correctors and every batch is accepted, but its kick directions are uncorrelated with the one-shot solve (26/64 point the right way) and its median kick is 0.05× the solve's size, so rewards scatter around zero (30 positive, 34 negative). Reference policies on the same tasks (raw fractional reduction): one-shot regularized response-matrix solve from the read +0.999, Xopt best of 64 +0.943, random kicks within ±0.35 mrad −26. Reproduce with:

```sh
gym/resources_servers/hackring_orbit/.venv/bin/python gym/resources_servers/hackring_orbit/analyze_runs.py \
  var/gym/LFM2.5-230M/validation-20261017-181953 var/gym/LFM2.5-1.2B-Instruct/validation-20261017-183220 \
  var/gym/LFM2.5-230M-hackring-v001/validation-20261017-191459
```

# Licensing information
Code: same as this repository.
Data: generated from this repository's `datasets/v001`.

Dependencies
- nemo_gym: Apache 2.0
- accelerator-toolbox: BSD-3-Clause
