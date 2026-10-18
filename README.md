# Hackathon FODO ring

This repo uses one 20-cell, 1 GeV PyAT storage ring. Four BPMs and eight corrector setpoints expose a fixed [24-PV contract](data/pv_contract.json): positions in mm and kicks in mrad. `src/hackring/lattice.py` generates the [facility deck](data/facility/decks/RING.json); its four BPM elements are PyAT `Monitor`s. The default demo connects the model chat, controls, and Blueprint/Three viewer to **one local PyAT fixture**. Osprey/EPICS remains available as a separate integration path for data generation and accelerator testing.

## Setup

Use Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js/npm, and a local vLLM installation. On the development Mac, vLLM Metal runs from `~/.venv-vllm-metal/bin/vllm`; set `VLLM_BIN` if yours is elsewhere. The default command installs Python and frontend dependencies, checks the ring, serves the pinned [v004 checkpoint](https://huggingface.co/zack-r-s/hackring-LFM2.5-230M/tree/v004) on port 8000 if needed, and starts the shared fixture, chat bridge, and viewer:

```sh
scripts/bootstrap_demo.sh --check
scripts/bootstrap_demo.sh
# Add --no-open on a headless host.
```

The viewer opens at `http://127.0.0.1:5173/?driver=1`; the same PyAT state is exposed at `http://127.0.0.1:8001/state` and used by the chat at `http://127.0.0.1:8002/chat`. The agent status and goal panel is populated from confirmed tool outcomes and measured before/after RMS. `POST /chat/reset` resets the seeded plant and clears the chat session. The test driver is available only under `?driver=1`; replay mode disables live controls. The browser displays measured BPM samples as red dots and animates a green cursor along a separate, densely sampled PyAT 4D closed-orbit profile from the local fixture. The stepped speed slider controls display playback only; Max shows the sampled path as a continuous green line. The profile is omitted from Osprey/EPICS state because its 24 public PVs do not expose hidden model faults; no trajectory is inferred from four BPMs.

Try `Report beam position.`, `Set RING:MAG:H01:KICK:SP to 0.1 mrad, then read RING:DIAG:BPM01:POSITION:X.`, and `Center the beam in both planes.` The exact numeric setpoint takes the harness's bounded read → write → read path and is labeled `channel_write · harness`; the model chooses `orbit_correct(plane, effort)` for orbit goals. The latter skill uses a bounded response-matrix solver and PyAT measures the outcome. Requests beyond ±0.35 mrad are refused. The [four-case live smoke check](evals/mark-iii/live-fixture-v004.json) records the actual model, tools, measured plant, and result.

`scripts/dev_up.sh --live` starts only the local fixture, bridge, and viewer when a matching model already serves port 8000. `scripts/dev_up.sh` starts the scripted bridge for frontend development without inference. `scripts/demo.sh --manual` and plain `scripts/demo.sh` retain the manual and Xopt fixture drivers. The fixture demo needs neither Docker nor Osprey. For the separate Osprey/EPICS path, use `scripts/bootstrap_demo.sh --osprey`; it builds and starts this checkout's Osprey deployment and web terminal, and has its own port/volume lifecycle. Use `--check` before that path. The checked-in lattice and PV contract are shared; the runtime services are distinct.

## Evaluation and QA

[Mark III](evals/mark-iii/README.md) is the post-reconciliation evaluation series. It freezes 28 seed-disjoint PyAT correction cases, a direct `channel_write` Gym task, and a separate `orbit_correct` Gym task. Each Gym rollout starts from a fresh seeded plant and allows one tool attempt; requested-plane BPM RMS determines reward and success. Four stochastic repeats per task are reported separately from greedy local PyAT scores. The [model/data chart](training/README.md) points to each published version and its matching evaluation. Mark I and Mark II are historical and are not relabeled as Mark III.

Run `PYTHONPATH=src:scripts .venv/bin/python -m pytest -q tests` and `npm --prefix frontend run build` for code validation. `scripts/qa_state_match.py` compares all 24 displayed PVs with the fixture at desktop and mobile sizes, while `scripts/qa_poll_order.py` checks serial polling; pass `--chrome /path/to/Chrome` on a host with system Chrome, or install Playwright Chromium first. [Frontend QA notes](frontend/QA.md) record the latest checks. The viewer is read-only except for the explicit development driver and chat controls; chat writes update the same plant that the viewer polls.

## Generate and validate Xopt traces

The data lane records read → write → read attempts against the served PyAT accelerator through EPICS, using the same 24-PV contract as the chat and viewer. `datasets/v001/` contains 64 episodes and 4,096 Xopt attempts. The v002 pass fits a response-matrix teacher from v001 train data and measures fresh correction and exact-write tasks on the served ring, with separate Liquid-format conversation exports. The v003 pass retains canonical served-plant examples and adds measured local-PyAT examples for the evaluator's error-kick disturbance. V004 uses fresh local PyAT measurements to train a natural-language `orbit_correct(plane, effort)` skill call; the bounded response-matrix solver supplies numerical PV settings. Physical traces remain separate from language annotations and model training. [Dataset documentation](datasets/README.md) describes the schemas, provenance, and commands; the [training and release chart](training/README.md) points to each run's data, checkpoint, and evaluation.

Public releases use one [versioned dataset repository](https://huggingface.co/datasets/zack-r-s/hackring-pyat-control) and one [versioned LFM model repository](https://huggingface.co/zack-r-s/hackring-LFM2.5-230M). Matching dataset/model tags `v002`, `v003`, and `v004` identify each pass. The [release chart](training/README.md) maps the local training-run directories, legacy tags, and each evaluation. Mark III separates the model's raw intent choice from the deployed correction skill and the explicit-write harness.

The first collection used an isolated Osprey deployment at `/private/tmp/hackring-data-v001-monitor`. At that time the checked-in deck had `Marker` BPMs, so staging converted only those four zero-length elements to `Monitor` and verified the test orbit was unchanged. The merged main deck now uses `Monitor` directly; staging accepts it without changing the checked-in lattice or facility files. The recorded v001 manifest retains the historical conversion provenance.

To stage a new isolated deployment, use a fresh `--deployment` path. It requires free CA/PVA ports 5064/5075 and the [pinned Osprey source checkout](https://github.com/als-apg/osprey/tree/d639869). On this Mac, Rancher Desktop supplies the Docker socket:

```sh
PYTHONPATH=src .venv/bin/python scripts/prepare_dataset_va.py --deployment /private/tmp/hackring-data-next --build
DOCKER_HOST=unix://$HOME/.rd/docker.sock .venv/bin/osprey up --repo /private/tmp/hackring-data-next --dev -d
PYTHONPATH=src .venv/bin/python scripts/generate_dataset.py --deployment /private/tmp/hackring-data-next --output datasets/v005
PYTHONPATH=src .venv/bin/python scripts/validate_dataset.py datasets/v005
```

The generator refuses to overwrite an output version. The full default pass is 64 episodes × 64 sequential Xopt evaluations with bounded ±0.35 mrad correctors. The driver sets hidden seeded disturbances, reads all public PVs, applies EPICS writes, and records the settled public PVs and objective after each attempt. Run `PYTHONPATH=src .venv/bin/python scripts/validate_dataset.py datasets/v001` to verify the checked-in first pass. The annotation script writes a separate version under `datasets/v001/annotations/`; it does not edit the raw traces. No fine-tuning is performed here.
