# Hackathon FODO ring starter

This repository fixes one small accelerator model for the agent, optimizer, and viewer lanes. The source of truth is `src/hackring/lattice.py`: a 20-cell, 1 GeV FODO storage ring with four BPMs and four horizontal plus four vertical correctors. It generates a PyAT deck and Osprey facility records in `data/facility/`. The public contract has **24 PVs**: eight BPM X/Y readings and eight corrector SP/RB pairs, with positions in mm and kicks in mrad. Four hidden error kicks create repeatable disturbances without becoming agent controls. `data/pv_contract.json` names every public PV.

No training data or model fine-tuning is started here. The local Xopt driver exists to move a deterministic PyAT machine while the frontend is built. The frontend is only a Blueprint + React Three Fiber starter; its next-task prompt is in `frontend/CLAUDE_PROMPT.md`.

## Setup and checks

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). The Osprey source version used to validate this starter is `als-apg/osprey` commit `d639869` (October 2026). Install it separately from this project's dependencies:

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python 'git+https://github.com/als-apg/osprey.git@d639869'
PYTHONPATH=src .venv/bin/python -m hackring.lattice
PYTHONPATH=src .venv/bin/python scripts/check_ring.py
.venv/bin/osprey validate
.venv/bin/osprey facility validate
.venv/bin/osprey build
```

The Osprey profile uses its **PyAT virtual accelerator in-process** and the controls tool server. It selects the `vllm` provider at `http://host.docker.internal:8000/v1`, requesting the model ID `LFM2.5-230M`. Start a local vLLM OpenAI-compatible server with `--served-model-name LFM2.5-230M`, or update `providers.yml` to match the ID your server exposes. Copy `.env.example` to `.env` to override `VLLM_BASE_URL` for your host. This setup only validates/renders Osprey; it does not launch vLLM or invoke inference. The profile deliberately has no planning workflows, subagents, or approval hooks. Its local simulator and hard PV limits remain the boundary for writes.

## Frontend demonstration without inference

Run these in separate terminals after setup:

```sh
PYTHONPATH=src .venv/bin/uvicorn hackring.api:app --host 127.0.0.1 --port 8001
cd frontend && npm install && npm run dev
PYTHONPATH=src .venv/bin/python scripts/xopt_demo.py --steps 35 --delay 0.35
```

The API exposes `GET /state`, `POST /write`, and `POST /scenario`. The viewer polls `/state` and shows the same `hackring.state/1` shape that the future EPICS adapter will expose. Xopt runs bounded Nelder–Mead over eight kicks (±0.35 mrad), resetting a seeded disturbance before it starts. The script does not save rows; each candidate changes the live state for visual testing. The optimizer may temporarily increase orbit RMS while it explores.

`HACKRING_BACKEND=epics` switches the API to PyEPICS reads and writes for a **served** Osprey virtual accelerator using the same PV names. The committed Osprey profile currently uses the simpler in-process mode, so EPICS mode requires a future served-VA deployment and its Channel Access configuration. Scenario reset in EPICS mode also awaits wiring to Osprey's model RPC. The local PyAT API and Osprey simulator are separate machine instances; the former is the frontend fixture until this bridge is completed.

## Parallel lanes

- **Machine/Osprey:** Keep `lattice.py`, generated facility records, and `pv_contract.json` in sync. Validate the PA wiring and add served-VA access when needed.
- **Frontend:** Start with `frontend/CLAUDE_PROMPT.md`; preserve `hackring.state/1` and the PV names.
- **Data/agent:** Use the fixed PV contract and bounded VOCS. Later record complete episodes as ordered read → write → read transitions with scenario seed and optimizer metadata; add operator language after the physical traces exist. No data-generation pipeline is included yet.

The `ara/` research artifact is local and gitignored. `build/`, `var/`, and secrets in `.env` are also ignored.
