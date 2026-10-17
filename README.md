# Hackathon FODO ring starter

This repository fixes one small accelerator model for the agent, optimizer, and viewer lanes. The source of truth is `src/hackring/lattice.py`: a 20-cell, 1 GeV FODO storage ring with four BPMs and four horizontal plus four vertical correctors. It generates a PyAT deck and Osprey facility records in `data/facility/`. The public contract has **24 PVs**: eight BPM X/Y readings and eight corrector SP/RB pairs, with positions in mm and kicks in mrad. Four hidden error kicks create repeatable disturbances without becoming agent controls. `data/pv_contract.json` names every public PV.

No training data or model fine-tuning is started here. The local Xopt driver exists to move a deterministic PyAT machine while the frontend is built. The frontend is only a Blueprint + React Three Fiber starter; its next-task prompt is in `frontend/CLAUDE_PROMPT.md`.

## Developer setup

Run the commands below from the repository root. You need Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js with npm, and Git. Osprey is pinned to the source commit used to validate this starter; it is installed separately from this project's Python package.

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python 'git+https://github.com/als-apg/osprey.git@d639869'
cp .env.example .env
cd frontend && npm ci && cd ..
```

The tracked `data/facility/` files already contain the generated lattice and PV mapping. Regenerate them after editing `src/hackring/lattice.py`, then run the physical and Osprey checks:

```sh
PYTHONPATH=src .venv/bin/python -m hackring.lattice
PYTHONPATH=src .venv/bin/python scripts/check_ring.py
.venv/bin/osprey validate
.venv/bin/osprey facility validate
.venv/bin/osprey build
cd frontend && npm run build
```

`osprey build` renders ignored output under `build/`; rerun it after changing `profile.yml`, `providers.yml`, or the facility data. The Osprey source version used here is [`als-apg/osprey` commit `d639869`](https://github.com/als-apg/osprey/tree/d639869).

## Inspect the Osprey interface

From the repository root, check the rendered deployment and launch its web terminal:

```sh
.venv/bin/osprey status
.venv/bin/osprey web --shell zsh
```

Open the `Open:` URL printed by `osprey web`. The `--shell zsh` form checks the interface without starting an agent session or requiring inference. Stop the foreground server with Ctrl-C. `osprey status` may report that Docker is stopped even when this web-only check works. For the agent terminal, run `.venv/bin/osprey web` instead; that path also needs a working local vLLM server and the agent CLI Osprey launches.

The profile uses Osprey's **in-process PyAT virtual accelerator** and controls tools. Its `vllm` provider requests `LFM2.5-230M` at the `VLLM_BASE_URL` in `.env` (default `http://host.docker.internal:8000/v1`). The example env file also supplies `VLLM_API_KEY`, which Osprey expects even if your local vLLM server does not check it. On a supported vLLM host, serve the [instruction-tuned checkpoint](https://huggingface.co/LiquidAI/LFM2.5-230M) under that API name using [vLLM's serve command](https://docs.vllm.ai/en/stable/cli/serve/):

```sh
vllm serve LiquidAI/LFM2.5-230M --served-model-name LFM2.5-230M --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/v1/models
```

Set `VLLM_BASE_URL` to an address reachable from the Osprey process if the default host alias does not resolve in your environment. The model ID returned by `/v1/models` must match `providers.yml`. This starter's Osprey build was validated, but vLLM inference was not launched or tested. The profile has no planning workflows, subagents, or approval hooks; simulator PV limits are defined in `data/facility/limits.yaml`.

## Develop the viewer without inference

Use three terminals, starting each command from the repository root. Terminal 1 serves the local PyAT machine:

```sh
PYTHONPATH=src .venv/bin/uvicorn hackring.api:app --host 127.0.0.1 --port 8001
```

Terminal 2 starts the minimal Blueprint/Three viewer; open the Vite URL it prints (normally `http://127.0.0.1:5173`):

```sh
cd frontend && npm run dev
```

Terminal 3 creates a seeded disturbance and moves the ring with Xopt so the viewer has changing measurements:

```sh
PYTHONPATH=src .venv/bin/python scripts/xopt_demo.py --steps 35 --delay 0.35
```

To check the API directly, run `curl http://127.0.0.1:8001/state`. It exposes `GET /state`, `POST /scenario`, and `POST /write`. The viewer polls `/state` every half-second and consumes the `hackring.state/1` contract from `src/hackring/machine.py`; `data/pv_contract.json` lists its PVs and units. Xopt uses bounded Nelder–Mead over eight kicks (±0.35 mrad). This demo writes no dataset, and exploratory steps may temporarily increase orbit RMS. Re-run the driver to reset the disturbance with the same seed.

`HACKRING_BACKEND=epics` switches the API to PyEPICS reads and writes for a **served** Osprey virtual accelerator using the same PV names. The committed Osprey profile currently uses the simpler in-process mode, so EPICS mode requires a future served-VA deployment and its Channel Access configuration. Scenario reset in EPICS mode also awaits wiring to Osprey's model RPC. The local PyAT API and Osprey simulator are separate machine instances; the former is the frontend fixture until this bridge is completed.

One-command visual demo: `scripts/demo.sh` starts the API and viewer, opens the browser, runs Xopt, and cleans up. Options: `--epics` (caproto IOC + PyEPICS path), `--steps`, `--delay`, `--seed`, `--loops N`, `--keep`, and `--manual` (no Xopt; opens the viewer with the test-driver panel).

**Manual test driver:** `npm run dev` with `?driver=1` (e.g. `http://127.0.0.1:5173/?driver=1`) shows a drop-down of preset beam states (seeded disturbances, single/limit corrector kicks, zeroing, and an out-of-limit write that must be rejected) with Apply/Reset buttons. It is dev-only, in `frontend/src/TestDriver.tsx`, the sole module that calls `/scenario` and `/write`; production builds and the default viewer stay read-only. Scenario presets need the local backend (`/scenario` is 501 in EPICS mode).

### Viewer states, replay, and EPICS-mode testing

The viewer is read-only (`GET /state` only) and shows **Live**, **Stale**, **Disconnected**, or **Replay**. Set `VITE_MACHINE_URL` to point at a different API. Open `http://127.0.0.1:5173/?replay=/replay/demo.json` to replay recorded snapshots (re-record with `scripts/record_replay.py --seed 7` while `xopt_demo.py` runs). Agent intent, action status, and target/goal panels read "Unavailable" until the backend provides those contracts.

To exercise the PyEPICS path deterministically without a served Osprey VA, install dev extras (`uv pip install --python .venv/bin/python -e '.[dev]'`) and run a caproto soft IOC over the same PyAT plant. Use these env vars in every terminal: `EPICS_CA_ADDR_LIST=127.0.0.1 EPICS_CA_AUTO_ADDR_LIST=NO EPICS_CAS_INTF_ADDR_LIST=127.0.0.1 EPICS_CAS_AUTO_BEACON_ADDR_LIST=NO EPICS_CAS_BEACON_ADDR_LIST=127.0.0.1`.

```sh
PYTHONPATH=src .venv/bin/python scripts/ca_fixture.py --seed 7
HACKRING_BACKEND=epics PYTHONPATH=src .venv/bin/uvicorn hackring.api:app --port 8001
PYTHONPATH=src .venv/bin/python scripts/xopt_demo.py   # /scenario is 501 in EPICS mode; the IOC seed sets the disturbance
PYTHONPATH=src .venv/bin/python -m pytest tests
```

`frontend/osprey-panel.yml` holds the `profile.yml` snippet that mounts the viewer as an Osprey URL panel (not applied yet). See `frontend/QA.md` for visual QA notes.

## Parallel lanes

- **Machine/Osprey:** Keep `lattice.py`, generated facility records, and `pv_contract.json` in sync. Validate the PA wiring and add served-VA access when needed.
- **Frontend:** Start with `frontend/CLAUDE_PROMPT.md`; preserve `hackring.state/1` and the PV names.
- **Data/agent:** Use the fixed PV contract and bounded VOCS. Later record complete episodes as ordered read → write → read transitions with scenario seed and optimizer metadata; add operator language after the physical traces exist. No data-generation pipeline is included yet.

The `ara/` research artifact is local and gitignored. `build/`, `var/`, and secrets in `.env` are also ignored.
