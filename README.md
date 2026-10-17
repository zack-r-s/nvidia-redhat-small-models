# Hackathon FODO ring

This repo uses one 20-cell, 1 GeV PyAT storage ring for the Osprey controls agent and the Blueprint/Three visualizer. Four BPMs and eight corrector setpoints expose a fixed [24-PV contract](data/pv_contract.json): positions in mm and kicks in mrad. `src/hackring/lattice.py` generates the [facility deck](data/facility/decks/RING.json); its four BPM elements are PyAT `Monitor`s so Osprey can serve them. The visualizer reads measurements but does not issue writes. The local Xopt fixture remains available for frontend development without inference.

## Setup

Use Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js/npm, and an Osprey-supported Docker engine. From the repo root:

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[dev]'
git clone https://github.com/als-apg/osprey.git /private/tmp/osprey-hackring-d639869
git -C /private/tmp/osprey-hackring-d639869 checkout d639869
SETUPTOOLS_SCM_PRETEND_VERSION=2026.6.2+gd639869 uv pip install --python .venv/bin/python \
  -e /private/tmp/osprey-hackring-d639869 \
  -e /private/tmp/osprey-hackring-d639869/packages/osprey-connectors \
  -e /private/tmp/osprey-hackring-d639869/packages/pyaml-cs-osprey
cp .env.example .env
cd frontend && npm ci && cd ..
PYTHONPATH=src .venv/bin/python scripts/check_ring.py
.venv/bin/osprey validate
.venv/bin/osprey facility validate
```

Osprey is pinned to [`d639869`](https://github.com/als-apg/osprey/tree/d639869). Its source checkout reports a lower wheel version than its workspace dependency floor. For this pinned commit, the metadata override below lets the local `--dev` build use the same source; it does not change the source code:

```sh
SETUPTOOLS_SCM_PRETEND_VERSION=2026.6.2+gd639869 .venv/bin/osprey build --skip-deps --skip-lifecycle --dev
cd frontend && npm run build && cd ..
```

Build output is ignored under `build/`. Rebuild after changing the profile, provider, or facility data. `.env` supplies `VLLM_BASE_URL` and `VLLM_API_KEY`; the model ID returned by the endpoint must be `LFM2.5-230M` as configured in `providers.yml`.

## Shared Osprey chat and viewer demo

Use a supported vLLM host to serve the instruction-tuned model, then set `VLLM_BASE_URL` to its `/v1` endpoint reachable from the host running the Osprey web terminal. For a server on the same host:

```sh
vllm serve LiquidAI/LFM2.5-230M --served-model-name LFM2.5-230M \
  --enable-auto-tool-choice --tool-call-parser lfm2 --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/v1/models
```

The `lfm2` parser is required for the chat's forced tool calls. On an Apple Silicon host running vLLM Metal alongside Docker, `--max-model-len 4096 --gpu-memory-utilization 0.05` kept this demo's memory use low; tune those values for your host.

The checked-in profile follows the small served deployment used for data collection: `hello-world` preset, one PyAT virtual accelerator, a recorded archive, controls MCP tools, and the custom small-model web terminal. It has no planning, rule, or approval hooks. Start it from a separate terminal after the build:

```sh
# If Rancher Desktop is your Docker engine, first export DOCKER_HOST=unix://$HOME/.rd/docker.sock
.venv/bin/osprey up --dev -d
scripts/shared_demo.sh
```

`shared_demo.sh` checks that all 24 EPICS PVs answer before starting the API at `http://127.0.0.1:8001/state` and the viewer at `http://127.0.0.1:5173/`. It does not launch a fixture IOC. In another terminal run `.venv/bin/osprey web` and open its chat separately from the visualizer. The chat's controls MCP server and the viewer API use the same served accelerator through Channel Access ports 5064 and 5075. Do not start this deployment on those ports while another accelerator, including the data-generation deployment, is using them.

For the end-to-end check, record `/state`, ask chat to read `RING:MAG:H01:KICK:SP`, set it to `0.05` mrad, then read `RING:MAG:H01:KICK:RB`. The readback, at least one BPM measurement, `/state`, and the viewer must change together. An out-of-limit request above ±0.35 mrad must leave the setpoint unchanged. The API's `backend` field and viewer label should say `osprey-epics`/“Osprey” in this mode. `POST /scenario` is unavailable in EPICS mode; do not use its dev-only controls for this test.

The small-model chat currently needs an explicit corrector setpoint and kick for a beam move. A vague “Move the beam” request asks for those details; target-orbit optimization is not yet wired into this chat loop.

## Frontend fixture and QA

For UI work without Docker or inference, `scripts/demo.sh` runs the local PyAT machine and Xopt; `scripts/demo.sh --epics` uses a **caproto fixture IOC** on isolated CA port 6064, not Osprey. `?driver=1` exposes development-only presets for the local machine. The production viewer only issues `GET /state`; replay remains at `/?replay=/replay/demo.json`. The agent intent, action status, and target panels explicitly say “Unavailable” until an event contract exists. [Frontend QA notes](frontend/QA.md) describe the browser checks; `scripts/qa_state_match.py` and `scripts/qa_poll_order.py` cover displayed values and polling order. Pass `--chrome` to those QA scripts to use a system Chrome executable instead of Playwright's bundled browser.

The API emits `hackring.state/1`. `EpicsMachine` gets every measured value from EPICS and reads the checked-in deck only for static device positions and circumference. `LocalMachine` is a separate PyAT fixture. [The frontend handoff prompt](frontend/CLAUDE_PROMPT.md) remains the starting point for later Blueprint/Three work; [the Osprey panel snippet](frontend/osprey-panel.yml) is intentionally not enabled.

Run the physical and fixture checks with:

```sh
PYTHONPATH=src .venv/bin/python scripts/check_ring.py
PYTHONPATH=src .venv/bin/python -m pytest tests
cd frontend && npm run build
```

The `ara/` research artifact, `build/`, `var/`, virtual environments, frontend dependencies, and `.env` secrets are gitignored.
