# Hackathon FODO ring

This repo uses one 20-cell, 1 GeV PyAT storage ring for the Osprey controls agent and the Blueprint/Three visualizer. Four BPMs and eight corrector setpoints expose a fixed [24-PV contract](data/pv_contract.json): positions in mm and kicks in mrad. `src/hackring/lattice.py` generates the [facility deck](data/facility/decks/RING.json); its four BPM elements are PyAT `Monitor`s so Osprey can serve them. In the shared demo, the chat and read-only visualizer use the same served accelerator. The local Xopt fixture is separate and needs no inference server.

## Setup

Use Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js/npm, and an Osprey-supported Docker engine. From the repo root:

For a clean local demo on a new machine, install and start Docker, install uv and Node.js/npm, and install vLLM separately. On Apple Silicon, the [vLLM Metal installer](https://docs.vllm.ai/projects/vllm-metal/en/latest/installation/) supplies `~/.venv-vllm-metal/bin/vllm`; on a supported NVIDIA Linux host, follow the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/). Then run:

```sh
scripts/bootstrap_demo.sh --check
scripts/bootstrap_demo.sh
```

The script installs the Python, pinned Osprey, and frontend dependencies; resets **only this checkout's Osprey deployment** with `osprey reset`; builds and starts the accelerator; starts local LFM2.5-230M if it is not already served; and opens the viewer/API and authenticated web terminal. It prints the chat login URL and viewer URL. Set `VLLM_BIN=/path/to/vllm` if the executable is outside `PATH`. Docker must already be running. The reset removes this deployment's containers, volumes, rendered `build/`, and agent memory, while retaining the source and audit log. The script saves any existing `.env` to ignored `var/bootstrap/env.before-bootstrap` before replacing it with the local vLLM example. Osprey refuses the reset when a different checkout owns containers with this deployment's project name.

The bootstrap selects a free Osprey port block and separate EPICS CA/PVA ports before resetting. It writes those choices to the ignored host overlay `profiles/bootstrap-local.yml` and selects it with `.env.variant`; the checked-in profile and lattice stay unchanged. The chat and viewer read the rendered deployment ports, so they follow this local overlay. This lets the demo coexist with the Xopt data-generation Osprey stack. If ports 8001 or 5173 are already occupied, stop that viewer/API first; those two frontend ports are fixed.

If the earlier `hackring-live-230m` demo is still running, the bootstrap stops it automatically before startup, freeing its ports and resources while keeping its volumes and leaving the Xopt data stack alone. The old checkout defaults to a sibling directory named `hackring-live-230m`; set `LEGACY_DEMO_REPO=/path/to/checkout` if it lives elsewhere. Use `--keep-old-demo` to leave it running. A failed `osprey up` also stops this bootstrap's partial deployment, so a retry does not leave its containers holding ports. ClickHouse's internal port 9000 is unrelated to host-port selection; a timeout there can also indicate that the Docker VM is overloaded.

The manual setup below remains useful when working on only one part of the stack:

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

Serve LFM2.5-230M with vLLM's `lfm2` tool parser. The custom web-terminal agent calls this endpoint directly; `.env` sets `VLLM_BASE_URL`. For a server on the same host:

```sh
vllm serve LiquidAI/LFM2.5-230M --served-model-name LFM2.5-230M \
  --enable-auto-tool-choice --tool-call-parser lfm2 --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/v1/models
```

The `lfm2` parser is required for structured tool calls. On the tested Apple Silicon host running vLLM Metal alongside Docker, `--max-model-len 4096 --gpu-memory-utilization 0.05` worked for 230M; tune these resource flags for your host. A 1.2B comparison server is optional and does not change the checked-in 230M provider configuration.

The checked-in profile has one served PyAT virtual accelerator, a recorded archive, controls MCP tools, and the custom small-model web terminal. It has no planning, rule, or approval hooks. Start the accelerator after building, then start the viewer API in a separate terminal:

```sh
# If Rancher Desktop is your Docker engine, first export DOCKER_HOST=unix://$HOME/.rd/docker.sock
.venv/bin/osprey up --dev -d
scripts/shared_demo.sh
```

`shared_demo.sh` checks the deployment container and all 24 EPICS PVs before starting the API at `http://127.0.0.1:8001/state` and the viewer at `http://127.0.0.1:5173/`. It does not launch a fixture IOC. In another terminal run `.venv/bin/osprey web`; open the authenticated URL it prints. The chat and viewer remain in separate tabs. Both use the accelerator named by this worktree's rendered `build/config.yml`; the checked-in default CA/PVA ports are 5064/5075. The agent and `shared_demo.sh` read the project name and accelerator ports from that rendered config, so a host-local overlay can move them after rebuilding. Do not launch two accelerators on the same CA/PVA ports, or run the viewer API while another service owns 8001/5173.

For manual setup when data generation already owns 5064/5075, make a separate demo checkout and give that copy a unique `project_name`, `virtual_accelerator.port`, and `virtual_accelerator.pva_port` through a host overlay before building it. The bootstrap above does this automatically. Keep the checked-in lattice and facility contract unchanged. The caproto UI fixture also uses 6064, so stop it before using that particular demo port.

For the end-to-end check, note the starting values at `/state`, then ask the chat: `Set RING:MAG:H01:KICK:SP to 0.05, then read RING:DIAG:BPM01:POSITION:X`. The named setpoint's unit is mrad; an explicit `mrad` is also accepted. Look for a `channel_read · pre-write`, a `channel_write`, and Osprey's reported outcome. Compare the H01 setpoint and readback and at least one BPM in chat, `/state`, and the viewer. A later setpoint command replaces the earlier value. The API's `backend` field and viewer label should say `osprey-epics`/“Osprey”. An out-of-limit request above ±0.35 mrad should leave the setpoint unchanged and report the rejection. `POST /scenario` is unavailable in EPICS mode.

The small-model chat currently needs an explicit corrector setpoint and absolute kick for a beam move. Vague requests such as “Move the beam” or “Disturb the beam” ask for those details; target-orbit optimization is not yet wired into this chat loop. The agent offers `channel_write` only for an explicit setpoint request, checks its proposed channel and value, reads the setpoint before writing, and reads any PV explicitly requested after the write before reporting Osprey's outcome. It stores concise verified replies in conversation history so raw tool results do not fill the 230M model's context. A read-only prompt cannot write through this agent. If no write occurred, the response says so.

## Frontend fixture and QA

For UI work without Docker or inference, `scripts/demo.sh --manual` starts the local PyAT machine and a development-only control panel; plain `scripts/demo.sh` also drives it with Xopt. `scripts/demo.sh --epics` uses a **caproto fixture IOC** on CA port 6064, not Osprey. Stop any other accelerator using 6064 before starting that fixture. `?driver=1` exposes local-fixture presets only. The production viewer only issues `GET /state`; replay remains at `/?replay=/replay/demo.json`. The agent intent, action status, and target panels show “Unavailable” until an event contract exists. [Frontend QA notes](frontend/QA.md) describe the browser checks; `scripts/qa_state_match.py` and `scripts/qa_poll_order.py` cover displayed values and polling order. Pass `--chrome` to those QA scripts to use a system Chrome executable instead of Playwright's bundled browser.

The API emits `hackring.state/1`. `EpicsMachine` gets every measured value from EPICS and reads the checked-in deck only for static device positions and circumference. `LocalMachine` is a separate PyAT fixture. [The frontend handoff prompt](frontend/CLAUDE_PROMPT.md) describes current boundaries and remaining UI work; [the Osprey panel snippet](frontend/osprey-panel.yml) remains optional and is not enabled in `profile.yml`.

Run the physical and fixture checks with:

```sh
PYTHONPATH=src .venv/bin/python scripts/check_ring.py
PYTHONPATH=src .venv/bin/python -m pytest tests
cd frontend && npm run build
```

The `ara/` research artifact, `build/`, `var/`, virtual environments, frontend dependencies, and `.env` secrets are gitignored.
