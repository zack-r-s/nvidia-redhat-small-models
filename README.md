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

The chat also accepts explicit goals such as “Center the horizontal orbit,” “Correct the vertical beam,” or “Reduce the beam orbit.” It reads the requested plane's BPMs and corrector setpoints first, then asks the model for one batch of absolute kicks. The harness accepts only distinct correctors in the requested plane and finite values within ±0.35 mrad. It reads the written correctors' readbacks and the same BPMs after the attempt, reporting measured RMS before and after. A worsening proposal is still measured and reported; the model does not get an invented improvement. Vague requests such as “Move the beam” still ask for a specific setpoint and kick. Exact setpoint commands retain their verified read/write/read flow, and read-only prompts cannot write. The 230M base model may not yet choose good corrections; this interface enables evaluation and later fine-tuning.

For read-only interpretation, “Which orbit plane has the larger measured RMS?” reads all eight BPM channels and reports the measured X and Y RMS values and the larger plane.

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

## Generate and validate Xopt traces

The data lane records read → write → read attempts against the served PyAT accelerator through EPICS, using the same 24-PV contract as the chat and viewer. `datasets/v001/` contains 64 episodes and 4,096 attempts with episode-level train, validation, and test splits. The physical traces are separate from language annotations and model training. [Dataset documentation](datasets/README.md) describes the row schema, provenance, and annotation pass.

The first collection used an isolated Osprey deployment at `/private/tmp/hackring-data-v001-monitor`. At that time the checked-in deck had `Marker` BPMs, so staging converted only those four zero-length elements to `Monitor` and verified the test orbit was unchanged. The merged main deck now uses `Monitor` directly; staging accepts it without changing the checked-in lattice or facility files. The recorded v001 manifest retains the historical conversion provenance.

To stage a new isolated deployment, use a fresh `--deployment` path. It requires free CA/PVA ports 5064/5075 and the pinned Osprey source checkout described above. On this Mac, Rancher Desktop supplies the Docker socket:

```sh
PYTHONPATH=src .venv/bin/python scripts/prepare_dataset_va.py --deployment /private/tmp/hackring-data-next --build
DOCKER_HOST=unix://$HOME/.rd/docker.sock .venv/bin/osprey up --repo /private/tmp/hackring-data-next --dev -d
PYTHONPATH=src .venv/bin/python scripts/generate_dataset.py --deployment /private/tmp/hackring-data-next --output datasets/v002
PYTHONPATH=src .venv/bin/python scripts/validate_dataset.py datasets/v002
```

The generator refuses to overwrite an output version. The full default pass is 64 episodes × 64 sequential Xopt evaluations with bounded ±0.35 mrad correctors. The driver sets hidden seeded disturbances, reads all public PVs, applies EPICS writes, and records the settled public PVs and objective after each attempt. Run `PYTHONPATH=src .venv/bin/python scripts/validate_dataset.py datasets/v001` to verify the checked-in first pass. The annotation script writes a separate version under `datasets/v001/annotations/`; it does not edit the raw traces. No fine-tuning is performed here.
