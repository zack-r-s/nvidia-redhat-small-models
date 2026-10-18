#!/usr/bin/env bash
# Fixture-first demo; --osprey retains the separate served-accelerator workflow.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ " $* " != *" --osprey "* ]]; then
  exec "$ROOT/scripts/bootstrap_fixture.sh" "$@"
fi
legacy_args=()
for argument in "$@"; do
  [[ $argument == --osprey ]] || legacy_args+=("$argument")
done
set -- "${legacy_args[@]}"
cd "$ROOT"
PIN=d639869
OSPREY_SRC="$ROOT/var/bootstrap/osprey-src"
BOOT="$ROOT/var/bootstrap"
PY="$ROOT/.venv/bin/python"
OSPREY="$ROOT/.venv/bin/osprey"
MODEL_URL=http://127.0.0.1:8000/v1/models

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'bootstrap: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing $1. Install it and rerun."; }

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_demo.sh --osprey [--check] [--keep-old-demo] [--dev-chat]
                                 [--model MODEL] [--revision REVISION]
                                 [--served-name NAME] [--tool-parser PARSER]

Resets only this checkout's Osprey deployment, installs pinned project
dependencies, starts the selected vLLM model if needed, and brings up the
Osprey accelerator, viewer/API, and authenticated web terminal.

--check        Check host prerequisites without changing anything.
--dev-chat     Serve the viewer with its chat panel through scripts/dev_up.sh
               (fixture API 8001, chat bridge 8002, viewer 5173) instead of the
               Osprey viewer. The viewer and chat then show the local PyAT dev
               fixture, not Osprey; the Osprey stack and web terminal still start.
--model        Hugging Face ID or checkpoint directory (relative to repo root)
               (env VLLM_MODEL; default LiquidAI/LFM2.5-230M).
--revision     Hugging Face model tag or commit (env VLLM_REVISION).
--served-name  Model ID the endpoint reports and the chat requests
               (env VLLM_SERVED_NAME; default the model's last path part).
--tool-parser  vLLM tool-call parser (env VLLM_TOOL_PARSER; default lfm2,
               which only fits LFM models).
VLLM_SERVE_ARGS replaces the extra serve flags; on macOS the default is
"--max-model-len 4096 --gpu-memory-utilization 0.05", tuned for 230M.
The old hackring-live-230m demo is stopped automatically when it is running.
Its volumes and the Xopt stack remain. --keep-old-demo skips that cleanup.

If vllm is outside PATH, set VLLM_BIN to its executable. On Apple Silicon the
script also checks ~/.venv-vllm-metal/bin/vllm. Docker must already be running.
To serve v004, pass --model zack-r-s/hackring-LFM2.5-230M --revision v004.
The reset destroys this deployment's containers, volumes, build/, and agent
memory. It preserves the source and audit log. This script backs up any .env
before replacing it with the local vLLM example.
EOF
}

MODE=run
CLEANUP_OLD_DEMO=1
DEV_CHAT=0
VLLM_MODEL=${VLLM_MODEL:-${HACKRING_VLLM_MODEL:-LiquidAI/LFM2.5-230M}}
VLLM_REVISION=${VLLM_REVISION:-${HACKRING_VLLM_REVISION:-}}
MODEL_ID=${VLLM_SERVED_NAME:-}
VLLM_TOOL_PARSER=${VLLM_TOOL_PARSER:-lfm2}
while (( $# )); do
  case "$1" in
    --check) MODE=check ;;
    --cleanup-old-demo) CLEANUP_OLD_DEMO=1 ;;
    --keep-old-demo) CLEANUP_OLD_DEMO=0 ;;
    --dev-chat) DEV_CHAT=1 ;;
    --model|--revision|--served-name|--tool-parser)
      (( $# >= 2 )) && [[ -n $2 ]] || { usage >&2; exit 2; }
      case "$1" in
        --model) VLLM_MODEL=$2 ;;
        --revision) VLLM_REVISION=$2 ;;
        --served-name) MODEL_ID=$2 ;;
        --tool-parser) VLLM_TOOL_PARSER=$2 ;;
      esac
      shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done
MODEL_ID=${MODEL_ID:-$(basename "${VLLM_MODEL%/}")}
if [[ -n ${VLLM_SERVE_ARGS+set} ]]; then
  read -r -a serve_extra <<<"$VLLM_SERVE_ARGS"
elif [[ $(uname -s) == Darwin ]]; then
  serve_extra=(--max-model-len 4096 --gpu-memory-utilization 0.05)
else
  serve_extra=()
fi

[[ -f profile.yml && -f .env.example && -f data/pv_contract.json ]] || die 'Run from a complete hackring checkout.'
for cmd in git uv npm docker curl; do need "$cmd"; done

# Rancher Desktop on macOS may not set DOCKER_HOST for non-interactive shells.
if [[ -z ${DOCKER_HOST:-} && -S "$HOME/.rd/docker.sock" ]]; then
  export DOCKER_HOST="unix://$HOME/.rd/docker.sock"
fi
docker info >/dev/null 2>&1 || die 'Docker is unavailable. Start Docker Desktop or Rancher Desktop and check docker info.'
if ! command -v python3.12 >/dev/null 2>&1; then
  uv python find 3.12 >/dev/null 2>&1 || die 'Python 3.12 is unavailable. Run: uv python install 3.12'
fi

VLLM_CMD=${VLLM_BIN:-}
if [[ -z $VLLM_CMD ]]; then
  VLLM_CMD=$(command -v vllm || true)
fi
if [[ -z $VLLM_CMD && -x "$HOME/.venv-vllm-metal/bin/vllm" ]]; then
  VLLM_CMD="$HOME/.venv-vllm-metal/bin/vllm"
fi
if [[ -n $VLLM_CMD && ! -x $VLLM_CMD ]]; then
  die "VLLM_BIN is not executable: $VLLM_CMD"
fi

model_ready() {
  curl -fsS --max-time 3 "$MODEL_URL" 2>/dev/null |
    "$PY" -c 'import json,sys; assert any(m.get("id") == sys.argv[1] for m in json.load(sys.stdin)["data"])' "$MODEL_ID" 2>/dev/null || return 1
  # /models identifies the served name, not the checkpoint revision. When pinned,
  # trust only a process this checkout launched with the requested source and tag.
  if [[ -n $VLLM_REVISION ]]; then
    [[ -f "$BOOT/vllm.pid" ]] || return 1
    local pid command_line
    pid=$(cat "$BOOT/vllm.pid")
    [[ $pid =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null || return 1
    command_line=" $(ps -p "$pid" -o command= 2>/dev/null || true) "
    [[ $command_line == *" $VLLM_MODEL "* && $command_line == *" --revision $VLLM_REVISION "* ]] || return 1
  fi
}

port_free() {
  "$PY" - "$1" <<'PY'
import socket, sys
with socket.socket() as sock:
    # Like uvicorn and vite: lingering TIME_WAIT sockets do not block a new server, a listener does.
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', int(sys.argv[1])))
PY
}

if [[ $MODE == check ]]; then
  if [[ -z $VLLM_CMD ]] && ! curl -fsS --max-time 3 "$MODEL_URL" >/dev/null 2>&1; then
    die 'vLLM is missing and no model server is available on 127.0.0.1:8000.'
  fi
  say 'Host checks passed: git, uv/Python 3.12, npm, Docker, curl, and vLLM or an existing model endpoint.'
  printf 'Model: %s revision %s served as %s (tool parser %s)\n' "$VLLM_MODEL" "${VLLM_REVISION:-default}" "$MODEL_ID" "$VLLM_TOOL_PARSER"
  if [[ -n $VLLM_CMD ]]; then
    printf 'vLLM: %s\n' "$VLLM_CMD"
  else
    printf 'Model endpoint: %s\n' "$MODEL_URL"
  fi
  exit 0
fi

if [[ -f .env.variant ]]; then
  selected_variant=$(sed -n 's/^OSPREY_PROFILE_VARIANT=//p' .env.variant | tail -1)
  [[ -z $selected_variant || $selected_variant == bootstrap-local ]] ||
    die ".env.variant selects '$selected_variant'; bootstrap will not replace another host variant."
  if [[ $selected_variant == bootstrap-local && ! -f profiles/bootstrap-local.yml ]]; then
    die '.env.variant selects bootstrap-local, but its profile overlay is missing.'
  fi
fi

# Fail before changing deployment state if the model cannot be served locally.
if curl -fsS --max-time 3 "$MODEL_URL" >/dev/null 2>&1; then
  : # Validate its exact model after the project Python environment is ready.
elif [[ -z $VLLM_CMD ]]; then
  die 'No vLLM server on 127.0.0.1:8000 and no vllm executable. Install vLLM and rerun (or set VLLM_BIN).'
fi

say 'Installing project and pinned Osprey dependencies'
mkdir -p "$BOOT"
uv venv --allow-existing --python 3.12 "$ROOT/.venv"
uv pip install --python "$PY" -e '.[dev]'
if [[ ! -d "$OSPREY_SRC/.git" ]]; then
  git clone https://github.com/als-apg/osprey.git "$OSPREY_SRC"
  git -C "$OSPREY_SRC" checkout --detach "$PIN"
fi
[[ $(git -C "$OSPREY_SRC" rev-parse HEAD) == "$PIN"* ]] ||
  die "$OSPREY_SRC is not at pinned commit $PIN; move that source cache aside and rerun."
SETUPTOOLS_SCM_PRETEND_VERSION=2026.6.2+gd639869 \
  uv pip install --python "$PY" -e "$OSPREY_SRC" \
  -e "$OSPREY_SRC/packages/osprey-connectors" \
  -e "$OSPREY_SRC/packages/pyaml-cs-osprey"
npm --prefix frontend ci

old_demo_running=$(docker ps -q --filter label=com.docker.compose.project=hackring-live-230m)
if [[ $CLEANUP_OLD_DEMO == 1 && -n $old_demo_running ]]; then
  legacy_repo=${LEGACY_DEMO_REPO:-"$ROOT/../hackring-live-230m"}
  [[ -f "$legacy_repo/profile.yml" ]] ||
    die "Old demo checkout not found at $legacy_repo; set LEGACY_DEMO_REPO to its path."
  legacy_project=$("$PY" - "$legacy_repo" <<'PY'
import sys
import yaml
from pathlib import Path
repo = Path(sys.argv[1])
rendered = repo / "build" / "config.yml"
source = rendered if rendered.is_file() else repo / "profile.yml"
print(yaml.safe_load(source.read_text())["project_name"])
PY
)
  [[ $legacy_project == hackring-live-230m ]] ||
    die "Refusing to stop $legacy_repo: its project is '$legacy_project', not hackring-live-230m."
  say 'Stopping the earlier live demo to free its ports and Docker resources (volumes kept)'
  "$OSPREY" web stop --repo "$legacy_repo" >/dev/null 2>&1 || true
  "$OSPREY" down --repo "$legacy_repo"
  [[ -z $(docker ps -q --filter label=com.docker.compose.project=hackring-live-230m) ]] ||
    die 'The earlier live demo still has running containers after osprey down.'
elif [[ $CLEANUP_OLD_DEMO == 0 && -n $old_demo_running ]]; then
  say 'Keeping the earlier live demo running; Docker will host three Osprey stacks.'
fi

# A model switch replaces the vLLM server an earlier bootstrap started; other servers are left alone.
if [[ -f "$BOOT/vllm.pid" ]] && ! model_ready; then
  old_pid=$(cat "$BOOT/vllm.pid")
  if [[ $old_pid =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null &&
     [[ $(ps -p "$old_pid" -o command= 2>/dev/null || true) == *serve* ]]; then
    say "Stopping this checkout's previous vLLM server (pid $old_pid) to serve $MODEL_ID"
    kill "$old_pid"
    for _ in {1..30}; do
      kill -0 "$old_pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$old_pid" 2>/dev/null && die "Previous vLLM server $old_pid did not exit; stop it and rerun."
  fi
  rm -f "$BOOT/vllm.pid"
fi
if curl -fsS --max-time 3 "$MODEL_URL" >/dev/null 2>&1 && ! model_ready; then
  die "Port 8000 serves a different model or an unverified revision; expected $MODEL_ID from $VLLM_MODEL ${VLLM_REVISION:-default}. Stop that server and rerun."
fi
if ! model_ready && ! port_free 8000; then
  die "Port 8000 is occupied without a working $MODEL_ID endpoint. Stop that process and rerun."
fi

say 'Stopping this checkout’s previous web terminal and viewer, if present'
"$OSPREY" web stop --repo "$ROOT" >/dev/null 2>&1 || true
if [[ -f "$BOOT/viewer.pid" ]]; then
  old_pid=$(cat "$BOOT/viewer.pid")
  if [[ $old_pid =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    old_command=$(ps -p "$old_pid" -o command= 2>/dev/null || true)
    # Either viewer mode: the Osprey viewer, or dev_up.sh, which stops its children on TERM.
    if [[ $old_command == *"$ROOT/scripts/shared_demo.sh"* || $old_command == *"scripts/dev_up.sh"* ]]; then
      kill "$old_pid"
      for _ in {1..20}; do kill -0 "$old_pid" 2>/dev/null || break; sleep 0.25; done
    fi
  fi
  rm -f "$BOOT/viewer.pid"
fi

viewer_ports=(8001 5173)
(( DEV_CHAT )) && viewer_ports+=(8002)
for port in "${viewer_ports[@]}"; do
  # A stopped viewer's servers can take a few seconds to release their ports.
  for _ in {1..40}; do port_free "$port" && break; sleep 0.5; done
  if ! port_free "$port"; then
    die "Port $port is occupied by another process. Stop it before resetting this deployment."
  fi
done

# Reserve a complete Osprey port block plus both EPICS protocol ports. The
# selected overlay is host-local and never changes the checked-in profile.
read -r PORT_BASE CA_PORT PVA_PORT < <("$PY" - <<'PY'
import socket

def free(port):
    for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
        with socket.socket(socket.AF_INET, kind) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                return False
    return True

for index in range(1, 21):
    base = 10000 + 1000 * index
    ca = 5064 + 1000 * index
    pva = 5075 + 1000 * index
    if all(free(port) for port in range(base, base + 1000)) and free(ca) and free(pva):
        print(base, ca, pva)
        break
else:
    raise SystemExit("No free Osprey port block and EPICS port pair found in the bootstrap range")
PY
)
[[ -n ${PORT_BASE:-} ]] || die 'Could not select free deployment ports.'
say "Selected isolated ports: block $PORT_BASE-$((PORT_BASE + 999)), CA $CA_PORT, PVA $PVA_PORT"

say 'Resetting only this Osprey deployment'
"$OSPREY" reset --repo "$ROOT" --dry-run
if [[ -f .env ]]; then
  cp .env "$BOOT/env.before-bootstrap"
  chmod 600 "$BOOT/env.before-bootstrap"
fi
"$OSPREY" reset --repo "$ROOT" --yes
cp .env.example .env
chmod 600 .env
printf 'VLLM_MODEL_NAME=%s\n' "$MODEL_ID" >> .env
if [[ -n $VLLM_REVISION ]]; then
  printf 'VLLM_MODEL_REVISION=%s\n' "$VLLM_REVISION" >> .env
fi

mkdir -p profiles
"$PY" - "$ROOT" "$PORT_BASE" "$CA_PORT" "$PVA_PORT" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
base, ca, pva = map(int, sys.argv[2:])
project = "hackring-demo-" + hashlib.sha256(str(root).encode()).hexdigest()[:8]
(root / "profiles" / "bootstrap-local.yml").write_text(
    f"project_name: {project}\n"
    f"virtual_accelerator:\n  port: {ca}\n  pva_port: {pva}\n"
    f"config:\n  deployment.port_base: {base}\n"
)
print(f"Host-local Osprey project: {project}")
PY
printf 'OSPREY_PROFILE_VARIANT=bootstrap-local\n' > .env.variant

say 'Validating and rebuilding Osprey and frontend'
PYTHONPATH=src "$PY" scripts/check_ring.py
"$OSPREY" validate
"$OSPREY" facility validate
SETUPTOOLS_SCM_PRETEND_VERSION=2026.6.2+gd639869 \
  "$OSPREY" build --skip-deps --skip-lifecycle --dev
npm --prefix frontend run build

say "Starting local $MODEL_ID if needed"
if ! model_ready; then
  if curl -fsS --max-time 3 "$MODEL_URL" >/dev/null 2>&1; then
    die "Port 8000 serves a different model; expected $MODEL_ID. Stop that server and rerun."
  fi
  [[ -n $VLLM_CMD ]] || die 'vLLM executable disappeared.'
  model_args=(serve "$VLLM_MODEL" --served-model-name "$MODEL_ID"
    --enable-auto-tool-choice --tool-call-parser "$VLLM_TOOL_PARSER" --host 127.0.0.1 --port 8000
    ${serve_extra[@]+"${serve_extra[@]}"})
  if [[ -n $VLLM_REVISION ]]; then
    model_args+=(--revision "$VLLM_REVISION")
  fi
  nohup "$VLLM_CMD" "${model_args[@]}" >"$BOOT/vllm.log" 2>&1 &
  echo $! >"$BOOT/vllm.pid"
  for _ in {1..180}; do
    model_ready && break
    kill -0 "$(cat "$BOOT/vllm.pid")" 2>/dev/null || die "vLLM exited; see $BOOT/vllm.log"
    sleep 2
  done
  model_ready || die "vLLM did not become ready; see $BOOT/vllm.log"
fi

say 'Starting the served PyAT accelerator'
if ! "$OSPREY" up --dev -d; then
  say 'Stopping this checkout’s partial deployment to release its ports (volumes kept)'
  "$OSPREY" down --repo "$ROOT" || true
  die 'Osprey did not become healthy. See the startup error above and var/logs.'
fi

if (( DEV_CHAT )); then
  say 'Starting dev-fixture viewer, machine API, and chat bridge (not Osprey)'
  # dev_up.sh runs from the repo root and reads VLLM_BASE_URL; the bridge takes the served name from .env.
  nohup "$ROOT/scripts/dev_up.sh" --live --no-open >"$BOOT/viewer.log" 2>&1 &
  viewer_urls=(http://127.0.0.1:8001/state http://127.0.0.1:8002/chat/health http://127.0.0.1:5173/)
else
  say 'Starting viewer API and frontend'
  nohup "$ROOT/scripts/shared_demo.sh" >"$BOOT/viewer.log" 2>&1 &
  viewer_urls=(http://127.0.0.1:8001/state http://127.0.0.1:5173/)
fi
echo $! >"$BOOT/viewer.pid"
for url in "${viewer_urls[@]}"; do
  ready=0
  for _ in {1..60}; do
    if curl -fsS --max-time 3 "$url" >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$(cat "$BOOT/viewer.pid")" 2>/dev/null || die "Viewer exited; see $BOOT/viewer.log"
    sleep 1
  done
  [[ $ready == 1 ]] || die "$url did not become ready; see $BOOT/viewer.log"
done

say 'Starting authenticated Osprey web terminal'
"$OSPREY" web --repo "$ROOT" --detach
if (( DEV_CHAT )); then
  printf '\nViewer: http://127.0.0.1:5173/?driver=1  (dev fixture with chat panel, not Osprey)\n'
  printf 'Machine: http://127.0.0.1:8001/state  (local PyAT fixture)\nChat: http://127.0.0.1:8002/chat\n'
  printf 'Osprey web terminal (above) still drives the served accelerator, which this viewer does not show.\n'
else
  printf '\nViewer: http://127.0.0.1:5173/\nMachine: http://127.0.0.1:8001/state\n'
fi
printf 'Logs: %s\n' "$BOOT"
