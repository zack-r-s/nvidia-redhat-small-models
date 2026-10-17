#!/usr/bin/env bash
# Recreate this checkout's Osprey deployment and start the shared live demo.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PIN=d639869
OSPREY_SRC="$ROOT/var/bootstrap/osprey-src"
BOOT="$ROOT/var/bootstrap"
PY="$ROOT/.venv/bin/python"
OSPREY="$ROOT/.venv/bin/osprey"
MODEL_URL=http://127.0.0.1:8000/v1/models
MODEL_ID=LFM2.5-230M

say() { printf '\n==> %s\n' "$*"; }
die() { printf 'bootstrap: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing $1. Install it and rerun."; }

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_demo.sh [--check | --cleanup-old-demo]

Resets only this checkout's Osprey deployment, installs pinned project
dependencies, starts the local LFM2.5-230M vLLM server if needed, and brings
up the Osprey accelerator, viewer/API, and authenticated web terminal.

--check  Check host prerequisites without changing anything.
--cleanup-old-demo  Stop the old hackring-live-230m Osprey deployment before
                    starting this one. Its volumes and the Xopt stack remain.

If vllm is outside PATH, set VLLM_BIN to its executable. On Apple Silicon the
script also checks ~/.venv-vllm-metal/bin/vllm. Docker must already be running.
The reset destroys this deployment's containers, volumes, build/, and agent
memory. It preserves the source and audit log. This script backs up any .env
before replacing it with the local vLLM example.
EOF
}

MODE=run
CLEANUP_OLD_DEMO=0
case "${1:-}" in
  '') ;;
  --check) MODE=check ;;
  --cleanup-old-demo) CLEANUP_OLD_DEMO=1 ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
(( $# <= 1 )) || { usage >&2; exit 2; }

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
    "$PY" -c 'import json,sys; assert any(m.get("id") == "LFM2.5-230M" for m in json.load(sys.stdin)["data"])' 2>/dev/null
}

port_free() {
  "$PY" - "$1" <<'PY'
import socket, sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
PY
}

if [[ $MODE == check ]]; then
  if [[ -z $VLLM_CMD ]] && ! curl -fsS --max-time 3 "$MODEL_URL" >/dev/null 2>&1; then
    die 'vLLM is missing and no model server is available on 127.0.0.1:8000.'
  fi
  say 'Host checks passed: git, uv/Python 3.12, npm, Docker, curl, and vLLM or an existing model endpoint.'
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
uv venv --python 3.12 "$ROOT/.venv"
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

if [[ $CLEANUP_OLD_DEMO == 1 ]]; then
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
fi

if curl -fsS --max-time 3 "$MODEL_URL" >/dev/null 2>&1 && ! model_ready; then
  die "Port 8000 serves a different model; expected $MODEL_ID. Stop that server and rerun."
fi
if ! model_ready && ! port_free 8000; then
  die 'Port 8000 is occupied without a working LFM2.5-230M endpoint. Stop that process and rerun.'
fi

say 'Stopping this checkout’s previous web terminal and viewer, if present'
"$OSPREY" web stop --repo "$ROOT" >/dev/null 2>&1 || true
if [[ -f "$BOOT/viewer.pid" ]]; then
  old_pid=$(cat "$BOOT/viewer.pid")
  if [[ $old_pid =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    old_command=$(ps -p "$old_pid" -o command= 2>/dev/null || true)
    if [[ $old_command == *"$ROOT/scripts/shared_demo.sh"* ]]; then
      kill "$old_pid"
      for _ in {1..20}; do kill -0 "$old_pid" 2>/dev/null || break; sleep 0.25; done
    fi
  fi
  rm -f "$BOOT/viewer.pid"
fi

for port in 8001 5173; do
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

say 'Starting local LFM2.5-230M if needed'
if ! model_ready; then
  if curl -fsS --max-time 3 "$MODEL_URL" >/dev/null 2>&1; then
    die "Port 8000 serves a different model; expected $MODEL_ID. Stop that server and rerun."
  fi
  [[ -n $VLLM_CMD ]] || die 'vLLM executable disappeared.'
  model_args=(serve LiquidAI/LFM2.5-230M --served-model-name "$MODEL_ID"
    --enable-auto-tool-choice --tool-call-parser lfm2 --host 127.0.0.1 --port 8000)
  if [[ $(uname -s) == Darwin ]]; then
    model_args+=(--max-model-len 4096 --gpu-memory-utilization 0.05)
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

say 'Starting viewer API and frontend'
nohup "$ROOT/scripts/shared_demo.sh" >"$BOOT/viewer.log" 2>&1 &
echo $! >"$BOOT/viewer.pid"
for url in http://127.0.0.1:8001/state http://127.0.0.1:5173/; do
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
printf '\nViewer: http://127.0.0.1:5173/\nMachine: http://127.0.0.1:8001/state\n'
printf 'Logs: %s\n' "$BOOT"
