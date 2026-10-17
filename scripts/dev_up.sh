#!/usr/bin/env bash
# Dev mode for chat-UI work: fixture machine API + chat bridge + viewer. No Docker, no Osprey.
# Usage: scripts/dev_up.sh [--scripted | --live] [--seed N] [--no-open]
#   (default)  live if a vLLM server answers on VLLM_BASE_URL (default http://127.0.0.1:8000/v1), else scripted
#   --scripted model-free deterministic replies (build the UI without a model)
#   --live     require the model server; fail if it is not up
#   --seed N   seed the starting disturbance (default 7); the machine is the local PyAT fixture
# Everything here is a DEV FIXTURE: replies are not from Osprey. Stop with Ctrl-C.
set -euo pipefail
cd "$(dirname "$0")/.."

CHAT_MODE=auto SEED=7 OPEN=1
while [[ $# -gt 0 ]]; do
  case $1 in
    --scripted) CHAT_MODE=scripted ;;
    --live) CHAT_MODE=live ;;
    --seed) SEED=$2; shift ;;
    --no-open) OPEN=0 ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

PY=.venv/bin/python
[[ -x $PY && -d frontend/node_modules ]] || { echo "Run the README developer setup first (.venv and frontend/node_modules)." >&2; exit 1; }
for port in 8001 8002 5173; do
  if lsof -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then echo "Port $port is already in use; stop that process first." >&2; exit 1; fi
done

VLLM_BASE_URL=${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}
model_up() { curl -fsS --max-time 2 "$VLLM_BASE_URL/models" >/dev/null 2>&1; }
if [[ $CHAT_MODE == live ]] && ! model_up; then
  echo "No model server at $VLLM_BASE_URL. Start vLLM (on an 8 GiB Mac: --gpu-memory-utilization 0.4), or use --scripted." >&2
  exit 1
fi
RESOLVED=$CHAT_MODE
[[ $CHAT_MODE != auto ]] || { model_up && RESOLVED=live || RESOLVED=scripted; }

export PYTHONPATH=src HACKRING_API_URL=http://127.0.0.1:8001 VLLM_BASE_URL CHAT_MODE
unset HACKRING_BACKEND  # the dev fixture is always the local PyAT machine
LOGS=$(mktemp -d); PIDS=()
cleanup() { trap - EXIT INT TERM; for p in "${PIDS[@]:-}"; do [[ -n $p ]] && kill "$p" 2>/dev/null || true; done; echo "Stopped. Logs: $LOGS"; }
trap cleanup EXIT INT TERM
wait_for() { for _ in $(seq 60); do curl -fs "$1" >/dev/null 2>&1 && return 0; sleep 0.5; done; echo "$2 did not start; see $LOGS" >&2; exit 1; }

$PY -m uvicorn hackring.api:app --host 127.0.0.1 --port 8001 >"$LOGS/api.log" 2>&1 & PIDS+=($!)
wait_for http://127.0.0.1:8001/state "Fixture machine API"
curl -fs -XPOST http://127.0.0.1:8001/scenario -H 'content-type: application/json' -d "{\"seed\": $SEED}" >/dev/null
$PY -m uvicorn hackring.chat_bridge:app --host 127.0.0.1 --port 8002 >"$LOGS/bridge.log" 2>&1 & PIDS+=($!)
wait_for http://127.0.0.1:8002/chat/health "Chat bridge"
(cd frontend && exec node_modules/.bin/vite --host 127.0.0.1 >"$LOGS/vite.log" 2>&1) & PIDS+=($!)
wait_for http://127.0.0.1:5173 "Viewer"

URL="http://127.0.0.1:5173/?driver=1"
cat <<EOF
Dev mode (fixture, not Osprey)
  viewer   $URL
  machine  http://127.0.0.1:8001/state   (seed $SEED)
  chat     http://127.0.0.1:8002/chat    (mode: $RESOLVED)
  logs     $LOGS
Ctrl-C to stop.
EOF
if [[ $OPEN == 1 ]]; then if command -v open >/dev/null; then open "$URL"; else xdg-open "$URL" 2>/dev/null || true; fi; fi
wait
