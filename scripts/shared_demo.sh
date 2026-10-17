#!/usr/bin/env bash
# Connect the read-only visualizer to an already-running Osprey served accelerator.
# Start the accelerator with `osprey up --dev -d` first; this never starts a fixture IOC.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
[[ -x $PY && -d frontend/node_modules ]] || { echo "Install the Python and frontend dependencies first." >&2; exit 1; }
for port in 8001 5173; do
  if lsof -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then echo "Port $port is already in use." >&2; exit 1; fi
done

export PYTHONPATH=src HACKRING_BACKEND=epics HACKRING_EPICS_SOURCE=osprey-epics
export EPICS_CA_NAME_SERVERS=127.0.0.1:5064 EPICS_CA_AUTO_ADDR_LIST=NO EPICS_CA_ADDR_LIST=
export EPICS_PVA_NAME_SERVERS=127.0.0.1:5075 EPICS_PVA_AUTO_ADDR_LIST=NO EPICS_PVA_ADDR_LIST=

# Do not mistake another project's VA on the same ports for this deployment.
health=$(docker inspect --format '{{.State.Health.Status}}' hackring-virtual-accelerator 2>/dev/null || true)
[[ $health == healthy ]] || { echo "hackring-virtual-accelerator is not healthy; start this worktree's Osprey deployment first." >&2; exit 1; }

# Fail before starting either UI if the real 24-PV served machine is unavailable.
$PY - <<'PY'
from hackring.machine import EpicsMachine
state = EpicsMachine().read()
assert state["backend"] == "osprey-epics" and len(state["pvs"]) == 24
assert len([d for d in state["lattice"]["devices"] if d["kind"] == "bpm"]) == 4
print("Served EPICS machine is responding with 24 PVs and four BPM positions.")
PY

LOGS=$(mktemp -d); PIDS=()
cleanup() { trap - EXIT INT TERM; for p in "${PIDS[@]:-}"; do [[ -n $p ]] && kill "$p" 2>/dev/null || true; done; echo "Stopped API and viewer. Logs: $LOGS"; }
trap cleanup EXIT INT TERM

$PY -m uvicorn hackring.api:app --host 127.0.0.1 --port 8001 >"$LOGS/api.log" 2>&1 & PIDS+=($!)
(cd frontend && exec node_modules/.bin/vite --host 127.0.0.1) >"$LOGS/vite.log" 2>&1 & PIDS+=($!)
for url in http://127.0.0.1:8001/state http://127.0.0.1:5173; do
  ready=0
  for _ in $(seq 60); do
    if curl -fsS "$url" >/dev/null 2>&1; then ready=1; break; fi
    sleep 0.5
  done
  [[ $ready == 1 ]] || { echo "$url did not start; see $LOGS" >&2; exit 1; }
done

echo "Viewer: http://127.0.0.1:5173/"
echo "Machine: http://127.0.0.1:8001/state (Osprey via EPICS)"
echo "Run .venv/bin/osprey web in another terminal for the LFM chat. Ctrl-C stops this API and viewer."
wait
