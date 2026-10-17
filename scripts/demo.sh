#!/usr/bin/env bash
# Start the machine API, the viewer, open the browser, and drive the beam with Xopt.
# Usage: scripts/demo.sh [--epics] [--steps N] [--delay S] [--seed N] [--loops N] [--keep]
#   --epics  run the PyEPICS path: caproto soft IOC -> EPICS-backed API -> Xopt
#   --loops  repeat the Xopt run N times (default 1); each reuses the same seeded disturbance
#   --manual start servers and open the viewer with the manual test-driver panel; no Xopt run
#   --keep   leave servers running after Xopt finishes (Ctrl-C to stop)
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=local STEPS=35 DELAY=0.35 SEED=7 LOOPS=1 KEEP=0 MANUAL=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --epics) MODE=epics ;;
    --steps) STEPS=$2; shift ;;
    --delay) DELAY=$2; shift ;;
    --seed) SEED=$2; shift ;;
    --loops) LOOPS=$2; shift ;;
    --keep) KEEP=1 ;;
    --manual) MANUAL=1 ;;
    -h|--help) sed -n 2,8p "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

PY=.venv/bin/python
[[ -x $PY && -d frontend/node_modules ]] || { echo "Run the README developer setup first (.venv and frontend/node_modules)." >&2; exit 1; }
for port in 8001 5173; do
  if lsof -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1; then echo "Port $port is already in use; stop that process first." >&2; exit 1; fi
done

export PYTHONPATH=src
if [[ $MODE == epics ]]; then export EPICS_CA_SERVER_PORT=6064 EPICS_CA_REPEATER_PORT=6065 HACKRING_EPICS_SOURCE=caproto-epics; fi
export EPICS_CA_ADDR_LIST=127.0.0.1 EPICS_CA_AUTO_ADDR_LIST=NO EPICS_CAS_INTF_ADDR_LIST=127.0.0.1
export EPICS_CAS_AUTO_BEACON_ADDR_LIST=NO EPICS_CAS_BEACON_ADDR_LIST=127.0.0.1
LOGS=$(mktemp -d); PIDS=()
cleanup() { trap - EXIT INT TERM; for p in "${PIDS[@]:-}"; do [[ -n $p ]] && kill "$p" 2>/dev/null || true; done; echo "Stopped. Logs: $LOGS"; }
trap cleanup EXIT INT TERM

wait_for() { # url, label
  for _ in $(seq 60); do curl -fs "$1" >/dev/null 2>&1 && return 0; sleep 0.5; done
  echo "$2 did not start; see $LOGS" >&2; exit 1
}

if [[ $MODE == epics ]]; then
  $PY scripts/ca_fixture.py --seed "$SEED" >"$LOGS/ioc.log" 2>&1 & PIDS+=($!)
  sleep 4
  HACKRING_BACKEND=epics $PY -m uvicorn hackring.api:app --host 127.0.0.1 --port 8001 >"$LOGS/api.log" 2>&1 & PIDS+=($!)
else
  $PY -m uvicorn hackring.api:app --host 127.0.0.1 --port 8001 >"$LOGS/api.log" 2>&1 & PIDS+=($!)
fi
wait_for http://127.0.0.1:8001/state "Machine API"
(cd frontend && exec node_modules/.bin/vite --host 127.0.0.1 >"$LOGS/vite.log" 2>&1) & PIDS+=($!)
wait_for http://127.0.0.1:5173 "Viewer"

URL=http://127.0.0.1:5173/?driver=1
echo "Mode: $MODE · viewer: $URL · logs: $LOGS"
if command -v open >/dev/null; then open "$URL"; else xdg-open "$URL" 2>/dev/null || echo "Open $URL in a browser."; fi
if [[ $MANUAL == 1 ]]; then echo "Manual mode: use the test-driver panel in the browser. Ctrl-C to stop."; wait; fi
sleep 3  # let the viewer show the starting state

for i in $(seq "$LOOPS"); do
  echo "--- Xopt run $i/$LOOPS (seed $SEED) ---"
  $PY scripts/xopt_demo.py --steps "$STEPS" --delay "$DELAY" --seed "$SEED"
  sleep 2
done

if [[ $KEEP == 1 ]]; then echo "Servers still running; Ctrl-C to stop."; wait; fi
