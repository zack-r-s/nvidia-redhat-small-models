#!/usr/bin/env bash
# One-plant demo: local PyAT fixture, live small model, chat bridge, and viewer.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

MODEL=${VLLM_MODEL:-zack-r-s/hackring-LFM2.5-230M}
REVISION=${VLLM_REVISION:-v004}
SERVED=${VLLM_SERVED_NAME:-LFM2.5-230M}
PARSER=${VLLM_TOOL_PARSER:-lfm2}
CHECK=0
SEED=7
OPEN=1
while (( $# )); do
  case "$1" in
    --check) CHECK=1 ;;
    --no-open) OPEN=0 ;;
    --dev-chat) : ;; # Former spelling for the default fixture path.
    --model|--revision|--served-name|--tool-parser|--seed)
      (( $# >= 2 )) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --model) MODEL=$2 ;;
        --revision) REVISION=$2 ;;
        --served-name) SERVED=$2 ;;
        --tool-parser) PARSER=$2 ;;
        --seed) SEED=$2 ;;
      esac
      shift ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/bootstrap_demo.sh [--check] [--model HF_ID] [--revision TAG]
                                 [--served-name NAME] [--seed N] [--no-open]
Starts the pinned model, shared local PyAT fixture, chat bridge, and viewer.
No Docker or Osprey is needed. Use --osprey for the separate accelerator demo.
EOF
      exit 0 ;;
    *) echo "Unknown fixture option: $1" >&2; exit 2 ;;
  esac
  shift
done

for command in uv npm curl; do command -v "$command" >/dev/null || { echo "Missing $command" >&2; exit 1; }; done
VLLM_CMD=${VLLM_BIN:-$(command -v vllm || true)}
[[ -n $VLLM_CMD ]] || VLLM_CMD="$HOME/.venv-vllm-metal/bin/vllm"
[[ -x $VLLM_CMD ]] || { echo "vLLM executable is missing; set VLLM_BIN" >&2; exit 1; }
if (( CHECK )); then
  echo "Fixture prerequisites found: uv, npm, curl, $VLLM_CMD (no Docker required)."
  exit 0
fi

uv venv --allow-existing --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev,gym]'
npm --prefix frontend ci
npm --prefix frontend run build
PYTHONPATH=src .venv/bin/python scripts/check_ring.py

export VLLM_BASE_URL=http://127.0.0.1:8000/v1 VLLM_MODEL_NAME="$SERVED"
model_ready() {
  curl -fsS --max-time 3 "$VLLM_BASE_URL/models" 2>/dev/null |
    .venv/bin/python -c 'import json,sys; assert any(m.get("id")==sys.argv[1] for m in json.load(sys.stdin)["data"])' "$SERVED" 2>/dev/null
}
STARTED_PID=""
cleanup() { [[ -z $STARTED_PID ]] || kill "$STARTED_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
if model_ready; then
  listener=$(lsof -nP -iTCP:8000 -sTCP:LISTEN -t | head -1)
  command_line=$(ps -p "$listener" -o command= 2>/dev/null || true)
  [[ " $command_line " == *" $MODEL "* && " $command_line " == *" --revision $REVISION "* ]] || {
    echo "Port 8000 reports $SERVED but its model/revision cannot be verified as $MODEL@$REVISION." >&2; exit 1;
  }
else
  if lsof -nP -iTCP:8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Port 8000 is occupied by another service; stop it before starting the fixture demo." >&2; exit 1
  fi
  VLLM_SERVE_ARGS=${VLLM_SERVE_ARGS:---max-model-len 4096 --gpu-memory-utilization 0.4}
  read -r -a extra <<<"$VLLM_SERVE_ARGS"
  mkdir -p var/bootstrap-fixture
  "$VLLM_CMD" serve "$MODEL" --revision "$REVISION" --served-model-name "$SERVED" \
    --enable-auto-tool-choice --tool-call-parser "$PARSER" --host 127.0.0.1 --port 8000 \
    "${extra[@]}" >var/bootstrap-fixture/vllm.log 2>&1 &
  STARTED_PID=$!
  for _ in {1..180}; do
    model_ready && break
    kill -0 "$STARTED_PID" 2>/dev/null || { echo "vLLM exited; see var/bootstrap-fixture/vllm.log" >&2; exit 1; }
    sleep 2
  done
  model_ready || { echo "vLLM did not become ready; see var/bootstrap-fixture/vllm.log" >&2; exit 1; }
fi

echo "Starting shared PyAT fixture demo with $MODEL@$REVISION. Ctrl-C stops this launch."
demo_args=(--live --seed "$SEED")
(( OPEN )) || demo_args+=(--no-open)
"$ROOT/scripts/dev_up.sh" "${demo_args[@]}"
