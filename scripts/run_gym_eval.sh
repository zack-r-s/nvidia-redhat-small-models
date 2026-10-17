#!/usr/bin/env bash
# Evaluate the vLLM model the bootstrap serves on the hackring_orbit NeMo Gym environment.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SERVER="$ROOT/gym/resources_servers/hackring_orbit"
PY="$ROOT/.venv/bin/python"

usage() {
  cat <<'EOF'
Usage: scripts/run_gym_eval.sh [--split validation|train|example] [gym eval run options...]

Runs `gym eval run` for hackring_orbit against the endpoint in .env: VLLM_BASE_URL,
VLLM_API_KEY, and VLLM_MODEL_NAME (written by scripts/bootstrap_demo.sh; falls back
to providers.yml's default); values already set in the shell take precedence. The validation split is the last 25% of v001 episodes.
Task rows are regenerated from datasets/v001/splits/ft75-eval25/split.json each run.
Sampling defaults match the small agent (temperature 0.2, 512 output tokens); extra
options pass through and override them, e.g. --limit 2 --num-repeats 4.
Each run writes var/gym/<model>/<split>-<timestamp>/: run.json, rollouts.jsonl,
rollouts_materialized_inputs.jsonl, and rollouts_aggregate_metrics.json.
EOF
}

SPLIT=validation
if [[ ${1:-} == -h || ${1:-} == --help ]]; then usage; exit 0; fi
if [[ ${1:-} == --split ]]; then
  [[ -n ${2:-} ]] || { usage >&2; exit 2; }
  SPLIT=$2
  shift 2
fi
[[ $SPLIT =~ ^(validation|train|example)$ ]] || { usage >&2; exit 2; }
command -v gym >/dev/null 2>&1 || { echo 'run_gym_eval: install NeMo Gym so `gym` is on PATH.' >&2; exit 1; }

# Variables already set in the shell win over .env, as in small_agent.py, so another endpoint
# can be evaluated without editing .env: VLLM_BASE_URL=... VLLM_MODEL_NAME=... scripts/run_gym_eval.sh
preset=()
for name in VLLM_BASE_URL VLLM_API_KEY VLLM_MODEL_NAME; do
  [[ -n ${!name+set} ]] && preset+=("$name=${!name}")
done
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
for assignment in ${preset[@]+"${preset[@]}"}; do
  export "$assignment"
done
BASE_URL=${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}
API_KEY=${VLLM_API_KEY:-local-vllm}
MODEL=${VLLM_MODEL_NAME:-$("$PY" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["providers"]["vllm"]["default_model"])' "$ROOT/providers.yml")}

curl -fsS --max-time 5 "$BASE_URL/models" |
  "$PY" -c 'import json, sys; assert any(m.get("id") == sys.argv[1] for m in json.load(sys.stdin)["data"])' "$MODEL" ||
  { echo "run_gym_eval: $BASE_URL does not serve $MODEL; start it with scripts/bootstrap_demo.sh." >&2; exit 1; }

# Regenerate every run so the tasks track the agent's current prompt and the split record.
PYTHONPATH="$ROOT/src:$ROOT/scripts" "$PY" "$ROOT/scripts/prepare_gym_data.py" \
  "$ROOT/datasets/v001/splits/ft75-eval25/split.json"

# One folder per served model name, one subfolder per run: var/gym/<model>/<split>-<timestamp>/.
RUN_DIR="$ROOT/var/gym/$(printf '%s' "$MODEL" | tr -c 'A-Za-z0-9._-' '_')/$SPLIT-$(date +%Y%m%d-%H%M%S)"
OUT="$RUN_DIR/rollouts.jsonl"
mkdir -p "$RUN_DIR"
"$PY" - "$(curl -fsS --max-time 5 "$BASE_URL/models")" "$MODEL" "$BASE_URL" "$SPLIT" \
  "$(git -C "$ROOT" rev-parse --short HEAD)" "$@" >"$RUN_DIR/run.json" <<'PY'
import json, sys
from datetime import datetime, timezone
listing, model, base_url, split, commit, *extra = sys.argv[1:]
served = next(m for m in json.loads(listing)["data"] if m.get("id") == model)
print(json.dumps({"served_model_name": model, "model_source": served.get("root"),
                  "max_model_len": served.get("max_model_len"), "base_url": base_url, "split": split,
                  "repo_commit": commit, "extra_args": extra,
                  "started_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
PY
echo "Model: $MODEL @ $BASE_URL"
echo "Run folder: $RUN_DIR"
# NeMo Gym 0.4.0 finds out-of-tree servers relative to the working directory.
cd "$ROOT/gym"
gym eval run \
  --resources-server hackring_orbit \
  --model-type vllm_model \
  --agent hackring_orbit_simple_agent \
  --input "resources_servers/hackring_orbit/data/$SPLIT.jsonl" \
  --output "$OUT" \
  --split "$([[ $SPLIT == train ]] && echo train || echo validation)" \
  --model "$MODEL" --model-url "$BASE_URL" --model-api-key "$API_KEY" \
  --temperature 0.2 --max-output-tokens 512 \
  "$@" \
  ++policy_model.responses_api_models.vllm_model.uses_reasoning_parser=false
