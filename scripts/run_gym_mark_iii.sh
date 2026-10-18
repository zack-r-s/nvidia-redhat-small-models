#!/usr/bin/env bash
# Evaluate one pinned model on one frozen Mark III interface.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
TRACK=${1:-}
[[ $TRACK == direct || $TRACK == skill ]] || { echo "Usage: $0 direct|skill [Gym options]" >&2; exit 2; }
shift
export PATH="$ROOT/var/gym-cli/bin:$PATH"
GYM="$ROOT/var/gym-cli/bin/gym"
[[ -x $GYM ]] || { echo "Install nemo-gym==0.4.0 into var/gym-cli first." >&2; exit 1; }
"$GYM" --version | head -1 | grep -q 'NeMo Gym v0.4.0' || { echo "Expected NeMo Gym 0.4.0" >&2; exit 1; }
PY="$ROOT/.venv/bin/python"
MPLCONFIGDIR=${MPLCONFIGDIR:-$ROOT/var/mpl-cache}
export MPLCONFIGDIR
PYTHONPATH="$ROOT/src:$ROOT/scripts" "$PY" scripts/prepare_gym_mark_iii.py
MODEL=${VLLM_MODEL_NAME:?Set VLLM_MODEL_NAME to the served ID}
BASE=${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}
curl -fsS --max-time 5 "$BASE/models" |
  "$PY" -c 'import json,sys; assert any(m.get("id")==sys.argv[1] for m in json.load(sys.stdin)["data"])' "$MODEL"
SOURCE=${VLLM_MODEL_SOURCE:?Set VLLM_MODEL_SOURCE to the pinned Hugging Face repo}
REVISION=${VLLM_MODEL_REVISION:?Set VLLM_MODEL_REVISION to the pinned revision}
MODEL_DIR=$(printf '%s' "$MODEL" | tr -c 'A-Za-z0-9._-' '_')
RUN="$ROOT/var/gym/mark-iii/$TRACK/$MODEL_DIR/validation-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"
"$PY" scripts/gym_run_metadata.py write "$RUN" "$MODEL" "$BASE" validation "$@"
"$PY" - "$RUN" "$TRACK" "$SOURCE" "$REVISION" <<'PY'
import hashlib,json,sys
from pathlib import Path
run,track,source,revision=sys.argv[1:]
root=Path.cwd(); path=Path(run)/'run.json'; data=json.loads(path.read_text())
task=root/'gym/resources_servers/hackring_mark_iii/data'/f'{track}.jsonl'
cases=root/'evals/mark-i/pyat-v002/cases.jsonl'
data.update(evaluation_series='mark-iii',evaluation_interface='channel_write' if track=='direct' else 'orbit_correct',
            track=track,model_source=source,model_revision=revision,
            task_sha256=hashlib.sha256(task.read_bytes()).hexdigest(),
            cases_sha256=hashlib.sha256(cases.read_bytes()).hexdigest(),
            objective='requested-plane BPM RMS-squared fractional reduction; one tool attempt')
path.write_text(json.dumps(data,indent=2)+'\n')
PY
cd gym
"$GYM" eval run \
  --resources-server hackring_mark_iii \
  --model-type vllm_model \
  --agent hackring_mark_iii_simple_agent \
  --input "resources_servers/hackring_mark_iii/data/$TRACK.jsonl" \
  --output "$RUN/rollouts.jsonl" \
  --split validation \
  --model "$MODEL" --model-url "$BASE" --model-api-key "${VLLM_API_KEY:-local-vllm}" \
  --temperature 0.2 --max-output-tokens 512 \
  "$@" \
  ++policy_model.responses_api_models.vllm_model.uses_reasoning_parser=false
echo "Saved Mark III $TRACK run to $RUN"
