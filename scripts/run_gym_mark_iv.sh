#!/usr/bin/env bash
# Evaluate one pinned model on all 36 frozen Mark IV prompts through NeMo Gym.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PATH="$ROOT/var/gym-cli/bin:$PATH"
GYM="$ROOT/var/gym-cli/bin/gym"
[[ -x $GYM ]] || { echo "Install nemo-gym==0.4.0 into var/gym-cli first." >&2; exit 1; }
"$GYM" --version | head -1 | grep -q 'NeMo Gym v0.4.0' || { echo "Expected NeMo Gym 0.4.0" >&2; exit 1; }
PY="$ROOT/.venv/bin/python"
export MPLCONFIGDIR=${MPLCONFIGDIR:-$ROOT/var/mpl-cache}
PYTHONPATH="$ROOT/src:$ROOT/scripts" "$PY" scripts/prepare_mark_iv.py
MODEL=${VLLM_MODEL_NAME:?Set VLLM_MODEL_NAME to the served ID}
BASE=${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}
SOURCE=${VLLM_MODEL_SOURCE:?Set VLLM_MODEL_SOURCE to the pinned Hugging Face repo}
REVISION=${VLLM_MODEL_REVISION:?Set VLLM_MODEL_REVISION to the pinned revision}
curl -fsS --max-time 5 "$BASE/models" |
  "$PY" -c 'import json,sys; assert any(m.get("id")==sys.argv[1] for m in json.load(sys.stdin)["data"])' "$MODEL"
MODEL_DIR=$(printf '%s' "$MODEL" | tr -c 'A-Za-z0-9._-' '_')
RUN="$ROOT/var/gym/mark-iv/$MODEL_DIR/validation-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN"
"$PY" scripts/gym_run_metadata.py write "$RUN" "$MODEL" "$BASE" validation "$@"
"$PY" - "$RUN" "$SOURCE" "$REVISION" <<'PY'
import hashlib,json,sys
from pathlib import Path
run,source,revision=sys.argv[1:]
root=Path.cwd(); path=Path(run)/'run.json'; data=json.loads(path.read_text())
task=root/'gym/resources_servers/hackring_mark_iv/data/validation.jsonl'
cases=root/'evals/mark-iv/cases.jsonl'
data.update(evaluation_series='mark-iv',evaluation_interface='raw_four_tool_operator',
            model_source=source,model_revision=revision,
            task_sha256=hashlib.sha256(task.read_bytes()).hexdigest(),
            cases_sha256=hashlib.sha256(cases.read_bytes()).hexdigest(),
            grader_sha256=hashlib.sha256((root/'scripts/mark_iv_common.py').read_bytes()).hexdigest())
path.write_text(json.dumps(data,indent=2)+'\n')
PY
cd gym
"$GYM" eval run \
  --config resources_servers/hackring_mark_iv/configs/hackring_mark_iv.yaml \
  --model-type vllm_model \
  --agent hackring_mark_iv_agent \
  --input resources_servers/hackring_mark_iv/data/validation.jsonl \
  --output "$RUN/rollouts.jsonl" \
  --split validation \
  --model "$MODEL" --model-url "$BASE" --model-api-key "${VLLM_API_KEY:-local-vllm}" \
  --temperature 0.2 --max-output-tokens 512 --num-repeats 4 \
  "$@" \
  ++policy_model.responses_api_models.vllm_model.uses_reasoning_parser=false
"$PY" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
folder=Path(sys.argv[1]); prepared=[json.loads(x) for x in (folder/'preprocessed_datasets/validation.jsonl').open()]
assert len(prepared)==36, f'Expected 36 prepared tasks, found {len(prepared)}'
expected={'channel_read','channel_write','channel_limits','orbit_correct'}
assert all({tool['name'] for tool in row['responses_create_params']['tools']}==expected for row in prepared)
rows=[json.loads(x) for x in (folder/'rollouts.jsonl').open()]
assert len(rows)==144, f'Expected 144 rollouts, found {len(rows)}'
assert not (folder/'rollouts_failures.jsonl').read_text().strip(), 'Gym recorded rollout failures'
print(f'Verified 36 Mark IV prompts x 4 repeats: {folder}')
PY
