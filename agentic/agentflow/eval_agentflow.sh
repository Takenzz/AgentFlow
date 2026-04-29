#!/bin/bash
# AgentFlow standalone evaluation.
# Default deployment:
#   - Planner: local SGLang small model, or API when USE_API_FOR_PLANNER=1.
#   - Executor / verifier / base_generator / python_coder / rewarder: API.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
export PYTHONPATH="/root/Megatron-LM/:${SCRIPT_DIR}:${SLIME_ROOT}:${PYTHONPATH:-}"

# Planner model in HF format. Used only when USE_API_FOR_PLANNER=0 and AUTO_START=1.
MODEL_PATH=${MODEL_PATH:-"/data/AgentFlow_Qwen25-1.5B-RL-HF/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/data/models/qwen25_1.5b"}

EVAL_DATA=(
    aime /data/aime-2024/aime-2024.jsonl
)

OUTPUT=${OUTPUT:-"${SCRIPT_DIR}/eval_results.json"}
TRAJECTORY_DIR=${TRAJECTORY_DIR:-""}

PLANNER_PORT=${PLANNER_PORT:-30000}
if [ -n "${PLANNER_URL:-}" ]; then
    AUTO_START=${AUTO_START:-0}
else
    AUTO_START=${AUTO_START:-1}
    PLANNER_URL="http://127.0.0.1:${PLANNER_PORT}/generate"
fi
USE_API_FOR_PLANNER=${USE_API_FOR_PLANNER:-0}
USE_API_FOR_NON_PLANNER=${USE_API_FOR_NON_PLANNER:-1}

TP=${TP:-1}
MEM_FRACTION=${MEM_FRACTION:-0.7}
CTX_LEN=${CTX_LEN:-65536}

CONCURRENCY=${CONCURRENCY:-16}
MAX_STEPS=${MAX_STEPS:-5}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
NUM_SAMPLES=${NUM_SAMPLES:-0}

# Shared API defaults. Role-specific variables below can override them.
AGENTFLOW_API_BASE=${AGENTFLOW_API_BASE:-"https://api.openai.com/v1"}
AGENTFLOW_API_KEY=${AGENTFLOW_API_KEY:-"${OPENAI_API_KEY:-}"}
AGENTFLOW_API_MODEL=${AGENTFLOW_API_MODEL:-"gpt-4o-mini"}
AGENTFLOW_API_TIMEOUT=${AGENTFLOW_API_TIMEOUT:-180}
AGENTFLOW_API_MAX_RETRIES=${AGENTFLOW_API_MAX_RETRIES:-3}
AGENTFLOW_API_ENABLE_THINKING=${AGENTFLOW_API_ENABLE_THINKING:-false}
AGENTFLOW_API_THINKING_BUDGET=${AGENTFLOW_API_THINKING_BUDGET:-""}

PLANNER_API_BASE=${PLANNER_API_BASE:-"${AGENTFLOW_PLANNER_API_BASE:-${AGENTFLOW_API_BASE}}"}
PLANNER_API_KEY=${PLANNER_API_KEY:-"${AGENTFLOW_PLANNER_API_KEY:-${AGENTFLOW_API_KEY}}"}
PLANNER_MODEL=${PLANNER_MODEL:-"${AGENTFLOW_PLANNER_MODEL:-${AGENTFLOW_API_MODEL}}"}

EXECUTOR_API_BASE=${EXECUTOR_API_BASE:-"${AGENTFLOW_EXECUTOR_API_BASE:-${AGENTFLOW_API_BASE}}"}
EXECUTOR_API_KEY=${EXECUTOR_API_KEY:-"${AGENTFLOW_EXECUTOR_API_KEY:-${AGENTFLOW_API_KEY}}"}
EXECUTOR_MODEL=${EXECUTOR_MODEL:-"${AGENTFLOW_EXECUTOR_MODEL:-${AGENTFLOW_API_MODEL}}"}

CODER_API_BASE=${CODER_API_BASE:-"${AGENTFLOW_CODER_API_BASE:-${EXECUTOR_API_BASE}}"}
CODER_API_KEY=${CODER_API_KEY:-"${AGENTFLOW_CODER_API_KEY:-${EXECUTOR_API_KEY}}"}
CODER_MODEL=${CODER_MODEL:-"${AGENTFLOW_CODER_MODEL:-${EXECUTOR_MODEL}}"}

REWARDER_API_BASE=${REWARDER_API_BASE:-"${AGENTFLOW_REWARDER_API_BASE:-${EXECUTOR_API_BASE}}"}
REWARDER_API_KEY=${REWARDER_API_KEY:-"${AGENTFLOW_REWARDER_API_KEY:-${EXECUTOR_API_KEY}}"}
REWARDER_MODEL=${REWARDER_MODEL:-"${AGENTFLOW_REWARDER_MODEL:-${EXECUTOR_MODEL}}"}

PY_ARGS=(
    --tokenizer "${TOKENIZER_PATH}"
    --eval-data "${EVAL_DATA[@]}"
    --input-key prompt
    --label-key label
    --output "${OUTPUT}"
    --concurrency "${CONCURRENCY}"
    --max-steps "${MAX_STEPS}"
    --temperature "${TEMPERATURE}"
    --top-p "${TOP_P}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --api-timeout "${AGENTFLOW_API_TIMEOUT}"
    --api-max-retries "${AGENTFLOW_API_MAX_RETRIES}"
    --api-enable-thinking "${AGENTFLOW_API_ENABLE_THINKING}"
)

if [ -n "${AGENTFLOW_API_THINKING_BUDGET}" ]; then
    PY_ARGS+=(--api-thinking-budget "${AGENTFLOW_API_THINKING_BUDGET}")
fi

if [ "${USE_API_FOR_PLANNER}" = "1" ]; then
    PY_ARGS+=(
        --planner-backend api
        --planner-api-base "${PLANNER_API_BASE}"
        --planner-api-key "${PLANNER_API_KEY}"
        --planner-model "${PLANNER_MODEL}"
    )
else
    if [ "${AUTO_START}" = "1" ]; then
        PY_ARGS+=(
            --model "${MODEL_PATH}"
            --start-servers
            --planner-port "${PLANNER_PORT}"
            --tp "${TP}"
            --mem-fraction "${MEM_FRACTION}"
            --ctx-len "${CTX_LEN}"
        )
    else
        PY_ARGS+=(--planner-url "${PLANNER_URL}")
    fi
fi

if [ "${USE_API_FOR_NON_PLANNER}" = "1" ]; then
    if [ -z "${EXECUTOR_MODEL}" ]; then
        echo "AGENTFLOW_EXECUTOR_MODEL/EXECUTOR_MODEL must be set when USE_API_FOR_NON_PLANNER=1"
        exit 1
    fi
    PY_ARGS+=(
        --use-api-for-non-planner
        --executor-api-base "${EXECUTOR_API_BASE}"
        --executor-api-key "${EXECUTOR_API_KEY}"
        --executor-model "${EXECUTOR_MODEL}"
        --coder-api-base "${CODER_API_BASE}"
        --coder-api-key "${CODER_API_KEY}"
        --coder-model "${CODER_MODEL}"
        --rewarder-api-base "${REWARDER_API_BASE}"
        --rewarder-api-key "${REWARDER_API_KEY}"
        --rewarder-model "${REWARDER_MODEL}"
    )
fi

if [ -n "${TRAJECTORY_DIR}" ]; then
    PY_ARGS+=(--trajectory-dir "${TRAJECTORY_DIR}")
fi

if [ "${NUM_SAMPLES}" -gt 0 ] 2>/dev/null; then
    PY_ARGS+=(--num-samples "${NUM_SAMPLES}")
fi

echo "Starting AgentFlow evaluation"
if [ "${USE_API_FOR_PLANNER}" = "1" ]; then
    echo "  planner: api ${PLANNER_MODEL}"
elif [ "${AUTO_START}" = "1" ]; then
    echo "  planner: auto-start local ${MODEL_PATH}"
else
    echo "  planner: existing local service ${PLANNER_URL}"
fi
echo "  support roles: $([ "${USE_API_FOR_NON_PLANNER}" = "1" ] && echo "api ${EXECUTOR_MODEL}" || echo "local SGLang URLs")"
echo "  output: ${OUTPUT}"

python3 "${SCRIPT_DIR}/eval_agentflow.py" "${PY_ARGS[@]}"
