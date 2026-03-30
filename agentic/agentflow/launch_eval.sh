#!/bin/bash
# AgentFlow 评估一键启动脚本
# 自动启动三个 SGLang 服务，等待就绪后执行评估，评估完毕后关闭服务。
#
# 使用方式：
#   bash /data/slime-agentic/agentic/agentflow/launch_eval.sh
#   MODEL_PATH=/data/my_model bash launch_eval.sh
#
# GPU 分配：
#   GPU 0,1 : SGLang Planner       port=30000  (训练后的模型)
#   GPU 2,3 : SGLang Executor/Base port=30001  (Qwen2.5-7B-Instruct，executor/verifier/final_output)
#   GPU 4,5 : SGLang Coder         port=30002  (Qwen2.5-Coder-7B-Instruct，python_coder)

set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

LOG_DIR="/tmp/agentflow_eval_logs"
mkdir -p "$LOG_DIR"

# ── 可配置项 ──────────────────────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-"/data/AgentFlow_pro-Qwen25-7B-RL/"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/data/models/qwen25_7b"}
MODEL_BASE=${MODEL_BASE:-"/data/models/qwen25_7b"}
MODEL_CODER=${MODEL_CODER:-"/data/models/qwen2.5_7b_codeer"}

CTX_LEN=${CTX_LEN:-131072}
MEM_FRACTION=${MEM_FRACTION:-0.7}
CONCURRENCY=${CONCURRENCY:-16}
MAX_STEPS=${MAX_STEPS:-5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
NUM_SAMPLES=${NUM_SAMPLES:-0}

OUTPUT=${OUTPUT:-"${SCRIPT_DIR}/eval_results.json"}

# ── 工具函数 ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_port() {
    local name=$1 host=$2 port=$3 timeout=${4:-600} interval=5 elapsed=0
    log "等待 $name (${host}:${port}) 就绪..."
    while [ $elapsed -lt $timeout ]; do
        if bash -c "echo >/dev/tcp/${host}/${port}" 2>/dev/null; then
            log "$name 已就绪 (${elapsed}s)"
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
        if (( elapsed % 30 == 0 )); then
            log "  还在等待 $name... (${elapsed}s)"
        fi
    done
    log "ERROR: $name 在 ${timeout}s 内未能启动，请检查日志: $LOG_DIR"
    return 1
}

cleanup() {
    log "关闭 SGLang 服务..."
    pkill -f "sglang.launch_server.*port 3000" 2>/dev/null || true
    sleep 2
    log "清理完成。"
}
trap cleanup EXIT

# ── Step 1: 清理旧进程 ────────────────────────────────────────────────────────
log "清理旧 SGLang 进程..."
pkill -f "sglang.launch_server.*port 3000" 2>/dev/null || true
sleep 2

# ── Step 2: 启动三个 SGLang 服务 ──────────────────────────────────────────────

# GPU 0,1 — 训练后模型，port 30000（planner）
CUDA_VISIBLE_DEVICES=0,1 conda run -n sglang --no-capture-output \
    bash -c "
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python3 -m sglang.launch_server \
            --model-path ${MODEL_PATH} \
            --port 30000 \
            --context-length ${CTX_LEN} \
            --tp 2
    " > "$LOG_DIR/sglang_30000.log" 2>&1 &
log "Planner (trained)     PID=$! (GPU 0,1, port 30000)，日志: $LOG_DIR/sglang_30000.log"

# GPU 2,3 — Qwen2.5-7B-Instruct，port 30001（executor/verifier/final_output）
CUDA_VISIBLE_DEVICES=2,3 conda run -n sglang --no-capture-output \
    bash -c "
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python3 -m sglang.launch_server \
            --model-path ${MODEL_BASE} \
            --port 30001 \
            --context-length ${CTX_LEN} \
            --tp 2
    " > "$LOG_DIR/sglang_30001.log" 2>&1 &
log "Executor/Base (7B)    PID=$! (GPU 2,3, port 30001)，日志: $LOG_DIR/sglang_30001.log"

# GPU 4,5 — Qwen2.5-Coder-7B-Instruct，port 30002（python_coder）
CUDA_VISIBLE_DEVICES=4,5 conda run -n sglang --no-capture-output \
    bash -c "
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python3 -m sglang.launch_server \
            --model-path ${MODEL_CODER} \
            --port 30002 \
            --context-length ${CTX_LEN} \
            --tp 2
    " > "$LOG_DIR/sglang_30002.log" 2>&1 &
log "Coder (7B)            PID=$! (GPU 4,5, port 30002)，日志: $LOG_DIR/sglang_30002.log"

# ── Step 3: 等待所有服务就绪 ──────────────────────────────────────────────────
log "等待所有 SGLang 服务启动..."
wait_port "Planner-30000"  127.0.0.1 30000 600
wait_port "Executor-30001" 127.0.0.1 30001 600
wait_port "Coder-30002"    127.0.0.1 30002 600
log "所有服务已就绪，开始评估。"

# ── Step 4: 运行评估 ──────────────────────────────────────────────────────────
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
export PYTHONPATH="/root/Megatron-LM/:${SCRIPT_DIR}:${SLIME_ROOT}:${PYTHONPATH:-}"

PY_ARGS=(
    --tokenizer      "${TOKENIZER_PATH}"
    --eval-data      aime /data/aime-2024/aime-2024.jsonl
    --input-key      prompt
    --label-key      label
    --output         "${OUTPUT}"
    --concurrency    "${CONCURRENCY}"
    --max-steps      "${MAX_STEPS}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --temperature    0.7
    --top-p          0.95
    --planner-url    "http://127.0.0.1:30000/generate"
    --executor-url   "http://127.0.0.1:30001/generate"
    --coder-url      "http://127.0.0.1:30002/generate"
)

if [ "${NUM_SAMPLES}" -gt 0 ] 2>/dev/null; then
    PY_ARGS+=(--num-samples "${NUM_SAMPLES}")
fi

log "开始评估（model=${MODEL_PATH}）..."
python3 "${SCRIPT_DIR}/eval_agentflow.py" "${PY_ARGS[@]}"

log "评估完成，结果保存至：${OUTPUT}"
