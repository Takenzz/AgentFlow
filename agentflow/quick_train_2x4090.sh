#!/bin/bash
# Smoke-test AgentFlow training on two RTX 4090 cards.
# GPU 0: student trainer + student Planner rollout
# GPU 1: teacher SGLang logprob server

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
LOG_DIR=${LOG_DIR:-"/tmp/agentflow_quick_2x4090"}
mkdir -p "$LOG_DIR"

STUDENT_CUDA_VISIBLE_DEVICES=${STUDENT_CUDA_VISIBLE_DEVICES:-0}
TEACHER_CUDA_VISIBLE_DEVICES=${TEACHER_CUDA_VISIBLE_DEVICES:-1}

STUDENT_HF_CHECKPOINT=${STUDENT_HF_CHECKPOINT:-"/data/models/student-0.8b"}
STUDENT_REF_LOAD=${STUDENT_REF_LOAD:-"/data/models/student-0.8b_torch_dist"}
STUDENT_MODEL_CONFIG=${STUDENT_MODEL_CONFIG:-"${SCRIPT_DIR}/model_configs/qwen3-0.6B.sh"}

TEACHER_HF_CHECKPOINT=${TEACHER_HF_CHECKPOINT:-"/data/models/teacher-2b"}
TEACHER_HOST=${TEACHER_HOST:-"127.0.0.1"}
TEACHER_PORT=${TEACHER_PORT:-"30080"}
TEACHER_CONTEXT_LENGTH=${TEACHER_CONTEXT_LENGTH:-8192}
TEACHER_MEM_FRACTION_STATIC=${TEACHER_MEM_FRACTION_STATIC:-0.72}

PROMPT_DATA=${PROMPT_DATA:-"${REPO_ROOT}/examples/tiny_math.jsonl"}
EVAL_PROMPT_DATA=${EVAL_PROMPT_DATA:-"${PROMPT_DATA}"}
SAVE_DIR=${SAVE_DIR:-"/tmp/agentflow_student_quick_ckpt"}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_port() {
    local name=$1 host=$2 port=$3 timeout=${4:-600} interval=5 elapsed=0
    log "等待 ${name} (${host}:${port}) 就绪..."
    while [ $elapsed -lt $timeout ]; do
        if bash -c "echo >/dev/tcp/${host}/${port}" 2>/dev/null; then
            log "${name} 已就绪 (${elapsed}s)"
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
    done
    log "ERROR: ${name} 在 ${timeout}s 内未就绪，日志: ${LOG_DIR}/teacher.log"
    return 1
}

cleanup_old_processes() {
    log "清理旧 Ray 和 SGLang 进程..."
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    pkill -9 -f 'sglang\.launch_server' 2>/dev/null || true
    sleep 2
}

start_teacher() {
    log "启动 teacher SGLang：${TEACHER_HF_CHECKPOINT} -> http://${TEACHER_HOST}:${TEACHER_PORT}/generate"
    CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}" \
        python3 -m sglang.launch_server \
        --model-path "${TEACHER_HF_CHECKPOINT}" \
        --host "${TEACHER_HOST}" \
        --port "${TEACHER_PORT}" \
        --tp-size 1 \
        --context-length "${TEACHER_CONTEXT_LENGTH}" \
        --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}" \
        --trust-remote-code \
        ${TEACHER_EXTRA_ARGS:-} \
        > "${LOG_DIR}/teacher.log" 2>&1 &
    echo $! > "${LOG_DIR}/teacher.pid"
    wait_port "teacher" "${TEACHER_HOST}" "${TEACHER_PORT}" 900
}

run_student_training() {
    log "启动 student quick training，日志: ${LOG_DIR}/train.log"
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}" \
    SKIP_PROCESS_KILL=1 \
    MODEL_CONFIG_SCRIPT="${STUDENT_MODEL_CONFIG}" \
    BASE_HF_CHECKPOINT="${STUDENT_HF_CHECKPOINT}" \
    REF_LOAD="${STUDENT_REF_LOAD}" \
    SAVE_DIR="${SAVE_DIR}" \
    PROMPT_DATA="${PROMPT_DATA}" \
    EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA}" \
    TRAIN_GPUS=1 \
    TRAIN_TP=1 \
    ROLLOUT_ENGINE_GPUS=1 \
    NUM_ROLLOUT=${NUM_ROLLOUT:-1} \
    ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1} \
    N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2} \
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-2} \
    ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-2048} \
    SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-8192} \
    SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.58} \
    MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096} \
    SAVE_INTERVAL=${SAVE_INTERVAL:-1} \
    EVAL_INTERVAL=${EVAL_INTERVAL:-1} \
    USE_OPD=1 \
    OPD_TYPE=sglang \
    OPD_KL_COEF=${OPD_KL_COEF:-1.0} \
    USE_KL_LOSS=${USE_KL_LOSS:-0} \
    RM_URL="http://${TEACHER_HOST}:${TEACHER_PORT}/generate" \
        bash "${SCRIPT_DIR}/train_agentflow.sh" \
        > "${LOG_DIR}/train.log" 2>&1
}

cleanup_old_processes
start_teacher
run_student_training
log "quick training 完成。日志目录: ${LOG_DIR}"
