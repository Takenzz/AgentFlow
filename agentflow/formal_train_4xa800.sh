#!/bin/bash
# Formal AgentFlow training launcher for a 4xA800 node.
#
# Default mode:
#   - Use all 4 A800 GPUs for student training + student Planner rollout.
#   - Use an external teacher SGLang service via RM_URL.
#
# Local-teacher mode:
#   LOCAL_TEACHER=1 bash agentic/agentflow/formal_train_4xa800.sh
#   - GPU 0,1,2: student training + student Planner rollout.
#   - GPU 3: local 2B teacher SGLang logprob server.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
LOG_DIR=${LOG_DIR:-"/tmp/agentflow_formal_4xa800"}
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

is_true() {
   case "${1:-}" in
      1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
      *) return 1 ;;
   esac
}

count_csv_items() {
    local value="${1// /}"
    if [ -z "$value" ]; then
        echo 0
    else
        awk -F',' '{print NF}' <<< "$value"
    fi
}

wait_port() {
    local name=$1 host=$2 port=$3 timeout=${4:-900} interval=5 elapsed=0
    log "等待 ${name} (${host}:${port}) 就绪..."
    while [ $elapsed -lt $timeout ]; do
        if bash -c "echo >/dev/tcp/${host}/${port}" 2>/dev/null; then
            log "${name} 已就绪 (${elapsed}s)"
            return 0
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
        if (( elapsed % 60 == 0 )); then
            log "  还在等待 ${name}... (${elapsed}s)"
        fi
    done
    log "ERROR: ${name} 在 ${timeout}s 内未就绪。日志: ${LOG_DIR}/teacher.log"
    return 1
}

LOCAL_TEACHER=${LOCAL_TEACHER:-0}
CLEAN_SGLANG=${CLEAN_SGLANG:-${LOCAL_TEACHER}}

if is_true "$LOCAL_TEACHER"; then
    STUDENT_CUDA_VISIBLE_DEVICES=${STUDENT_CUDA_VISIBLE_DEVICES:-0,1,2}
    TEACHER_CUDA_VISIBLE_DEVICES=${TEACHER_CUDA_VISIBLE_DEVICES:-3}
else
    STUDENT_CUDA_VISIBLE_DEVICES=${STUDENT_CUDA_VISIBLE_DEVICES:-0,1,2,3}
fi

DEFAULT_TRAIN_GPUS="$(count_csv_items "$STUDENT_CUDA_VISIBLE_DEVICES")"
TRAIN_GPUS=${TRAIN_GPUS:-$DEFAULT_TRAIN_GPUS}

STUDENT_HF_CHECKPOINT=${STUDENT_HF_CHECKPOINT:-"/data/models/student-0.8b"}
STUDENT_REF_LOAD=${STUDENT_REF_LOAD:-"/data/models/student-0.8b_torch_dist"}
STUDENT_MODEL_CONFIG=${STUDENT_MODEL_CONFIG:-"${SCRIPT_DIR}/model_configs/qwen3-0.6B.sh"}

PROMPT_DATA=${PROMPT_DATA:-"/data/dapo-math-17k/dapo-math-17k.jsonl"}
EVAL_PROMPT_DATA=${EVAL_PROMPT_DATA:-"/data/aime-2024/aime-2024.jsonl"}
SAVE_DIR=${SAVE_DIR:-"/data/agentflow_runs/student_0.8b_4xa800"}

TEACHER_HF_CHECKPOINT=${TEACHER_HF_CHECKPOINT:-"/data/models/teacher-2b"}
TEACHER_HOST=${TEACHER_HOST:-"127.0.0.1"}
TEACHER_PORT=${TEACHER_PORT:-"30080"}
TEACHER_CONTEXT_LENGTH=${TEACHER_CONTEXT_LENGTH:-32768}
TEACHER_MEM_FRACTION_STATIC=${TEACHER_MEM_FRACTION_STATIC:-0.78}

if is_true "$LOCAL_TEACHER"; then
    RM_URL="http://${TEACHER_HOST}:${TEACHER_PORT}/generate"
else
    if [ -z "${RM_URL:-}" ]; then
        echo "ERROR: External-teacher mode requires RM_URL=http://teacher-host:port/generate." >&2
        echo "Set LOCAL_TEACHER=1 to launch a local teacher on GPU ${TEACHER_CUDA_VISIBLE_DEVICES:-3}." >&2
        exit 1
    fi
fi

cleanup_old_processes() {
    log "清理旧 Ray 进程..."
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
    if is_true "$CLEAN_SGLANG"; then
        log "清理旧 SGLang 进程..."
        pkill -9 -f 'sglang\.launch_server' 2>/dev/null || true
    fi
    sleep 2
}

start_local_teacher() {
    log "启动本地 teacher SGLang：${TEACHER_HF_CHECKPOINT} -> ${RM_URL}"
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
    wait_port "teacher" "${TEACHER_HOST}" "${TEACHER_PORT}" 1200
}

run_training() {
    local default_rollout_batch_size default_global_batch_size
    if [ "$TRAIN_GPUS" -ge 4 ]; then
        default_rollout_batch_size=8
        default_global_batch_size=64
    else
        default_rollout_batch_size=6
        default_global_batch_size=48
    fi

    log "启动 4xA800 formal training。student GPUs=${STUDENT_CUDA_VISIBLE_DEVICES}, TRAIN_GPUS=${TRAIN_GPUS}"
    log "训练日志: ${LOG_DIR}/train.log"
    cd "$REPO_ROOT"
    CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}" \
    SKIP_PROCESS_KILL=1 \
    MODEL_CONFIG_SCRIPT="${STUDENT_MODEL_CONFIG}" \
    BASE_HF_CHECKPOINT="${STUDENT_HF_CHECKPOINT}" \
    REF_LOAD="${STUDENT_REF_LOAD}" \
    SAVE_DIR="${SAVE_DIR}" \
    PROMPT_DATA="${PROMPT_DATA}" \
    EVAL_PROMPT_DATA="${EVAL_PROMPT_DATA}" \
    TRAIN_GPUS="${TRAIN_GPUS}" \
    TRAIN_TP=${TRAIN_TP:-1} \
    ROLLOUT_ENGINE_GPUS=${ROLLOUT_ENGINE_GPUS:-1} \
    NUM_EPOCH=${NUM_EPOCH:-1} \
    ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-$default_rollout_batch_size} \
    N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8} \
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$default_global_batch_size} \
    ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-8192} \
    SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-32768} \
    SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.72} \
    MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384} \
    SAVE_INTERVAL=${SAVE_INTERVAL:-50} \
    EVAL_INTERVAL=${EVAL_INTERVAL:-20} \
    USE_OPD=${USE_OPD:-1} \
    OPD_TYPE=${OPD_TYPE:-sglang} \
    OPD_KL_COEF=${OPD_KL_COEF:-1.0} \
    USE_KL_LOSS=${USE_KL_LOSS:-0} \
    RM_URL="${RM_URL}" \
        bash "${SCRIPT_DIR}/train_agentflow.sh" \
        > "${LOG_DIR}/train.log" 2>&1
}

cleanup_old_processes
if is_true "$LOCAL_TEACHER"; then
    start_local_teacher
else
    log "使用外部 teacher: ${RM_URL}"
fi
run_training
log "formal training 完成。日志目录: ${LOG_DIR}"
