#!/bin/bash
# run_eval.sh — 一键启动 SGLang 服务 + MemAgent 评测
# =====================================================
# 用法：
#   bash run_eval.sh                          # 用默认配置跑全套
#   TASKS=hqa LENGTH="50 200 800" bash run_eval.sh
#   MODEL_PATH=/data/my_ckpt TP=4 bash run_eval.sh
#
# 主要环境变量（均有默认值，按需覆盖）：
#   MODEL_PATH    模型路径（必填，或在脚本内修改默认值）
#   TP            tensor parallel size（默认 1）
#   SERVE_PORT    SGLang 服务端口（默认 8000）
#   TASKS         评测套件：hqa | general | all（默认 hqa）
#   LENGTH        HQA 的 doc 数列表（默认 "50 100 200 400 800 1600 3200 6400"）
#   SPLITS        RULER general 的 split 列表
#   RULER_LENGTHS RULER general 的长度列表
#   DATA_ROOT     eval_{length}.json 所在目录
#   RULER_DATA_ROOT eval_{split}_{length}.json 所在目录
#   SAVE_DIR      结果输出目录（默认 results/）
#   SAVE_FILE     结果文件名前缀（默认取模型目录名）
#   N_PROC        并发请求数（默认 64）
#   TEMPERATURE   采样温度（默认 0.7）
#   TOP_P         top-p（默认 0.95）
#   OVER1M        是否额外跑 12800/25600 doc 测试（默认 0）
#   API           推理模式：recurrent | openai（默认 recurrent）
#   FORCE         是否忽略缓存重新评测（默认 0）
#   CONTEXT_LEN   SGLang context 长度（默认 131072）
#   MEM_CHUNK_TOKENS / MEM_MAX_MEMORY / MEM_MAX_FINAL / MEM_MAX_CHUNKS
# =====================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 配置（可通过环境变量覆盖） ──────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-""}            # 必填
TP=${TP:-1}
SERVE_HOST=${SERVE_HOST:-"127.0.0.1"}
SERVE_PORT=${SERVE_PORT:-"8000"}

TASKS=${TASKS:-"hqa"}                  # hqa | general | all
LENGTH=${LENGTH:-"50 100 200 400 800 1600 3200 6400"}
OVER1M=${OVER1M:-0}                    # 1 则追加 12800 25600

SPLITS=${SPLITS:-"niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt fwe qa_1"}
RULER_LENGTHS=${RULER_LENGTHS:-"8192 16384 32768 65536 131072 262144 524288"}

DATA_ROOT=${DATA_ROOT:-${DATAROOT:-"/data/hotpotqa"}}
RULER_DATA_ROOT=${RULER_DATA_ROOT:-${RULER_DATAROOT:-"/data/ruler"}}
SAVE_DIR=${SAVE_DIR:-"${SCRIPT_DIR}/results"}

API=${API:-"recurrent"}
N_PROC=${N_PROC:-64}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_P=${TOP_P:-0.95}
MAX_INPUT_LEN=${MAX_INPUT_LEN:-120000}   # openai 模式专用
MAX_OUTPUT_LEN=${MAX_OUTPUT_LEN:-10000}  # openai 模式专用
FORCE=${FORCE:-0}

# SGLang 服务参数
CONTEXT_LEN=${CONTEXT_LEN:-131072}
MEM_CHUNK_TOKENS=${MEM_CHUNK_TOKENS:-5000}
MEM_MAX_MEMORY=${MEM_MAX_MEMORY:-1024}
MEM_MAX_FINAL=${MEM_MAX_FINAL:-256}
MEM_MAX_CHUNKS=${MEM_MAX_CHUNKS:-512}

# ── 参数校验 ─────────────────────────────────────────────────────────────────
if [[ -z "${MODEL_PATH}" ]]; then
    echo "[error] MODEL_PATH is not set."
    echo "  Usage: MODEL_PATH=/path/to/ckpt bash run_eval.sh"
    exit 1
fi

if [[ ! -e "${MODEL_PATH}" ]] && [[ "${MODEL_PATH}" != */* ]]; then
    # 允许 HuggingFace 格式 "org/repo" 直接传入
    echo "[info] MODEL_PATH looks like a HuggingFace repo id: ${MODEL_PATH}"
fi

# SAVE_FILE 默认取 MODEL_PATH 的最后一级目录名
SAVE_FILE=${SAVE_FILE:-"$(basename "${MODEL_PATH%/}")"}

# MODEL_NAME 是 SGLang 服务中注册的 id
# SGLang 以完整 --model-path 作为 model id，所以这里保留完整路径
MODEL_NAME="${MODEL_PATH}"

# ── 工具函数 ─────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_server() {
    local url="http://${SERVE_HOST}:${SERVE_PORT}/v1/models"
    local model_id="${MODEL_NAME}"
    log "Waiting for SGLang server at ${url} (model=${model_id}) ..."
    local attempts=0
    while true; do
        resp=$(curl -sf --max-time 10 "${url}" 2>/dev/null || true)
        if echo "${resp}" | grep -q "${model_id}" 2>/dev/null; then
            log "Server is ready."
            break
        fi
        attempts=$((attempts + 1))
        # 每 6 次（30s）打印一次当前服务返回的 model id，方便排查名字不匹配
        if (( attempts % 6 == 0 )); then
            local found_ids
            found_ids=$(echo "${resp}" | grep -o '"id":"[^"]*"' 2>/dev/null || echo "(no response)")
            log "Still waiting... server returned: ${found_ids}"
        fi
        sleep 5
    done
}

kill_server() {
    if [[ -n "${SGLANG_PID:-}" ]]; then
        log "Shutting down SGLang server (pid=${SGLANG_PID}) ..."
        kill -TERM -- "-${SGLANG_PID}" 2>/dev/null || kill "${SGLANG_PID}" 2>/dev/null || true
    fi
}

# ── 构建公共 eval 参数 ────────────────────────────────────────────────────────
build_common_args() {
    local extra_args=()
    extra_args+=(--model     "${MODEL_NAME}")
    extra_args+=(--tokenizer "${MODEL_PATH}")
    extra_args+=(--api       "${API}")
    extra_args+=(--n-proc    "${N_PROC}")
    extra_args+=(--temperature "${TEMPERATURE}")
    extra_args+=(--top-p    "${TOP_P}")
    if [[ "${FORCE}" == "1" ]]; then
        extra_args+=(--force)
    fi
    if [[ "${API}" == "openai" ]]; then
        extra_args+=(--max-input-len  "${MAX_INPUT_LEN}")
        extra_args+=(--max-output-len "${MAX_OUTPUT_LEN}")
    fi
    echo "${extra_args[@]}"
}

# ── 评测函数 ─────────────────────────────────────────────────────────────────
run_hqa() {
    local lengths=($LENGTH)
    if [[ "${OVER1M}" == "1" ]]; then
        lengths+=(12800 25600)
    fi

    local common
    read -ra common <<< "$(build_common_args)"

    for length in "${lengths[@]}"; do
        local save_subdir="${SAVE_DIR}/ruler_hqa_${length}"
        log "==> ruler_hqa [n_docs=${length}]"
        python "${SCRIPT_DIR}/eval_ruler_hqa.py" \
            "${common[@]}" \
            --length    "${length}" \
            --data-root "${DATA_ROOT}" \
            --save-dir  "${save_subdir}" \
            --save-file "${SAVE_FILE}"
    done
}

run_general() {
    local splits=($SPLITS)
    local lengths=($RULER_LENGTHS)

    local common
    read -ra common <<< "$(build_common_args)"

    for split in "${splits[@]}"; do
        for length in "${lengths[@]}"; do
            # ruler_general.py 原版跳过 qa_1 > 262144
            if [[ "${split}" == "qa_1" ]] && (( length > 262144 )); then
                continue
            fi
            local save_subdir="${SAVE_DIR}/ruler_${split}_${length}"
            log "==> ruler_general [split=${split}  length=${length}]"
            python "${SCRIPT_DIR}/eval_ruler_general.py" \
                "${common[@]}" \
                --split     "${split}" \
                --length    "${length}" \
                --data-root "${RULER_DATA_ROOT}" \
                --save-dir  "${save_subdir}" \
                --save-file "${SAVE_FILE}"
        done
    done
}

# ── 主流程 ───────────────────────────────────────────────────────────────────
export SERVE_HOST SERVE_PORT
export MEM_CHUNK_TOKENS MEM_MAX_MEMORY MEM_MAX_FINAL MEM_MAX_CHUNKS
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

log "=== MemAgent Eval ==="
log "  MODEL_PATH  = ${MODEL_PATH}"
log "  TP          = ${TP}"
log "  TASKS       = ${TASKS}"
log "  API         = ${API}"
log "  SERVE_PORT  = ${SERVE_PORT}"
log "  SAVE_DIR    = ${SAVE_DIR}"
log "  SAVE_FILE   = ${SAVE_FILE}"
log "  MEM_CHUNK_TOKENS=${MEM_CHUNK_TOKENS}  MAX_MEMORY=${MEM_MAX_MEMORY}  MAX_FINAL=${MEM_MAX_FINAL}"

mkdir -p "${SAVE_DIR}"

# ── 启动 SGLang ───────────────────────────────────────────────────────────────
SGLANG_LOG="${SAVE_DIR}/sglang_server.log"

# ── 清理可能残留的旧 SGLang 进程 ─────────────────────────────────────────────
log "Killing any existing SGLang processes on port ${SERVE_PORT} ..."
pkill -f "sglang.launch_server" 2>/dev/null || true
pkill -f "sglang_router"        2>/dev/null || true
sleep 2   # 等待端口释放

log "Starting SGLang server (tp=${TP}, port=${SERVE_PORT}, ctx=${CONTEXT_LEN}) ..."
log "SGLang logs -> ${SGLANG_LOG}"
python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --tp "${TP}" \
    --host "${SERVE_HOST}" \
    --port "${SERVE_PORT}" \
    --context-length "${CONTEXT_LEN}" \
    --trust-remote-code \
    > "${SGLANG_LOG}" 2>&1 &
# 记录进程组 id 以便 trap 清理
SGLANG_PID=$!
trap 'kill_server' EXIT INT TERM

wait_for_server

# ── 运行评测 ─────────────────────────────────────────────────────────────────
case "${TASKS}" in
    hqa)
        run_hqa
        ;;
    general)
        run_general
        ;;
    all)
        run_hqa
        run_general
        ;;
    *)
        log "[error] Unknown TASKS value: ${TASKS}. Use hqa | general | all."
        exit 1
        ;;
esac

log "=== All evaluations finished. Results in: ${SAVE_DIR} ==="
