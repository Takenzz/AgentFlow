#!/bin/bash
# eval_all_checkpoints.sh — 遍历所有 checkpoint，每个跑 N 次 ruler_hqa，记录 max/avg
#
# 用法：
#   bash eval_all_checkpoints.sh
#   CKPT_DIR=/data/my_ckpt_hf RUNS=3 LENGTH="50 200" bash eval_all_checkpoints.sh
#
# 环境变量（均有默认值）：
#   CKPT_DIR      HF checkpoint 根目录（内含 iter_* 子目录）
#   LENGTH        要评测的 doc 数列表（空格分隔）
#   RUNS          每个 checkpoint 重复跑几次（默认 5）
#   DATA_ROOT     eval_{length}.json 所在目录
#   SAVE_BASE     结果根目录（默认 results/checkpoint_sweep）
#   TP            SGLang tensor parallel size（默认 1）
#   SERVE_PORT    SGLang 服务端口（默认 8000）
#   N_PROC        并发请求数（默认 64）
#   CONTEXT_LEN   SGLang context 长度（默认 131072）
#   MEM_CHUNK_TOKENS / MEM_MAX_MEMORY / MEM_MAX_FINAL / MEM_MAX_CHUNKS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 配置 ──────────────────────────────────────────────────────────────────────
CKPT_DIR="${CKPT_DIR:-/data/MemAgent_Qwen25-7B-RL-HF}"
LENGTH="${LENGTH:-50}"
RUNS="${RUNS:-5}"
DATA_ROOT="${DATA_ROOT:-/data/hotpotqa_dataset/files}"
SAVE_BASE="${SAVE_BASE:-${SCRIPT_DIR}/results/checkpoint_sweep}"
TP="${TP:-1}"
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
SERVE_PORT="${SERVE_PORT:-8000}"
N_PROC="${N_PROC:-64}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
CONTEXT_LEN="${CONTEXT_LEN:-131072}"
MEM_CHUNK_TOKENS="${MEM_CHUNK_TOKENS:-5000}"
MEM_MAX_MEMORY="${MEM_MAX_MEMORY:-1024}"
MEM_MAX_FINAL="${MEM_MAX_FINAL:-256}"
MEM_MAX_CHUNKS="${MEM_MAX_CHUNKS:-512}"

# ── 工具函数 ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

export SERVE_HOST SERVE_PORT MEM_CHUNK_TOKENS MEM_MAX_MEMORY MEM_MAX_FINAL MEM_MAX_CHUNKS
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# ── 收集 checkpoint ───────────────────────────────────────────────────────────
mapfile -t CKPTS < <(ls -d "${CKPT_DIR}"/iter_* 2>/dev/null | sort)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "[error] No iter_* checkpoints found in ${CKPT_DIR}"
    exit 1
fi

log "Found ${#CKPTS[@]} checkpoints, RUNS=${RUNS}, LENGTH=${LENGTH}"
mkdir -p "${SAVE_BASE}"

SUMMARY_FILE="${SAVE_BASE}/summary.tsv"
# 写表头（仅当文件不存在时）
if [[ ! -f "${SUMMARY_FILE}" ]]; then
    {
        printf "checkpoint"
        for length in ${LENGTH}; do
            printf "\tlength=%s_run1\tlength=%s_run2\tlength=%s_run3\tlength=%s_run4\tlength=%s_run5\tlength=%s_max\tlength=%s_avg" \
                "${length}" "${length}" "${length}" "${length}" "${length}" "${length}" "${length}"
        done
        printf "\n"
    } > "${SUMMARY_FILE}"
fi

# ── 启动 / 重启 SGLang ────────────────────────────────────────────────────────
start_server() {
    local model_path="$1"
    local sglang_log="${SAVE_BASE}/sglang_${model_name}.log"

    log "Killing any existing SGLang processes ..."
    pkill -f "sglang.launch_server" 2>/dev/null || true
    pkill -f "sglang_router"        2>/dev/null || true
    sleep 3

    log "Starting SGLang for ${model_name} ..."
    python -m sglang.launch_server \
        --model-path "${model_path}" \
        --tp "${TP}" \
        --host "${SERVE_HOST}" \
        --port "${SERVE_PORT}" \
        --context-length "${CONTEXT_LEN}" \
        --trust-remote-code \
        > "${sglang_log}" 2>&1 &
    SGLANG_PID=$!

    # 等待服务就绪
    local url="http://${SERVE_HOST}:${SERVE_PORT}/v1/models"
    log "Waiting for SGLang server (model=${model_path}) ..."
    local attempts=0
    while true; do
        local resp
        resp=$(curl -sf --max-time 10 "${url}" 2>/dev/null || true)
        if echo "${resp}" | grep -q "${model_name}" 2>/dev/null; then
            log "Server ready."
            break
        fi
        attempts=$((attempts + 1))
        if (( attempts % 6 == 0 )); then
            local found
            found=$(echo "${resp}" | grep -o '"id":"[^"]*"' 2>/dev/null || echo "(no response)")
            log "Still waiting... got: ${found}"
        fi
        sleep 5
    done
}

stop_server() {
    if [[ -n "${SGLANG_PID:-}" ]]; then
        log "Stopping SGLang (pid=${SGLANG_PID}) ..."
        kill -TERM -- "-${SGLANG_PID}" 2>/dev/null || kill "${SGLANG_PID}" 2>/dev/null || true
        wait "${SGLANG_PID}" 2>/dev/null || true
        SGLANG_PID=""
    fi
}

trap 'stop_server' EXIT INT TERM

# ── 解析 sub_EM ───────────────────────────────────────────────────────────────
# 从 eval_ruler_hqa.py 的输出中提取 sub_em 值（百分制）
parse_sub_em() {
    local output="$1"
    echo "${output}" | grep -oP '(?<=sub_em: )\d+(\.\d+)?' | tail -1
}

# ── 主循环 ────────────────────────────────────────────────────────────────────
SGLANG_PID=""

for ckpt_path in "${CKPTS[@]}"; do
    model_name="$(basename "${ckpt_path}")"
    log "========================================"
    log "Checkpoint: ${model_name}"
    log "========================================"

    start_server "${ckpt_path}"

    # 当前 checkpoint 的汇总行（先写 checkpoint 名）
    row="${model_name}"

    for length in ${LENGTH}; do
        log "--- length=${length} ---"
        run_scores=()

        for (( run=1; run<=RUNS; run++ )); do
            save_dir="${SAVE_BASE}/${model_name}/ruler_hqa_${length}"
            save_file="${model_name}_run${run}"

            log "  run ${run}/${RUNS}  (save_file=${save_file})"
            output=$(
                python "${SCRIPT_DIR}/eval_ruler_hqa.py" \
                    --model       "${ckpt_path}" \
                    --tokenizer   "${ckpt_path}" \
                    --length      "${length}" \
                    --data-root   "${DATA_ROOT}" \
                    --save-dir    "${save_dir}" \
                    --save-file   "${save_file}" \
                    --api         recurrent \
                    --n-proc      "${N_PROC}" \
                    --temperature "${TEMPERATURE}" \
                    --top-p       "${TOP_P}" \
                    --force \
                    2>&1
            )
            echo "${output}"

            score=$(parse_sub_em "${output}")
            if [[ -z "${score}" ]]; then
                log "  [warn] Could not parse sub_em from run ${run}, using 0"
                score="0"
            fi
            run_scores+=("${score}")
            log "  run ${run} sub_EM = ${score}"
        done

        # 计算 max / avg（用 python 做浮点运算）
        scores_csv=$(IFS=,; echo "${run_scores[*]}")
        read -r max_score avg_score < <(python3 - <<EOF
scores = [float(x) for x in "${scores_csv}".split(",")]
print(f"{max(scores):.2f} {sum(scores)/len(scores):.2f}")
EOF
        )

        log "  length=${length}  scores=[${scores_csv}]  max=${max_score}  avg=${avg_score}"

        # 追加到行
        for s in "${run_scores[@]}"; do
            row+="\t${s}"
        done
        row+="\t${max_score}\t${avg_score}"

        # 实时把当前 length 的结果写到独立文件
        length_result="${SAVE_BASE}/${model_name}/result_length${length}.txt"
        mkdir -p "$(dirname "${length_result}")"
        {
            echo "checkpoint: ${model_name}"
            echo "length:     ${length}"
            echo "runs:       ${scores_csv}"
            echo "max:        ${max_score}"
            echo "avg:        ${avg_score}"
        } > "${length_result}"
    done

    # 写入汇总 TSV
    printf "%b\n" "${row}" >> "${SUMMARY_FILE}"
    log "Results written to ${SUMMARY_FILE}"

    stop_server
done

log "========================================"
log "All done. Summary: ${SUMMARY_FILE}"
log "========================================"
cat "${SUMMARY_FILE}"
