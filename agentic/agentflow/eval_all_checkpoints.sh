#!/bin/bash
# 对所有转换后的 HF checkpoint 循环评估，每个跑 NUM_RUNS 次，取最高分，记录为 JSONL。
#
# 用法：
#   bash eval_all_checkpoints.sh
#   NUM_RUNS=5 bash eval_all_checkpoints.sh          # 只跑 5 次
#   CHECKPOINT_DIR=/other/path bash eval_all_checkpoints.sh

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${SLIME_ROOT}:${PYTHONPATH:-}"

# ── 配置 ──────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/data/AgentFlow_Qwen25-7B-RL-HF"}
TOKENIZER_PATH=${TOKENIZER_PATH:-"/data/models/qwen25_7b"}
MODEL_BASE=${MODEL_BASE:-"/data/models/qwen25_7b"}
MODEL_CODER=${MODEL_CODER:-"/data/models/qwen2.5_7b_codeer"}

CTX_LEN=${CTX_LEN:-131072}
MEM_FRACTION=${MEM_FRACTION:-0.7}
CONCURRENCY=${CONCURRENCY:-16}
MAX_STEPS=${MAX_STEPS:-5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}
NUM_SAMPLES=${NUM_SAMPLES:-0}
NUM_RUNS=${NUM_RUNS:-10}

LOG_DIR="/tmp/agentflow_ckpt_eval_logs"
TMP_DIR="/tmp/agentflow_ckpt_eval_tmp"
RESULTS_JSONL="${SCRIPT_DIR}/checkpoint_eval_results.jsonl"

mkdir -p "$LOG_DIR" "$TMP_DIR"

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
        (( elapsed % 30 == 0 )) && log "  还在等 $name... (${elapsed}s)"
    done
    log "ERROR: $name 在 ${timeout}s 内未就绪，日志: $LOG_DIR"
    return 1
}

stop_planner() {
    pkill -f "sglang.launch_server.*--port 30000" 2>/dev/null || true
    sleep 3
}

# ── 全局 cleanup ──────────────────────────────────────────────────────────────
cleanup() {
    log "关闭所有 SGLang 服务..."
    pkill -f "sglang.launch_server.*--port 3000" 2>/dev/null || true
    sleep 2
    log "清理完成。"
}
trap cleanup EXIT

# ── Step 1: 清理旧进程 ────────────────────────────────────────────────────────
log "清理旧 SGLang 进程..."
pkill -f "sglang.launch_server.*--port 3000" 2>/dev/null || true
sleep 2

# ── Step 2: 启动 Executor + Coder（只启动一次，整个脚本共享）────────────────
# GPU 2,3 — base 模型，port 30001
CUDA_VISIBLE_DEVICES=2,3 conda run -n sglang --no-capture-output \
    bash -c "
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python3 -m sglang.launch_server \
            --model-path ${MODEL_BASE} \
            --port 30001 \
            --context-length ${CTX_LEN} \
            --tp 2
    " > "$LOG_DIR/sglang_30001.log" 2>&1 &
log "Executor (base)  PID=$! (GPU 2,3, port 30001)"

# GPU 4,5 — coder 模型，port 30002
CUDA_VISIBLE_DEVICES=4,5 conda run -n sglang --no-capture-output \
    bash -c "
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python3 -m sglang.launch_server \
            --model-path ${MODEL_CODER} \
            --port 30002 \
            --context-length ${CTX_LEN} \
            --tp 2
    " > "$LOG_DIR/sglang_30002.log" 2>&1 &
log "Coder            PID=$! (GPU 4,5, port 30002)"

wait_port "Executor-30001" 127.0.0.1 30001 600
wait_port "Coder-30002"    127.0.0.1 30002 600
log "Executor + Coder 已就绪。"

# ── Step 3: 收集所有 checkpoint ───────────────────────────────────────────────
mapfile -t CKPTS < <(ls -d "${CHECKPOINT_DIR}"/iter_* 2>/dev/null | sort)

if [ ${#CKPTS[@]} -eq 0 ]; then
    log "ERROR: 在 ${CHECKPOINT_DIR} 中未找到 iter_* checkpoint，请先运行转换脚本。"
    exit 1
fi

log "共找到 ${#CKPTS[@]} 个 checkpoint，每个跑 ${NUM_RUNS} 次评估。"
log "结果将追加写入：${RESULTS_JSONL}"
echo ""

# 构建公共 eval 参数（手动 URL 模式，不让 eval_agentflow.py 自己管服务）
COMMON_ARGS=(
    --tokenizer      "${TOKENIZER_PATH}"
    --eval-data      aime /data/aime-2024/aime-2024.jsonl
    --input-key      prompt
    --label-key      label
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
    COMMON_ARGS+=(--num-samples "${NUM_SAMPLES}")
fi

# ── Step 4: 逐 checkpoint 评估 ────────────────────────────────────────────────
for ckpt_path in "${CKPTS[@]}"; do
    ckpt_name=$(basename "$ckpt_path")

    # 如果 JSONL 中已有此 checkpoint 的记录，跳过
    if [ -f "$RESULTS_JSONL" ] && grep -q "\"checkpoint\":\"${ckpt_name}\"" "$RESULTS_JSONL" 2>/dev/null; then
        log "[SKIP] ${ckpt_name} 已有记录，跳过。"
        continue
    fi

    log "════════════════════════════════════════"
    log "开始评估：${ckpt_name}"
    log "模型路径：${ckpt_path}"

    # 启动 Planner（GPU 0,1，port 30000）
    CUDA_VISIBLE_DEVICES=0,1 conda run -n sglang --no-capture-output \
        bash -c "
            export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
            python3 -m sglang.launch_server \
                --model-path ${ckpt_path} \
                --port 30000 \
                --context-length ${CTX_LEN} \
                --tp 2
        " > "$LOG_DIR/sglang_30000_${ckpt_name}.log" 2>&1 &
    PLANNER_PID=$!
    log "Planner PID=${PLANNER_PID}，日志: $LOG_DIR/sglang_30000_${ckpt_name}.log"

    if ! wait_port "Planner-30000" 127.0.0.1 30000 600; then
        log "[FAILED] ${ckpt_name} Planner 启动失败，跳过。"
        kill $PLANNER_PID 2>/dev/null || true
        stop_planner
        continue
    fi

    # 跑 NUM_RUNS 次，收集 accuracy
    scores=()
    run_details=()
    for ((run=1; run<=NUM_RUNS; run++)); do
        out_file="${TMP_DIR}/${ckpt_name}_run${run}.json"
        log "  [${ckpt_name}] Run ${run}/${NUM_RUNS}..."

        if python3 "${SCRIPT_DIR}/eval_agentflow.py" "${COMMON_ARGS[@]}" --output "${out_file}"; then
            # 提取 accuracy（取第一个数据集的分数）
            acc=$(python3 -c "
import json, sys
d = json.load(open('${out_file}'))
first = next(iter(d.values()))
print(first['accuracy'])
" 2>/dev/null || echo "0.0")
            scores+=("$acc")
            run_details+=("{\"run\":${run},\"accuracy\":${acc}}")
            log "  [${ckpt_name}] Run ${run} accuracy=${acc}"
        else
            log "  [${ckpt_name}] Run ${run} 评估失败，记为 0.0"
            scores+=("0.0")
            run_details+=("{\"run\":${run},\"accuracy\":0.0,\"error\":true}")
        fi
    done

    # 停止 Planner，释放 GPU 0,1
    log "停止 Planner (PID=${PLANNER_PID})..."
    kill $PLANNER_PID 2>/dev/null || true
    stop_planner

    # 计算最高分（用 IFS=, 将数组元素以逗号拼接为合法 Python 列表）
    scores_csv=$(IFS=,; echo "${scores[*]}")
    best_score=$(python3 -c "scores=[${scores_csv}]; print(max(scores) if scores else 0.0)")
    avg_score=$(python3  -c "scores=[${scores_csv}]; print(sum(scores)/len(scores) if scores else 0.0)")

    # 写入 JSONL（纯 Python，无需 jq）
    python3 - << PYEOF >> "$RESULTS_JSONL"
import json, sys
runs = [$(IFS=,; echo "${run_details[*]}")]
record = {
    "checkpoint": "${ckpt_name}",
    "path":       "${ckpt_path}",
    "best_score": ${best_score},
    "avg_score":  ${avg_score},
    "num_runs":   len(runs),
    "runs":       runs,
    "timestamp":  $(date +%s),
}
print(json.dumps(record, ensure_ascii=False))
PYEOF

    log "[DONE] ${ckpt_name}  best=${best_score}  avg=${avg_score}"
    echo ""
done

log "════════════════════════════════════════"
log "全部评估完成！结果文件：${RESULTS_JSONL}"
log ""
log "各 checkpoint 最高分一览："
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/data/slime-agentic/agentic/agentflow/checkpoint_eval_results.jsonl")
if not p.exists():
    print("  (文件不存在)")
else:
    for line in p.read_text().strip().splitlines():
        r = json.loads(line)
        print(f"  {r['checkpoint']:20s}  best={r['best_score']:.3f}  avg={r['avg_score']:.3f}")
PY
