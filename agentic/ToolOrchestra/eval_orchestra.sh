#!/bin/bash
# ToolOrchestra 一键评测脚本
# ==========================
# 自动启动所有依赖服务（检索 + Expert SGLang × 3 + Orchestrator SGLang），
# 运行评测，评测完自动停止本脚本启动的服务。
#
# GPU 分配（默认，可通过环境变量覆盖）：
#   GPU 0       : 检索服务 + Orchestrator  (conda: orche / sglang，8B 模型小，可共存)
#   GPU 1       : SGLang Qwen3-32B-FP8      port=30001 (conda: sglang)
#   GPU 2       : SGLang Qwen2.5-Coder-32B  port=30002 (conda: sglang)
#   GPU 3       : SGLang Qwen2.5-Math-7B    port=30003 (conda: sglang)
#   GPU 4       : SGLang Qwen3-14B          port=30004 (conda: sglang)
#   GPU 5       : SGLang DeepSeek-R1-32B    port=30005 (conda: sglang)
#   GPU 6       : SGLang Qwen3-30B-A3B      port=30006 (conda: sglang)
#   GPU 7       : SGLang Qwen2.5-Math-72B   port=30007 (conda: sglang)
#
# 使用方式：
#
#   # 最简单：直接指定 checkpoint（同时用作 tokenizer 路径）
#   ORCH_CKPT=/data/checkpoints/step_100 bash eval_orchestra.sh
#
#   # tokenizer 和 ckpt 不同路径时（如 ckpt 是合并后的路径，tokenizer 用原始 HF 模型）
#   ORCH_CKPT=/data/checkpoints/step_100 ORCH_MODEL=Qwen/Qwen3-8B bash eval_orchestra.sh
#
#   # 评测 FRAMES benchmark
#   ORCH_CKPT=/data/checkpoints/step_100 BENCHMARK=frames bash eval_orchestra.sh
#
#   # 评测 HLE benchmark（默认读 evaluation/hle.jsonl）
#   ORCH_CKPT=/data/checkpoints/step_100 BENCHMARK=hle bash eval_orchestra.sh
#
#   # HLE 用 Parquet 或 HF 多 shard 目录（需 pandas/pyarrow）
#   ORCH_CKPT=/path/to/ckpt BENCHMARK=hle \
#     EVAL_DATA=/path/to/hle.parquet MAX_TURNS=15 bash eval_orchestra.sh
#   # 或 EVAL_DATA=/path/to/hle_parquet_shards_dir/
#
#   # 快速冒烟测试（5 条样本）
#   ORCH_CKPT=/data/checkpoints/step_100 MAX_EXAMPLES=5 bash eval_orchestra.sh
#
#   # Orchestrator 已有 SGLang 服务在运行，跳过自动启动（必须指定 ORCH_MODEL）
#   SKIP_SERVICES=1 ORCH_URL=http://127.0.0.1:30000/v1 \
#       ORCH_MODEL=Qwen/Qwen3-8B bash eval_orchestra.sh
#
#   # 不启动 expert/检索（服务已在运行），仅自动启动 orchestrator
#   SKIP_EXPERT_SERVICES=1 ORCH_CKPT=/data/checkpoints/step_100 bash eval_orchestra.sh
#
#   # 与其它评测并行、勿杀残留 eval 进程时
#   KILL_STALE_EVAL=0 bash eval_orchestra.sh

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/orchestra_eval_logs}"
mkdir -p "$LOG_DIR"

# ──────────────────────────── 选择正确的 Python ──────────────────────────── #
# eval_orchestra.py 需要 transformers（tokenizer）、httpx、openai 等，
# 这些通常在 sglang conda 环境中。tau2 子进程通过 sys.executable 继承同
# 一个 Python，确保所有依赖可用。
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-sglang}"
EVAL_PYTHON_CMD="conda run -n ${EVAL_CONDA_ENV} --no-capture-output python"
log_python() { echo "[python] Using conda env: $EVAL_CONDA_ENV"; }

# ──────────────────────────── 可配置参数 ──────────────────────────────────── #
BENCHMARK="${BENCHMARK:-tau2}"          # tau2 | hle | frames

ORCH_CKPT="${ORCH_CKPT:-}"             # 若设置，自动启动 orchestrator SGLang
ORCH_URL="${ORCH_URL:-http://127.0.0.1:30000/v1}"
ORCH_MODEL="${ORCH_MODEL:-}"           # HF 模型名或本地路径，用于加载 tokenizer
                                       # 留空时自动取 ORCH_CKPT
ORCH_PORT="${ORCH_PORT:-30000}"
ORCH_GPU="${ORCH_GPU:-0}"              # orchestrator 与检索服务共用 GPU 0（8B 模型占用小）
ORCH_TP="${ORCH_TP:-1}"               # tensor parallel 数（跨多 GPU 需改）
ORCH_CTX="${ORCH_CTX:-131072}"         # 必须和 SGLang --context-length 一致

EVAL_DATA="${EVAL_DATA:-}"             # 留空则 tau2/hle/frames 用脚本目录下默认 jsonl；
                                       # HLE 可用 Parquet 文件或含 *.parquet 的目录
CONCURRENCY="${CONCURRENCY:-4}"
MAX_TURNS="${MAX_TURNS:-12}"
TEMPERATURE="${TEMPERATURE:-0.7}"
USER_LLM="${USER_LLM:-qwen-turbo-latest}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
DOMAINS="${DOMAINS:-}"

# 是否跳过自动启动（服务已在运行时使用）
SKIP_SERVICES="${SKIP_SERVICES:-0}"        # =1 跳过全部自动启动
SKIP_EXPERT_SERVICES="${SKIP_EXPERT_SERVICES:-0}"  # =1 只跳过 expert/检索

# 启动前是否杀掉残留的 eval_orchestra.py（避免与 --clean-previous 互相删目录）
KILL_STALE_EVAL="${KILL_STALE_EVAL:-1}"

# Expert 服务资源
RETRIEVAL_GPU="${RETRIEVAL_GPU:-0}"
EXPERT1_GPU="${EXPERT1_GPU:-1}"   # Qwen3-32B-FP8       → search-1 / reasoner-1 / answer-1 / expert-1
EXPERT2_GPU="${EXPERT2_GPU:-2}"   # Qwen2.5-Coder-32B   → reasoner-2
EXPERT3_GPU="${EXPERT3_GPU:-3}"   # Qwen2.5-Math-7B     → answer-math-2
EXPERT4_GPU="${EXPERT4_GPU:-4}"   # Qwen3-14B           → expert-3 / reasoner-3
EXPERT5_GPU="${EXPERT5_GPU:-5}"   # DeepSeek-R1-32B     → answer-2
EXPERT6_GPU="${EXPERT6_GPU:-6}"   # Qwen3-30B-A3B       → expert-2 / answer-3
EXPERT7_GPU="${EXPERT7_GPU:-7}"   # Qwen2.5-Math-72B    → answer-math-1

RETRIEVAL_MODEL_PATH="${RETRIEVAL_MODEL_PATH:-${SCRIPT_DIR}/retrieval_general_thought.py}"
EXPERT1_MODEL="${EXPERT1_MODEL:-/data/models/qwen3_32b_fp8}"
EXPERT2_MODEL="${EXPERT2_MODEL:-/data/models/qwen2.5_32b_coder}"
EXPERT3_MODEL="${EXPERT3_MODEL:-/data/models/qwen2.5_math_7b}"
EXPERT4_MODEL="${EXPERT4_MODEL:-/data/models/qwen3_14b}"
EXPERT5_MODEL="${EXPERT5_MODEL:-/data/models/qwen_32b_distill}"
EXPERT6_MODEL="${EXPERT6_MODEL:-/data/models/qwen3_30b_a3b}"
EXPERT7_MODEL="${EXPERT7_MODEL:-/data/models/qwen_72b_math}"

# eval_orchestra.py 会在 OUTPUT_DIR 下自动创建 {benchmark}_{timestamp}/ 子目录
OUTPUT_DIR="${OUTPUT_DIR:-/data/eval_results}"

# ──────────────────────────── 工具函数 ────────────────────────────────────── #
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
        if (( elapsed % 60 == 0 )); then
            log "  还在等待 $name... (${elapsed}s)"
        fi
    done
    log "ERROR: $name 在 ${timeout}s 内未能启动，日志: $LOG_DIR"
    return 1
}

port_alive() {
    bash -c "echo >/dev/tcp/127.0.0.1/$1" 2>/dev/null
}

# 记录本脚本启动的 PID，用于最终清理
STARTED_PIDS=()

cleanup() {
    log "========= 清理本次启动的服务 ========="
    for pid in "${STARTED_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            log "  停止 PID=$pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    # 清理 tau2 临时文件
    rm -rf /tmp/tau2_eval_* 2>/dev/null || true
    log "清理完成"
}
trap cleanup EXIT

# ──────────────────────────── Step -1: 终止残留的评测主进程 ────────────────── #
if [ "$KILL_STALE_EVAL" = "1" ]; then
    log "============================================"
    log "Step -1: 终止残留的评测任务 (eval_orchestra.py)"
    log "============================================"
    _EV_PY="${SCRIPT_DIR}/eval_orchestra.py"
    if pgrep -f "$_EV_PY" >/dev/null 2>&1; then
        log "发现残留进程，发送 SIGKILL ..."
        pkill -9 -f "$_EV_PY" 2>/dev/null || true
        sleep 2
        log "残留 eval_orchestra.py 已清理"
    else
        log "无残留的 eval_orchestra.py"
    fi
fi

# ──────────────────────────── Step 0: 杀死所有旧服务 ─────────────────────── #
if [ "$SKIP_SERVICES" = "1" ]; then
    log "SKIP_SERVICES=1，跳过所有服务启动，假定服务已就绪"
elif [ "$SKIP_EXPERT_SERVICES" = "1" ]; then
    log "SKIP_EXPERT_SERVICES=1，跳过 expert/检索服务启动"
else
    log "============================================"
    log "Step 0: 清理所有旧进程"
    log "============================================"
    pkill -9 -f sglang 2>/dev/null || true
    sleep 2
    ray stop --force 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "retrieval_general" 2>/dev/null || true
    sleep 3
    rm -rf /tmp/tau2_eval_* /tmp/tau2_orch_* /tmp/tau2_transfer_* /tmp/tau2_output* 2>/dev/null || true
    log "旧进程已清理"

    # ──────────────────────────── Step 1: 启动 Expert 服务 ────────────────── #
    log "============================================"
    log "Step 1: 启动 Expert 服务 & 检索服务"
    log "============================================"

    # ── 检索服务（GPU $RETRIEVAL_GPU, port 8000）
    log "启动检索服务 (GPU $RETRIEVAL_GPU, port 8000)..."
    CUDA_VISIBLE_DEVICES=$RETRIEVAL_GPU \
    REPO_PATH=${SCRIPT_DIR} \
    INDEX_DIR=/data/dataset/index \
    conda run -n orche --no-capture-output \
        python "$RETRIEVAL_MODEL_PATH" --port 8000 \
        > "$LOG_DIR/retrieval.log" 2>&1 &
    pid=$!
    STARTED_PIDS+=("$pid")
    log "检索服务 PID=$pid，日志: $LOG_DIR/retrieval.log"

    # ── Qwen3-32B-FP8（GPU $EXPERT1_GPU, port 30001）
    log "启动 Qwen3-32B-FP8 (GPU $EXPERT1_GPU, port 30001)..."
    CUDA_VISIBLE_DEVICES=$EXPERT1_GPU conda run -n sglang --no-capture-output \
        bash -c "
            export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
            python -m sglang.launch_server \
                --model-path '$EXPERT1_MODEL' \
                --port 30001 \
                --context-length 163840 \
                --tp 1
        " > "$LOG_DIR/sglang_30001.log" 2>&1 &
    pid=$!
    STARTED_PIDS+=("$pid")
    log "Qwen3-32B-FP8 PID=$pid，日志: $LOG_DIR/sglang_30001.log"

    # ── Qwen2.5-Coder-32B（GPU $EXPERT2_GPU, port 30002）
    log "启动 Qwen2.5-Coder-32B (GPU $EXPERT2_GPU, port 30002)..."
    CUDA_VISIBLE_DEVICES=$EXPERT2_GPU conda run -n sglang --no-capture-output \
        bash -c "
            export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
            python -m sglang.launch_server \
                --model-path '$EXPERT2_MODEL' \
                --port 30002 \
                --context-length 163840 \
                --tp 1
        " > "$LOG_DIR/sglang_30002.log" 2>&1 &
    pid=$!
    STARTED_PIDS+=("$pid")
    log "Qwen2.5-Coder-32B PID=$pid，日志: $LOG_DIR/sglang_30002.log"

    # ── Qwen2.5-Math-7B（GPU $EXPERT3_GPU, port 30003）
    log "启动 Qwen2.5-Math-7B (GPU $EXPERT3_GPU, port 30003)..."
    CUDA_VISIBLE_DEVICES=$EXPERT3_GPU conda run -n sglang --no-capture-output \
        bash -c "
            export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
            python -m sglang.launch_server \
                --model-path '$EXPERT3_MODEL' \
                --port 30003 \
                --context-length 163840 \
                --tp 1
        " > "$LOG_DIR/sglang_30003.log" 2>&1 &
    pid=$!
    STARTED_PIDS+=("$pid")
    log "Qwen2.5-Math-7B PID=$pid，日志: $LOG_DIR/sglang_30003.log"

    # ── Qwen3-14B（GPU $EXPERT4_GPU, port 30004）
    log "启动 Qwen3-14B (GPU $EXPERT4_GPU, port 30004)..."
    CUDA_VISIBLE_DEVICES=$EXPERT4_GPU conda run -n sglang --no-capture-output \
        bash -c "
            export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
            python -m sglang.launch_server \
                --model-path '$EXPERT4_MODEL' \
                --port 30004 \
                --context-length 163840 \
                --tp 1
        " > "$LOG_DIR/sglang_30004.log" 2>&1 &
    pid=$!
    STARTED_PIDS+=("$pid")
    log "Qwen3-14B PID=$pid，日志: $LOG_DIR/sglang_30004.log"

    # ── DeepSeek-R1-Distill-Qwen-32B（GPU $EXPERT5_GPU, port 30005）
    log "启动 DeepSeek-R1-Distill-32B (GPU $EXPERT5_GPU, port 30005)..."
    CUDA_VISIBLE_DEVICES=$EXPERT5_GPU conda run -n sglang --no-capture-output \
        bash -c "
            export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
            python -m sglang.launch_server \
                --model-path '$EXPERT5_MODEL' \
                --port 30005 \
                --context-length 163840 \
                --tp 1
        " > "$LOG_DIR/sglang_30005.log" 2>&1 &
    pid=$!
    STARTED_PIDS+=("$pid")
    log "DeepSeek-R1-Distill-32B PID=$pid，日志: $LOG_DIR/sglang_30005.log"

    # ── Qwen3-30B-A3B（GPU $EXPERT6_GPU, port 30006）
    log "启动 Qwen3-30B-A3B (GPU $EXPERT6_GPU, port 30006)..."
    CUDA_VISIBLE_DEVICES=$EXPERT6_GPU conda run -n sglang --no-capture-output \
        bash -c "
            export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
            python -m sglang.launch_server \
                --model-path '$EXPERT6_MODEL' \
                --port 30006 \
                --context-length 163840 \
                --tp 1
        " > "$LOG_DIR/sglang_30006.log" 2>&1 &
    pid=$!
    STARTED_PIDS+=("$pid")
    log "Qwen3-30B-A3B PID=$pid，日志: $LOG_DIR/sglang_30006.log"

    # ── Qwen2.5-Math-72B（GPU $EXPERT7_GPU, port 30007）
    log "启动 Qwen2.5-Math-72B (GPU $EXPERT7_GPU, port 30007)..."
    CUDA_VISIBLE_DEVICES=$EXPERT7_GPU conda run -n sglang --no-capture-output \
        bash -c "
            export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
            python -m sglang.launch_server \
                --model-path '$EXPERT7_MODEL' \
                --port 30007 \
                --context-length 163840 \
                --tp 1
        " > "$LOG_DIR/sglang_30007.log" 2>&1 &
    pid=$!
    STARTED_PIDS+=("$pid")
    log "Qwen2.5-Math-72B PID=$pid，日志: $LOG_DIR/sglang_30007.log"
fi

# ──────────────────────────── Step 2: 启动 Orchestrator SGLang ──────────────── #
if [ "$SKIP_SERVICES" != "1" ] && [ -n "$ORCH_CKPT" ]; then
    log "============================================"
    log "Step 2: 启动 Orchestrator SGLang"
    log "============================================"

    log "启动 Orchestrator (GPU $ORCH_GPU, port $ORCH_PORT)..."
    log "  ckpt: $ORCH_CKPT"

    # 构建 GPU 列表（支持 tp > 1）
    if [ "$ORCH_TP" -gt 1 ]; then
        gpu_list=""
        for i in $(seq 0 $((ORCH_TP - 1))); do
            g=$((ORCH_GPU + i))
            gpu_list="${gpu_list:+${gpu_list},}${g}"
        done
    else
        gpu_list="$ORCH_GPU"
    fi

    CUDA_VISIBLE_DEVICES=$gpu_list conda run -n sglang --no-capture-output \
        bash -c "
            export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
            python -m sglang.launch_server \
                --model-path '$ORCH_CKPT' \
                --port '$ORCH_PORT' \
                --context-length '$ORCH_CTX' \
                --tp '$ORCH_TP'
        " > "$LOG_DIR/sglang_orch.log" 2>&1 &
    pid=$!
    STARTED_PIDS+=("$pid")
    log "Orchestrator PID=$pid，日志: $LOG_DIR/sglang_orch.log"

    ORCH_URL="http://127.0.0.1:${ORCH_PORT}/v1"
fi

# ──────────────────────────── Step 3: 等待所有服务就绪 ──────────────────────── #
if [ "$SKIP_SERVICES" != "1" ]; then
    log "============================================"
    log "Step 3: 等待所有服务就绪（模型加载需要几分钟）"
    log "============================================"

    wait_port "检索服务"     127.0.0.1 8000  600
    wait_port "SGLang-30001(Qwen3-32B)"      127.0.0.1 30001 600
    wait_port "SGLang-30002(Coder-32B)"      127.0.0.1 30002 600
    wait_port "SGLang-30003(Math-7B)"        127.0.0.1 30003 600
    wait_port "SGLang-30004(Qwen3-14B)"      127.0.0.1 30004 600
    wait_port "SGLang-30005(DeepSeek-R1-32B)" 127.0.0.1 30005 600
    wait_port "SGLang-30006(Qwen3-30B-A3B)"  127.0.0.1 30006 600
    wait_port "SGLang-30007(Math-72B)"       127.0.0.1 30007 600

    if [ -n "$ORCH_CKPT" ]; then
        wait_port "Orchestrator" 127.0.0.1 "$ORCH_PORT" 600
    fi

    log "所有服务已就绪！"
fi

# ──────────────────────────── Step 4: 运行评测 ──────────────────────────────── #
log "============================================"
log "Step 4: 开始评测 benchmark=$BENCHMARK"
log "============================================"

# --model-name 必须指定（用于加载 tokenizer），默认使用 ORCH_CKPT
if [ -z "$ORCH_MODEL" ]; then
    if [ -n "$ORCH_CKPT" ]; then
        ORCH_MODEL="$ORCH_CKPT"
    else
        log "ERROR: ORCH_MODEL 或 ORCH_CKPT 必须设置（eval_orchestra.py 需要加载 tokenizer）"
        exit 1
    fi
fi

mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="${SCRIPT_DIR}:/data/slime-agentic/agentic:/data/slime-agentic:${PYTHONPATH:-}"

log_python

CLEAN_PREVIOUS="${CLEAN_PREVIOUS:-1}"      # 默认清理旧评测结果

CMD=(
    $EVAL_PYTHON_CMD "${SCRIPT_DIR}/eval_orchestra.py"
    --benchmark      "$BENCHMARK"
    --model-url      "$ORCH_URL"
    --model-name     "$ORCH_MODEL"
    --output-dir     "$OUTPUT_DIR"
    --context-length "$ORCH_CTX"
    --concurrency    "$CONCURRENCY"
    --max-turns      "$MAX_TURNS"
    --temperature    "$TEMPERATURE"
    --user-llm       "$USER_LLM"
)

[ "$CLEAN_PREVIOUS" = "1" ] && CMD+=(--clean-previous)

[ -n "$EVAL_DATA"     ] && CMD+=(--eval-data     "$EVAL_DATA")
[ -n "$MAX_EXAMPLES"  ] && CMD+=(--max-examples  "$MAX_EXAMPLES")
[ -n "$DOMAINS"       ] && CMD+=(--domains       $DOMAINS)

log "模型路径(tokenizer): $ORCH_MODEL"
log "输出目录: $OUTPUT_DIR/${BENCHMARK}_<timestamp>/"
log "命令: ${CMD[*]}"
echo ""

"${CMD[@]}" 2>&1
EXIT_CODE=$?

# ──────────────────────────── 结果汇总 ──────────────────────────────────────── #
# 找到 Python 脚本创建的实际带时间戳子目录
ACTUAL_OUTDIR=$(ls -dt "${OUTPUT_DIR}/${BENCHMARK}_"* 2>/dev/null | head -1)
if [ -z "$ACTUAL_OUTDIR" ]; then
    ACTUAL_OUTDIR="$OUTPUT_DIR"
fi

echo ""
log "============================================"
log "评测完成"
log "  benchmark  : $BENCHMARK"
log "  输出目录   : $ACTUAL_OUTDIR"
log "  结果摘要   : $ACTUAL_OUTDIR/_summary.json"
log "  完整日志   : $OUTPUT_DIR/_eval.log"
log "  服务日志   : $LOG_DIR/"
log "============================================"

exit $EXIT_CODE
