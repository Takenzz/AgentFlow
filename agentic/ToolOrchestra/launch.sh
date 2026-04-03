#!/bin/bash
# ToolOrchestra 一键启动脚本（B300 × 8 卡）
#
# 使用方式：
#   bash /data/slime-agentic/agentic/ToolOrchestra/launch.sh
#
# GPU 分配：
#   GPU 0       : 检索服务 (qwen3_8b_emb, port 8000)
#                 + SGLang Qwen3-32B-FP8               port=30001 (conda: sglang)
#                 + SGLang Qwen2.5-Math-7B             port=30003 (conda: sglang)
#   GPU 1       : SGLang DeepSeek-R1-Distill-Qwen-32B  port=30005 (conda: sglang, mem=0.45)
#                 + SGLang Qwen3-30B-A3B               port=30006 (conda: sglang, mem=0.45)
#   GPU 2       : SGLang Qwen2.5-Coder-32B             port=30002 (conda: sglang, mem=0.45)
#                 + SGLang Qwen3-14B                   port=30007 (conda: sglang, mem=0.45)
#   GPU 3       : SGLang Qwen2.5-Math-72B              port=30004 (conda: sglang)
#   GPU 4,5,6,7 : 训练 Qwen3-8B (TP=2, DP=2)
#
# 上下文长度（B300 优化）：
#   Expert 模型: 163840 (160K) — 原生 128K 的 1.25x, YaRN 扩展质量可靠
#   训练模型 8B: 131072 (128K) — 8B 原生 40K, 已 3.2x 扩展
#
# 显存估算（B300, 288GB/卡）：
#   GPU 0 : emb ~16G + fp8-32B ~33G + 7B ~15G + KV caches ≈ 230G
#   GPU 1 : distill-32B ~62G + 30B-A3B ~60G + KV caches ≈ 260G (各占约 130G)
#   GPU 2 : coder-32B  ~62G + 14B ~28G + KV caches ≈ 260G (各占约 130G)
#   GPU 3 : 72B ~136G + KV cache ≈ 260G

set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

LOG_DIR="/tmp/orchestra_logs"
mkdir -p "$LOG_DIR"

# ── 工具函数 ─────────────────────────────────────────────────────────────────
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

# ── Step 1: 清理旧进程 ───────────────────────────────────────────────────────
# 检索服务（port 8000）加载耗时长（8B emb 模型 + 大型 FAISS 索引 + 语料），
# 若已就绪则跳过重启，显著节省等待时间。
# 强制重启可在环境变量中设置：FORCE_RESTART_RETRIEVAL=1
log "清理旧进程..."
# 精确杀 SGLang 服务进程（匹配命令行），不误杀检索服务（retrieval_general_thought.py）
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
pkill -9 sglang 2>/dev/null || true
sleep 2
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 3
# 等待 SGLang 端口完全释放，避免新进程绑端口失败
for _port in 30001 30002 30003 30004 30005 30006 30007; do
    _waited=0
    while bash -c "echo >/dev/tcp/127.0.0.1/${_port}" 2>/dev/null; do
        sleep 1; _waited=$((_waited+1))
        if [ $_waited -ge 15 ]; then
            log "警告: port ${_port} 仍被占用，强制杀占用进程..."
            fuser -k "${_port}/tcp" 2>/dev/null || true
            sleep 2; break
        fi
    done
done
rm -rf /tmp/tau2_orch_* /tmp/tau2_transfer_* /tmp/tau2_output* 2>/dev/null
rm -rf /data/rollout_logs/train /data/rollout_logs/eval 2>/dev/null
log "清理完成"

# ── Step 2: 启动检索服务（GPU 0，conda: orche）──────────────────────────────
_retrieval_alive() {
    bash -c "echo >/dev/tcp/127.0.0.1/8000" 2>/dev/null
}

if [ "${FORCE_RESTART_RETRIEVAL:-0}" = "1" ]; then
    log "FORCE_RESTART_RETRIEVAL=1，强制重启检索服务..."
    pkill -f "retrieval_general_thought.py" 2>/dev/null || true
    sleep 2
fi

if _retrieval_alive; then
    log "检索服务已在运行 (port 8000)，跳过重启。如需强制重启请设置 FORCE_RESTART_RETRIEVAL=1"
    RETRIEVAL_PID=""
else
    log "启动检索服务 (GPU 0, port 8000)..."
    CUDA_VISIBLE_DEVICES=0 \
    REPO_PATH=${SCRIPT_DIR} \
    INDEX_DIR=/data/dataset/index \
    conda run -n orche --no-capture-output \
        python ${SCRIPT_DIR}/retrieval_general_thought.py --port 8000 \
        > "$LOG_DIR/retrieval.log" 2>&1 &
    RETRIEVAL_PID=$!
    log "检索服务 PID=$RETRIEVAL_PID，日志: $LOG_DIR/retrieval.log"
fi

# ── Step 3: 启动独占 GPU 的 SGLang 服务（GPUs 1,2,3 可并行启动）─────────────
log "启动 SGLang 服务（GPUs 1,2,3）..."

# GPU 1 — DeepSeek-R1-Distill-Qwen-32B，port 30005（与 Qwen3-30B-A3B 共享，各占约 0.45）
CUDA_VISIBLE_DEVICES=1 conda run -n sglang --no-capture-output \
    bash -c "
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python -m sglang.launch_server \
            --model-path /data/models/qwen_32b_distill \
            --port 30005 \
            --context-length 163840 \
            --mem-fraction-static 0.45 \
            --tp 1
    " > "$LOG_DIR/sglang_30005.log" 2>&1 &
log "DeepSeek-R1-Distill-Qwen-32B PID=$! (GPU 1, port 30005)，日志: $LOG_DIR/sglang_30005.log"

# GPU 2 — Qwen2.5-Coder-32B，port 30002（与 Qwen3-14B 共享，各占约 0.45）
CUDA_VISIBLE_DEVICES=2 conda run -n sglang --no-capture-output \
    bash -c "
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python -m sglang.launch_server \
            --model-path /data/models/qwen2.5_32b_coder \
            --port 30002 \
            --context-length 163840 \
            --mem-fraction-static 0.45 \
            --tp 1
    " > "$LOG_DIR/sglang_30002.log" 2>&1 &
log "Qwen2.5-Coder-32B PID=$! (GPU 2, port 30002)，日志: $LOG_DIR/sglang_30002.log"

# GPU 3 — Qwen2.5-Math-72B，port 30004
CUDA_VISIBLE_DEVICES=3 conda run -n sglang --no-capture-output \
    bash -c "
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python -m sglang.launch_server \
            --model-path /data/models/qwen_72b_math \
            --port 30004 \
            --context-length 163840 \
            --mem-fraction-static 0.88 \
            --tp 1
    " > "$LOG_DIR/sglang_30004.log" 2>&1 &
log "Qwen2.5-Math-72B PID=$! (GPU 3, port 30004)，日志: $LOG_DIR/sglang_30004.log"

# ── Step 4: 启动共享 GPU 0 的 SGLang 服务（需依次启动，避免显存分配冲突）────
# GPU 0 已被检索服务占用 ~16G，先等检索加载完再启动 sglang
log "等待检索服务加载完成后再启动 GPU 0 的 SGLang 服务..."
wait_port "检索服务" 127.0.0.1 8000 600

# GPU 0 — Qwen3-32B-FP8，port 30001（与检索服务共享，限制 KV cache 比例）
log "启动 Qwen3-32B-FP8 (GPU 0, port 30001)..."
CUDA_VISIBLE_DEVICES=0 conda run -n sglang --no-capture-output \
    bash -c "
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python -m sglang.launch_server \
            --model-path /data/models/qwen3_32b_fp8 \
            --port 30001 \
            --context-length 163840 \
            --mem-fraction-static 0.55 \
            --tp 1
    " > "$LOG_DIR/sglang_30001.log" 2>&1 &
log "Qwen3-32B-FP8 PID=$! (GPU 0, port 30001)，日志: $LOG_DIR/sglang_30001.log"

# 等 fp8 加载完成并分配好 KV cache 后再启动 7B，防止两个 sglang 同时争抢显存
wait_port "SGLang-30001" 127.0.0.1 30001 600

# GPU 0 — Qwen2.5-Math-7B，port 30003（GPU 0 第三个服务，用剩余显存）
log "启动 Qwen2.5-Math-7B (GPU 0, port 30003)..."
CUDA_VISIBLE_DEVICES=0 conda run -n sglang --no-capture-output \
    bash -c "
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python -m sglang.launch_server \
            --model-path /data/models/qwen2.5_math_7b \
            --port 30003 \
            --context-length 163840 \
            --mem-fraction-static 0.40 \
            --tp 1
    " > "$LOG_DIR/sglang_30003.log" 2>&1 &
log "Qwen2.5-Math-7B PID=$! (GPU 0, port 30003)，日志: $LOG_DIR/sglang_30003.log"

# ── Step 4.5: 等待 distill/coder 就绪后再启动同卡第二个模型 ─────────────────
# 必须等第一个模型完成显存分配后再启动同卡第二个，避免 OOM 冲突
wait_port "SGLang-30005" 127.0.0.1 30005 600

# GPU 1 — Qwen3-30B-A3B，port 30006（distill 就绪后启动，共享 GPU 1）
log "启动 Qwen3-30B-A3B (GPU 1, port 30006)..."
CUDA_VISIBLE_DEVICES=1 conda run -n sglang --no-capture-output \
    bash -c "
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python -m sglang.launch_server \
            --model-path /data/models/qwen3_30b_a3b \
            --port 30006 \
            --context-length 163840 \
            --mem-fraction-static 0.45 \
            --tp 1
    " > "$LOG_DIR/sglang_30006.log" 2>&1 &
log "Qwen3-30B-A3B PID=$! (GPU 1, port 30006)，日志: $LOG_DIR/sglang_30006.log"

wait_port "SGLang-30002" 127.0.0.1 30002 600

# GPU 2 — Qwen3-14B，port 30007（coder 就绪后启动，共享 GPU 2）
log "启动 Qwen3-14B (GPU 2, port 30007)..."
CUDA_VISIBLE_DEVICES=2 conda run -n sglang --no-capture-output \
    bash -c "
        export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python -m sglang.launch_server \
            --model-path /data/models/qwen3_14b \
            --port 30007 \
            --context-length 163840 \
            --mem-fraction-static 0.45 \
            --tp 1
    " > "$LOG_DIR/sglang_30007.log" 2>&1 &
log "Qwen3-14B PID=$! (GPU 2, port 30007)，日志: $LOG_DIR/sglang_30007.log"

# ── Step 5: 等待所有服务就绪 ─────────────────────────────────────────────────
log "等待所有 SGLang 服务启动..."
wait_port "SGLang-30003" 127.0.0.1 30003 600
wait_port "SGLang-30004" 127.0.0.1 30004 600
wait_port "SGLang-30006" 127.0.0.1 30006 600
wait_port "SGLang-30007" 127.0.0.1 30007 600
log "所有服务已就绪！"

# ── Step 6: 启动训练（GPUs 4,5,6,7）─────────────────────────────────────────
log "开始训练（GPUs 4,5,6,7）..."
cd /data/slime-agentic
CUDA_VISIBLE_DEVICES=4,5,6,7 SKIP_PROCESS_KILL=1 \
    bash "${SCRIPT_DIR}/train_orchestra.sh"
