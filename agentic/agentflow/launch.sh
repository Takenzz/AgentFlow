#!/bin/bash
# AgentFlow 一键启动脚本
# 自动启动训练及所有依赖的 SGLang 推理服务

set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# 提高文件描述符上限，避免 "Too many open files"
ulimit -n 65536 2>/dev/null || true

LOG_DIR="/tmp/agentflow_logs"
mkdir -p "$LOG_DIR"

# 模型路径（本地离线目录）
MODEL_BASE="/data/models/qwen25_7b"
MODEL_CODER="/data/models/qwen2.5_7b_codeer"

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

# Step 1: 同步清理旧进程，避免竞态
log "清理旧进程..."
pkill -9 sglang 2>/dev/null || true
sleep 1
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 -f 'sglang\.launch_server' 2>/dev/null || true
sleep 2
pkill -9 ray 2>/dev/null || true
pkill -9 -f 'sglang\.launch_server' 2>/dev/null || true

log "等待旧 ray 端口关闭..."
for i in $(seq 1 30); do
    if ! bash -c "echo >/dev/tcp/127.0.0.1/8265" 2>/dev/null; then
        break
    fi
    sleep 1
done
log "旧进程已清理。"

# Step 2: 启动训练（跳过训练脚本中的 kill）
log "启动训练脚本..."
cd /data/slime-agentic
CUDA_VISIBLE_DEVICES=0,1,2,3 SKIP_PROCESS_KILL=1 \
    bash "${SCRIPT_DIR}/agentflow_qwen25_7b_rl_v2.sh" \
    > "$LOG_DIR/train.log" 2>&1 &
TRAIN_PID=$!
log "训练进程 PID=$TRAIN_PID，日志: $LOG_DIR/train.log"

# Step 3: 等待 ray
log "等待 ray dashboard 就绪..."
wait_port "ray-dashboard" 127.0.0.1 8265 120
log "ray 已就绪，开始启动 SGLang 服务..."

# Step 4: 启动 SGLang 服务
# 用 setsid 创建独立会话，防止 launch.sh 退出/Ctrl+C 时信号传播杀掉 SGLang
setsid bash -c "
    export CUDA_VISIBLE_DEVICES=6,7
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    conda run -n sglang --no-capture-output \
        python3 -m sglang.launch_server \
            --model-path ${MODEL_BASE} \
            --port 30000 \
            --context-length 131072 \
            --tp 2
" > "$LOG_DIR/sglang_30000.log" 2>&1 &
SGLANG_30000_PID=$!
log "Qwen2.5-7B-Instruct  PID=$SGLANG_30000_PID (GPU 6,7, port 30000)，日志: $LOG_DIR/sglang_30000.log"

setsid bash -c "
    export CUDA_VISIBLE_DEVICES=4,5
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    conda run -n sglang --no-capture-output \
        python3 -m sglang.launch_server \
            --model-path ${MODEL_CODER} \
            --port 30001 \
            --context-length 131072 \
            --tp 2
" > "$LOG_DIR/sglang_30001.log" 2>&1 &
SGLANG_30001_PID=$!
log "Qwen2.5-Coder-7B     PID=$SGLANG_30001_PID (GPU 4,5, port 30001)，日志: $LOG_DIR/sglang_30001.log"

# Step 5: 等待 SGLang 就绪
log "等待 SGLang 服务启动（模型加载可能需要几分钟）..."
wait_port "SGLang-30000" 127.0.0.1 30000 600
wait_port "SGLang-30001" 127.0.0.1 30001 600
log "所有 SGLang 服务已就绪，训练正在进行中。"
log "训练日志: tail -f $LOG_DIR/train.log"

# Step 6: 等待训练进程结束（不让 set -e 因训练退出码非零而提前退出）
TRAIN_EXIT=0
wait $TRAIN_PID || TRAIN_EXIT=$?
if [ $TRAIN_EXIT -eq 0 ]; then
    log "训练完成。"
else
    log "训练退出，exit code=$TRAIN_EXIT，请查看日志: $LOG_DIR/train.log"
fi
