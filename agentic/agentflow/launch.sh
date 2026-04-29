#!/bin/bash
# AgentFlow training launcher.
# Local GPUs are reserved for trainer + small Planner rollout. Support roles
# are served through OpenAI-compatible APIs configured by AGENTFLOW_* variables.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
LOG_DIR=${LOG_DIR:-"/tmp/agentflow_logs"}
mkdir -p "$LOG_DIR"

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
ulimit -n 65536 2>/dev/null || true

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

if [ -z "${AGENTFLOW_API_MODEL:-}" ]; then
    export AGENTFLOW_API_MODEL=${EXECUTOR_MODEL:-"gpt-4o-mini"}
fi
export AGENTFLOW_API_BASE=${AGENTFLOW_API_BASE:-"${EXECUTOR_API_BASE:-https://api.openai.com/v1}"}
export AGENTFLOW_API_KEY=${AGENTFLOW_API_KEY:-"${EXECUTOR_API_KEY:-${OPENAI_API_KEY:-}}"}

log "清理旧 Ray 和旧 Planner rollout 进程..."
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 -f 'sglang\.launch_server' 2>/dev/null || true
sleep 2

log "启动训练脚本..."
cd "$REPO_ROOT"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} SKIP_PROCESS_KILL=1 \
    bash "${SCRIPT_DIR}/agentflow_qwen25_7b_rl_v2.sh" \
    > "$LOG_DIR/train.log" 2>&1 &
TRAIN_PID=$!
log "训练进程 PID=$TRAIN_PID，日志: $LOG_DIR/train.log"

log "等待 Ray dashboard 就绪..."
wait_port "ray-dashboard" 127.0.0.1 8265 180
log "Ray 已就绪。非 Planner 角色将使用 API：${AGENTFLOW_API_BASE} (${AGENTFLOW_API_MODEL})"
log "训练日志: tail -f $LOG_DIR/train.log"

TRAIN_EXIT=0
wait $TRAIN_PID || TRAIN_EXIT=$?
if [ $TRAIN_EXIT -eq 0 ]; then
    log "训练完成。"
else
    log "训练退出，exit code=$TRAIN_EXIT，请查看日志: $LOG_DIR/train.log"
fi
exit $TRAIN_EXIT
