#!/bin/bash
# ToolOrchestra 全 checkpoint 评测扫描脚本
# =========================================
# 对所有已转换的 HF checkpoint 依次评测，每个 checkpoint 跑 N 次，
# 记录最大值和平均值，输出汇总表格。
#
# Expert 服务（ports 30001-30007）和检索服务（port 8000）只启动一次，
# 每个 checkpoint 只重启 Orchestrator SGLang（port 30000）。
#
# 使用方式：
#   # 全量扫描（默认 tau2 benchmark，3次/checkpoint）
#   bash sweep_eval.sh
#
#   # 指定 benchmark
#   BENCHMARK=frames bash sweep_eval.sh
#
#   # 只评测某几个 checkpoint（iter 名，空格分隔）
#   ONLY_ITERS="iter_0000099 iter_0000129" bash sweep_eval.sh
#
#   # 快速测试（每次只跑5条样本）
#   MAX_EXAMPLES=5 N_RUNS=1 bash sweep_eval.sh
#
#   # Expert 服务已在运行，跳过启动
#   SKIP_EXPERT_SERVICES=1 bash sweep_eval.sh
#
#   # 与其它评测并行、勿杀残留 eval_orchestra.py 时
#   KILL_STALE_EVAL=0 bash sweep_eval.sh

set -uo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ──────────────────────────── 可配置参数 ────────────────────────────────────── #
BENCHMARK="${BENCHMARK:-tau2}"
N_RUNS="${N_RUNS:-3}"                    # 每个 checkpoint 跑几次

HF_CKPT_BASE="${HF_CKPT_BASE:-/data/checkpoints/orchestra_qwen3_8b_rl_hf}"
SWEEP_OUTPUT_BASE="${SWEEP_OUTPUT_BASE:-/data/eval_results/sweep_${BENCHMARK}}"

ORCH_PORT="${ORCH_PORT:-30000}"
ORCH_GPU="${ORCH_GPU:-0}"
ORCH_TP="${ORCH_TP:-1}"
ORCH_CTX="${ORCH_CTX:-163840}"

CONCURRENCY="${CONCURRENCY:-4}"
MAX_TURNS="${MAX_TURNS:-12}"
TEMPERATURE="${TEMPERATURE:-0.7}"
USER_LLM="${USER_LLM:-qwen-turbo-latest}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
DOMAINS="${DOMAINS:-}"

SKIP_EXPERT_SERVICES="${SKIP_EXPERT_SERVICES:-0}"   # =1 跳过 expert/检索启动（服务已在运行）
KILL_STALE_EVAL="${KILL_STALE_EVAL:-1}"             # =0 保留其它正在跑的 eval_orchestra.py

# GPU / 模型路径（与 eval_orchestra.sh 保持一致）
RETRIEVAL_GPU="${RETRIEVAL_GPU:-0}"
EXPERT1_GPU="${EXPERT1_GPU:-1}"
EXPERT2_GPU="${EXPERT2_GPU:-2}"
EXPERT3_GPU="${EXPERT3_GPU:-3}"
EXPERT4_GPU="${EXPERT4_GPU:-4}"
EXPERT5_GPU="${EXPERT5_GPU:-5}"
EXPERT6_GPU="${EXPERT6_GPU:-6}"
EXPERT7_GPU="${EXPERT7_GPU:-7}"

RETRIEVAL_MODEL_PATH="${RETRIEVAL_MODEL_PATH:-${SCRIPT_DIR}/training/retrieval_general_thought.py}"
EXPERT1_MODEL="${EXPERT1_MODEL:-/data/models/qwen3_32b_fp8}"
EXPERT2_MODEL="${EXPERT2_MODEL:-/data/models/qwen2.5_32b_coder}"
EXPERT3_MODEL="${EXPERT3_MODEL:-/data/models/qwen2.5_math_7b}"
EXPERT4_MODEL="${EXPERT4_MODEL:-/data/models/qwen3_14b}"
EXPERT5_MODEL="${EXPERT5_MODEL:-/data/models/qwen_32b_distill}"
EXPERT6_MODEL="${EXPERT6_MODEL:-/data/models/qwen3_30b_a3b}"
EXPERT7_MODEL="${EXPERT7_MODEL:-/data/models/qwen_72b_math}"

LOG_DIR="${LOG_DIR:-/tmp/sweep_eval_logs}"
mkdir -p "$LOG_DIR" "$SWEEP_OUTPUT_BASE"

# eval_orchestra.py 需要 sglang 环境（transformers, httpx, openai）
EVAL_CONDA_ENV="${EVAL_CONDA_ENV:-sglang}"
EVAL_PYTHON_CMD="conda run -n ${EVAL_CONDA_ENV} --no-capture-output python"

# 简单 JSON 解析用系统 Python 即可
if [ -x /usr/bin/python3 ]; then
    SYS_PYTHON=/usr/bin/python3
else
    SYS_PYTHON=python3
fi

# ──────────────────────────── 工具函数 ──────────────────────────────────────── #
log() { echo "[$(date '+%H:%M:%S')] $*"; }

if [ "$KILL_STALE_EVAL" = "1" ]; then
    log "============================================"
    log "终止残留的评测任务 (eval_orchestra.py)"
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

kill_orchestrator() {
    log "停止 Orchestrator (port $ORCH_PORT)..."
    pkill -f "sglang.*port.*$ORCH_PORT" 2>/dev/null || true
    pkill -f "sglang.*$ORCH_PORT" 2>/dev/null || true
    # 等待端口释放
    local elapsed=0
    while port_alive "$ORCH_PORT" && [ $elapsed -lt 60 ]; do
        sleep 2
        elapsed=$((elapsed + 2))
    done
    if port_alive "$ORCH_PORT"; then
        log "WARN: 端口 $ORCH_PORT 60s 后仍未释放，强制 kill"
        fuser -k "${ORCH_PORT}/tcp" 2>/dev/null || true
        sleep 3
    fi
    log "Orchestrator 已停止"
}

start_orchestrator() {
    local ckpt=$1
    log "启动 Orchestrator (GPU $ORCH_GPU, port $ORCH_PORT)..."
    log "  ckpt: $ckpt"

    if [ "$ORCH_TP" -gt 1 ]; then
        local gpu_list=""
        for i in $(seq 0 $((ORCH_TP - 1))); do
            local g=$((ORCH_GPU + i))
            gpu_list="${gpu_list:+${gpu_list},}${g}"
        done
    else
        local gpu_list="$ORCH_GPU"
    fi

    CUDA_VISIBLE_DEVICES=$gpu_list conda run -n sglang --no-capture-output \
        bash -c "
            export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
            python -m sglang.launch_server \
                --model-path '$ckpt' \
                --port '$ORCH_PORT' \
                --context-length '$ORCH_CTX' \
                --tp '$ORCH_TP'
        " > "$LOG_DIR/sglang_orch.log" 2>&1 &
    ORCH_PID=$!
    log "Orchestrator PID=$ORCH_PID"
}

EXPERT_PIDS=()

start_expert_services() {
    log "============================================"
    log "启动 Expert 服务 & 检索服务"
    log "============================================"

    pkill -9 -f sglang 2>/dev/null || true
    sleep 2
    pkill -9 -f "retrieval_general" 2>/dev/null || true
    sleep 2
    rm -rf /tmp/tau2_eval_* /tmp/tau2_orch_* /tmp/tau2_transfer_* /tmp/tau2_output* 2>/dev/null || true

    # 检索服务
    log "启动检索服务 (GPU $RETRIEVAL_GPU, port 8000)..."
    CUDA_VISIBLE_DEVICES=$RETRIEVAL_GPU \
    REPO_PATH=${SCRIPT_DIR} \
    INDEX_DIR=/data/dataset/index \
    conda run -n orche --no-capture-output \
        python "$RETRIEVAL_MODEL_PATH" --port 8000 \
        > "$LOG_DIR/retrieval.log" 2>&1 &
    EXPERT_PIDS+=($!)

    # Qwen3-32B-FP8
    log "启动 Qwen3-32B-FP8 (GPU $EXPERT1_GPU, port 30001)..."
    CUDA_VISIBLE_DEVICES=$EXPERT1_GPU conda run -n sglang --no-capture-output \
        bash -c "export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python -m sglang.launch_server --model-path '$EXPERT1_MODEL' --port 30001 --context-length 163840 --tp 1" \
        > "$LOG_DIR/sglang_30001.log" 2>&1 &
    EXPERT_PIDS+=($!)

    # Qwen2.5-Coder-32B
    log "启动 Qwen2.5-Coder-32B (GPU $EXPERT2_GPU, port 30002)..."
    CUDA_VISIBLE_DEVICES=$EXPERT2_GPU conda run -n sglang --no-capture-output \
        bash -c "export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python -m sglang.launch_server --model-path '$EXPERT2_MODEL' --port 30002 --context-length 163840 --tp 1" \
        > "$LOG_DIR/sglang_30002.log" 2>&1 &
    EXPERT_PIDS+=($!)

    # Qwen2.5-Math-7B
    log "启动 Qwen2.5-Math-7B (GPU $EXPERT3_GPU, port 30003)..."
    CUDA_VISIBLE_DEVICES=$EXPERT3_GPU conda run -n sglang --no-capture-output \
        bash -c "export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python -m sglang.launch_server --model-path '$EXPERT3_MODEL' --port 30003 --context-length 163840 --tp 1" \
        > "$LOG_DIR/sglang_30003.log" 2>&1 &
    EXPERT_PIDS+=($!)

    # Qwen3-14B
    log "启动 Qwen3-14B (GPU $EXPERT4_GPU, port 30004)..."
    CUDA_VISIBLE_DEVICES=$EXPERT4_GPU conda run -n sglang --no-capture-output \
        bash -c "export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python -m sglang.launch_server --model-path '$EXPERT4_MODEL' --port 30004 --context-length 163840 --tp 1" \
        > "$LOG_DIR/sglang_30004.log" 2>&1 &
    EXPERT_PIDS+=($!)

    # DeepSeek-R1-Distill-32B
    log "启动 DeepSeek-R1-Distill-32B (GPU $EXPERT5_GPU, port 30005)..."
    CUDA_VISIBLE_DEVICES=$EXPERT5_GPU conda run -n sglang --no-capture-output \
        bash -c "export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python -m sglang.launch_server --model-path '$EXPERT5_MODEL' --port 30005 --context-length 163840 --tp 1" \
        > "$LOG_DIR/sglang_30005.log" 2>&1 &
    EXPERT_PIDS+=($!)

    # Qwen3-30B-A3B
    log "启动 Qwen3-30B-A3B (GPU $EXPERT6_GPU, port 30006)..."
    CUDA_VISIBLE_DEVICES=$EXPERT6_GPU conda run -n sglang --no-capture-output \
        bash -c "export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python -m sglang.launch_server --model-path '$EXPERT6_MODEL' --port 30006 --context-length 163840 --tp 1" \
        > "$LOG_DIR/sglang_30006.log" 2>&1 &
    EXPERT_PIDS+=($!)

    # Qwen2.5-Math-72B
    log "启动 Qwen2.5-Math-72B (GPU $EXPERT7_GPU, port 30007)..."
    CUDA_VISIBLE_DEVICES=$EXPERT7_GPU conda run -n sglang --no-capture-output \
        bash -c "export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        python -m sglang.launch_server --model-path '$EXPERT7_MODEL' --port 30007 --context-length 163840 --tp 1" \
        > "$LOG_DIR/sglang_30007.log" 2>&1 &
    EXPERT_PIDS+=($!)

    log "等待所有 Expert 服务就绪..."
    wait_port "检索服务"              127.0.0.1 8000  600
    wait_port "SGLang-30001(Qwen3-32B)"       127.0.0.1 30001 600
    wait_port "SGLang-30002(Coder-32B)"       127.0.0.1 30002 600
    wait_port "SGLang-30003(Math-7B)"         127.0.0.1 30003 600
    wait_port "SGLang-30004(Qwen3-14B)"       127.0.0.1 30004 600
    wait_port "SGLang-30005(DeepSeek-R1-32B)" 127.0.0.1 30005 600
    wait_port "SGLang-30006(Qwen3-30B-A3B)"   127.0.0.1 30006 600
    wait_port "SGLang-30007(Math-72B)"        127.0.0.1 30007 600
    log "所有 Expert 服务已就绪！"
}

cleanup_all() {
    log "======= 清理所有服务 ======="
    kill_orchestrator 2>/dev/null || true
    for pid in "${EXPERT_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    pkill -f sglang 2>/dev/null || true
    pkill -f "retrieval_general" 2>/dev/null || true
    rm -rf /tmp/tau2_eval_* /tmp/tau2_orch_* /tmp/tau2_transfer_* /tmp/tau2_output* 2>/dev/null || true
    log "清理完成"
}
trap cleanup_all EXIT

# ──────────────────────────── 收集 checkpoint 列表 ──────────────────────────── #
if [ -n "${ONLY_ITERS:-}" ]; then
    CKPT_LIST=()
    for iter in $ONLY_ITERS; do
        p="$HF_CKPT_BASE/$iter"
        if [ -d "$p" ] && [ -f "$p/config.json" ]; then
            CKPT_LIST+=("$p")
        else
            log "WARN: $p 不存在或未完成转换，跳过"
        fi
    done
else
    mapfile -t CKPT_LIST < <(
        ls -d "$HF_CKPT_BASE"/iter_* 2>/dev/null \
        | sort \
        | while read -r p; do
            [ -f "$p/config.json" ] && echo "$p"
        done
    )
fi

if [ ${#CKPT_LIST[@]} -eq 0 ]; then
    log "ERROR: 在 $HF_CKPT_BASE 下未找到已转换的 checkpoint（需含 config.json）"
    log "请先运行: bash convert_to_hf.sh 或 CONVERT_ALL=1 bash convert_to_hf.sh"
    exit 1
fi

log "============================================"
log "Sweep 配置"
log "  benchmark   : $BENCHMARK"
log "  n_runs/ckpt : $N_RUNS"
log "  checkpoint 数: ${#CKPT_LIST[@]}"
log "  输出根目录  : $SWEEP_OUTPUT_BASE"
log "============================================"
for p in "${CKPT_LIST[@]}"; do log "  $p"; done
echo ""

# ──────────────────────────── 启动 Expert 服务 ──────────────────────────────── #
if [ "$SKIP_EXPERT_SERVICES" = "1" ]; then
    log "SKIP_EXPERT_SERVICES=1，假定 Expert 服务已在运行"
    # 仍需确认端口可达
    wait_port "检索服务"              127.0.0.1 8000  60
    wait_port "SGLang-30001"          127.0.0.1 30001 60
    wait_port "SGLang-30002"          127.0.0.1 30002 60
    wait_port "SGLang-30003"          127.0.0.1 30003 60
    wait_port "SGLang-30004"          127.0.0.1 30004 60
    wait_port "SGLang-30005"          127.0.0.1 30005 60
    wait_port "SGLang-30006"          127.0.0.1 30006 60
    wait_port "SGLang-30007"          127.0.0.1 30007 60
else
    start_expert_services
fi

# ──────────────────────────── 主循环 ────────────────────────────────────────── #
export PYTHONPATH="${SCRIPT_DIR}:/data/slime-agentic/agentic:/data/slime-agentic:${PYTHONPATH:-}"

# 结果记录：ckpt_name -> 准确率列表
declare -A SWEEP_ACCURACIES   # "iter_xxx" -> "0.72 0.75 0.71"
declare -A SWEEP_CORRECT
declare -A SWEEP_TOTAL
ORCH_PID=""

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

for ckpt_path in "${CKPT_LIST[@]}"; do
    iter_name=$(basename "$ckpt_path")
    log ""
    log "════════════════════════════════════════════"
    log "Checkpoint: $iter_name"
    log "════════════════════════════════════════════"

    # 停止上一个 orchestrator
    if [ -n "$ORCH_PID" ] && kill -0 "$ORCH_PID" 2>/dev/null; then
        kill_orchestrator
    elif port_alive "$ORCH_PORT"; then
        kill_orchestrator
    fi

    # 启动新 orchestrator
    start_orchestrator "$ckpt_path"
    wait_port "Orchestrator" 127.0.0.1 "$ORCH_PORT" 600

    run_accuracies=()
    run_corrects=()
    run_totals=()

    for run in $(seq 1 "$N_RUNS"); do
        log "--- $iter_name  Run $run/$N_RUNS ---"

        run_output_dir="${SWEEP_OUTPUT_BASE}/${iter_name}/run${run}"
        mkdir -p "$run_output_dir"

        # 清理上一次 tau2 临时文件（避免串号）
        rm -rf /tmp/tau2_eval_* /tmp/tau2_orch_* /tmp/tau2_transfer_* /tmp/tau2_output* 2>/dev/null || true

        CMD=(
            $EVAL_PYTHON_CMD "${SCRIPT_DIR}/eval_orchestra.py"
            --benchmark      "$BENCHMARK"
            --model-url      "http://127.0.0.1:${ORCH_PORT}/v1"
            --model-name     "$ckpt_path"
            --output-dir     "$run_output_dir"
            --context-length "$ORCH_CTX"
            --concurrency    "$CONCURRENCY"
            --max-turns      "$MAX_TURNS"
            --temperature    "$TEMPERATURE"
            --user-llm       "$USER_LLM"
        )
        [ -n "$MAX_EXAMPLES" ] && CMD+=(--max-examples "$MAX_EXAMPLES")
        [ -n "$DOMAINS"      ] && CMD+=(--domains $DOMAINS)

        log "命令: ${CMD[*]}"
        "${CMD[@]}" 2>&1 | tee "${run_output_dir}/_eval.log"
        eval_exit=${PIPESTATUS[0]}

        if [ $eval_exit -ne 0 ]; then
            log "WARN: Run $run 评测失败（exit=$eval_exit），跳过"
            continue
        fi

        # 找到 eval_orchestra.py 创建的带时间戳子目录
        actual_output=$(ls -dt "${run_output_dir}/${BENCHMARK}_"* 2>/dev/null | head -1)
        if [ -z "$actual_output" ]; then
            actual_output="$run_output_dir"
        fi

        summary_file="${actual_output}/_summary.json"
        if [ -f "$summary_file" ]; then
            acc=$($SYS_PYTHON -c "
import json, sys
d = json.load(open('$summary_file'))
o = d.get('overall', {})
print(o.get('accuracy', 0))
")
            correct=$($SYS_PYTHON -c "
import json
d = json.load(open('$summary_file'))
o = d.get('overall', {})
print(o.get('correct', 0))
")
            total=$($SYS_PYTHON -c "
import json
d = json.load(open('$summary_file'))
o = d.get('overall', {})
print(o.get('count', 0))
")
            run_accuracies+=("$acc")
            run_corrects+=("$correct")
            run_totals+=("$total")
            log "$iter_name Run $run: accuracy=$acc  ($correct/$total)"
        else
            log "WARN: _summary.json 不存在，跳过 Run $run"
        fi
    done

    # 计算 max / avg
    if [ ${#run_accuracies[@]} -gt 0 ]; then
        stats=$($SYS_PYTHON -c "
import sys
accs = [float(x) for x in '${run_accuracies[*]}'.split()]
corrects = [int(x) for x in '${run_corrects[*]}'.split()]
totals = [int(x) for x in '${run_totals[*]}'.split()]
max_acc = max(accs)
avg_acc = sum(accs) / len(accs)
max_corr = corrects[accs.index(max_acc)]
max_tot  = totals[accs.index(max_acc)]
print(f'{max_acc:.4f} {avg_acc:.4f} {max_corr} {max_tot}')
")
        max_acc=$(echo $stats | awk '{print $1}')
        avg_acc=$(echo $stats | awk '{print $2}')
        max_corr=$(echo $stats | awk '{print $3}')
        max_tot=$(echo $stats | awk '{print $4}')

        SWEEP_ACCURACIES["$iter_name"]="${run_accuracies[*]}"
        SWEEP_CORRECT["$iter_name"]="$max_corr/$max_tot"
        log "[$iter_name] max=$max_acc  avg=$avg_acc  runs=${run_accuracies[*]}"

        # 写入 per-checkpoint 汇总
        $SYS_PYTHON -c "
import json
data = {
    'checkpoint': '$iter_name',
    'benchmark': '$BENCHMARK',
    'n_runs': ${#run_accuracies[@]},
    'accuracies': [float(x) for x in '${run_accuracies[*]}'.split()],
    'max_accuracy': float('$max_acc'),
    'avg_accuracy': float('$avg_acc'),
    'max_correct_str': '$max_corr/$max_tot',
}
with open('${SWEEP_OUTPUT_BASE}/${iter_name}/_ckpt_summary.json', 'w') as f:
    json.dump(data, f, indent=2)
"
    else
        log "[$iter_name] 所有 Run 均失败，跳过"
        SWEEP_ACCURACIES["$iter_name"]=""
    fi
done

# 停止最后一个 orchestrator
kill_orchestrator 2>/dev/null || true

# ──────────────────────────── 汇总输出 ──────────────────────────────────────── #
log ""
log "════════════════════════════════════════════"
log "          Sweep 评测汇总  ($BENCHMARK)"
log "════════════════════════════════════════════"

SWEEP_RESULT_FILE="${SWEEP_OUTPUT_BASE}/_sweep_summary.json"
all_results=()

printf "%-22s  %-8s  %-8s  %-10s  %s\n" "Checkpoint" "Max Acc" "Avg Acc" "Best" "All Runs"
printf "%-22s  %-8s  %-8s  %-10s  %s\n" "──────────────────────" "───────" "───────" "─────────" "────────────────────"

for ckpt_path in "${CKPT_LIST[@]}"; do
    iter_name=$(basename "$ckpt_path")
    accs_str="${SWEEP_ACCURACIES[$iter_name]:-}"
    if [ -z "$accs_str" ]; then
        printf "%-22s  %-8s  %-8s  %-10s  %s\n" "$iter_name" "FAILED" "-" "-" "-"
        continue
    fi

    result=$($SYS_PYTHON -c "
accs = [float(x) for x in '$accs_str'.split()]
print(f'{max(accs):.4f} {sum(accs)/len(accs):.4f}')
")
    max_acc=$(echo $result | awk '{print $1}')
    avg_acc=$(echo $result | awk '{print $2}')
    best_str="${SWEEP_CORRECT[$iter_name]:-}"
    printf "%-22s  %-8s  %-8s  %-10s  %s\n" "$iter_name" "$max_acc" "$avg_acc" "$best_str" "$accs_str"

    runs_csv=$($SYS_PYTHON -c "print(','.join('$accs_str'.split()))")
    all_results+=("{\"checkpoint\":\"$iter_name\",\"max\":$max_acc,\"avg\":$avg_acc,\"runs\":[$runs_csv]}")
done

echo ""
log "结果已保存至: $SWEEP_OUTPUT_BASE/"
log "  每个 checkpoint: ${SWEEP_OUTPUT_BASE}/<iter>/run{1..N}/_summary.json"
log "  总汇总:          $SWEEP_RESULT_FILE"

# 写入总汇总 JSON
$SYS_PYTHON -c "
import json, os, glob

results = []
base = '$SWEEP_OUTPUT_BASE'
for ckpt_dir in sorted(glob.glob(os.path.join(base, 'iter_*'))):
    summary_file = os.path.join(ckpt_dir, '_ckpt_summary.json')
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            results.append(json.load(f))

with open('$SWEEP_RESULT_FILE', 'w') as f:
    json.dump({'benchmark': '$BENCHMARK', 'checkpoints': results}, f, indent=2)
print('Sweep summary saved.')
"
