#!/bin/bash
# ToolOrchestra RL 训练脚本
# 基于 slime + GRPO，训练 Qwen3-8B 作为 Orchestrator
#
# 使用方式：
#   cd /data/slime-agentic
#   bash agentic/ToolOrchestra/train_orchestra.sh
#
# 前置条件：
#   1. 检索服务已启动（retrieval_general_thought.py --port 8000）
#   2. Expert 模型服务已启动（可与训练共用同一批 GPU，训练时暂停 expert 服务）

# 若由 launch.sh 统一管理，则跳过此处的进程清理
if [ "${SKIP_PROCESS_KILL:-0}" != "1" ]; then
    pkill -9 sglang 2>/dev/null || true
    sleep 3
    ray stop --force
    pkill -9 ray 2>/dev/null || true
    pkill -9 python 2>/dev/null || true
    sleep 3
fi

# 清理上一轮 tau2 残留的临时文件
rm -rf /tmp/tau2_orch_* /tmp/tau2_transfer_* /tmp/tau2_output* 2>/dev/null
echo "Cleaned up tau2 temp files."

# 清理上一轮 rollout 日志（重新开始记录）
rm -rf /data/rollout_logs/train /data/rollout_logs/eval 2>/dev/null
echo "Cleaned up old rollout logs."

ulimit -n 65536
set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="/data/slime-agentic"

# NVLink 检测
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK"

# ── 模型架构（Qwen3-8B）─────────────────────────────────────────────────────
source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"

# ── 路径配置 ─────────────────────────────────────────────────────────────────
HF_CKPT="/data/models/qwen3_8b"       # HuggingFace 格式原始权重
REF_CKPT="/data/qwen3_8b_dist"        # Megatron 分布式格式（ref model）
SAVE_DIR="/data/checkpoints/orchestra_qwen3_8b_rl"
DATA_PATH="${DATA_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data/data_slime_full.jsonl}"

# ── Checkpoint ───────────────────────────────────────────────────────────────
CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --ref-load       "${REF_CKPT}"
    --save           "${SAVE_DIR}"
    --save-interval  10
)

# ── Rollout（数据 + 采样）────────────────────────────────────────────────────
ROLLOUT_ARGS=(
    --prompt-data    "${DATA_PATH}"
    --input-key      problem
    --label-key      answer
    --metadata-key   metadata
    --tool-key       tools
    --rollout-shuffle
    --reward-key     score

    --num-epoch              2
    --rollout-batch-size     32
    --n-samples-per-prompt   8
    --rollout-max-response-len 16384
    --rollout-temperature    0.7
    --global-batch-size      128
    --balance-data
)

# # ── 评估 ─────────────────────────────────────────────────────────────────────
# EVAL_ARGS=(
#     --eval-interval             50
#     --eval-prompt-data          orchestra_qa "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data/eval_100.jsonl"
#     --n-samples-per-eval-prompt 1
#     --eval-max-response-len     32768
#     --eval-top-p                0.95
# )

# ── 并行 & 性能 ───────────────────────────────────────────────────────────────
# 8B 模型，4× GPU（GPU 4-7），TP=2（DP=2），降低单卡显存
PERF_ARGS=(
    --tensor-model-parallel-size   2
    --pipeline-model-parallel-size 1
    --context-parallel-size        1
    --expert-model-parallel-size   1
    --expert-tensor-parallel-size  1

    --use-dynamic-batch-size
    --max-tokens-per-gpu         8192
    --log-probs-max-tokens-per-gpu 131072

    --train-memory-margin-bytes  0
    --recompute-granularity      full
    --recompute-method           uniform
    --recompute-num-layers       1
    --recompute-loss-function
    --log-probs-chunk-size       4096
)

# ── GRPO 算法 ─────────────────────────────────────────────────────────────────
GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef        0.001
    --kl-loss-type        low_var_kl
    --entropy-coef        0.0
    --eps-clip            0.2
    --eps-clip-high       0.3
)

# ── 优化器 ───────────────────────────────────────────────────────────────────
OPTIMIZER_ARGS=(
    --optimizer      adam
    --lr             1e-6
    --lr-decay-style constant
    --weight-decay   0.1
    --adam-beta1     0.9
    --adam-beta2     0.98
)

# ── SGLang 推理引擎 ───────────────────────────────────────────────────────────
# 与 Megatron TP=2 对齐：rollout 引擎跨 2 卡
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 2
    --sglang-mem-fraction-static  0.50
    --sglang-context-length       131072
    --sglang-server-concurrency   512
)

# ── 其他 ─────────────────────────────────────────────────────────────────────
MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout    0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

# ── 自定义函数路径 ─────────────────────────────────────────────────────────────
CUSTOM_ARGS=(
    --custom-generate-function-path              rollout.generate
    --custom-rm-path                             rollout.reward_func
    --custom-convert-samples-to-train-data-path  custom_convert.custom_convert
)

# ── WandB（按需开启）─────────────────────────────────────────────────────────
WANDB_ARGS=()
# WANDB_ARGS=(
#     --use-wandb
#     --wandb-project ToolOrchestra
#     --wandb-group   orchestra_qwen3_8b_rl
#     --wandb-key     "${WANDB_KEY:-your_key_here}"
# )

# ── 启动 Ray ──────────────────────────────────────────────────────────────────
# 禁用 OpenTelemetry gRPC metrics 导出，避免 Ray InfoActor SIGSEGV（gRPC getenv 竞态）
export OTEL_SDK_DISABLED=true
export RAY_DISABLE_EXPORT_METRICS=1
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus 4 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

# PYTHONPATH:
#   - ToolOrchestra 目录：rollout.py / reward.py / custom_convert.py
#   - agentflow 目录：SGLangEngine / GenerationOutput
#   - slime-agentic 根目录
#   - Megatron-LM
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SCRIPT_DIR}:/root/Megatron-LM:${SLIME_DIR}/agentic:${SLIME_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"PYTORCH_ALLOC_CONF\": \"max_split_size_mb:128\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"CUDA_VISIBLE_DEVICES\": \"${CUDA_VISIBLE_DEVICES:-4,5,6,7}\",
    \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\": \"1\",
    \"OTEL_SDK_DISABLED\": \"true\",
    \"RAY_DISABLE_EXPORT_METRICS\": \"1\"
  }
}"
# ── 提交训练任务 ──────────────────────────────────────────────────────────────
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_DIR}/train.py" \
    --actor-num-nodes         1 \
    --actor-num-gpus-per-node 4 \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}"
