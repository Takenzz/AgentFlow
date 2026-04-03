#!/bin/bash
# MemAgent 训练启动脚本（7B 模型参考配置）
# 用法：bash run_memagent_7b.sh

if [ "${SKIP_PROCESS_KILL}" != "1" ]; then
    pkill -9 sglang 2>/dev/null || true
    sleep 3
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
fi

set -ex

export PYTHONBUFFERED=16
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# NVLink 检测
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ── 路径配置 ──────────────────────────────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-"/data/models/qwen25_7b"}
REF_PATH=${REF_PATH:-"/data/models/qwen2.5_7b_dist/"}
SAVE_PATH=${SAVE_PATH:-"/data/MemAgent_Qwen25-7B-RL/"}

# HotpotQA 数据集（由 prepare_data.py 从 MemAgent parquet 转换而来）
# 转换命令示例（二选一）：
#   python agentic/memagent/prepare_data.py \
#       --input /data/MemAgent/taskutils/memory_data/hotpotqa_train_process.parquet \
#       --output /data/hotpotqa_slime/train.jsonl
#
#   python agentic/memagent/prepare_data.py \
#       --hf-dataset BytedTsinghua-SIA/hotpotqa --hf-split train \
#       --output /data/hotpotqa_slime/train.jsonl
TRAIN_DATA=${TRAIN_DATA:-"/data/hotpotqa_slime/train.jsonl"}
EVAL_DATA=${EVAL_DATA:-"/data/hotpotqa_slime/dev.jsonl"}

source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-7B.sh"

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --ref-load "${REF_PATH}"
    --save "${SAVE_PATH}"
    --save-interval 100
)

# ── Rollout 参数 ───────────────────────────────────────────────────────────────
# rollout-max-response-len 覆盖单次 LLM 调用的最大 token 数，
# 实际每轮最大 token 由 MEM_MAX_MEMORY / MEM_MAX_FINAL 环境变量控制。
ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key prompt
    --label-key label
    --rollout-shuffle
    --reward-key score
    --num-epoch 1
    --rollout-batch-size 16
    --n-samples-per-prompt 16
    --rollout-max-response-len 8192
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --global-batch-size 256
    --balance-data
)

EVAL_ARGS=(
    # eval 已禁用，如需开启请取消注释并准备 dev.jsonl
    # --eval-interval 50
    # --eval-prompt-data hotpotqa_dev "${EVAL_DATA}"
    # --n-samples-per-eval-prompt 1
    # --eval-max-response-len 4096
    # --eval-top-p 0.95
)

PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.001
    --kl-loss-type low_var_kl
    --eps-clip 0.2
    --eps-clip-high 0.3
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --colocate
    --rollout-num-gpus-per-engine 4
    --sglang-mem-fraction-static 0.75
    # MemAgent 需要处理长上下文，开启 YaRN
    --sglang-context-length 131072
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

# memagent/rollout.py 和 agentflow/custom_convert.py 均在 PYTHONPATH 中
CUSTOM_ARGS=(
    --custom-generate-function-path rollout.generate
    --custom-rm-path rollout.reward_func
    --custom-convert-samples-to-train-data-path custom_convert.custom_convert
)

ulimit -n 65536

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus 8 \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:${SCRIPT_DIR}/..:${SLIME_ROOT}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\": \"1\",
    \"MEM_CHUNK_TOKENS\": \"${MEM_CHUNK_TOKENS:-5000}\",
    \"MEM_MAX_MEMORY\": \"${MEM_MAX_MEMORY:-1024}\",
    \"MEM_MAX_FINAL\": \"${MEM_MAX_FINAL:-256}\",
    \"MEM_MAX_CHUNKS\": \"${MEM_MAX_CHUNKS:-512}\"
  }
}"

SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_ROOT}/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    ${MODEL_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${OPTIMIZER_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${PERF_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${MISC_ARGS[@]} \
    ${CUSTOM_ARGS[@]}
