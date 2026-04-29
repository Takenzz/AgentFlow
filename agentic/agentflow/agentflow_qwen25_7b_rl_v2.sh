#!/bin/bash


# Cleanup previous runs
if [ "${SKIP_PROCESS_KILL}" != "1" ]; then
    pkill -9 sglang
    sleep 3
    ray stop --force
    pkill -9 ray
    pkill -9 python
    sleep 3
    pkill -9 ray
    pkill -9 python
fi

set -ex

# Set to "1" to save training trajectories to trajectories/ folder (default: off)
SAVE_TRAJECTORY=${SAVE_TRAJECTORY:-"0"}

# Reset trajectories directory (only when saving is enabled)
TRAJECTORIES_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)/trajectories"
if [ "${SAVE_TRAJECTORY:-0}" = "1" ]; then
    rm -rf "${TRAJECTORIES_DIR}"
    mkdir -p "${TRAJECTORIES_DIR}"
fi

# Prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Detect NVLink availability
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Get script directory
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Load model configuration. Default targets the "about 2B or smaller" setup;
# override MODEL_CONFIG_SCRIPT for another small planner.
MODEL_CONFIG_SCRIPT=${MODEL_CONFIG_SCRIPT:-"${SCRIPT_DIR}/../../scripts/models/qwen2.5-1.5B.sh"}
source "${MODEL_CONFIG_SCRIPT}"

BASE_HF_CHECKPOINT=${BASE_HF_CHECKPOINT:-"/data/models/qwen25_1.5b"}
REF_LOAD=${REF_LOAD:-"/data/models/qwen2.5_1.5b_dist/"}
SAVE_DIR=${SAVE_DIR:-"/data/AgentFlow_Qwen25-1.5B-RL/"}
PROMPT_DATA=${PROMPT_DATA:-"/data/dapo-math-17k/dapo-math-17k.jsonl"}
EVAL_PROMPT_DATA=${EVAL_PROMPT_DATA:-"/data/aime-2024/aime-2024.jsonl"}
TRAIN_GPUS=${TRAIN_GPUS:-4}
TRAIN_TP=${TRAIN_TP:-1}
ROLLOUT_ENGINE_GPUS=${ROLLOUT_ENGINE_GPUS:-1}

# Checkpoint arguments
CKPT_ARGS=(
   --hf-checkpoint ${BASE_HF_CHECKPOINT}
   --ref-load ${REF_LOAD}
   --save ${SAVE_DIR}
   --save-interval ${SAVE_INTERVAL:-100}
)

# Rollout arguments
ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA}
   --input-key prompt
   --label-key label
   --rollout-shuffle
   --reward-key score
   --num-epoch ${NUM_EPOCH:-1}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-8}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-8}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-32768}
   --rollout-temperature ${ROLLOUT_TEMPERATURE:-0.7}
   --global-batch-size ${GLOBAL_BATCH_SIZE:-64}
   --balance-data
)

# Evaluation arguments
EVAL_ARGS=(
   --eval-interval ${EVAL_INTERVAL:-20}
   --eval-prompt-data aime ${EVAL_PROMPT_DATA}
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 32768

   --eval-top-p 0.95
)

# Performance arguments
PERF_ARGS=(
   --tensor-model-parallel-size ${TRAIN_TP}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full    
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-16384}
)

# GRPO arguments
GRPO_ARGS=(
   --advantage-estimator grpo     
   --use-kl-loss                  
   --kl-loss-coef 0.001           
   --kl-loss-type low_var_kl
   --entropy-coef 0.0              
   --eps-clip 0.2                 
   --eps-clip-high 0.3            
)

# Optimizer arguments
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6                      
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# WandB arguments
# config: trainer.logger=['console','wandb'], PROJECT_NAME='AgentFlow_pro'
# 启用 wandb：请确保环境变量 WANDB_KEY 已设置，或手动填入 key
# WANDB_ARGS=(
#    --use-wandb
#    --wandb-project AgentFlow_pro
#    --wandb-group AgentFlow_pro-Qwen25-1.5B-RL
#    --wandb-key ${WANDB_KEY:-"your_wandb_key_here"}
# )
# 如不需要 wandb，注释上面 4 行并取消注释下一行：
WANDB_ARGS=()

# SGLang arguments
# config: ROLLOUT_TP_SIZE=1, gpu_memory_utilization=0.6
# SGLang arguments
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine ${ROLLOUT_ENGINE_GPUS}
   --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION_STATIC:-0.75}
   --sglang-context-length ${SGLANG_CONTEXT_LENGTH:-65536}
)

# Misc arguments
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Custom ReAct generation and reward functions
CUSTOM_ARGS=(
   --custom-generate-function-path rollout.generate
   --custom-rm-path rollout.reward_func
   --custom-eval-rollout-log-function-path rollout.eval_log
   --custom-convert-samples-to-train-data-path custom_convert.custom_convert
)

# Launch Ray head node
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${TRAIN_GPUS} --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build runtime environment JSON
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:/root/slime\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SAVE_TRAJECTORY\": \"${SAVE_TRAJECTORY}\",
    \"AGENTFLOW_API_BASE\": \"${AGENTFLOW_API_BASE:-https://api.openai.com/v1}\",
    \"AGENTFLOW_API_KEY\": \"${AGENTFLOW_API_KEY:-${OPENAI_API_KEY:-}}\",
    \"AGENTFLOW_API_MODEL\": \"${AGENTFLOW_API_MODEL:-gpt-4o-mini}\",
    \"AGENTFLOW_API_ENABLE_THINKING\": \"${AGENTFLOW_API_ENABLE_THINKING:-false}\",
    \"AGENTFLOW_API_THINKING_BUDGET\": \"${AGENTFLOW_API_THINKING_BUDGET:-}\",
    \"AGENTFLOW_EXECUTOR_API_BASE\": \"${AGENTFLOW_EXECUTOR_API_BASE:-${AGENTFLOW_API_BASE:-https://api.openai.com/v1}}\",
    \"AGENTFLOW_EXECUTOR_API_KEY\": \"${AGENTFLOW_EXECUTOR_API_KEY:-${AGENTFLOW_API_KEY:-${OPENAI_API_KEY:-}}}\",
    \"AGENTFLOW_EXECUTOR_MODEL\": \"${AGENTFLOW_EXECUTOR_MODEL:-${AGENTFLOW_API_MODEL:-gpt-4o-mini}}\",
    \"AGENTFLOW_CODER_API_BASE\": \"${AGENTFLOW_CODER_API_BASE:-${AGENTFLOW_EXECUTOR_API_BASE:-${AGENTFLOW_API_BASE:-https://api.openai.com/v1}}}\",
    \"AGENTFLOW_CODER_API_KEY\": \"${AGENTFLOW_CODER_API_KEY:-${AGENTFLOW_EXECUTOR_API_KEY:-${AGENTFLOW_API_KEY:-${OPENAI_API_KEY:-}}}}\",
    \"AGENTFLOW_CODER_MODEL\": \"${AGENTFLOW_CODER_MODEL:-${AGENTFLOW_EXECUTOR_MODEL:-${AGENTFLOW_API_MODEL:-gpt-4o-mini}}}\",
    \"AGENTFLOW_REWARDER_API_BASE\": \"${AGENTFLOW_REWARDER_API_BASE:-${AGENTFLOW_EXECUTOR_API_BASE:-${AGENTFLOW_API_BASE:-https://api.openai.com/v1}}}\",
    \"AGENTFLOW_REWARDER_API_KEY\": \"${AGENTFLOW_REWARDER_API_KEY:-${AGENTFLOW_EXECUTOR_API_KEY:-${AGENTFLOW_API_KEY:-${OPENAI_API_KEY:-}}}}\",
    \"AGENTFLOW_REWARDER_MODEL\": \"${AGENTFLOW_REWARDER_MODEL:-${AGENTFLOW_EXECUTOR_MODEL:-${AGENTFLOW_API_MODEL:-gpt-4o-mini}}}\",
    \"AGENTFLOW_API_TIMEOUT\": \"${AGENTFLOW_API_TIMEOUT:-180}\",
    \"AGENTFLOW_API_MAX_RETRIES\": \"${AGENTFLOW_API_MAX_RETRIES:-3}\",
    \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\": \"1\"
  }
}"

# Submit training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${TRAIN_GPUS} \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
