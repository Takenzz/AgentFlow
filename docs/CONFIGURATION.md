# 训练配置说明

主脚本是 `agentic/agentflow/train_agentflow.sh`。它被 `launch.sh` 和 `quick_train_2x4090.sh` 调用，也可以直接调用。

## 必填路径

```bash
MODEL_CONFIG_SCRIPT=/path/to/model_config.sh
BASE_HF_CHECKPOINT=/path/to/student_hf
REF_LOAD=/path/to/student_torch_dist
SAVE_DIR=/path/to/output_ckpt
PROMPT_DATA=/path/to/train.jsonl
EVAL_PROMPT_DATA=/path/to/eval.jsonl
```

## 外部依赖路径

```bash
SLIME_PATH=/path/to/slime
MEGATRON_PATH=/path/to/Megatron-LM
```

如果 slime 已经 pip 安装，`SLIME_PATH` 可以不设置。

## AgentFlow API 角色

非 Planner 角色默认走 OpenAI-compatible API：

```bash
AGENTFLOW_API_BASE=https://api.openai.com/v1
AGENTFLOW_API_KEY=$OPENAI_API_KEY
AGENTFLOW_API_MODEL=gpt-4o-mini
```

也可以按角色覆盖：

```bash
AGENTFLOW_EXECUTOR_MODEL=gpt-4o-mini
AGENTFLOW_CODER_MODEL=gpt-4o-mini
AGENTFLOW_REWARDER_MODEL=gpt-4o-mini
```

## 训练规模

常用参数：

```bash
TRAIN_GPUS=1
TRAIN_TP=1
ROLLOUT_ENGINE_GPUS=1
ROLLOUT_BATCH_SIZE=1
N_SAMPLES_PER_PROMPT=2
GLOBAL_BATCH_SIZE=2
NUM_ROLLOUT=1
MAX_TOKENS_PER_GPU=4096
```

`NUM_ROLLOUT` 优先级高于 `NUM_EPOCH`。快速测试建议用 `NUM_ROLLOUT=1`。

## OPD Teacher

使用 SGLang teacher logprob：

```bash
USE_OPD=1
OPD_TYPE=sglang
OPD_KL_COEF=1.0
RM_URL=http://127.0.0.1:30080/generate
```

`rollout.reward_func` 会按 AgentFlow turn 分别请求 teacher logprob，`custom_convert.custom_convert` 会把这些 teacher logprob 写进训练 batch。

如果不设置 `RM_URL`，`rollout.reward_func` 会退回到轻量答案匹配 reward：从 `final_output` 中抽取 `\boxed{...}`，再和 `label` 归一化后做精确比较。这适合先跑通普通 GRPO 链路；需要 teacher distillation 时再打开 `USE_OPD=1`。

## 显存相关

4090 quick test 建议：

```bash
SGLANG_CONTEXT_LENGTH=8192
ROLLOUT_MAX_RESPONSE_LEN=2048
SGLANG_MEM_FRACTION_STATIC=0.58
MAX_TOKENS_PER_GPU=4096
USE_KL_LOSS=0
```

如果 OOM，优先降低：

- `ROLLOUT_MAX_RESPONSE_LEN`
- `SGLANG_CONTEXT_LENGTH`
- `MAX_TOKENS_PER_GPU`
- `N_SAMPLES_PER_PROMPT`

## 日志

`launch.sh` 默认日志：

```text
/tmp/agentflow_logs/train.log
```

`quick_train_2x4090.sh` 默认日志：

```text
/tmp/agentflow_quick_2x4090/teacher.log
/tmp/agentflow_quick_2x4090/train.log
```

`formal_train_4xa800.sh` 默认日志：

```text
/tmp/agentflow_formal_4xa800/train.log
/tmp/agentflow_formal_4xa800/teacher.log
```
