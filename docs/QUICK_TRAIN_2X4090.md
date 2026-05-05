# 两张 4090 快速训练测试

目标：用两张 RTX 4090 做一个最小 smoke test。

- GPU 0: student 训练 + student Planner rollout。
- GPU 1: 2B teacher SGLang logprob server。
- student: 0.8B 级小模型。
- teacher: 2B 小模型。
- 训练信号: OPD KL，teacher 返回 student 轨迹的 token logprob。

## 1. 前提

确认依赖：

```bash
python3 -c "import slime; print(slime.__file__)"
python3 -c "import ray, torch, sglang, openai, aiohttp"
nvidia-smi
```

如果 slime/Megatron 是源码路径：

```bash
export SLIME_PATH=/path/to/slime
export MEGATRON_PATH=/path/to/Megatron-LM
```

重要：teacher 和 student 必须 tokenizer 兼容，最好同一模型家族。OPD 是按 student token id 请求 teacher logprob，tokenizer 不兼容会让 teacher logprob 没有训练意义。

## 2. 准备 student torch_dist

student 需要 HF 和 torch_dist 两份：

```bash
export STUDENT_HF_CHECKPOINT=/data/models/student-0.8b
export STUDENT_REF_LOAD=/data/models/student-0.8b_torch_dist
```

如果你的 0.8B student 与 `qwen3-0.6B.sh` 结构不一致，请先写一个自己的 model config：

```bash
cp agentic/agentflow/model_configs/qwen3-0.6B.sh agentic/agentflow/model_configs/student-0.8B.sh
```

然后按真实结构修改 `num-layers`、`hidden-size`、attention heads、vocab 等参数。

转换示例：

```bash
export SLIME_REPO=/path/to/slime
export MEGATRON_PATH=/path/to/Megatron-LM
export STUDENT_MODEL_CONFIG=$PWD/agentic/agentflow/model_configs/student-0.8B.sh

source $STUDENT_MODEL_CONFIG

PYTHONPATH=$MEGATRON_PATH:$SLIME_REPO \
python3 $SLIME_REPO/tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint $STUDENT_HF_CHECKPOINT \
  --save $STUDENT_REF_LOAD
```

如果你实际用 Qwen3-0.6B 做 0.8B 级 smoke test，可以直接：

```bash
export STUDENT_MODEL_CONFIG=$PWD/agentic/agentflow/model_configs/qwen3-0.6B.sh
```

## 3. 准备 teacher

teacher 只需要 HF checkpoint，quick script 会自动在 GPU 1 拉起 SGLang：

```bash
export TEACHER_HF_CHECKPOINT=/data/models/teacher-2b
```

默认 teacher 地址：

```text
http://127.0.0.1:30080/generate
```

如果端口冲突：

```bash
export TEACHER_PORT=30180
```

## 4. 设置辅助 API

AgentFlow 里 Executor、Verifier、工具内部 LLM 默认使用 OpenAI-compatible API。快速测试也需要这些 API 可用：

```bash
export AGENTFLOW_API_BASE=https://api.openai.com/v1
export AGENTFLOW_API_KEY=$OPENAI_API_KEY
export AGENTFLOW_API_MODEL=gpt-4o-mini
```

如果你用本地兼容 API，把 base/model/key 换成自己的即可。

## 5. 启动 quick test

最小命令：

```bash
STUDENT_HF_CHECKPOINT=/data/models/student-0.8b \
STUDENT_REF_LOAD=/data/models/student-0.8b_torch_dist \
STUDENT_MODEL_CONFIG=$PWD/agentic/agentflow/model_configs/student-0.8B.sh \
TEACHER_HF_CHECKPOINT=/data/models/teacher-2b \
bash agentic/agentflow/quick_train_2x4090.sh
```

如果你先用仓库自带的 tiny 数据，默认会读取：

```text
examples/tiny_math.jsonl
```

如果要换自己的小数据：

```bash
export PROMPT_DATA=/data/my_tiny_train.jsonl
export EVAL_PROMPT_DATA=/data/my_tiny_eval.jsonl
```

## 6. 默认 quick 参数

`quick_train_2x4090.sh` 默认：

```bash
TRAIN_GPUS=1
TRAIN_TP=1
ROLLOUT_ENGINE_GPUS=1
NUM_ROLLOUT=1
ROLLOUT_BATCH_SIZE=1
N_SAMPLES_PER_PROMPT=2
GLOBAL_BATCH_SIZE=2
ROLLOUT_MAX_RESPONSE_LEN=2048
SGLANG_CONTEXT_LENGTH=8192
MAX_TOKENS_PER_GPU=4096
USE_OPD=1
OPD_TYPE=sglang
OPD_KL_COEF=1.0
USE_KL_LOSS=0
```

这不是追性能的配置，只用于快速验证整条链路能跑通。

## 7. 看日志

```bash
tail -f /tmp/agentflow_quick_2x4090/teacher.log
tail -f /tmp/agentflow_quick_2x4090/train.log
```

成功时应看到：

- teacher SGLang server 就绪。
- Ray dashboard 就绪。
- `rollout.generate` 生成 AgentFlow trajectory。
- `rollout.reward_func` 请求 teacher `/generate`。
- `custom_convert` 输出包含 `teacher_log_probs`。
- 训练保存到 `SAVE_DIR`，默认 `/tmp/agentflow_student_quick_ckpt`。

## 8. 常见问题

OOM：

```bash
ROLLOUT_MAX_RESPONSE_LEN=1024 \
SGLANG_CONTEXT_LENGTH=4096 \
MAX_TOKENS_PER_GPU=2048 \
bash agentic/agentflow/quick_train_2x4090.sh
```

teacher 没起来：

```bash
cat /tmp/agentflow_quick_2x4090/teacher.log
```

Ray 卡住或 GPU 被占：

```bash
ray stop --force
pkill -9 ray
pkill -9 -f 'sglang\.launch_server'
```

teacher/student tokenizer 不一致：

换同系列模型，或不要使用 OPD SGLang teacher logprob 模式。
