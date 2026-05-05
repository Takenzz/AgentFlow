# 4xA800 正式训练

这份文档面向正式训练环境：单机 4 张 A800，student 为 0.8B 级模型，teacher 为 2B 模型。

脚本：

```text
agentic/agentflow/formal_train_4xa800.sh
```

## 1. 推荐部署

有两种模式。

### 模式 A：外部 teacher，4 张 A800 全部训练 student

这是默认推荐模式：

- GPU 0,1,2,3: student 训练 + student Planner rollout。
- teacher 2B: 已经在另一台机器或另一个服务中启动。
- 训练脚本通过 `RM_URL` 请求 teacher `/generate` 拿 token logprob。

优点：

- 4 张 A800 全部给 student，吞吐更高。
- teacher 服务和训练任务解耦，正式训练更稳定。
- teacher 可以复用给多次实验。

### 模式 B：本机 local teacher，3 张 A800 训练 student

如果只有这一台 4xA800：

- GPU 0,1,2: student 训练 + student Planner rollout。
- GPU 3: teacher 2B SGLang logprob server。

优点：

- 不需要单独 teacher 机器。
- 和 2x4090 quick test 的拓扑一致，便于迁移。

代价：

- student 只用 3 张 A800。
- 本机同时跑 teacher 和训练，日志/进程管理要更注意。

## 2. 前置检查

```bash
python3 -c "import slime; print(slime.__file__)"
python3 -c "import ray, torch, sglang, openai, aiohttp"
nvidia-smi
```

如果 slime/Megatron 是源码目录：

```bash
export SLIME_PATH=/path/to/slime
export MEGATRON_PATH=/path/to/Megatron-LM
```

## 3. Student checkpoint

正式训练需要：

```bash
export STUDENT_HF_CHECKPOINT=/data/models/student-0.8b
export STUDENT_REF_LOAD=/data/models/student-0.8b_torch_dist
export STUDENT_MODEL_CONFIG=$PWD/agentic/agentflow/model_configs/student-0.8B.sh
```

`STUDENT_REF_LOAD` 必须由同一份 `STUDENT_MODEL_CONFIG` 转换得到。模型结构不一致会导致 checkpoint load 失败或训练错误。

如果你的 0.8B 模型不是仓库已有配置，复制最接近的配置再改：

```bash
cp agentic/agentflow/model_configs/qwen3-0.6B.sh \
   agentic/agentflow/model_configs/student-0.8B.sh
```

然后按真实 HF config 修改：

- `--num-layers`
- `--hidden-size`
- `--ffn-hidden-size`
- `--num-attention-heads`
- `--num-query-groups`
- `--vocab-size`
- RoPE/RMSNorm/GQA 等结构开关

## 4. Teacher 要求

OPD SGLang 模式会把 student token id 发给 teacher，请求 teacher 对同一段 token 序列计算 logprob。

因此：

- teacher/student 必须 tokenizer 兼容。
- 最好使用同一模型家族。
- teacher 大小可以是 2B，student 可以是 0.8B，但 tokenizer 不能乱。

## 5. 模式 A：外部 teacher 启动

先在 teacher 服务上启动 SGLang，例如：

```bash
CUDA_VISIBLE_DEVICES=0 \
python3 -m sglang.launch_server \
  --model-path /data/models/teacher-2b \
  --host 0.0.0.0 \
  --port 30080 \
  --tp-size 1 \
  --context-length 32768 \
  --mem-fraction-static 0.78 \
  --trust-remote-code
```

在 4xA800 训练机上：

```bash
export RM_URL=http://teacher-host:30080/generate
export STUDENT_HF_CHECKPOINT=/data/models/student-0.8b
export STUDENT_REF_LOAD=/data/models/student-0.8b_torch_dist
export STUDENT_MODEL_CONFIG=$PWD/agentic/agentflow/model_configs/student-0.8B.sh
export PROMPT_DATA=/data/dapo-math-17k/dapo-math-17k.jsonl
export EVAL_PROMPT_DATA=/data/aime-2024/aime-2024.jsonl
export SAVE_DIR=/data/agentflow_runs/student_0.8b_4xa800

bash agentic/agentflow/formal_train_4xa800.sh
```

默认会使用：

```bash
STUDENT_CUDA_VISIBLE_DEVICES=0,1,2,3
TRAIN_GPUS=4
ROLLOUT_BATCH_SIZE=8
N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE=64
ROLLOUT_MAX_RESPONSE_LEN=8192
SGLANG_CONTEXT_LENGTH=32768
MAX_TOKENS_PER_GPU=16384
USE_OPD=1
USE_KL_LOSS=0
```

## 6. 模式 B：本机 local teacher 启动

```bash
export LOCAL_TEACHER=1
export TEACHER_HF_CHECKPOINT=/data/models/teacher-2b
export STUDENT_HF_CHECKPOINT=/data/models/student-0.8b
export STUDENT_REF_LOAD=/data/models/student-0.8b_torch_dist
export STUDENT_MODEL_CONFIG=$PWD/agentic/agentflow/model_configs/student-0.8B.sh
export PROMPT_DATA=/data/dapo-math-17k/dapo-math-17k.jsonl
export EVAL_PROMPT_DATA=/data/aime-2024/aime-2024.jsonl
export SAVE_DIR=/data/agentflow_runs/student_0.8b_3xa800_local_teacher

bash agentic/agentflow/formal_train_4xa800.sh
```

默认会使用：

```bash
TEACHER_CUDA_VISIBLE_DEVICES=3
STUDENT_CUDA_VISIBLE_DEVICES=0,1,2
TRAIN_GPUS=3
ROLLOUT_BATCH_SIZE=6
N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE=48
```

这里 `GLOBAL_BATCH_SIZE=48` 是为了适配 3 张 student GPU，避免数据并行切分时 batch 不整齐。

## 7. API 角色

正式训练仍需要 Executor、Verifier、工具内部 LLM 的 OpenAI-compatible API：

```bash
export AGENTFLOW_API_BASE=https://api.openai.com/v1
export AGENTFLOW_API_KEY=$OPENAI_API_KEY
export AGENTFLOW_API_MODEL=gpt-4o-mini
```

如果你有本地兼容 API：

```bash
export AGENTFLOW_API_BASE=http://api-host:8000/v1
export AGENTFLOW_API_KEY=EMPTY
export AGENTFLOW_API_MODEL=your-api-model
```

## 8. 扩大训练规模

确认跑通后，可以逐步放大：

```bash
ROLLOUT_MAX_RESPONSE_LEN=16384 \
SGLANG_CONTEXT_LENGTH=65536 \
MAX_TOKENS_PER_GPU=32768 \
bash agentic/agentflow/formal_train_4xa800.sh
```

如果显存紧张，反向降低：

```bash
ROLLOUT_BATCH_SIZE=4 \
N_SAMPLES_PER_PROMPT=4 \
GLOBAL_BATCH_SIZE=32 \
ROLLOUT_MAX_RESPONSE_LEN=4096 \
MAX_TOKENS_PER_GPU=8192 \
bash agentic/agentflow/formal_train_4xa800.sh
```

## 9. 日志

默认日志目录：

```text
/tmp/agentflow_formal_4xa800
```

查看训练：

```bash
tail -f /tmp/agentflow_formal_4xa800/train.log
```

local teacher 模式查看 teacher：

```bash
tail -f /tmp/agentflow_formal_4xa800/teacher.log
```

## 10. 进程清理

外部 teacher 模式默认只清 Ray，不清本机 SGLang，避免误杀你手动启动的 teacher。需要强制清理本机 SGLang 时：

```bash
CLEAN_SGLANG=1 bash agentic/agentflow/formal_train_4xa800.sh
```

local teacher 模式默认会清理旧 SGLang，因为脚本会自己重新启动 teacher。
