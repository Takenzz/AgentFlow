# 环境与数据准备

这份文档说明如何把最小 AgentFlow 仓库接到外部安装的 slime 训练环境。

## 1. 依赖边界

本仓库提供：

- AgentFlow 多轮 rollout。
- Planner turn 拆分后的 custom convert。
- OpenAI-compatible API engine。
- 两个内置工具。
- 训练 launcher 和 quick test 脚本。

本仓库不再提供：

- slime 源码。
- Megatron-LM。
- SGLang 训练/推理依赖。
- HF/Megatron checkpoint 转换工具。
- 大模型结构脚本全集。

## 2. 安装检查

在你准备训练的 Python 环境里确认：

```bash
python3 -c "import slime; print(slime.__file__)"
python3 -c "import ray, torch, sglang, openai, aiohttp"
```

如果 `slime` 或 `Megatron-LM` 不是 pip 安装，而是源码目录：

```bash
export SLIME_PATH=/path/to/slime
export MEGATRON_PATH=/path/to/Megatron-LM
```

`agentic/agentflow/train_agentflow.sh` 会把这些路径加入 Ray runtime 的 `PYTHONPATH`。

## 3. 模型准备

训练需要两个 student 路径：

- `BASE_HF_CHECKPOINT`: student 的原始 HF checkpoint。
- `REF_LOAD`: student 转换后的 slime/Megatron `torch_dist` checkpoint。

如果你的 slime 源码在 `$SLIME_REPO`，可以按 slime 自己的转换工具执行：

```bash
export SLIME_REPO=/path/to/slime
export MEGATRON_PATH=/path/to/Megatron-LM

source agentic/agentflow/model_configs/qwen3-0.6B.sh

PYTHONPATH=$MEGATRON_PATH:$SLIME_REPO \
python3 $SLIME_REPO/tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /data/models/student_hf \
  --save /data/models/student_torch_dist
```

如果 student 不是 Qwen3-0.6B/Qwen2.5-0.5B/Qwen2.5-1.5B，请新增一个匹配模型结构的脚本到 `agentic/agentflow/model_configs/`，并在训练时设置 `MODEL_CONFIG_SCRIPT`。

## 4. Teacher 要求

当前 quick test 使用 OPD SGLang 模式：

- teacher 通过 SGLang `/generate` 提供每个 token 的 logprob。
- `rollout.reward_func` 会把 student 生成的 token 序列发给 teacher。
- teacher 和 student 必须使用兼容 tokenizer，最好是同一模型家族，否则 logprob 对齐没有意义。

如果你使用 2B teacher、0.8B student，建议二者来自同一系列，例如同 tokenizer 的 Qwen 系列小模型。

## 5. 数据格式

默认 JSONL 字段：

```json
{"prompt": "question text", "label": "answer"}
```

默认训练参数使用：

```bash
INPUT_KEY=prompt
LABEL_KEY=label
```

如果你的 prompt 已经是 OpenAI messages 结构，设置：

```bash
APPLY_CHAT_TEMPLATE=1
INPUT_KEY=messages
```
