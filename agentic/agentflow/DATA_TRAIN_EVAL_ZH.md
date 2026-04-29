# AgentFlow 数据、训练与测评流程

这份文档按“数据 -> 训练 -> 测评 -> 蒸馏优化”的顺序说明当前 AgentFlow 项目的推荐用法。当前部署假设是：本地最多使用 4 张卡训练和部署一个 2B 级别或更小的 Planner；Executor、Verifier、base_generator、python_coder、Rewarder 默认全部使用 OpenAI 兼容 API 服务。这样可以把本地显存集中给训练，同时保留用大模型 API 做对照实验的能力。

## 1. 角色与部署方式

| 角色 | 训练时 | 评测时 | 是否参与 Planner loss |
|---|---|---|---|
| Planner | 本地 slime/SGLang rollout，训练 2B 级小模型 | 本地小模型或 API 大模型 | 是 |
| Executor | API | API | 否 |
| Verifier | API | API | 否 |
| base_generator / final_output | API | API | 否 |
| python_coder | API | API | 否 |
| Rewarder | API | API | 否 |

注意：Planner API 模式只用于评测或大模型对照。RL 训练需要本地 Planner 返回 token、logprob 和可回传的 rollout，因此不能直接用闭源 API 模型做被训练对象。

## 2. 数据准备

训练集使用 `dapo-math-17k`，评测集默认使用 `aime-2024`。

```bash
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /data/dapo-math-17k

huggingface-cli download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /data/aime-2024
```

默认字段：

| 字段 | 说明 |
|---|---|
| `prompt` | 数学问题，支持 string 或 chat messages |
| `label` | 标准答案 |

如数据字段不同，在训练脚本中修改 `--input-key`、`--label-key`，或通过脚本变量覆盖。

## 3. 模型与格式转换

默认训练 1.5B 级 Qwen2.5 Planner。你可以把它视为 2B 级小模型配置；如后续换成真正 2B 模型，只需要提供对应的 Megatron model config 和 checkpoint 路径。

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
  --local-dir /data/models/qwen25_1.5b
```

训练前需要把 HF checkpoint 转成 slime/Megatron 使用的 torch_dist 格式：

```bash
cd /path/to/slime-agentic
source scripts/models/qwen2.5-1.5B.sh

python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /data/models/qwen25_1.5b \
  --save /data/models/qwen2.5_1.5b_dist/
```

训练后如果要用 `eval_agentflow.sh` 自动拉起本地 Planner，需要把训练产物转换回 HF 格式：

```bash
cd /path/to/slime-agentic
source scripts/models/qwen2.5-1.5B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  ${MODEL_ARGS[@]} \
  --load /data/AgentFlow_Qwen25-1.5B-RL/ \
  --hf-checkpoint /data/models/qwen25_1.5b \
  --save /data/AgentFlow_Qwen25-1.5B-RL-HF/
```

## 4. API 配置

所有非 Planner 角色默认走 API。最小配置如下：

```bash
export AGENTFLOW_API_BASE=https://api.openai.com/v1
export AGENTFLOW_API_KEY=你的_api_key
export AGENTFLOW_API_MODEL=gpt-4o-mini
```

也可以按角色单独指定：

```bash
export AGENTFLOW_EXECUTOR_MODEL=gpt-4o-mini
export AGENTFLOW_CODER_MODEL=gpt-4o-mini
export AGENTFLOW_REWARDER_MODEL=gpt-4o-mini
export AGENTFLOW_API_TIMEOUT=180
export AGENTFLOW_API_MAX_RETRIES=3
```

API 调用已经在 `core/llm_engine.py` 中加入了有上限的重试、指数退避、空候选结果检查、空响应检查和可配置超时。第三方兼容服务也可以使用同一套变量，只要实现 `/chat/completions`。

## 5. 训练启动

推荐入口：

```bash
cd /path/to/slime-agentic

export AGENTFLOW_API_BASE=https://api.openai.com/v1
export AGENTFLOW_API_KEY=你的_api_key
export AGENTFLOW_API_MODEL=gpt-4o-mini

bash agentic/agentflow/launch.sh
```

`launch.sh` 会清理旧 Ray 和旧 SGLang 进程，然后启动 `agentflow_qwen25_7b_rl_v2.sh`。脚本名保留了历史命名，但默认配置已经切到 1.5B/2B 级小 Planner。

关键参数：

| 变量 / 参数 | 默认值 | 说明 |
|---|---:|---|
| `TRAIN_GPUS` | `4` | 训练可见 GPU 数，适合你后续 4 卡训练 |
| `TRAIN_TP` | `1` | 小 Planner 默认不做 tensor parallel，把 4 卡留给数据并行/训练吞吐 |
| `ROLLOUT_ENGINE_GPUS` | `1` | Planner rollout engine 使用 1 张卡 |
| `BASE_HF_CHECKPOINT` | `/data/models/qwen25_1.5b` | 原始 HF 模型 |
| `REF_LOAD` | `/data/models/qwen2.5_1.5b_dist/` | torch_dist 参考模型 |
| `SAVE_DIR` | `/data/AgentFlow_Qwen25-1.5B-RL/` | 训练输出目录 |
| `PROMPT_DATA` | `/data/dapo-math-17k/dapo-math-17k.jsonl` | 训练数据 |
| `ROLLOUT_BATCH_SIZE` | `8` | rollout batch size |
| `N_SAMPLES_PER_PROMPT` | `8` | GRPO 每题采样条数 |
| `GLOBAL_BATCH_SIZE` | `64` | 全局 batch size |
| `ROLLOUT_TEMPERATURE` | `0.7` | rollout 采样温度 |
| `MAX_TOKENS_PER_GPU` | `16384` | 动态 batch 的单卡 token 上限 |
| `EVAL_INTERVAL` | `20` | 训练中自动评测间隔 |

常用启动示例：

```bash
TRAIN_GPUS=4 \
TRAIN_TP=1 \
ROLLOUT_ENGINE_GPUS=1 \
BASE_HF_CHECKPOINT=/data/models/qwen25_1.5b \
REF_LOAD=/data/models/qwen2.5_1.5b_dist/ \
SAVE_DIR=/data/AgentFlow_Qwen25-1.5B-RL/ \
bash agentic/agentflow/launch.sh
```

如需要保存轨迹排查 Planner 行为：

```bash
SAVE_TRAJECTORY=1 bash agentic/agentflow/launch.sh
```

## 6. 测评

### 6.1 评测本地训练后的 Planner

默认只自动启动一个本地 Planner SGLang，其他角色走 API：

```bash
MODEL_PATH=/data/AgentFlow_Qwen25-1.5B-RL-HF/ \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
bash agentic/agentflow/eval_agentflow.sh
```

快速 smoke test：

```bash
NUM_SAMPLES=5 CONCURRENCY=4 bash agentic/agentflow/eval_agentflow.sh
```

### 6.2 用 API 大模型作为 Planner 对照

这个模式不启动任何本地 Planner，用于比较“2B Planner vs 大模型 Planner”的 agent 规划能力：

```bash
USE_API_FOR_PLANNER=1 \
PLANNER_API_BASE=https://api.openai.com/v1 \
PLANNER_API_KEY=你的_api_key \
PLANNER_MODEL=gpt-4o \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
bash agentic/agentflow/eval_agentflow.sh
```

`TOKENIZER_PATH` 仍然需要提供，因为框架会用本地 tokenizer 统一渲染 prompt 和记录 token 信息。API Planner 评测不参与训练 loss，所以不会依赖 API 的 token logprob。

### 6.3 批量评测 checkpoint

先把每个 `iter_*` 转成 HF 格式，然后运行：

```bash
CHECKPOINT_DIR=/data/AgentFlow_Qwen25-1.5B-RL-HF \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
bash agentic/agentflow/eval_all_checkpoints.sh
```

脚本会逐个启动本地 Planner，并复用 API 支持角色。结果写入：

```text
agentic/agentflow/checkpoint_eval_results.jsonl
```

## 7. OPD 是否适合这个项目

适合。这里的 OPD 指 On-Policy Distillation：学生 Planner 先用当前策略 rollout，产生它真实会访问到的状态、工具选择和中间轨迹；教师大模型再针对这些 on-policy 样本给出更优的下一步规划、工具选择、停止判断、最终答案整合，或者给出偏好比较信号。核心不是“离线模仿一批教师轨迹”，而是“在学生当前分布上让教师纠偏”。

这比普通离线蒸馏更适合 AgentFlow，因为小 Planner 在训练中遇到的状态分布会不断变化。离线教师轨迹通常很干净，但学生 rollout 里会有错误上下文、错误工具结果、半完成推理和格式抖动。OPD 正好可以覆盖这些真实错误状态，降低训练和推理之间的分布偏移。

推荐训练链路：

1. 可选 SFT 热启动：用少量高质量教师轨迹让 2B Planner 学会基本输出格式和工具协议。
2. On-policy 采样：用当前 2B Planner 在训练题上 rollout，保存 `analysis`、`next_step`、工具调用、工具结果、Verifier 结论和 `final_output`。
3. 教师纠偏：把学生当前轨迹和中间状态发给 API 大模型，让它给出更优下一步、修正后的规划片段，或两条轨迹的偏好判断。
4. OPD 更新：用教师输出对学生做蒸馏更新，重点学习学生自己容易走到的状态下应该如何恢复、继续规划、停止或整合答案。
5. GRPO 奖励优化：在 OPD 之后继续用真实 reward 做强化学习，让模型不只是“学教师”，还要为最终任务指标优化。

推荐对照实验：

| 实验组 | 目的 |
|---|---|
| 原始 2B Planner | 小模型原始规划能力 |
| SFT 热启动 2B Planner，可选 | 验证格式和工具协议是否更稳 |
| OPD 2B Planner | 验证 on-policy 教师纠偏是否改善真实 rollout |
| OPD + GRPO 2B Planner | 验证蒸馏后再强化学习是否继续提升指标 |
| API 大模型 Planner | 作为教师上限和系统上限参考 |

需要注意的风险：

| 风险 | 处理方式 |
|---|---|
| 教师纠偏成本高 | 单卡 4090 阶段只采小样本，优先采失败或低分轨迹 |
| 教师直接给完整答案，学生学不到规划 | Prompt 中要求教师只修正当前步骤或给偏好解释，保留学生执行闭环 |
| 学生 on-policy 轨迹太差 | 先做很小规模 SFT 热启动，保证基本格式和工具调用可用 |
| 蒸馏压制探索 | OPD 后继续 GRPO，并在 rollout 保留温度采样 |
| 教师偏好与 reward 不一致 | 最终仍以 Rewarder/规则判分和 AIME 准确率做主指标 |

面试中的讲法可以更精准：我不是简单离线蒸馏大模型答案，而是让 2B Planner 按当前策略生成轨迹，再用大模型对这些 on-policy 状态进行纠偏和偏好指导，最后接 GRPO 做任务奖励优化。这个方案能体现我理解分布偏移、在线采样、教师-学生训练和强化学习闭环。
