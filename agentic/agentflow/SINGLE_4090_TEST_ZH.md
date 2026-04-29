# 单卡 4090 快速测试指南

这份文档用于你现在只有一张 4090 时先验证 AgentFlow 链路。目标不是跑完整训练，而是先确认：Planner 能启动、其他角色能走 API、一次小样本评测能产出结果、轨迹能保存并分析。

## 1. 推荐测试目标

单卡 4090 建议分三步测：

| 阶段 | 目的 | 是否需要训练 |
|---|---|---|
| API Planner 对照 | 确认数据、工具、Rewarder、评测流程能跑通 | 否 |
| 本地小 Planner 评测 | 确认单卡能部署小 Planner，其他角色走 API | 否 |
| 极小规模训练冒烟 | 确认 slime/Ray/rollout/GRPO 闭环能启动 | 可选 |

如果你只是为了准备面试项目，前两步最重要。第三步可以只跑极少样本，用来证明你理解训练闭环，不一定追求指标。

## 2. API 配置

所有非 Planner 角色默认走 API。先设置：

```bash
export AGENTFLOW_API_BASE=https://api.openai.com/v1
export AGENTFLOW_API_KEY=你的_api_key
export AGENTFLOW_API_MODEL=gpt-4o-mini
export AGENTFLOW_API_TIMEOUT=180
export AGENTFLOW_API_MAX_RETRIES=3
```

如果你有更适合代码生成或 judge 的模型，可以单独覆盖：

```bash
export AGENTFLOW_CODER_MODEL=gpt-4o-mini
export AGENTFLOW_REWARDER_MODEL=gpt-4o-mini
```

## 3. 测试一：纯 API Planner 对照

这一步不占用本地显存，适合先确认数据和评测链路。

```bash
cd /path/to/slime-agentic

USE_API_FOR_PLANNER=1 \
PLANNER_API_BASE=https://api.openai.com/v1 \
PLANNER_API_KEY=你的_api_key \
PLANNER_MODEL=gpt-4o \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
NUM_SAMPLES=5 \
CONCURRENCY=2 \
MAX_STEPS=3 \
MAX_NEW_TOKENS=2048 \
bash agentic/agentflow/eval_agentflow.sh
```

预期结果：

| 输出 | 说明 |
|---|---|
| `agentic/agentflow/eval_results.json` | 每条样本的 pred、label、score、final_output |
| 控制台 summary | 数据集准确率和耗时 |

如果这一步失败，优先检查 API key、API base、数据路径、tokenizer 路径。

## 4. 测试二：单卡本地 Planner + API 支持角色

这一步会在 4090 上启动一个本地 Planner。建议先用 0.5B/1.5B 模型，不要一开始上 7B。

```bash
cd /path/to/slime-agentic

CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/data/AgentFlow_Qwen25-1.5B-RL-HF/ \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
TP=1 \
CTX_LEN=32768 \
MEM_FRACTION=0.75 \
NUM_SAMPLES=5 \
CONCURRENCY=2 \
MAX_STEPS=3 \
MAX_NEW_TOKENS=2048 \
bash agentic/agentflow/eval_agentflow.sh
```

如果显存紧张，可以继续降低：

```bash
CTX_LEN=16384
MAX_NEW_TOKENS=1024
CONCURRENCY=1
MAX_STEPS=2
```

建议先确认本地 Planner 能完成 5 条样本，再逐步增加 `NUM_SAMPLES` 和 `MAX_STEPS`。

## 5. 测试三：保存轨迹排查 Planner 行为

保存轨迹可以帮助你讲清楚 agent 在每一步做了什么：

```bash
TRAJECTORY_DIR=/tmp/agentflow_4090_traces \
NUM_SAMPLES=3 \
CONCURRENCY=1 \
MAX_STEPS=3 \
bash agentic/agentflow/eval_agentflow.sh
```

轨迹中重点看：

| 字段 | 看什么 |
|---|---|
| `analysis` | Planner 初始分析是否有明确解题策略 |
| `steps` | 是否选择了合理工具，子目标是否清楚 |
| `tool_command` | 工具调用格式是否稳定 |
| `execution_result` | 工具结果是否被 Planner 正确吸收 |
| `final_output` | 最终答案是否遵守数学题格式 |

## 6. 可选：单卡训练冒烟

单卡 4090 做完整 GRPO 训练很吃紧，建议只做“能启动”的极小规模冒烟，不作为正式实验指标。

```bash
cd /path/to/slime-agentic

CUDA_VISIBLE_DEVICES=0 \
TRAIN_GPUS=1 \
TRAIN_TP=1 \
ROLLOUT_ENGINE_GPUS=1 \
ROLLOUT_BATCH_SIZE=1 \
N_SAMPLES_PER_PROMPT=2 \
GLOBAL_BATCH_SIZE=2 \
MAX_TOKENS_PER_GPU=4096 \
SGLANG_CONTEXT_LENGTH=16384 \
ROLLOUT_MAX_RESPONSE_LEN=4096 \
EVAL_INTERVAL=100000 \
SAVE_INTERVAL=100000 \
BASE_HF_CHECKPOINT=/data/models/qwen25_1.5b \
REF_LOAD=/data/models/qwen2.5_1.5b_dist/ \
SAVE_DIR=/data/AgentFlow_4090_smoke/ \
bash agentic/agentflow/launch.sh
```

注意事项：

| 问题 | 建议 |
|---|---|
| 4090 显存不足 | 先换 0.5B 或降低上下文、batch、response 长度 |
| 训练速度慢 | 单卡只用于验证闭环，正式指标留给 4 卡 |
| API 成本高 | `NUM_SAMPLES`、`MAX_STEPS`、`N_SAMPLES_PER_PROMPT` 都先设小 |
| 指标波动大 | 冒烟阶段不看最终准确率，只看链路是否稳定 |

## 7. OPD 是否适合这个项目

结论：适合，而且是一个很好的面试项目方向。

原因有三点：

1. 你的资源约束很真实：本地只能稳定训练/部署 2B 级 Planner，但你可以用 API 大模型当教师模型。
2. AgentFlow 的核心能力正好是“规划轨迹”，OPD 可以在学生当前 rollout 到的真实状态上让教师纠偏，而不是只模仿离线干净轨迹。
3. OPD 后再接 GRPO，能展示从 on-policy 采样、教师纠偏到奖励优化的完整训练经验，而不是只做 prompt 或只跑评测。

推荐实验设计：

| 实验组 | 目的 |
|---|---|
| `原始 2B Planner` | 小模型原始能力 |
| `API 大模型 Planner` | 教师模型上限 |
| `SFT 热启动 2B Planner，可选` | 证明格式和工具协议是否更稳 |
| `OPD 2B Planner` | 证明 on-policy 教师纠偏能改善真实 rollout |
| `OPD + GRPO 2B Planner` | 证明奖励优化能继续提升任务指标 |

单卡 4090 阶段不建议一开始完整实现大规模 OPD，可以先做小样本验证：

1. 用当前 2B Planner 跑 `NUM_SAMPLES=5` 到 `20` 的轨迹。
2. 保存失败、低分、工具调用错误或过早停止的轨迹。
3. 把这些轨迹发给 API 教师模型，让它只修正“下一步规划/工具选择/停止判断”，不要直接替学生完成整题。
4. 用修正片段构造小规模 OPD 数据，先验证学生在同类状态下是否更稳定。

OPD 数据建议优先保留有训练价值的 on-policy 样本：

| 过滤条件 | 原因 |
|---|---|
| 学生原轨迹失败或低分 | 教师纠偏的收益更明显 |
| 工具调用可解析 | 保证小模型学到可执行格式 |
| 教师修正后能提升 reward 或人工检查更合理 | 避免教师错误被学生模仿 |
| 步数不过长 | 避免 2B 学到冗长规划 |
| 同一状态有教师偏好解释 | 便于构造偏好蒸馏或步骤级监督 |

面试时可以这样讲：我让 2B Planner 先按当前策略生成真实轨迹，再用大模型 API 对这些 on-policy 状态做步骤级纠偏或偏好指导，最后用 GRPO 在真实奖励下继续优化。这个方案既考虑了资源限制，也体现了分布偏移、在线采样、教师-学生训练、评测和系统部署的完整闭环。
