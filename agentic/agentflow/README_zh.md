# AgentFlow 中文说明

基于 slime 框架复现 AgentFlow，在数学推理任务上使用 GRPO 对多步 agent 轨迹训练小 Planner。

## 当前推荐部署方式

| 角色 | 默认部署 |
|---|---|
| Planner | 本地 2B 级或更小模型，用于训练；评测时也可切到 API 大模型 |
| Executor / Verifier / base_generator | OpenAI 兼容 API |
| python_coder | OpenAI 兼容 API |
| Rewarder | OpenAI 兼容 API |

这样本地 4 张卡可以主要用于小 Planner 的训练和 rollout，不再需要旧版三路/多 GPU SGLang 服务部署。

完整流程见 [DATA_TRAIN_EVAL_ZH.md](./DATA_TRAIN_EVAL_ZH.md)，里面包含：

- 数据下载与字段格式
- HF <-> Megatron/torch_dist 转换
- 4 卡训练启动脚本和关键参数
- 本地 Planner 与 API Planner 两种评测方式
- checkpoint 批量评测
- OPD 蒸馏大模型 Planner 到 2B Planner 的项目讲法

单卡 4090 快速测试见 [SINGLE_4090_TEST_ZH.md](./SINGLE_4090_TEST_ZH.md)。

## 快速开始

### 1. 配置 API

```bash
export AGENTFLOW_API_BASE=https://api.openai.com/v1
export AGENTFLOW_API_KEY=你的_api_key
export AGENTFLOW_API_MODEL=gpt-4o-mini
```

可按角色覆盖：

```bash
export AGENTFLOW_EXECUTOR_MODEL=gpt-4o-mini
export AGENTFLOW_CODER_MODEL=gpt-4o-mini
export AGENTFLOW_REWARDER_MODEL=gpt-4o-mini
```

### 2. 启动训练

默认脚本名保留历史命名，但默认配置已经切到 1.5B/2B 级小 Planner：

```bash
cd /path/to/slime-agentic

TRAIN_GPUS=4 \
TRAIN_TP=1 \
ROLLOUT_ENGINE_GPUS=1 \
BASE_HF_CHECKPOINT=/data/models/qwen25_1.5b \
REF_LOAD=/data/models/qwen2.5_1.5b_dist/ \
SAVE_DIR=/data/AgentFlow_Qwen25-1.5B-RL/ \
bash agentic/agentflow/launch.sh
```

### 3. 评测本地 Planner

```bash
MODEL_PATH=/data/AgentFlow_Qwen25-1.5B-RL-HF/ \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
bash agentic/agentflow/eval_agentflow.sh
```

### 4. 评测 API 大模型 Planner

```bash
USE_API_FOR_PLANNER=1 \
PLANNER_API_BASE=https://api.openai.com/v1 \
PLANNER_API_KEY=你的_api_key \
PLANNER_MODEL=gpt-4o \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
bash agentic/agentflow/eval_agentflow.sh
```

## 主要脚本

| 文件 | 作用 |
|---|---|
| `launch.sh` | 推荐训练入口，只清理/启动训练和 Ray，不再启动额外多 GPU 支持模型服务 |
| `agentflow_qwen25_7b_rl_v2.sh` | slime/Ray 训练参数脚本，支持 4 卡训练小 Planner |
| `rollout.py` | 训练 rollout，Planner 本地，其他角色 API |
| `eval_agentflow.py` | 独立评测核心，支持本地 Planner 或 API Planner |
| `eval_agentflow.sh` | 推荐评测入口 |
| `eval_all_checkpoints.sh` | 批量评测 HF checkpoint |
| `DATA_TRAIN_EVAL_ZH.md` | 数据、训练、测评和 OPD 蒸馏说明 |
| `SINGLE_4090_TEST_ZH.md` | 单卡 4090 快速测试说明 |

## OPD 蒸馏建议

这里的 OPD 指 On-Policy Distillation。推荐流程不是简单“先离线生成教师轨迹再 SFT”，而是让 2B Planner 用当前策略先 rollout，收集它真实访问到的中间状态、工具选择和失败轨迹，再让 API 大模型教师对这些 on-policy 状态给出更优动作、修正轨迹或偏好信号。这样更贴近 GRPO 的训练分布，也更能体现你对在线训练和分布偏移的理解。

SFT 可以作为可选热启动，但不要把 SFT 和 OPD 混成一个概念。更好的项目叙事是：`SFT 热启动（可选） -> OPD 纠偏当前策略分布 -> GRPO 奖励优化`。
