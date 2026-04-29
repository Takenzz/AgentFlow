# AgentFlow 中文说明

本目录是基于 slime 框架的 AgentFlow 复现，用 GRPO 训练一个小规模 Planner，让它在数学推理任务中学会多步规划、调用工具、整合最终答案。

## 当前推荐部署方式

| 角色 | 默认部署方式 |
|---|---|
| Planner | 本地 2B 级或更小模型，用于训练；评测时也可切到 API 大模型 |
| Executor / Verifier / base_generator | OpenAI 兼容 API |
| python_coder | OpenAI 兼容 API |
| Rewarder | OpenAI 兼容 API |

这样做的核心目标是节省本地显存：本地 GPU 只承担小 Planner 的训练和 rollout，其他辅助角色全部交给 API。旧版“三个本地 SGLang 服务、多组 GPU 部署”的路径已经不再作为推荐方式。

## 文档入口

| 文档 | 内容 |
|---|---|
| [DATA_TRAIN_EVAL_ZH.md](./DATA_TRAIN_EVAL_ZH.md) | 数据、模型转换、训练、评测、OPD 蒸馏的完整流程 |
| [SINGLE_4090_TEST_ZH.md](./SINGLE_4090_TEST_ZH.md) | 单卡 4090 快速测试方案 |
| [README_zh.md](./README_zh.md) | 简短中文入口，与本文件保持一致 |

## 最短训练命令

```bash
cd /path/to/slime-agentic

export AGENTFLOW_API_BASE=https://api.openai.com/v1
export AGENTFLOW_API_KEY=你的_api_key
export AGENTFLOW_API_MODEL=gpt-4o-mini

TRAIN_GPUS=4 \
TRAIN_TP=1 \
ROLLOUT_ENGINE_GPUS=1 \
BASE_HF_CHECKPOINT=/data/models/qwen25_1.5b \
REF_LOAD=/data/models/qwen2.5_1.5b_dist/ \
SAVE_DIR=/data/AgentFlow_Qwen25-1.5B-RL/ \
bash agentic/agentflow/launch.sh
```

## 单卡 4090 快速评测

如果只是先验证 AgentFlow 链路，不建议一上来跑完整训练。推荐先用单卡 4090 跑本地小 Planner 评测，并把其他角色全部切到 API：

```bash
NUM_SAMPLES=5 \
CONCURRENCY=2 \
TP=1 \
CTX_LEN=32768 \
MAX_NEW_TOKENS=2048 \
MODEL_PATH=/data/AgentFlow_Qwen25-1.5B-RL-HF/ \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
bash agentic/agentflow/eval_agentflow.sh
```

也可以不启动本地模型，直接用 API 大模型作为 Planner 上限对照：

```bash
USE_API_FOR_PLANNER=1 \
PLANNER_API_BASE=https://api.openai.com/v1 \
PLANNER_API_KEY=你的_api_key \
PLANNER_MODEL=gpt-4o \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
NUM_SAMPLES=5 \
bash agentic/agentflow/eval_agentflow.sh
```

## OPD 蒸馏结论

这里的 OPD 指的是 On-Policy Distillation，即让当前 2B Planner 先按自己的策略采样轨迹，再让 API 大模型在这些“学生真实会遇到的状态”上给出更优规划、动作修正或偏好信号。它不是简单离线收集大模型轨迹再做 SFT。

这个方向适合本项目：小 Planner 的问题往往不是完全不会生成，而是在自己 rollout 到的中间状态里容易选错工具、过早停止或最终整合不稳。OPD 正好能让教师模型针对学生当前分布纠偏，比纯离线蒸馏更贴近后续 GRPO 的训练分布。

推荐实验链路是：

1. `原始 2B Planner`
2. `可选 SFT 热启动后的 2B Planner`
3. `OPD 后的 2B Planner`
4. `OPD + GRPO 后的 2B Planner`
5. `API 大模型 Planner`，作为上限参考

这条线能体现数据构造、蒸馏、强化学习、评测对照和资源约束下的系统拆分能力。
