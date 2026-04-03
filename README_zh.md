# Agentic RL Reproductions with Slime
![Slime agentic icon](./imgs/icon.png)
基于 [slime](https://github.com/THUDM/slime) 框架，复现各类 **Agentic RL** 训练方法。

slime 提供了灵活的自定义 `generate` 函数与 reward 函数接口，使得将多步 agent rollout 接入 RL 训练流程变得简洁高效。本仓库在此基础上实现并复现若干有代表性的 Agentic RL 算法，所有实现均位于 [`agentic/`](./agentic) 目录下。

---

## 已实现方法

### AgentFlow — `agentic/agentflow/`

复现 [AgentFlow](https://arxiv.org/abs/2510.05592) 的核心思路：将单步 LLM 推理扩展为多轮 **Planner → Executor → Verifier** 的 agent 循环，并对 Planner 的生成轨迹施加 RL 信号（GRPO），从而在不依赖人工标注中间步骤的情况下提升模型的工具调用与推理能力。

#### 架构

```
问题输入
  │
  ▼
Planner.plan()          ← 分析问题，制定解题思路（loss_mask=1，参与训练）
  │
  └─► for step in range(max_steps):
        │
        ├─ Planner.generate_next_step()   ← 选择下一步工具及子目标（loss_mask=1）
        ├─ Executor.generate_tool_command() + execute_command()  ← 调用工具（不计入序列）
        ├─ Verifier.verificate_context()  ← 判断是否继续（不计入序列）
        └─ Memory.add_action()            ← 记录执行结果
  │
  ▼
Planner.generate_final_output()  ← 汇总结果，输出最终答案（loss_mask=0）
  │
  ▼
Rewarder.compute_reward()        ← LLM-as-Judge，对比模型答案与 ground truth
```

#### 工具（`tools/`）

| 工具名 | 描述 |
|---|---|
| `base_generator` | 通用文本生成工具，基于 LLM 直接回答子任务 |
| `python_coder` | Python 代码生成与执行工具，用于数学计算、算法求解等 |

#### 实验结果

| 模型 | 数据集 | Baseline | AgentFlow（复现） | 提升 |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | AIME 2024 | 10.0% | 30.0% | +20.0% |


训练后的模型权重已发布至 HuggingFace：[LMIS-ORG/AgentFlow_Slime_Agentic_Qwen2.5_7B](https://huggingface.co/LMIS-ORG/AgentFlow_Slime_Agentic_Qwen2.5_7B/tree/main)

#### 训练配置

- **算法**：GRPO，KL 散度约束（`low_var_kl`）
- **模型**：Qwen2.5-7B（可替换）
- **数据**：DAPO-Math-17K
- **推理引擎**：SGLang（Planner / Executor 分别使用独立端口）
- **评测集**：AIME 2024

#### 快速启动

具体的训练参数与启动方式均在 [`agentic/agentflow/`](./agentic/agentflow) 目录下。

---

### MemAgent — `agentic/memagent/`
训练后的模型权重已发布至 HuggingFace：[LMIS-ORG/MemAgent_Slime_Agentic_Qwen2.5_7B](https://huggingface.co/LMIS-ORG/MemAgent_Slime_Agentic_Qwen2.5_7B)

复现 [MemAgent](https://arxiv.org/abs/2507.02259) 的核心思路：通过逐 chunk 的 LLM 更新循环，将任意长度的文档压缩进固定大小的**循环记忆（recurrent memory）**，最终仅凭记忆回答问题。RL（GRPO）作用于所有记忆更新轮次，使用 **Multi-Conversation** 训练目标，让模型在不同时刻看到不同 chunk 时学会保留关键信息。

#### 架构

```
输入：问题 + 长文档
  │
  ▼
memory = "No previous memory"
  │
  └─► for chunk in split(document, chunk_tokens):
        │
        └─ LLM(problem, memory, chunk) → 更新后的 memory   (loss_mask=1，参与训练)
  │
  ▼
LLM(problem, memory) → 最终答案 \boxed{}                  (loss_mask=0)
  │
  ▼
Reward：与 ground truth 的精确匹配 / F1 分数
        （通过 custom_convert 均摊至所有记忆更新轮次）
```

每个记忆更新轮次作为独立训练序列；奖励在整个对话的所有轮次中均分（via `custom_convert`），与论文中 Multi-Conv RL 目标一致。

#### 实验结果

在 **RULER-HQA** 上，对 7K 至 448K token 的上下文长度进行评测：

| 模型                    | 7K    | 14K   | 28K   | 56K   | 112K  | 224K  | 448K  |
|-------------------------|-------|-------|-------|-------|-------|-------|-------|
| **MemAgent（复现）**    | **78.12** | **76.56** | **75.78** | **74.22** | **77.34** | **72.66** | **69.53** |
| QwenLong-L1-32B         | 72.66 | 75.00 | 72.66 | 60.94 | 31.25 | 17.19 | 13.28 |
| Qwen2.5-Instruct-14B-1M | 60.16 | 60.94 | 50.00 | 57.03 | 50.00 | 37.50 | 8.59  |
| Qwen2.5-Instruct-7B-1M  | 61.72 | 56.25 | 53.91 | 55.47 | 51.56 | 33.59 | 12.50 |
| DS-Distill-Qwen-32B     | 70.31 | 66.41 | 65.62 | 46.88 | 23.44 | 13.28 | 7.81  |
| DS-Distill-Qwen-14B     | 64.06 | 64.84 | 57.03 | 40.62 | 14.84 | 8.59  | 3.12  |
| DS-Distill-Qwen-7B      | 30.47 | 12.50 | 3.12  | 0.00  | 0.00  | 0.78  | 0.00  |

MemAgent（复现）基于 **7B** 模型，在全部长度上均大幅领先其他基线（含更大规模模型）。
训练后的模型权重已发布至 HuggingFace：[LMIS-ORG/MemAgent_Slime_Agentic_Qwen2.5_7B](https://huggingface.co/LMIS-ORG/MemAgent_Slime_Agentic_Qwen2.5_7B)

#### 训练配置

- **算法**：GRPO，KL 散度约束（`low_var_kl`）
- **模型**：Qwen2.5-7B（可替换）
- **数据**：HotpotQA（[BytedTsinghua-SIA/hotpotqa](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa)）
- **推理引擎**：SGLang，启用 YaRN（`--sglang-context-length 131072`）
- **评测集**：RULER-HQA（7K → 448K）

#### 快速启动

具体的训练参数与启动方式均在 [`agentic/memagent/`](./agentic/memagent) 目录下。

---

### ToolOrchestra — `agentic/ToolOrchestra/`

复现 [ToolOrchestra](https://arxiv.org/abs/2511.21689) 的核心思路：**Orchestrator-Expert** 多智能体框架，用于 RL 训练。中心化的 Orchestrator LLM 通过多轮工具调用学习将任务路由到最佳的专业专家模型和对应的工具。GRPO 施加在 Orchestrator 的决策轨迹上，使其在无需人工标注中间步骤的情况下提升工具调用与路由能力。

#### 架构

```
问题输入
  │
  ▼
Orchestrator LLM                        ← 决定调用哪个工具（loss_mask=1，参与训练）
  │
  └─► for turn in range(max_turns):
        │
        ├─ parse_tool_call()            ← 解析模型输出中的 <tool_call>
        │
        ├─ 工具调用                      ← 调用检索 / 外部工具（loss_mask=0）
        │    └─ FAISS 检索服务（port 8000）
        │
        ├─ call_expert ──────────────► 专家模型路由（loss_mask=0）
        │                               └─ 各专业模型运行于独立端口
        │
        └─ answer ──────────────────► 输出最终答案 → 终止循环
  │
  ▼
GenerationOutput
  - token_ids + log_probs（所有轮次拼接）
  - loss_mask：Orchestrator 输出 = 1 / 工具结果 = 0
```

#### 实验结果

| 模型 | 数据集 | 基线（Qwen3-8B） | ToolOrchestra（复现） | 提升 |
|---|---|---|---|---|
| Qwen3-8B | τ²-Bench | 0.278 | 0.388 | +0.110 |

#### 训练配置

- **算法**：GRPO，KL 散度约束（`low_var_kl`）
- **模型**：Qwen3-8B（可替换）
- **数据**：τ²-Bench
- **推理引擎**：SGLang（各专家模型运行在独立端口）
- **评测集**：τ²-Bench

#### 快速启动

具体的训练参数与启动方式均在 [`agentic/ToolOrchestra/`](./agentic/ToolOrchestra) 目录下。

---

## 框架接入方式

每个方法通过 slime 的三个自定义钩子接入训练流程：

```bash
--custom-generate-function-path  rollout.generate     # 自定义多步 rollout
--custom-rm-path                 rollout.reward_func  # 自定义 reward 计算
--custom-eval-rollout-log-function-path rollout.eval_log  # 自定义评测日志
```

`generate` 函数负责构造完整的 agent 轨迹并返回带 `loss_mask` 的 token 序列；仅 Planner 的生成 token 参与梯度计算，工具调用结果与最终输出的注入部分 mask 置 0。

---

## 环境依赖

推荐通过官方 Docker 镜像部署环境：

```bash
# 拉取最新镜像
docker pull slimerl/slime:latest

# 启动容器
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:latest /bin/bash
```

手动安装请参见项目根目录 [`requirements.txt`](./requirements.txt) 及 [`docker/`](./docker)。

核心依赖：

- `slime >= 0.2.2`
- `sglang`
- `ray`
- `transformers`
- `torch >= 2.0`

---

## 计划复现


欢迎 PR 或 Issue 讨论新方法的接入。
