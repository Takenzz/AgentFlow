# ToolOrchestra — Agentic RL with Slime

基于 [ToolOrchestra](https://arxiv.org/abs/2511.21689) 与 slime 框架的复现。**Orchestrator-Expert** 多智能体框架，用于 RL 训练。中心化的 Orchestrator LLM 通过多轮工具调用学习将任务路由到最佳的专业专家模型和对应的工具。GRPO 施加在 Orchestrator 的决策轨迹上，使其在无需人工标注中间步骤的情况下提升工具调用与路由能力。

[English](./README.md)

---

## 结果

| 模型 | 数据集 | 基线（Qwen3-8B） | ToolOrchestra（Ours） | 提升 |
|---|---|---|---|---|
| Qwen3-8B | τ²-Bench | 0.278 | 0.388 | +0.110 |

---

## 0. 前置条件

### LLM API Key

训练和评测中，τ² 用户模拟器（User Simulator）以及 QA 奖励评判模型均通过百炼（DashScope）API 调用 LLM。运行前需配置环境变量：

```bash
export DASHSCOPE_API_KEY=<your-dashscope-api-key>

# 可选覆盖（有默认值，通常无需修改）
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export DASHSCOPE_MODEL=qwen-turbo-latest       # 用户模拟器使用的模型
export QA_REWARD_JUDGE_MODEL=qwen-turbo-latest  # QA 奖励评判模型
```

> 百炼 API Key 可在 [百炼控制台](https://bailian.console.aliyun.com/) 申请。

### 环境安装

本目录提供两个 conda 环境依赖文件：

- `orche_requirement.txt` — **Orchestrator** 环境（`orche`）的依赖
- `sglang_requirement.txt` — **SGLang** 专家服务环境（`sglang`）的依赖

创建并配置环境：

```bash
# 创建并激活 orche 环境
conda create -n orche python=3.10 -y
conda activate orche
pip install -r agentic/ToolOrchestra/orche_requirement.txt

# 创建并激活 sglang 环境
conda create -n sglang python=3.10 -y
conda activate sglang
pip install -r agentic/ToolOrchestra/sglang_requirement.txt
```

> 检索服务使用 `orche` 环境；启动专家 SGLang 服务时使用 `sglang` 环境。

---

## 1. 下载模型

```bash
huggingface-cli download Qwen/Qwen3-8B \
  --local-dir /data/models/qwen3_8b
```

### 检索服务（FAISS 稠密检索）

HTTP 检索服务（`retrieval_general_thought.py`，端口 8000）使用 **[Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)** 对查询编码，并在 FAISS 索引与 JSONL 语料上做检索。

**代码中的默认路径**（[`retrieval_general_thought.py`](./retrieval_general_thought.py)）：

| 项 | 默认值 |
|---|---|
| 嵌入模型权重 | `/data/models/qwen3_8b_emb`（与 `Qwen/Qwen3-Embedding-8B` 一致；代码也支持直接传 HF 模型 id 字符串） |
| 索引目录（环境变量 `INDEX_DIR`） | `/data/dataset/index`（[`launch.sh`](./launch.sh)、[`eval_orchestra.sh`](./eval_orchestra.sh) 中已设置） |
| FAISS 索引文件 | `{INDEX_DIR}/train.index` |
| 段落语料 | `{INDEX_DIR}/train.jsonl` |

**语料来源：** 从 Hugging Face 数据集 **[multi-train/index](https://huggingface.co/datasets/multi-train/index)** 下载 `train.index` 与 `train.jsonl`，放到上述 `INDEX_DIR` 下（默认即 `/data/dataset/index`）。

示例：

```bash
# 嵌入模型（与 retrieval_general_thought.py 中的默认路径一致）
huggingface-cli download Qwen/Qwen3-Embedding-8B \
  --local-dir /data/models/qwen3_8b_emb

# FAISS 索引 + 语料（训练侧检索使用 train.*；勿整库拉取，wiki.* 体积很大）
mkdir -p /data/dataset/index
huggingface-cli download multi-train/index \
  --repo-type dataset \
  --local-dir /data/dataset/index \
  train.index train.jsonl
```

启动检索服务时可用环境变量 `INDEX_DIR` 覆盖索引目录；嵌入模型路径可在 `retrieval_general_thought.py` 里修改 `retrieval_model_path`（或自行改为从环境变量读取）。

## 2. 转换模型格式

slime 训练需要将 HuggingFace checkpoint 转换为 Megatron 分布式格式：

```bash
cd /path/to/slime-agentic
source scripts/models/qwen3-8B.sh

python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /data/models/qwen3_8b \
  --save /data/qwen3_8b_dist/
```

训练结束后转换回 HuggingFace 格式：

```bash
cd /path/to/slime-agentic
source scripts/models/qwen3-8B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  ${MODEL_ARGS[@]} \
  --load /data/checkpoints/orchestra_qwen3_8b_rl/ \
  --hf-checkpoint /data/models/qwen3_8b \
  --save /data/orchestra_qwen3_8b_hf/
```

- `--load`：训练产出的 torch_dist checkpoint 路径
- `--hf-checkpoint`：原始 HuggingFace 模型路径（用于补全配置文件）
- `--save`：转换后的 HuggingFace 格式保存路径

## 3. 启动训练

### 3.1 一键启动所有服务

ToolOrchestra 训练前需要检索服务和多个专家 SGLang 服务。推荐使用 `launch.sh` 统一管理：

```bash
bash agentic/ToolOrchestra/launch.sh
```

`launch.sh` 会在 GPU 0–3 启动所有专家服务，随后在 GPU 4–7 自动开始训练。

| GPU | 服务 | 端口 |
|---|---|---|
| 0 | 检索服务（FAISS） | 8000 |
| 0 | Qwen3-32B-FP8 | 30001 |
| 0 | Qwen2.5-Math-7B | 30003 |
| 1 | DeepSeek-R1-Distill-32B | 30005 |
| 1 | Qwen3-30B-A3B | 30006 |
| 2 | Qwen2.5-Coder-32B | 30002 |
| 2 | Qwen3-14B | 30007 |
| 3 | Qwen2.5-Math-72B | 30004 |
| 4–7 | 训练：Qwen3-8B Orchestrator（TP=2, DP=2） | — |

### 3.2 单独启动训练

专家服务已在运行时，可单独启动训练：

```bash
cd /path/to/slime-agentic
CUDA_VISIBLE_DEVICES=4,5,6,7 SKIP_PROCESS_KILL=1 \
  bash agentic/ToolOrchestra/train_orchestra.sh
```

训练权重默认保存至 `/data/checkpoints/orchestra_qwen3_8b_rl/`，可在脚本的 `CKPT_ARGS` 部分修改。

## 4. 关键训练参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--advantage-estimator` | `grpo` | 优势估计算法 |
| `--lr` | `1e-6` | 学习率 |
| `--n-samples-per-prompt` | `8` | 每条 prompt 采样数 |
| `--rollout-batch-size` | `32` | rollout batch size |
| `--global-batch-size` | `128` | 全局 batch size |
| `--rollout-temperature` | `0.7` | 采样温度 |
| `--rollout-max-response-len` | `16384` | Orchestrator 单次最大生成 token 数 |
| `--kl-loss-coef` | `0.001` | KL 散度系数 |
| `--eps-clip` / `--eps-clip-high` | `0.2` / `0.3` | PPO clip 范围 |
| `--sglang-context-length` | `131072` | SGLang 上下文长度（128K） |

## 5. 评测

### 5.1 转换 Checkpoint 为 HF 格式

评测前，使用一键转换脚本将训练保存的 torch_dist checkpoint 转换为 HuggingFace 格式：

```bash
# 转换最新 checkpoint
bash agentic/ToolOrchestra/convert_to_hf.sh

# 转换指定 iter
SINGLE_ITER=iter_0000129 bash agentic/ToolOrchestra/convert_to_hf.sh

# 转换所有 checkpoint
CONVERT_ALL=1 bash agentic/ToolOrchestra/convert_to_hf.sh

# 转换后直接运行评测
bash agentic/ToolOrchestra/convert_to_hf.sh && \
  ORCH_CKPT=/data/checkpoints/orchestra_qwen3_8b_rl_hf/iter_0000129 bash agentic/ToolOrchestra/eval_orchestra.sh
```

转换结果默认保存至 `/data/checkpoints/orchestra_qwen3_8b_rl_hf/`，已转换的 iter 会自动跳过。

### 5.2 自动启动服务并评测

```bash
# 评测 tau2 benchmark（默认）
ORCH_CKPT=/data/checkpoints/orchestra_qwen3_8b_rl_hf/iter_0000129 bash agentic/ToolOrchestra/eval_orchestra.sh

# 评测 FRAMES benchmark
ORCH_CKPT=/data/checkpoints/orchestra_qwen3_8b_rl_hf/iter_0000129 BENCHMARK=frames bash agentic/ToolOrchestra/eval_orchestra.sh

# 评测 HLE benchmark
ORCH_CKPT=/data/checkpoints/orchestra_qwen3_8b_rl_hf/iter_0000129 BENCHMARK=hle bash agentic/ToolOrchestra/eval_orchestra.sh

# 快速冒烟测试（5 条样本）
ORCH_CKPT=/data/checkpoints/orchestra_qwen3_8b_rl_hf/iter_0000129 MAX_EXAMPLES=5 bash agentic/ToolOrchestra/eval_orchestra.sh
```

结果保存至 `/data/eval_results/{benchmark}_{timestamp}/`。

### 5.2 跳过服务启动（服务已在运行）

```bash
# 专家服务已就绪，仅自动启动 Orchestrator
SKIP_EXPERT_SERVICES=1 ORCH_CKPT=/data/checkpoints/orchestra_qwen3_8b_rl bash eval_orchestra.sh

# 所有服务均已就绪
SKIP_SERVICES=1 ORCH_URL=http://127.0.0.1:30000/v1 \
  ORCH_MODEL=Qwen/Qwen3-8B bash eval_orchestra.sh
```

## 6. 架构

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
        │                               ├─ Qwen3-32B-FP8      (port 30001)
        │                               ├─ Qwen2.5-Coder-32B  (port 30002)
        │                               ├─ Qwen2.5-Math-7B    (port 30003)
        │                               ├─ Qwen2.5-Math-72B   (port 30004)
        │                               ├─ DeepSeek-R1-32B    (port 30005)
        │                               ├─ Qwen3-30B-A3B      (port 30006)
        │                               └─ Qwen3-14B          (port 30007)
        │
        └─ answer ──────────────────► 输出最终答案 → 终止循环
  │
  ▼
GenerationOutput
  - token_ids + log_probs（所有轮次拼接）
  - loss_mask：Orchestrator 输出 = 1 / 工具结果 = 0
```

**两种任务模式：**
- **QA**：Orchestrator 自主驱动循环，直到调用 `answer` 为止
- **Func_call**：循环由 tau2 仿真环境驱动（基于文件的进程间通信）

## 7. 训练配置

- **算法**：GRPO，KL 散度约束（`low_var_kl`）
- **Orchestrator 模型**：Qwen3-8B（可替换）
- **专家模型**：固定的专业 LLM，运行在独立 SGLang 端口上
- **检索服务**：基于 FAISS 的稠密向量检索
- **任务类型**：QA 推理 + 函数调用仿真（tau2）
- **推理引擎**：SGLang
