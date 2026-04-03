# AgentFlow — Agentic RL with Slime

基于 slime 框架复现 AgentFlow，在数学推理任务上使用 GRPO 对多步 agent 轨迹进行强化学习训练。

[English](./README.md)

---

> **启动失败排查**
> 最常见的原因是模型路径与实际不符。所有脚本使用以下默认路径：
> - 基础模型：`/data/models/qwen25_7b`
> - Coder 模型：`/data/models/qwen2.5_7b_codeer`
>
> 如果你的模型存放在其他位置，请修改对应脚本顶部的路径变量，或通过环境变量传入（例如 `MODEL_BASE=/your/path bash launch.sh`）。

---

## 0. 环境安装

本目录提供一个 conda 环境依赖文件：

- `sglang_requirement.txt` — **SGLang** 推理环境（`sglang`）的依赖

创建并配置环境：

```bash
conda create -n sglang python=3.10 -y
conda activate sglang
pip install -r agentic/agentflow/sglang_requirement.txt
```

> 训练和评测阶段启动 SGLang 推理服务时，均使用 `sglang` 环境。

## 1. 下载数据集

```bash
# 下载训练集（dapo-math-17k）
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /data/dapo-math-17k

# 下载评测集（aime-2024）
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /data/aime-2024
```

## 2. 下载模型

```bash
# 基础模型（Planner / Executor / Verifier）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/models/qwen25_7b

# Coder 模型（Executor 代码生成）
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir /data/models/qwen2.5_7b_codeer
```

## 3. 转换模型格式

slime 训练需要将 HuggingFace checkpoint 转换为 Megatron 分布式格式：

```bash
cd /path/to/slime-agentic
source scripts/models/qwen2.5-7B.sh 
python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /data/models/qwen25_7b \
  --save /data/qwen2.5_7b_dist/
```

将 Megatron 分布式格式模型转换为 HuggingFace 格式：
```bash
cd /path/to/slime-agentic
source scripts/models/qwen2.5-7B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py   --input-dir /data/AgentFlow_Qwen25-7B-RL/iter_00000xxx/   --output-dir /data/agentflow_xxx_hf   --origin-hf-dir /data/models/qwen25_7b
```

- `--input-dir`：训练产出的 torch_dist checkpoint 路径
- `--output-dir`：转换后的 HuggingFace 格式保存路径
- `--origin-hf-dir`：原始 HuggingFace 模型路径（用于补全配置文件）


## 4. 启动训练

### 4.1 一键启动（推荐）

使用 `launch.sh` 一条命令完成全部操作——自动清理旧进程、启动训练脚本、等待 Ray 就绪，再自动拉起 SGLang 服务：

```bash
bash agentic/agentflow/launch.sh
```

模型路径在脚本顶部通过 `MODEL_BASE` 和 `MODEL_CODER` 变量配置。训练日志写入 `/tmp/agentflow_logs/`。

### 4.2 手动启动

如需手动控制各步骤：

**第一步 — 启动 SGLang 服务**

训练需启动两个 SGLang 推理服务（分别用于 Executor/Verifier 和 Planner）：

注意，agentflow_qwen25_7b_rl_v2.sh 里面会杀掉所有的 python 进程，所以最好等训练脚本 kill 完进程、开始加载训练进程之后再启动 SGLang 推理服务。

```bash
# Executor / Verifier 使用的 coder 模型（端口 30001）
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
CUDA_VISIBLE_DEVICES=4,5 python3 -m sglang.launch_server \
  --model /data/models/qwen2.5_7b_codeer \
  --port 30001 \
  --context-length 131072 \
  --tp 2

# Planner 使用的基础模型（端口 30000，默认端口）
CUDA_VISIBLE_DEVICES=6,7 python3 -m sglang.launch_server \
  --model /data/models/qwen25_7b \
  --context-length 131072 \
  --tp 2
```

训练过程需要这两个服务。

**第二步 — 启动训练**

切换到项目根目录，再执行以下命令：

```bash
cd /path/to/slime
bash agentic/agentflow/agentflow_qwen25_7b_rl_v2.sh
```

训练权重默认保存至 `/data/AgentFlow_-Qwen25-7B-RL/`，可在脚本中的 `CKPT_ARGS` 部分修改。

如需保存每条样本的完整 agent 轨迹（用于调试），设置环境变量：

```bash
cd /path/to/slime
SAVE_TRAJECTORY=1 bash agentic/agentflow/agentflow_qwen25_7b_rl_v2.sh
```

轨迹以 JSON 格式保存至 `trajectories/`，每条记录包含 `analysis → steps → final_output` 完整链路。

## 5. 关键训练参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--advantage-estimator` | `grpo` | 优势估计算法 |
| `--lr` | `1e-6` | 学习率 |
| `--n-samples-per-prompt` | `8` | 每条 prompt 采样数 |
| `--rollout-batch-size` | `8` | rollout batch size |
| `--rollout-temperature` | `0.7` | 采样温度 |
| `--kl-loss-coef` | `0.001` | KL 散度系数 |
| `--eps-clip` / `--eps-clip-high` | `0.2` / `0.3` | PPO clip 范围 |
| `--sglang-context-length` | `131072` | SGLang 上下文长度 |

## 6. 评测

### 6.1 转换模型格式

训练过程中每 20 步在 AIME 2024 上自动评测，结果记录于 `eval_scores.json`。

#### 批量转换（推荐）

使用 `convert_agentflow_to_hf.sh` 一次性转换所有 `iter_*` checkpoint，已转换过的自动跳过：

```bash
bash agentic/agentflow/convert_agentflow_to_hf.sh
```

脚本顶部可配置的路径变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHECKPOINT_DIR` | `/data/AgentFlow_Qwen25-7B-RL` | 存放 `iter_*` Megatron checkpoint 的源目录 |
| `OUTPUT_BASE` | `/data/AgentFlow_Qwen25-7B-RL-HF` | 输出根目录，每个 checkpoint 保存为 `OUTPUT_BASE/iter_xxxxx/` |
| `ORIGIN_HF_DIR` | `/data/models/qwen25_7b` | 原始 HuggingFace 模型路径（用于补全配置文件） |

脚本结束后会打印汇总信息，列出失败的 checkpoint。

#### 单个转换

如需手动转换某个 checkpoint：

```bash
cd /path/to/slime
source scripts/models/qwen2.5-7B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    ${MODEL_ARGS[@]} \
    --load /data/AgentFlow_Qwen25-7B-RL/ \
    --hf-checkpoint /data/models/qwen25_7b \
    --save /data/AgentFlow_Qwen25-7B-RL-hf/
```

- `--load`：训练产出的 torch_dist checkpoint 路径
- `--hf-checkpoint`：原始 HuggingFace 模型路径（用于补全配置文件）
- `--save`：转换后的 HuggingFace 格式保存路径

转换完成后，可使用一键脚本或手动方式启动评测。

### 6.2 一键评测（推荐）

使用 `launch_eval.sh` 自动启动三个 SGLang 服务、等待就绪、执行评测，结束后自动关闭服务：

```bash
bash agentic/agentflow/launch_eval.sh
```

训练模型路径默认为 `/data/AgentFlow_pro-Qwen25-7B-RL/`，可通过环境变量覆盖：

```bash
MODEL_PATH=/data/my_model bash agentic/agentflow/launch_eval.sh
```

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_PATH` | `/data/AgentFlow_pro-Qwen25-7B-RL/` | 训练后的 AgentFlow 模型（Planner） |
| `MODEL_BASE` | `/data/models/qwen25_7b` | 基础模型（Executor/Verifier） |
| `MODEL_CODER` | `/data/models/qwen2.5_7b_codeer` | Coder 模型 |
| `OUTPUT` | `eval_results.json` | 评测结果保存路径 |
| `CONCURRENCY` | `16` | 并发请求数 |

日志写入 `/tmp/agentflow_eval_logs/`。

### 6.3 手动评测

如需手动控制各步骤：

**第一步 — 启动三个 SGLang 服务**

评测阶段的 agent 流程涉及三个独立角色，分别占用不同端口：

| 端口 | 角色 | 模型 |
|---|---|---|
| 30000 | Planner（主模型） | 训练后的 AgentFlow 模型 |
| 30001 | Executor / Verifier / base_generator | 基础对话模型 |
| 30002 | python_coder | Coder 模型 |

```bash
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Planner：使用训练后的模型（端口 30000）
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model /data/AgentFlow_Qwen25-7B-RL-hf/ \
  --port 30000 \
  --context-length 131072 \
  --tp 2 &

# Executor / Verifier / base_generator（端口 30001）
CUDA_VISIBLE_DEVICES=2,3 python3 -m sglang.launch_server \
  --model /data/models/qwen25_7b \
  --port 30001 \
  --context-length 131072 \
  --tp 2 &

# python_coder（端口 30002）
CUDA_VISIBLE_DEVICES=4,5 python3 -m sglang.launch_server \
  --model /data/models/qwen2.5_7b_codeer \
  --port 30002 \
  --context-length 131072 \
  --tp 2 &
```

等待三个服务均就绪后，执行评测。

**第二步 — 执行评测**

```bash
AUTO_START=0 bash agentic/agentflow/eval_agentflow.sh
```

结果保存至 `agentic/agentflow/eval_results.json`。

## 7. 批量评测所有 Checkpoint

将所有 checkpoint 批量转换为 HF 格式后（见 §6.1），使用 `eval_all_checkpoints.sh` 自动对每个 checkpoint 完成多次评测。Executor 和 Coder 服务**只启动一次**，在所有 checkpoint 间共享；每个 checkpoint 仅切换 Planner。

```bash
bash agentic/agentflow/eval_all_checkpoints.sh
```

每个 checkpoint 默认运行 `NUM_RUNS`（默认 10）次评测，记录最高分和平均分。已评测过的 checkpoint 再次运行时自动跳过。

**可配置的环境变量：**

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHECKPOINT_DIR` | `/data/AgentFlow_Qwen25-7B-RL-HF` | 存放转换后 `iter_*` HF checkpoint 的目录 |
| `MODEL_BASE` | `/data/models/qwen25_7b` | Executor/Verifier 使用的基础模型（端口 30001） |
| `MODEL_CODER` | `/data/models/qwen2.5_7b_codeer` | Coder 模型（端口 30002） |
| `NUM_RUNS` | `10` | 每个 checkpoint 的评测次数 |
| `CONCURRENCY` | `16` | 并发请求数 |
| `NUM_SAMPLES` | `0` | 每次评测的样本数（0 = 使用完整数据集） |

结果以追加模式写入 `agentic/agentflow/checkpoint_eval_results.jsonl`，每个 checkpoint 一条 JSON 记录：

```json
{
  "checkpoint": "iter_0000100",
  "path": "/data/AgentFlow_Qwen25-7B-RL-HF/iter_0000100",
  "best_score": 0.367,
  "avg_score": 0.341,
  "num_runs": 10,
  "runs": [{"run": 1, "accuracy": 0.333}, "..."],
  "timestamp": 1743200000
}
```

全部评测完成后打印汇总表：

```
iter_0000020        best=0.200  avg=0.187
iter_0000040        best=0.267  avg=0.251
iter_0000100        best=0.367  avg=0.341
```

各 SGLang 服务日志写入 `/tmp/agentflow_ckpt_eval_logs/`。

## 8. 评测 Baseline

Baseline 为单轮直接推理，不使用 AgentFlow 框架，只需一个 SGLang 服务。
如需手动指定已运行的服务：

```bash
# 先启动 SGLang 服务（端口 30000）
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model /data/models/qwen25_7b \
  --port 30000 \
  --context-length 131072 \
  --tp 2 &

# 再执行评测
AUTO_START=0 bash agentic/agentflow/eval_baseline.sh
```

结果保存至 `agentic/agentflow/baseline_results.json`。

## 9. 添加自定义工具

在 `tools/` 目录下新建一个子目录即可扩展工具，Planner 在启动时会自动扫描并加载所有工具，无需额外注册。

每个工具目录下需包含一个 `tool.py`，结构如下：

```python
from tools.base import BaseTool

# 必填：工具名称，Planner 通过此名称识别和调用该工具
TOOL_NAME = "My_Custom_Tool"

# 必填：工具描述，告诉 Planner 这个工具的用途和使用时机
TOOL_DESCRIPTION = """
简要描述该工具的功能及适用场景。
"""

# 必填：调用示例，帮助 Planner 理解如何构造调用命令
TOOL_DEMO_COMMANDS = [
    {
        "command": 'execution = tool.execute(query="示例查询")',
        "description": "该示例的说明。",
    },
]


class My_Custom_Tool(BaseTool):
    def __init__(self, llm_engine):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=TOOL_DESCRIPTION,
            demo_commands=TOOL_DEMO_COMMANDS,
        )
        self.llm_engine = llm_engine

    async def execute(self, query: str) -> str:
        # 统一执行入口，所有工具都必须实现此方法。
        # query：Planner 生成的子目标字符串。
        # 返回字符串结果，将被写入 Memory 并反馈给 Planner。
        ...
```

注意事项：
- `TOOL_NAME`、`TOOL_DESCRIPTION`、`TOOL_DEMO_COMMANDS` 由 Planner 读取，用于决定调用哪个工具以及如何构造调用命令。
- `execute(query)` 是**所有工具的统一执行入口**，由 Executor 统一调用，接收 Planner 生成的子目标字符串，必须返回一个字符串结果。
- 如果工具内部需要调用 LLM（如代码生成），`llm_engine` 会自动注入；纯确定性工具可以不使用。
