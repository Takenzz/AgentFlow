# AgentFlow Minimal Training Extension

这个仓库只保留 AgentFlow 接入 slime 训练所需的最小代码。`slime`、Megatron、SGLang、模型转换工具和大部分训练依赖请在运行环境中单独安装。

## 目录

- `train.py`: slime 训练入口，保持很薄，训练循环由外部 `slime` 提供。
- `agentic/agentflow/train_agentflow.sh`: 参数化训练脚本。
- `agentic/agentflow/launch.sh`: 常规训练 launcher，负责清理旧进程、提交 Ray job。
- `agentic/agentflow/quick_train_2x4090.sh`: 两张 4090 的快速训练测试脚本。
- `agentic/agentflow/formal_train_4xa800.sh`: 4 张 A800 的正式训练脚本。
- `agentic/agentflow/rollout.py`: AgentFlow 自定义 rollout、eval log 和 teacher logprob reward。
- `agentic/agentflow/custom_convert.py`: 把多轮 AgentFlow trajectory 拆成独立 Planner turn 训练样本。
- `agentic/agentflow/core/`: Planner、Executor、Verifier、Solver 和 LLM engine。
- `agentic/agentflow/tools/`: `Local_Math_Deduction_Tool` 和 `Python_Code_Generator_Tool`。
- `agentic/agentflow/model_configs/`: 常用小模型结构参数。
- `examples/tiny_math.jsonl`: smoke test 用的极小 JSONL 数据。
- `docs/`: 安装、配置和快速训练测试文档。

## 快速开始

先确认外部 slime 环境可用：

```bash
python3 -c "import slime; print(slime.__file__)"
python3 -c "import ray, torch, sglang"
```

如果 slime 或 Megatron 是源码安装，启动前设置：

```bash
export SLIME_PATH=/path/to/slime
export MEGATRON_PATH=/path/to/Megatron-LM
```

常规训练：

```bash
TRAIN_GPUS=4 \
TRAIN_TP=1 \
MODEL_CONFIG_SCRIPT=$PWD/agentic/agentflow/model_configs/qwen2.5-1.5B.sh \
BASE_HF_CHECKPOINT=/data/models/student_hf \
REF_LOAD=/data/models/student_torch_dist \
SAVE_DIR=/data/agentflow_runs/student_rl \
PROMPT_DATA=/data/dapo-math-17k/dapo-math-17k.jsonl \
EVAL_PROMPT_DATA=/data/aime-2024/aime-2024.jsonl \
AGENTFLOW_API_BASE=https://api.openai.com/v1 \
AGENTFLOW_API_KEY=$OPENAI_API_KEY \
AGENTFLOW_API_MODEL=gpt-4o-mini \
bash agentic/agentflow/launch.sh
```

两张 4090 的快速测试见 [docs/QUICK_TRAIN_2X4090.md](docs/QUICK_TRAIN_2X4090.md)。

## 文档

- [docs/SETUP.md](docs/SETUP.md): 环境、模型、数据和转换准备。
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md): 训练脚本参数和常用环境变量。
- [docs/TRAINING_SCRIPTS_ANALYSIS.md](docs/TRAINING_SCRIPTS_ANALYSIS.md): 所有训练脚本的逐层分析、技术原理和 Bash/Python 语法说明。
- [docs/QUICK_TRAIN_2X4090.md](docs/QUICK_TRAIN_2X4090.md): 2x4090、teacher 2B、student 0.8B 级别的快速训练测试。
- [docs/FORMAL_TRAIN_4XA800.md](docs/FORMAL_TRAIN_4XA800.md): 4xA800 正式训练入口、teacher 部署模式和推荐参数。

## AgentFlow 技术分析文档

这些是原项目里关于 AgentFlow 训练设计、系统原理、排障和工具调用的长文档，已经保留在 `agentic/agentflow/` 下：

- [01_FULL_TRAINING_PIPELINE_ZH.md](agentic/agentflow/01_FULL_TRAINING_PIPELINE_ZH.md): AgentFlow 全链路训练流程。
- [02_PRINCIPLES_INTERVIEW_TROUBLESHOOTING_ZH.md](agentic/agentflow/02_PRINCIPLES_INTERVIEW_TROUBLESHOOTING_ZH.md): 原理、面试问答和训练问题排查。
- [03_SINGLE_GPU_BASELINE_TEST_ZH.md](agentic/agentflow/03_SINGLE_GPU_BASELINE_TEST_ZH.md): 单卡 baseline 与冒烟测试思路。
- [04_AGENT_FRAMEWORK_TOOL_CALLING_ZH.md](agentic/agentflow/04_AGENT_FRAMEWORK_TOOL_CALLING_ZH.md): Agent 调度、Memory、Verifier 和工具调用详解。
