# 单卡全链路 baseline 测试流程

这份文档用于只有一张 GPU，尤其是一张 4090 时，快速验证 AgentFlow 的完整 baseline 链路。目标不是跑出最终训练指标，而是确认数据、模型服务、单轮 baseline、AgentFlow 推理、API 支持角色、轨迹保存和可选训练冒烟都能跑通。

## 1. 测试目标

推荐按下面顺序跑：

| 阶段 | 目的 | 是否需要本地 GPU | 是否需要 API |
|---|---|---:|---:|
| 环境和数据检查 | 确认路径、字段、tokenizer | 否 | 否 |
| 单轮 QA baseline | 测基础模型直接答题能力 | 是 | 否 |
| API Planner AgentFlow | 测工具链、Rewarder、数据链路 | 否 | 是 |
| 本地 Planner AgentFlow | 测单卡部署小 Planner 的完整 agent 链路 | 是 | 是 |
| 保存轨迹分析 | 看每一步行为 | 是或否 | 是 |
| 单卡训练冒烟 | 验证 Ray、rollout、GRPO 入口能启动 | 是 | 是 |

建议先跑 3 到 5 条样本，确认链路稳定后再扩大。

## 2. 基础环境变量

```bash
cd /path/to/slime-agentic

export AGENTFLOW_API_BASE=https://api.openai.com/v1
export AGENTFLOW_API_KEY=你的_api_key
export AGENTFLOW_API_MODEL=gpt-4o-mini
export AGENTFLOW_API_TIMEOUT=180
export AGENTFLOW_API_MAX_RETRIES=3
export AGENTFLOW_API_ENABLE_THINKING=false
```

如果代码生成或 judge 想用独立模型：

```bash
export AGENTFLOW_CODER_MODEL=gpt-4o-mini
export AGENTFLOW_REWARDER_MODEL=gpt-4o-mini
```

如果你使用 SiliconFlow/Qwen 的思考模型，建议单卡测试阶段先关闭 API 思考：

```bash
export AGENTFLOW_API_ENABLE_THINKING=false
```

这样可以减少超时和 `content` 为空的概率。若服务支持思考预算，也可以加：

```bash
export AGENTFLOW_API_THINKING_BUDGET=1024
```

单卡建议使用 0.5B/1.5B/2B 级模型，不要一开始上 7B。

默认路径示例：

```bash
export BASE_MODEL=/data/models/qwen25_1.5b
export RL_HF_MODEL=/data/AgentFlow_Qwen25-1.5B-RL-HF
export AIME_DATA=/data/aime-2024/aime-2024.jsonl
```

## 3. 数据和 tokenizer 检查

```bash
ls "$BASE_MODEL"
ls "$AIME_DATA"
head -n 2 "$AIME_DATA"
```

确认 JSONL 字段：

| 字段 | 预期 |
|---|---|
| `prompt` | 题目 |
| `label` | 标准答案 |

如果字段不同，需要修改脚本参数或临时改脚本里的 `--input-key`、`--label-key`。

## 4. 测试一：单轮 QA baseline

这个测试不使用 AgentFlow，只测模型直接回答数学题的能力。它能回答一个问题：基础模型不调用工具时大概是什么水平。

单卡 1.5B 示例：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/data/models/qwen25_1.5b \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
TP=1 \
CTX_LEN=16384 \
MEM_FRACTION=0.75 \
NUM_SAMPLES=5 \
CONCURRENCY=2 \
MAX_NEW_TOKENS=2048 \
OUTPUT=/tmp/agentflow_baseline/single_turn_baseline.json \
bash agentic/agentflow/eval_baseline.sh
```

预期：

| 输出 | 说明 |
|---|---|
| `/tmp/agentflow_baseline/single_turn_baseline.json` | 单轮回答结果 |
| 控制台 summary | accuracy、正确数、耗时 |

如果显存不足：

```bash
CTX_LEN=8192
MAX_NEW_TOKENS=1024
CONCURRENCY=1
MEM_FRACTION=0.65
```

## 5. 测试二：API Planner AgentFlow baseline

这个测试不启动本地 Planner，用 API 大模型当 Planner。它能验证数据、工具、Executor、Verifier、Rewarder、final_output 整条 AgentFlow 链路。

注意观察轨迹时不要只看最终 accuracy。现在工具被约束为局部工具：`Local_Math_Deduction_Tool` 只回答一个关系或定理，`Python_Code_Generator_Tool` 只做明确计算，`final_output` 只汇总 Memory 中已有结果；如果 Planner 把整题丢给工具，工具应该返回需要更小子目标的提示。这能更清楚地区分“大模型工具直接做题”和“Planner 学会拆解任务”。

```bash
USE_API_FOR_PLANNER=1 \
PLANNER_API_BASE=https://api.openai.com/v1 \
PLANNER_API_KEY=你的_api_key \
PLANNER_MODEL=gpt-4o \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
NUM_SAMPLES=5 \
CONCURRENCY=2 \
MAX_STEPS=3 \
MAX_NEW_TOKENS=2048 \
OUTPUT=/tmp/agentflow_baseline/api_planner_agentflow.json \
bash agentic/agentflow/eval_agentflow.sh
```

通过标准：

| 检查项 | 预期 |
|---|---|
| 脚本正常退出 | 是 |
| 输出 JSON 存在 | 是 |
| 每条样本有 `final_output` | 是 |
| 每条样本有 `score` | 是 |
| 错误率可解释 | 是 |

如果这一步失败，优先查 API key、API base、模型名、数据路径、tokenizer 路径。

## 6. 测试三：本地 Planner + API 支持角色

这个测试是单卡上最重要的全链路 baseline：Planner 在本地 GPU 上跑，其他角色走 API。

如果还没有 RL checkpoint，可以先用原始基础模型当 Planner：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/data/models/qwen25_1.5b \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
TP=1 \
CTX_LEN=16384 \
MEM_FRACTION=0.75 \
NUM_SAMPLES=5 \
CONCURRENCY=1 \
MAX_STEPS=3 \
MAX_NEW_TOKENS=2048 \
OUTPUT=/tmp/agentflow_baseline/local_raw_planner_agentflow.json \
bash agentic/agentflow/eval_agentflow.sh
```

如果已经有训练后的 HF checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/data/AgentFlow_Qwen25-1.5B-RL-HF/ \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
TP=1 \
CTX_LEN=16384 \
MEM_FRACTION=0.75 \
NUM_SAMPLES=5 \
CONCURRENCY=1 \
MAX_STEPS=3 \
MAX_NEW_TOKENS=2048 \
OUTPUT=/tmp/agentflow_baseline/local_rl_planner_agentflow.json \
bash agentic/agentflow/eval_agentflow.sh
```

建议对照：

| 文件 | 说明 |
|---|---|
| `single_turn_baseline.json` | 不使用 AgentFlow 的单轮能力 |
| `api_planner_agentflow.json` | API 大模型 Planner 上限 |
| `local_raw_planner_agentflow.json` | 原始小 Planner 的 agent 能力 |
| `local_rl_planner_agentflow.json` | RL 后小 Planner 的 agent 能力 |

## 7. 手动启动 Planner，反复评测

如果要反复调 `NUM_SAMPLES`、`MAX_STEPS`、`CONCURRENCY`，建议手动启动一次 SGLang，后续评测复用服务。

启动服务：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path /data/models/qwen25_1.5b \
  --port 30000 \
  --tp 1 \
  --context-length 16384 \
  --mem-fraction-static 0.75 \
  --trust-remote-code
```

复用服务评测：

```bash
AUTO_START=0 \
PLANNER_URL=http://127.0.0.1:30000/generate \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
NUM_SAMPLES=5 \
CONCURRENCY=1 \
MAX_STEPS=3 \
MAX_NEW_TOKENS=2048 \
OUTPUT=/tmp/agentflow_baseline/manual_server_eval.json \
bash agentic/agentflow/eval_agentflow.sh
```

停止服务：

```bash
pkill -f 'sglang.launch_server'
```

## 8. 保存轨迹

轨迹用于分析 Planner 每一步为什么成功或失败。

API Planner 保存轨迹：

```bash
USE_API_FOR_PLANNER=1 \
PLANNER_API_BASE=https://api.openai.com/v1 \
PLANNER_API_KEY=你的_api_key \
PLANNER_MODEL=gpt-4o \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
NUM_SAMPLES=3 \
CONCURRENCY=1 \
MAX_STEPS=3 \
TRAJECTORY_DIR=/tmp/agentflow_baseline/traces_api \
OUTPUT=/tmp/agentflow_baseline/api_trace_eval.json \
bash agentic/agentflow/eval_agentflow.sh
```

本地 Planner 保存轨迹：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL_PATH=/data/models/qwen25_1.5b \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
NUM_SAMPLES=3 \
CONCURRENCY=1 \
MAX_STEPS=3 \
TRAJECTORY_DIR=/tmp/agentflow_baseline/traces_local \
OUTPUT=/tmp/agentflow_baseline/local_trace_eval.json \
bash agentic/agentflow/eval_agentflow.sh
```

重点看：

| 字段 | 看什么 |
|---|---|
| `analysis` | 是否理解题意 |
| `steps[].next_step` | 工具选择和子目标是否合理 |
| `steps[].tool_command` | 是否能被解析执行 |
| `steps[].execution_result` | 工具结果是否有用 |
| `steps[].conclusion` | STOP/CONTINUE 是否合理 |
| `final_output` | 是否吸收工具结果并输出 `\boxed{}` |

## 9. 可选：单卡训练冒烟

单卡 4090 不建议直接跑正式 GRPO，只建议验证训练入口能启动。目标是证明 Ray、SGLang rollout、API 支持角色、Rewarder、custom_convert 和 trainer 能串起来。

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

冒烟通过标准：

| 检查项 | 预期 |
|---|---|
| Ray 能启动 | 是 |
| SGLang rollout engine 能启动 | 是 |
| API 支持角色能调用 | 是 |
| 至少产生一批 rollout | 是 |
| reward_func 能返回 score | 是 |
| trainer 没有立刻 OOM | 是 |

冒烟阶段不要看最终 accuracy。它只证明链路可运行。

## 10. 结果对比表

建议整理成这个表：

| 实验 | Planner | 是否 AgentFlow | 样本数 | Accuracy | 备注 |
|---|---|---:|---:|---:|---|
| 单轮 baseline | 本地基础模型 | 否 | 5 |  | 直接 QA |
| API Planner | API 大模型 | 是 | 5 |  | 系统上限 |
| 本地 raw Planner | 本地基础模型 | 是 | 5 |  | RL 前 |
| 本地 RL Planner | 本地训练 checkpoint | 是 | 5 |  | RL 后 |

如果本地 raw Planner 比单轮 baseline 更差，不一定说明 AgentFlow 没用，可能是 Planner 还没学会工具协议。此时优先看轨迹，而不是只看 accuracy。

## 11. 单卡常见问题

| 问题 | 表现 | 解决 |
|---|---|---|
| 显存不足 | SGLang 启动失败或 CUDA OOM | 降 `CTX_LEN`、`MAX_NEW_TOKENS`、`MEM_FRACTION`、`CONCURRENCY` |
| API 调用失败 | 429/401/timeout | 检查 key、base、模型名，降低并发 |
| 本地 Planner 输出乱 | 工具名不合法、格式不对 | 先用 API Planner 跑通，再做 SFT/OPD |
| 轨迹没有 final_output | final_output API 失败 | 查 API 配置和超时 |
| score 全 0 | 答案格式或 judge 问题 | 看 `final_output` 和 `pred` 抽取 |
| 评测太慢 | 每题多步 API 调用 | 降 `MAX_STEPS`、`NUM_SAMPLES`、`CONCURRENCY` |
| baseline 默认想用 7B/TP=4 | 单卡起不来 | 显式设置 `MODEL_PATH`、`TOKENIZER_PATH`、`TP=1` |

## 12. 最短可执行顺序

```bash
# 1. 单轮 baseline
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/data/models/qwen25_1.5b TOKENIZER_PATH=/data/models/qwen25_1.5b \
TP=1 CTX_LEN=16384 NUM_SAMPLES=5 CONCURRENCY=1 \
OUTPUT=/tmp/agentflow_baseline/single_turn.json \
bash agentic/agentflow/eval_baseline.sh

# 2. API Planner AgentFlow
USE_API_FOR_PLANNER=1 PLANNER_API_BASE=https://api.openai.com/v1 PLANNER_API_KEY=你的_api_key \
PLANNER_MODEL=gpt-4o TOKENIZER_PATH=/data/models/qwen25_1.5b \
NUM_SAMPLES=5 CONCURRENCY=1 MAX_STEPS=3 \
OUTPUT=/tmp/agentflow_baseline/api_agentflow.json \
bash agentic/agentflow/eval_agentflow.sh

# 3. 本地 Planner AgentFlow
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/data/models/qwen25_1.5b TOKENIZER_PATH=/data/models/qwen25_1.5b \
TP=1 CTX_LEN=16384 NUM_SAMPLES=5 CONCURRENCY=1 MAX_STEPS=3 \
OUTPUT=/tmp/agentflow_baseline/local_agentflow.json \
bash agentic/agentflow/eval_agentflow.sh
```

跑完这三步，就有了单卡条件下最小但完整的 baseline 对照。

## 13. 为什么单卡要按这个顺序测

顺序很重要，因为每一步都在排除一种风险。

| 顺序 | 测试 | 排除什么风险 |
|---|---|---|
| 1 | 数据和 tokenizer 检查 | 路径错、字段错、模型加载不了 |
| 2 | 单轮 baseline | 基础模型/SGLang 本身不能推理 |
| 3 | API Planner AgentFlow | AgentFlow 代码、工具、Rewarder 不通 |
| 4 | 本地 Planner AgentFlow | 单卡部署 Planner + API 支持角色不通 |
| 5 | 轨迹保存 | 不知道错误来自哪一步 |
| 6 | 训练冒烟 | Ray/slime/GRPO 接口不通 |

面试时可以这样说：

> 我在单卡上不会直接跑大规模 RL，而是先分层验证。先验证模型能单轮回答，再验证 AgentFlow 链路，再验证本地 Planner 能接入，最后只做极小训练冒烟。这样即使失败，也能知道失败发生在哪一层。

## 14. 三个 baseline 分别回答什么问题

### 14.1 单轮 QA baseline

回答：

```text
基础模型不用工具，直接答题有多强？
```

如果这个分数很低，不奇怪，因为 1.5B/2B 小模型数学能力有限。

### 14.2 API Planner AgentFlow

回答：

```text
如果 Planner 很强，这套 AgentFlow 工具链上限在哪里？
```

如果 API Planner 也很差，说明问题可能不在小 Planner，而在工具设计、final_output、Rewarder 或数据格式。

### 14.3 本地 Planner AgentFlow

回答：

```text
小 Planner 在同一套工具链里能做到什么程度？
```

RL 前后都跑这个，就能看训练是否改善 Planner 调度能力。

## 15. 如何读输出 JSON

评测 JSON 里重点看这些字段：

| 字段 | 含义 | 怎么分析 |
|---|---|---|
| `question` | 原题 | 看题型 |
| `label` | 标准答案 | 和 pred 对比 |
| `pred` | 抽取出的预测答案 | 是否格式正确 |
| `final_output` | 最终回答全文 | 是否真的解题 |
| `score` | reward/判分结果 | 是否合理 |
| `error` | 异常信息 | 判断链路是否失败 |

不要只看总 accuracy。小样本 5 条时，accuracy 波动很大，更重要的是看每条样本的轨迹是否合理。

## 16. 如何读 trajectory

trajectory 是面试准备最有价值的材料。你可以挑一条成功和一条失败来讲。

成功轨迹应该长这样：

```text
analysis 理解题意
  -> next_step 选择合适工具
  -> tool_command 可解析
  -> execution_result 有有效中间结果
  -> Verifier 判断 STOP 合理
  -> final_output 使用中间结果并给 \boxed{}
```

失败轨迹按这个模板记录：

| 检查点 | 你的结论 |
|---|---|
| analysis 是否错 |  |
| 第一步工具是否错 |  |
| Context 是否缺信息 |  |
| command 是否解析失败 |  |
| tool result 是否有用 |  |
| Verifier 是否过早 STOP |  |
| final_output 是否没吸收 memory |  |
| Rewarder 是否误判 |  |

这样你在面试里不会只说“模型答错了”，而是能说清楚“错在 Planner 选择工具、还是错在工具结果整合”。

## 17. 单卡参数怎么调

单卡 4090 常见调参顺序：

| 如果出现 | 先调什么 | 再调什么 |
|---|---|---|
| SGLang OOM | `CTX_LEN=8192` | `MEM_FRACTION=0.65` |
| 评测 OOM | `CONCURRENCY=1` | `MAX_NEW_TOKENS=1024` |
| API 太慢 | `MAX_STEPS=2` | `NUM_SAMPLES=3` |
| Planner 输出太短 | 增大 `MAX_NEW_TOKENS` | 检查 stop token |
| Planner 输出太乱 | 降 `TEMPERATURE` | 做 SFT 格式热启动 |
| 工具调用总失败 | 看 trajectory | 检查 Tool Name 和 command |

建议新手先用：

```bash
NUM_SAMPLES=3
CONCURRENCY=1
MAX_STEPS=2
MAX_NEW_TOKENS=1024
CTX_LEN=8192
```

跑通后再逐步加。

## 18. 面试里怎么解释“单卡 baseline”

可以这样回答：

> 单卡阶段我的目标不是追求最终指标，而是验证整个系统闭环。我先跑单轮 QA baseline，确认模型和数据能工作；再跑 API Planner AgentFlow，确认工具、Verifier、Rewarder 和 final_output 链路能工作；然后跑本地 Planner AgentFlow，确认小模型能接入调度；最后保存轨迹分析错误。如果这些都通过，再做极小 GRPO 冒烟，确认 slime、Ray、rollout、reward 和 custom_convert 能串起来。正式指标留到多卡训练后评测。

## 19. 单卡报告模板

你可以把结果整理成这样：

```text
环境：
- GPU: 1x RTX 4090
- Planner: Qwen2.5-1.5B
- Support roles: gpt-4o-mini API
- Dataset: AIME 2024, first 5 samples

实验：
1. Single-turn baseline: x/5
2. API Planner AgentFlow: x/5
3. Local raw Planner AgentFlow: x/5
4. Local RL Planner AgentFlow: x/5

观察：
- Local Planner 主要失败在工具名/STOP/final_output/计算错误
- API Planner 上限说明工具链是否有潜力
- 后续需要 SFT/OPD/GRPO 解决 Planner 调度问题
```

这个模板很适合面试前整理成讲稿。
