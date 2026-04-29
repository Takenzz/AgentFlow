# AgentFlow 全链路训练流程

这份文档只讲一件事：从数据、模型、服务配置，到 RL 训练、checkpoint 转换、评测和结果沉淀，如何把 AgentFlow 的完整训练闭环跑起来。

当前项目的推荐资源假设是：本地 GPU 主要训练一个 1.5B/2B 级 Planner，Executor、Verifier、局部推理工具、python_coder、Rewarder 默认使用 OpenAI 兼容 API。这样可以把显存留给被训练模型，同时保留大模型 API 做辅助能力和上限对照。

## 1. 总体链路

```text
训练数据 dapo-math-17k
  -> Planner 本地 rollout
  -> Executor / Tools / Verifier / final_output API 辅助
  -> Rewarder 判断 final_output 与 label 是否匹配
  -> slime 使用 GRPO 更新 Planner
  -> 保存 Megatron/torch_dist checkpoint
  -> 转 HF checkpoint
  -> AIME 评测和轨迹分析
```

核心原则：

| 模块 | 默认部署 | 是否被训练 | 说明 |
|---|---|---:|---|
| Planner | 本地 SGLang + slime | 是 | 产生 analysis 和 next_step，是 RL 优化目标 |
| Executor | API | 否 | 把 Planner 的工具选择转成可执行命令 |
| Local_Math_Deduction_Tool | API | 否 | 只推导一个局部数学关系，不负责完整解题 |
| python_coder | API | 否 | 只做明确的局部计算/符号检查 |
| Verifier | API | 否 | 判断当前 memory 是否足够，给出 STOP/CONTINUE |
| final_output | API | 否 | 只汇总 Memory 中已有工具结果，信息不足时输出不足 |
| Rewarder | API | 否 | 判断最终答案是否等价于 label |

注意：Planner API 模式只适合评测或教师上限对照。RL 训练必须让本地 Planner 返回 token、logprob 和 loss mask，否则无法回传梯度。

工具边界很重要：Planner 只负责拆步骤和选择工具，不能在 `analysis` 或 `next_step` 中直接解题；`Local_Math_Deduction_Tool` 只能回答一个局部定理、恒等式或关系；`Python_Code_Generator_Tool` 只能执行明确计算；`final_output` 只能汇总 Memory 里已有的工具结果，不能重新解题。未知工具名不会再自动兜底到 base_generator，这样 Planner 的工具选择错误会暴露在轨迹里，便于 SFT/OPD/RL 修正。只有评测旧 checkpoint 时，才建议临时设置 `AGENTFLOW_ALLOW_TOOL_ALIASES=true` 兼容旧工具名。

## 2. 环境准备

在项目根目录执行：

```bash
cd /path/to/slime-agentic
```

建议准备路径：

| 路径 | 作用 |
|---|---|
| `/data/models/qwen25_1.5b` | Planner 原始 HF 模型 |
| `/data/models/qwen2.5_1.5b_dist/` | 转换后的 torch_dist 参考模型 |
| `/data/dapo-math-17k/dapo-math-17k.jsonl` | RL 训练集 |
| `/data/aime-2024/aime-2024.jsonl` | 评测集 |
| `/data/AgentFlow_Qwen25-1.5B-RL/` | 训练输出 Megatron/torch_dist checkpoint |
| `/data/AgentFlow_Qwen25-1.5B-RL-HF/` | 转换后的 HF checkpoint |

需要的 API 环境变量：

```bash
export AGENTFLOW_API_BASE=https://api.openai.com/v1
export AGENTFLOW_API_KEY=你的_api_key
export AGENTFLOW_API_MODEL=gpt-4o-mini
export AGENTFLOW_API_TIMEOUT=180
export AGENTFLOW_API_MAX_RETRIES=3
export AGENTFLOW_API_ENABLE_THINKING=false
```

如果希望不同角色使用不同模型：

```bash
export AGENTFLOW_EXECUTOR_MODEL=gpt-4o-mini
export AGENTFLOW_CODER_MODEL=gpt-4o-mini
export AGENTFLOW_REWARDER_MODEL=gpt-4o-mini
```

如果使用 SiliconFlow/Qwen 这类支持 thinking 参数的 OpenAI 兼容 API，建议默认关闭思考，避免 reasoning token 占满输出预算导致 `content` 为空或请求超时：

```bash
export AGENTFLOW_API_ENABLE_THINKING=false
```

如果服务支持 thinking budget，也可以限制思考预算：

```bash
export AGENTFLOW_API_THINKING_BUDGET=1024
```

说明：这些参数会通过 OpenAI SDK 的 `extra_body` 传给 `/chat/completions`。不支持 `enable_thinking` 的服务可能会忽略它，也可能报参数错误；如果报错，可以设置 `AGENTFLOW_API_ENABLE_THINKING=auto`。

## 3. 数据准备

训练集默认使用 `dapo-math-17k`，评测集默认使用 `aime-2024`：

```bash
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /data/dapo-math-17k

huggingface-cli download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /data/aime-2024
```

默认字段：

| 字段 | 说明 |
|---|---|
| `prompt` | 数学问题，可以是字符串，也可以是 chat messages |
| `label` | 标准答案 |

如果数据字段不同，需要在训练参数或评测脚本中修改：

```bash
--input-key prompt
--label-key label
```

训练前先做数据检查：

```bash
head -n 2 /data/dapo-math-17k/dapo-math-17k.jsonl
head -n 2 /data/aime-2024/aime-2024.jsonl
```

重点确认：

| 检查项 | 预期 |
|---|---|
| JSONL 每行可解析 | 是 |
| `prompt` 不为空 | 是 |
| `label` 是可比较答案 | 是 |
| 数学题答案格式稳定 | 尽量稳定 |

## 4. 模型下载和格式转换

默认使用 Qwen2.5-1.5B-Instruct 作为 2B 级小 Planner：

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

转换后检查：

```bash
ls /data/models/qwen2.5_1.5b_dist/
```

如果换其他小模型，需要同步调整：

| 项 | 说明 |
|---|---|
| `MODEL_CONFIG_SCRIPT` | 模型结构参数脚本 |
| `BASE_HF_CHECKPOINT` | 原始 HF 模型目录 |
| `REF_LOAD` | torch_dist 参考模型目录 |
| tokenizer | 评测和 API prompt 渲染都依赖 |

## 5. 启动 RL 训练

推荐入口：

```bash
cd /path/to/slime-agentic

TRAIN_GPUS=4 \
TRAIN_TP=1 \
ROLLOUT_ENGINE_GPUS=1 \
BASE_HF_CHECKPOINT=/data/models/qwen25_1.5b \
REF_LOAD=/data/models/qwen2.5_1.5b_dist/ \
SAVE_DIR=/data/AgentFlow_Qwen25-1.5B-RL/ \
PROMPT_DATA=/data/dapo-math-17k/dapo-math-17k.jsonl \
EVAL_PROMPT_DATA=/data/aime-2024/aime-2024.jsonl \
bash agentic/agentflow/launch.sh
```

`launch.sh` 会做这些事：

1. 清理旧 Ray 和旧 SGLang Planner 进程。
2. 设置 API 默认环境变量。
3. 启动 `agentflow_qwen25_7b_rl_v2.sh`。
4. 拉起 Ray head。
5. 通过 `ray job submit` 启动 `train.py`。
6. 使用 `rollout.generate` 作为自定义多轮 AgentFlow rollout。
7. 使用 `rollout.reward_func` 计算最终奖励。
8. 使用 `custom_convert.custom_convert` 把多轮 trajectory 拆成训练样本。

关键训练参数：

| 变量 / 参数 | 默认值 | 说明 |
|---|---:|---|
| `TRAIN_GPUS` | `4` | 训练可见 GPU 数 |
| `TRAIN_TP` | `1` | 小 Planner 默认不开 tensor parallel |
| `ROLLOUT_ENGINE_GPUS` | `1` | Planner rollout engine 使用 GPU 数 |
| `ROLLOUT_BATCH_SIZE` | `8` | 每轮 rollout prompt 数 |
| `N_SAMPLES_PER_PROMPT` | `8` | GRPO 每题采样条数 |
| `GLOBAL_BATCH_SIZE` | `64` | trainer 全局 batch size |
| `ROLLOUT_TEMPERATURE` | `0.7` | Planner rollout 探索温度 |
| `ROLLOUT_MAX_RESPONSE_LEN` | `32768` | rollout 最大响应长度 |
| `MAX_TOKENS_PER_GPU` | `16384` | 动态 batch 单卡 token 上限 |
| `SAVE_INTERVAL` | `100` | checkpoint 保存间隔 |
| `EVAL_INTERVAL` | `20` | 训练中评测间隔 |
| `SGLANG_CONTEXT_LENGTH` | `65536` | SGLang 上下文长度 |
| `SGLANG_MEM_FRACTION_STATIC` | `0.75` | SGLang 静态显存比例 |

日志位置：

```text
/tmp/agentflow_logs/train.log
```

查看训练日志：

```bash
tail -f /tmp/agentflow_logs/train.log
```

查看 Ray：

```bash
ray status
```

## 6. 保存训练轨迹

如果要分析 Planner 每一步行为：

```bash
SAVE_TRAJECTORY=1 \
TRAIN_GPUS=4 \
TRAIN_TP=1 \
ROLLOUT_ENGINE_GPUS=1 \
bash agentic/agentflow/launch.sh
```

轨迹会写到：

```text
agentic/agentflow/trajectories/
```

轨迹字段：

| 字段 | 说明 |
|---|---|
| `question` | 原始题目 |
| `label` | 标准答案 |
| `analysis` | Planner 初始分析 |
| `steps` | 每一步 next_step、tool、command、result、conclusion |
| `final_output` | 最终答案 |
| `full_response` | 拼接后的完整链路日志 |

## 7. 训练后的 checkpoint 转 HF

训练输出默认是 Megatron/torch_dist 格式，SGLang 独立评测需要 HF 格式。

转换全部 checkpoint：

```bash
CHECKPOINT_DIR=/data/AgentFlow_Qwen25-1.5B-RL \
OUTPUT_BASE=/data/AgentFlow_Qwen25-1.5B-RL-HF \
ORIGIN_HF_DIR=/data/models/qwen25_1.5b \
bash agentic/agentflow/convert_agentflow_to_hf.sh
```

或者手动转换：

```bash
cd /path/to/slime-agentic
source scripts/models/qwen2.5-1.5B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  ${MODEL_ARGS[@]} \
  --load /data/AgentFlow_Qwen25-1.5B-RL/ \
  --hf-checkpoint /data/models/qwen25_1.5b \
  --save /data/AgentFlow_Qwen25-1.5B-RL-HF/
```

转换后检查：

```bash
ls /data/AgentFlow_Qwen25-1.5B-RL-HF/
```

至少应包含 `config.json`、tokenizer 相关文件和模型权重。

## 8. 评测本地 Planner

本地 Planner 使用训练后的 HF checkpoint，其他角色走 API：

```bash
MODEL_PATH=/data/AgentFlow_Qwen25-1.5B-RL-HF/ \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
TP=1 \
CTX_LEN=32768 \
MEM_FRACTION=0.75 \
NUM_SAMPLES=20 \
CONCURRENCY=4 \
MAX_STEPS=5 \
MAX_NEW_TOKENS=4096 \
OUTPUT=/tmp/agentflow_eval/local_planner_eval.json \
bash agentic/agentflow/eval_agentflow.sh
```

输出：

| 文件 | 说明 |
|---|---|
| `/tmp/agentflow_eval/local_planner_eval.json` | 每条样本的 pred、label、score、final_output |
| 控制台 summary | 准确率、正确数、耗时 |

显存紧张时降低：

```bash
CTX_LEN=16384
MAX_NEW_TOKENS=1024
CONCURRENCY=1
MAX_STEPS=2
```

## 9. API 大模型 Planner 上限对照

这个模式不启动本地 Planner，只用 API 大模型作为 Planner，对比小 Planner 和教师上限：

```bash
USE_API_FOR_PLANNER=1 \
PLANNER_API_BASE=https://api.openai.com/v1 \
PLANNER_API_KEY=你的_api_key \
PLANNER_MODEL=gpt-4o \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
NUM_SAMPLES=20 \
CONCURRENCY=4 \
MAX_STEPS=5 \
MAX_NEW_TOKENS=4096 \
OUTPUT=/tmp/agentflow_eval/api_planner_eval.json \
bash agentic/agentflow/eval_agentflow.sh
```

注意：`TOKENIZER_PATH` 仍然需要提供，因为框架会用本地 tokenizer 渲染 prompt 和记录 token 信息。API Planner 不参与训练 loss。

## 10. 批量评测 checkpoint

先把每个 `iter_*` 转成 HF 格式，然后运行：

```bash
CHECKPOINT_DIR=/data/AgentFlow_Qwen25-1.5B-RL-HF \
TOKENIZER_PATH=/data/models/qwen25_1.5b \
NUM_RUNS=3 \
NUM_SAMPLES=50 \
CONCURRENCY=4 \
bash agentic/agentflow/eval_all_checkpoints.sh
```

结果写入：

```text
agentic/agentflow/checkpoint_eval_results.jsonl
```

建议记录：

| 指标 | 用途 |
|---|---|
| AIME accuracy | 主指标 |
| 平均 step 数 | 判断是否过度规划 |
| STOP 比例 | 判断 Verifier 是否过早停止 |
| 工具调用解析失败率 | 判断 Planner/Executor 格式是否稳定 |
| final_output 无 `\boxed{}` 比例 | 判断答案格式稳定性 |

## 11. 推荐实验顺序

```text
1. 单轮 baseline：确认基础模型直接答题能力
2. API Planner AgentFlow：确认工具链和 Rewarder 跑通
3. 本地原始小 Planner AgentFlow：得到 RL 前指标
4. SFT 热启动，可选：让 Planner 学会工具协议和输出格式
5. OPD，小规模可选：用教师纠偏学生 on-policy 状态
6. GRPO RL：使用真实 reward 优化小 Planner
7. checkpoint 批量评测：选最优 checkpoint
8. 轨迹分析：解释提升来自哪里
```

## 12. 最终交付物

建议最终保留这些结果，方便复现实验和面试展示：

| 交付物 | 说明 |
|---|---|
| 训练命令和环境变量 | 能证明可复现 |
| `train.log` 关键片段 | 能证明训练闭环真实跑过 |
| checkpoint 路径 | 能证明有训练产物 |
| baseline 结果 JSON | 单轮对照 |
| API Planner 结果 JSON | 教师上限 |
| RL 前后本地 Planner 结果 JSON | 主要对比 |
| 典型成功/失败轨迹 | 用来解释机制和问题 |
| checkpoint 曲线 | 用来说明训练稳定性 |
