# 训练脚本详细分析

这份文档只分析训练相关脚本：它们如何串起来、每个参数为什么存在、背后的训练技术原理，以及脚本里用到的 Bash/Python 语法。

## 1. 总体调用链

常规训练：

```text
agentic/agentflow/launch.sh
  -> agentic/agentflow/train_agentflow.sh
    -> ray job submit
      -> python3 train.py
        -> slime 创建 Ray placement group / rollout manager / actor model
          -> rollout.generate
          -> rollout.reward_func
          -> custom_convert.custom_convert
```

两张 4090 快速测试：

```text
agentic/agentflow/quick_train_2x4090.sh
  -> GPU 1 启动 teacher SGLang server
  -> GPU 0 调用 train_agentflow.sh
    -> ray job submit
      -> python3 train.py
```

4xA800 正式训练：

```text
agentic/agentflow/formal_train_4xa800.sh
  -> 外部 teacher 模式: 使用 RM_URL
  -> local teacher 模式: GPU3 启动 teacher SGLang server
  -> GPU0-3 或 GPU0-2 调用 train_agentflow.sh
    -> ray job submit
      -> python3 train.py
```

模型结构参数：

```text
agentic/agentflow/model_configs/*.sh
  -> 被 train_agentflow.sh source
  -> 生成 MODEL_ARGS
  -> 传给 slime/Megatron 初始化 student 模型
```

核心设计是“薄仓库、厚外部框架”：本仓库只提供 AgentFlow 的自定义 rollout/reward/convert，完整分布式训练、权重同步、SGLang rollout engine 管理由外部安装的 slime 完成。

## 2. `train.py`

文件：`train.py`

这是 Python 层训练入口。它不是 AgentFlow 专属训练器，而是一个很薄的 slime 调用壳。

### 2.1 关键职责

```python
args = parse_args()
train(args)
```

`parse_args()` 来自 slime，它负责解析 `train_agentflow.sh` 最后拼出来的所有命令行参数。这样 AgentFlow 不需要自己维护完整训练参数系统。

```python
pgs = create_placement_groups(args)
```

Ray placement group 用来声明 GPU/CPU 资源如何分配给不同角色。对 RL 训练来说，至少有：

- actor/trainer: 更新 student 参数。
- rollout: SGLang engine 采样 student 的 response。
- 可选 critic: 如果启用 PPO critic，会创建 critic model。

placement group 的技术意义是避免 Ray 随机调度造成资源碎片，保证训练和 rollout worker 拿到预期 GPU。

```python
rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
```

rollout manager 负责调用自定义生成函数：

```bash
--custom-generate-function-path rollout.generate
```

也就是说，slime 的 manager 负责分布式调度，AgentFlow 的 `rollout.generate` 负责具体“一条样本怎么生成多轮 agent trajectory”。

```python
actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
```

actor model 是被训练的 student。critic model 默认不一定启用；GRPO 通常不需要 critic。

```python
actor_model.update_weights()
```

训练一开始先把 actor 权重同步到 rollout engine。技术上，rollout engine 是 SGLang 服务，它生成样本时必须用最新 student 权重；否则训练会变成“用旧策略采样，新策略更新”，off-policy 偏差会更大。

### 2.2 训练循环

```python
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
    actor_model.update_weights()
```

每轮做三件事：

1. rollout: student/Planner 生成 AgentFlow 多轮轨迹。
2. train: 使用 reward、logprob、loss mask 更新 actor。
3. sync: 把 actor 新权重同步回 SGLang rollout engine。

为什么是这个顺序：

- 先 rollout 才有 on-policy 数据。
- 用当前策略数据训练 GRPO/OPD。
- 训练后同步权重，下一轮 rollout 才能反映新策略。

### 2.3 eval/save/offload

```python
should_run_periodic_action(...)
```

这个函数根据 rollout id、interval、epoch 边界判断是否执行保存或评测。这样 `save_interval` 和 `eval_interval` 可以按训练步稳定触发。

```python
if args.offload_rollout:
    ray.get(rollout_manager.offload.remote())
```

offload 用于显存紧张场景，把 rollout 侧权重或 KV cache 暂时卸载。4090 quick test 默认不启用，因为脚本已经把 teacher 和 student 分到不同 GPU，并把 batch 设置得很小。

## 3. `train_agentflow.sh`

文件：`agentic/agentflow/train_agentflow.sh`

这是最重要的 shell 脚本。它把“用户友好的环境变量”转换成 “slime 训练器需要的命令行参数”。

### 3.1 Bash 安全开关

```bash
set -ex
```

- `-e`: 任意命令失败就退出，避免错误后继续训练。
- `-x`: 打印执行的命令，方便排查最终参数。

这里没有使用 `set -u`，因为大量参数允许通过 `${VAR:-default}` 读取未定义变量；`set -u` 会让一些外部环境差异更容易变成硬错误。

### 3.2 进程清理

```bash
if [ "${SKIP_PROCESS_KILL}" != "1" ]; then
    pkill -9 -f 'sglang\.launch_server' 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    pkill -9 ray 2>/dev/null || true
fi
```

训练前清理旧 Ray/SGLang 是为了避免：

- 旧 Ray head 占用 8265。
- 旧 SGLang server 占用 GPU 显存。
- 旧 worker 还在持有端口或 CUDA context。

`|| true` 的意思是“命令失败也不让脚本失败”。例如没有旧进程时 `pkill` 会返回非 0，但这是正常情况。

quick test 会设置 `SKIP_PROCESS_KILL=1`，因为它自己先启动 teacher SGLang。如果 `train_agentflow.sh` 再清理 SGLang，就会把 teacher 一起杀掉。

### 3.3 路径定位

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
```

这两行让脚本无论从哪个目录调用，都能找到自己的真实位置。

- `${BASH_SOURCE[0]}`: 当前脚本路径。
- `dirname`: 取脚本所在目录。
- `cd ... && pwd`: 转成绝对路径。
- `--`: 防止路径以 `-` 开头时被当成命令选项。

### 3.4 加载模型结构

```bash
MODEL_CONFIG_SCRIPT=${MODEL_CONFIG_SCRIPT:-"${SCRIPT_DIR}/model_configs/qwen2.5-1.5B.sh"}
source "${MODEL_CONFIG_SCRIPT}"
```

`source` 会在当前 shell 里执行模型配置脚本，让它定义 `MODEL_ARGS` 数组。

为什么模型结构要单独脚本化：

- Megatron 需要用结构参数实例化模型。
- `REF_LOAD` 的 torch_dist checkpoint 必须和这些结构参数一致。
- 不同 student 只需要切换 `MODEL_CONFIG_SCRIPT`，训练主脚本不用改。

### 3.5 Bash 默认值语法

```bash
TRAIN_GPUS=${TRAIN_GPUS:-4}
ROLLOUT_ENGINE_GPUS=${ROLLOUT_ENGINE_GPUS:-1}
```

`${VAR:-default}` 表示：如果 `VAR` 未设置或为空，就使用 `default`。

这让脚本同时支持：

```bash
bash train_agentflow.sh
```

和：

```bash
TRAIN_GPUS=2 ROLLOUT_BATCH_SIZE=1 bash train_agentflow.sh
```

### 3.6 布尔解析函数

```bash
is_true() {
   case "${1:-}" in
      1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
      *) return 1 ;;
   esac
}
```

Bash 里函数返回 `0` 表示 true，非 0 表示 false。这个函数允许用户用 `USE_OPD=1`、`USE_OPD=true`、`USE_OPD=yes` 等多种写法。

### 3.7 参数数组

脚本大量使用 Bash array：

```bash
CKPT_ARGS=(
   --hf-checkpoint ${BASE_HF_CHECKPOINT}
   --ref-load ${REF_LOAD}
   --save ${SAVE_DIR}
)
```

技术上这比拼字符串更安全，因为每个数组元素都是一个命令行 token。最后用：

```bash
${CKPT_ARGS[@]}
```

把数组展开成多个参数。

### 3.8 checkpoint 参数

```bash
--hf-checkpoint ${BASE_HF_CHECKPOINT}
--ref-load ${REF_LOAD}
--save ${SAVE_DIR}
--save-interval ${SAVE_INTERVAL:-100}
```

- `hf-checkpoint`: tokenizer、HF config、可能的初始化参考。
- `ref-load`: slime/Megatron torch_dist 格式 student checkpoint。
- `save`: 训练输出目录。
- `save-interval`: 每多少 rollout 保存一次。

`ref-load` 对 KL loss 很重要，因为 GRPO 的 KL 正则需要 reference model。OPD quick test 默认 `USE_KL_LOSS=0`，因为主要学习信号来自 teacher KL。

### 3.9 rollout 参数

```bash
--prompt-data ${PROMPT_DATA}
--input-key ${INPUT_KEY:-prompt}
--label-key ${LABEL_KEY:-label}
--rollout-batch-size ${ROLLOUT_BATCH_SIZE:-8}
--n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-8}
--rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-32768}
--global-batch-size ${GLOBAL_BATCH_SIZE:-64}
```

含义：

- `rollout-batch-size`: 每个 rollout step 取多少个 prompt。
- `n-samples-per-prompt`: 每个 prompt 采样多少条 trajectory。GRPO 需要同题多样本做组内比较。
- `global-batch-size`: trainer 侧全局 batch。AgentFlow 会把一条 trajectory 拆成多个 Planner turn，所以实际训练样本数会被 `custom_convert` 重新整理。
- `rollout-max-response-len`: 单条 rollout 最大 response token 数。AgentFlow 多轮会更长，显存紧张时先降这个。

```bash
if [ -n "${NUM_ROLLOUT:-}" ]; then
   ROLLOUT_ARGS+=(--num-rollout ${NUM_ROLLOUT})
else
   ROLLOUT_ARGS+=(--num-epoch ${NUM_EPOCH:-1})
fi
```

`NUM_ROLLOUT` 更适合 quick test，因为它直接控制训练轮数。`NUM_EPOCH` 更适合完整训练，因为它按数据集大小推导 rollout 数。

### 3.10 eval 参数

```bash
if ! is_true "${DISABLE_EVAL:-0}"; then
   EVAL_ARGS=(...)
fi
```

评测默认开启。quick test 用很小数据时，`EVAL_INTERVAL=1` 可以验证 eval hook 是否能跑。正式训练如果只想先看训练链路，可以设置：

```bash
DISABLE_EVAL=1
```

### 3.11 performance 参数

```bash
--tensor-model-parallel-size ${TRAIN_TP}
--sequence-parallel
--use-dynamic-batch-size
--max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-16384}
```

- `TRAIN_TP`: Tensor Parallel。小模型和 4090 quick test 通常用 1。
- `sequence-parallel`: 配合 Megatron 并行切分，减少部分激活显存。
- `use-dynamic-batch-size`: 按 token 数动态打包，避免长短样本混在一起造成 OOM。
- `max-tokens-per-gpu`: 动态 batch 的核心显存阈值。OOM 时优先降它。

### 3.12 GRPO 参数

```bash
--advantage-estimator grpo
--eps-clip 0.2
--eps-clip-high 0.3
```

GRPO 的核心思想是：同一个 prompt 采样多条 response，用组内 reward 做相对优势，不需要 critic。它适合数学/工具调用类任务，因为同题多样本天然可以比较。

`eps-clip`/`eps-clip-high` 是 PPO/GRPO 类目标里的 ratio clip，防止一次更新把策略推太远。

```bash
if is_true "${USE_KL_LOSS:-1}"; then
   GRPO_ARGS+=(--use-kl-loss --kl-loss-coef ${KL_LOSS_COEF:-0.001})
fi
```

KL loss 让 student 不要偏离 reference 太远。quick OPD 默认关掉 `USE_KL_LOSS`，因为 teacher OPD 已经额外提供 KL 信号，快速测试先保持简单。

### 3.13 OPD 参数

```bash
if [ -n "${RM_URL:-}" ]; then
   OPD_ARGS+=(--rm-url ${RM_URL})
fi
if is_true "${USE_OPD:-0}"; then
   OPD_ARGS+=(--use-opd --opd-type ${OPD_TYPE:-sglang} --opd-kl-coef ${OPD_KL_COEF:-1.0})
fi
```

OPD 这里是 on-policy distillation：

1. student 生成 AgentFlow trajectory。
2. `rollout.reward_func` 把每个 Planner turn 的 token id 发给 teacher SGLang `/generate`。
3. teacher 返回 input token logprob。
4. `custom_convert` 把 teacher logprob 对齐到每个 turn。
5. trainer 计算 student 与 teacher 的 KL，使 student 在自己采样出来的轨迹上向 teacher 分布靠近。

为什么用 `RM_URL`：slime 原本把 reward model 服务地址叫 `rm-url`。这里复用这个参数，但实际返回的是 teacher logprob，不是普通 scalar reward。

### 3.14 custom hook 参数

```bash
--custom-generate-function-path rollout.generate
--custom-rm-path rollout.reward_func
--custom-eval-rollout-log-function-path rollout.eval_log
--custom-convert-samples-to-train-data-path custom_convert.custom_convert
```

这是 AgentFlow 接入 slime 的关键。

- `rollout.generate`: 把单步 LLM rollout 改成 Planner/Executor/Verifier 多轮 agent loop。
- `rollout.reward_func`: 普通模式做答案匹配；OPD 模式请求 teacher logprob。
- `rollout.eval_log`: 记录 eval 明细。
- `custom_convert.custom_convert`: 把一条多轮 trajectory 拆成多个 Planner turn，每个 turn 都能训练。

如果没有 `custom_convert`，整条多轮轨迹会被当成一个长样本；这会让 prompt/token/loss mask 和 teacher logprob 对齐更困难。

### 3.15 Ray runtime env

```bash
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${AGENTFLOW_RUNTIME_PYTHONPATH}\",
    ...
  }
}"
```

Ray job 是另一个运行环境。你在当前 shell 里 `export PYTHONPATH` 不一定自动进入 Ray worker，所以脚本显式构造 runtime env。

`AGENTFLOW_RUNTIME_PYTHONPATH` 包含：

1. `agentic/agentflow`: 让 Ray worker 能 import `rollout`、`custom_convert`、`core`。
2. `MEGATRON_PATH`: 让 Megatron import 可见。
3. `SLIME_PATH`: 如果 slime 是源码安装，让 worker 能 import slime。
4. 原始 `PYTHONPATH`: 保留用户环境。

### 3.16 Ray job submit

```bash
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node ${TRAIN_GPUS} \
   --colocate \
   ...
```

`--` 之后是 Ray job 真正执行的命令。前面是 Ray CLI 参数，后面是训练程序参数。

`--colocate` 表示训练和 rollout 角色可以按 slime 的 colocate 策略共享节点资源。小模型/单机 quick test 通常希望 colocate，减少跨节点复杂度。

## 4. `launch.sh`

文件：`agentic/agentflow/launch.sh`

这是常规训练的外层 launcher，职责是让用户有一个简单入口，并把日志统一写到文件。

### 4.1 严格模式

```bash
set -euo pipefail
```

- `-e`: 出错退出。
- `-u`: 使用未定义变量时报错。
- `pipefail`: 管道中任一命令失败则整体失败。

这个脚本比 `train_agentflow.sh` 更适合严格模式，因为它只处理少量高层变量。

### 4.2 日志目录

```bash
LOG_DIR=${LOG_DIR:-"/tmp/agentflow_logs"}
mkdir -p "$LOG_DIR"
```

训练输出重定向到：

```text
/tmp/agentflow_logs/train.log
```

这样终端只显示启动状态，详细训练日志用 `tail -f` 看。

### 4.3 端口等待

```bash
wait_port "ray-dashboard" 127.0.0.1 8265 180
```

`wait_port` 用 `/dev/tcp/host/port` 检查服务端口是否可连接。Ray job submit 会启动 dashboard，外层脚本等待 8265 可用后告诉用户日志路径。

### 4.4 API 默认值

```bash
export AGENTFLOW_API_MODEL=${EXECUTOR_MODEL:-"gpt-4o-mini"}
export AGENTFLOW_API_BASE=${AGENTFLOW_API_BASE:-"${EXECUTOR_API_BASE:-https://api.openai.com/v1}"}
```

这里兼容旧变量 `EXECUTOR_*`，同时统一落到新的 `AGENTFLOW_API_*`。这些变量会被 `train_agentflow.sh` 写进 Ray runtime env，最终被 `rollout.py` 里的 `APIEngine` 读取。

### 4.5 后台训练进程

```bash
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} SKIP_PROCESS_KILL=1 \
    bash "${SCRIPT_DIR}/train_agentflow.sh" \
    > "$LOG_DIR/train.log" 2>&1 &
```

关键点：

- `CUDA_VISIBLE_DEVICES`: 限制训练脚本能看到哪些 GPU。
- `SKIP_PROCESS_KILL=1`: 外层已经清理过进程，内层不要重复清理。
- `> train.log 2>&1`: stdout 和 stderr 都写进日志。
- `&`: 后台运行，外层可以继续等待端口和监控 exit code。

## 5. `quick_train_2x4090.sh`

文件：`agentic/agentflow/quick_train_2x4090.sh`

这是为你的测试资源专门准备的 smoke test：两张 4090、teacher 2B、student 0.8B 级别。

### 5.1 GPU 分工

```bash
STUDENT_CUDA_VISIBLE_DEVICES=${STUDENT_CUDA_VISIBLE_DEVICES:-0}
TEACHER_CUDA_VISIBLE_DEVICES=${TEACHER_CUDA_VISIBLE_DEVICES:-1}
```

默认：

- GPU 0: student trainer + student Planner rollout。
- GPU 1: teacher SGLang server。

为什么这样分：

- teacher 只做 logprob，不更新参数。
- student 需要训练和 rollout 权重同步。
- 4090 没有 NVLink，强行多卡并行反而可能引入通信复杂度；quick test 先做单卡 student。

### 5.2 teacher SGLang server

```bash
CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}" \
    python3 -m sglang.launch_server \
    --model-path "${TEACHER_HF_CHECKPOINT}" \
    --host "${TEACHER_HOST}" \
    --port "${TEACHER_PORT}" \
    --tp-size 1 \
    --context-length "${TEACHER_CONTEXT_LENGTH}" \
    --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}"
```

teacher 用 SGLang 是因为它需要高效返回 token logprob。请求 payload 中会设置：

```json
{
  "input_ids": [...],
  "return_logprob": true,
  "logprob_start_len": 0,
  "sampling_params": {"max_new_tokens": 0}
}
```

`max_new_tokens=0` 表示 teacher 不生成新文本，只对输入序列计算 logprob。这就是 OPD teacher 的技术核心。

### 5.3 quick training 默认值

```bash
NUM_ROLLOUT=1
ROLLOUT_BATCH_SIZE=1
N_SAMPLES_PER_PROMPT=2
GLOBAL_BATCH_SIZE=2
ROLLOUT_MAX_RESPONSE_LEN=2048
SGLANG_CONTEXT_LENGTH=8192
MAX_TOKENS_PER_GPU=4096
USE_OPD=1
USE_KL_LOSS=0
```

这些值是为了“先跑通链路”：

- `NUM_ROLLOUT=1`: 只跑一轮。
- `ROLLOUT_BATCH_SIZE=1`: 每轮只取一个 prompt。
- `N_SAMPLES_PER_PROMPT=2`: GRPO 至少有同题多样本。
- `GLOBAL_BATCH_SIZE=2`: 和两条样本对齐。
- `ROLLOUT_MAX_RESPONSE_LEN=2048`: 降低 4090 OOM 风险。
- `USE_OPD=1`: 验证 teacher logprob 链路。
- `USE_KL_LOSS=0`: 避免 reference KL 和 OPD KL 同时叠加，先看 OPD 是否正常。

### 5.4 为什么 quick test 仍需要 API

student Planner 是本地训练模型；但 AgentFlow 的 Executor、Verifier、工具内部 LLM 默认走 OpenAI-compatible API。quick test 仍要设置：

```bash
AGENTFLOW_API_BASE=...
AGENTFLOW_API_KEY=...
AGENTFLOW_API_MODEL=...
```

这符合当前设计：只训练 Planner，辅助角色固定，降低显存占用。

## 6. model config 脚本

文件：

- `agentic/agentflow/model_configs/qwen3-0.6B.sh`
- `agentic/agentflow/model_configs/qwen2.5-0.5B.sh`
- `agentic/agentflow/model_configs/qwen2.5-1.5B.sh`

这些脚本只定义一个数组：

```bash
MODEL_ARGS=(...)
```

### 6.1 为什么需要结构参数

Megatron 不是直接读 HF `config.json` 来实例化所有结构，而是通过命令行参数描述模型：

- 层数：`--num-layers`
- 隐藏维度：`--hidden-size`
- FFN 维度：`--ffn-hidden-size`
- attention heads：`--num-attention-heads`
- GQA 分组：`--num-query-groups`
- vocab size：`--vocab-size`
- RoPE base：`--rotary-base`
- RMSNorm 等结构开关

这些参数必须和 `REF_LOAD` checkpoint 一致。否则轻则 load 失败，重则训练 silent mismatch。

### 6.2 student 0.8B 怎么处理

仓库提供的是常见小模型示例，不保证刚好等于你的 0.8B student。你应该：

1. 复制最接近的配置。
2. 按 student HF config 修改结构参数。
3. 用同一份 `MODEL_ARGS` 做 HF -> torch_dist 转换。
4. 训练时设置 `MODEL_CONFIG_SCRIPT` 指向这份配置。

示例：

```bash
cp agentic/agentflow/model_configs/qwen3-0.6B.sh \
   agentic/agentflow/model_configs/student-0.8B.sh

export MODEL_CONFIG_SCRIPT=$PWD/agentic/agentflow/model_configs/student-0.8B.sh
```

## 7. 最重要的参数关系

### 7.1 `CUDA_VISIBLE_DEVICES` 与 `TRAIN_GPUS`

`CUDA_VISIBLE_DEVICES=0` 表示进程只看见一张卡；此时脚本内部的 GPU 0 就是物理 GPU 0。

`TRAIN_GPUS=1` 告诉 Ray/slime 为 actor 节点申请 1 张 GPU。

两者必须一致。比如：

```bash
CUDA_VISIBLE_DEVICES=0,1 TRAIN_GPUS=2
```

表示训练进程可见两张卡，并申请两张卡。

quick test 中 student 是：

```bash
CUDA_VISIBLE_DEVICES=0 TRAIN_GPUS=1
```

teacher 单独是：

```bash
CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server ...
```

### 7.2 `BASE_HF_CHECKPOINT` 与 `REF_LOAD`

- `BASE_HF_CHECKPOINT`: HF 格式。
- `REF_LOAD`: Megatron/slime torch_dist 格式。

训练前必须用同一份 model config 把 HF 转成 torch_dist。否则模型结构和 checkpoint 参数不一致。

### 7.3 `RM_URL` 与 `USE_OPD`

```bash
USE_OPD=1
RM_URL=http://127.0.0.1:30080/generate
```

二者要一起出现。`USE_OPD=1` 告诉 slime 训练时使用 OPD KL；`RM_URL` 告诉 `rollout.reward_func` 去哪里拿 teacher logprob。

如果没有 `RM_URL`，reward 会退回普通答案匹配，只返回 scalar `score`。

### 7.4 tokenizer 兼容性

OPD SGLang 模式把 student token id 发给 teacher。teacher 必须能用同一 tokenizer 解释这些 token id。最好 teacher/student 来自同一模型家族，或者明确共享 tokenizer。

这是比模型大小更重要的约束。

## 8. 推荐阅读顺序

第一次读脚本建议按这个顺序：

1. `docs/QUICK_TRAIN_2X4090.md`: 先知道如何跑通。
2. `agentic/agentflow/quick_train_2x4090.sh`: 看 GPU 和 teacher/student 分工。
3. `docs/FORMAL_TRAIN_4XA800.md`: 再看正式训练部署。
4. `agentic/agentflow/formal_train_4xa800.sh`: 看 A800 GPU 拓扑和 teacher 模式。
5. `agentic/agentflow/train_agentflow.sh`: 看所有训练参数如何拼出来。
6. `train.py`: 看 slime 训练循环如何消费这些参数。
7. `agentic/agentflow/rollout.py`: 看 AgentFlow 如何接入 rollout/reward。
8. `agentic/agentflow/custom_convert.py`: 看多轮 trajectory 如何变成训练 batch。

## 9. `formal_train_4xa800.sh`

文件：`agentic/agentflow/formal_train_4xa800.sh`

这是正式训练环境的封装脚本，面向单机 4 张 A800。它不是替代 `train_agentflow.sh`，而是给正式训练设置更合适的 GPU 拓扑、batch 默认值、teacher 部署模式和日志路径。

### 9.1 两种 teacher 模式

默认模式是外部 teacher：

```bash
RM_URL=http://teacher-host:30080/generate \
bash agentic/agentflow/formal_train_4xa800.sh
```

这时：

- `STUDENT_CUDA_VISIBLE_DEVICES=0,1,2,3`
- `TRAIN_GPUS=4`
- teacher 不在训练机脚本内启动
- 4 张 A800 全部用于 student 训练和 student Planner rollout

这是正式训练更推荐的模式，因为 teacher 和训练任务解耦，teacher 服务可以独立重启、复用和监控。

本机 local teacher 模式：

```bash
LOCAL_TEACHER=1 \
TEACHER_HF_CHECKPOINT=/data/models/teacher-2b \
bash agentic/agentflow/formal_train_4xa800.sh
```

这时：

- `STUDENT_CUDA_VISIBLE_DEVICES=0,1,2`
- `TEACHER_CUDA_VISIBLE_DEVICES=3`
- `TRAIN_GPUS=3`
- 脚本自动启动 teacher SGLang server，并设置 `RM_URL=http://127.0.0.1:30080/generate`

这个模式适合只有一台 4xA800 的情况。

### 9.2 GPU 数自动推导

```bash
count_csv_items() {
    local value="${1// /}"
    if [ -z "$value" ]; then
        echo 0
    else
        awk -F',' '{print NF}' <<< "$value"
    fi
}
```

脚本用这个函数从 `STUDENT_CUDA_VISIBLE_DEVICES` 推导 `TRAIN_GPUS`。例如：

- `0,1,2,3` -> 4
- `0,1,2` -> 3

这样用户改 GPU 列表时，不必手动同步修改 `TRAIN_GPUS`。

### 9.3 为什么正式默认 batch 与 quick test 不同

外部 teacher、4 student GPU：

```bash
ROLLOUT_BATCH_SIZE=8
N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE=64
```

local teacher、3 student GPU：

```bash
ROLLOUT_BATCH_SIZE=6
N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE=48
```

这些值保持 `rollout_batch_size * n_samples_per_prompt == global_batch_size`，先让每轮 rollout 的 trajectory 数量和训练全局 batch 对齐。AgentFlow 后续还会按 turn 展开样本，但这个关系能让正式训练的第一层 batch 规模更可控。

3 GPU local teacher 模式下用 48，是为了更适配数据并行切分；64 对 3 张卡不整齐。

### 9.4 正式上下文长度

```bash
ROLLOUT_MAX_RESPONSE_LEN=8192
SGLANG_CONTEXT_LENGTH=32768
MAX_TOKENS_PER_GPU=16384
```

A800 显存比 4090 宽裕，但正式训练仍建议从中等长度开始。AgentFlow 是多轮轨迹，prompt、planner 输出、工具结果和 verifier 结果都会拉长序列；一上来开到 65536 context 更容易把问题混在一起，不利于定位。

跑通后可以逐步扩大：

```bash
ROLLOUT_MAX_RESPONSE_LEN=16384
SGLANG_CONTEXT_LENGTH=65536
MAX_TOKENS_PER_GPU=32768
```

### 9.5 进程清理策略

外部 teacher 模式默认不杀本机 SGLang：

```bash
CLEAN_SGLANG=${CLEAN_SGLANG:-${LOCAL_TEACHER}}
```

原因是：如果你手动在本机启动了一个 teacher 服务，并选择 `LOCAL_TEACHER=0` 但 `RM_URL=127.0.0.1:...`，脚本不应该默认误杀它。

local teacher 模式默认清理旧 SGLang，因为脚本要自己重新启动 teacher，避免端口和显存冲突。
