# MemAgent — Agentic RL with Slime

基于 slime 框架复现 MemAgent，在长上下文问答任务上使用 GRPO 对逐 chunk 记忆更新轨迹进行强化学习训练。

[English](./README.md)

---

> **启动失败排查**
> 最常见的原因是模型路径与实际不符。所有脚本使用以下默认路径：
> - 基础模型：`/data/models/qwen25_7b`
> - 参考模型：`/data/models/qwen2.5_7b_dist/`
>
> 如果你的模型存放在其他位置，请修改对应脚本顶部的路径变量，或通过环境变量传入（例如 `MODEL_PATH=/your/path bash run_memagent_7b.sh`）。

---

## 0. 环境安装

本目录提供一个 conda 环境依赖文件：

- `sglang_requirement.txt` — **SGLang** 推理环境（`sglang`）的依赖

创建并配置环境：

```bash
conda create -n sglang python=3.10 -y
conda activate sglang
pip install -r agentic/memagent/sglang_requirement.txt
```

> 训练和评测阶段启动 SGLang 推理服务时，均使用 `sglang` 环境。

## 1. 下载数据集

```bash
# 训练集（HotpotQA，通过 HuggingFace 下载）
huggingface-cli download --repo-type dataset BytedTsinghua-SIA/hotpotqa \
  --local-dir /data/hotpotqa_hf

# 评测集（RULER-HQA）
# 将 eval_<length>.json 文件放置于 /data/hotpotqa 下（格式见 prepare_data.py）
```

## 2. 下载模型

```bash
# 基础模型（训练 + 评测均使用）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/models/qwen25_7b
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

## 4. 准备训练数据

使用 `prepare_data.py` 将 HotpotQA 数据集转换为 slime 兼容的 JSONL 格式。

**从本地 parquet 文件转换：**
```bash
python agentic/memagent/prepare_data.py \
    --input  /data/hotpotqa_hf/hotpotqa_train_process.parquet \
    --output /data/hotpotqa_slime/train.jsonl
```

**直接从 HuggingFace 拉取：**
```bash
python agentic/memagent/prepare_data.py \
    --hf-dataset BytedTsinghua-SIA/hotpotqa \
    --hf-split   train \
    --output     /data/hotpotqa_slime/train.jsonl

python agentic/memagent/prepare_data.py \
    --hf-dataset BytedTsinghua-SIA/hotpotqa \
    --hf-split   dev \
    --output     /data/hotpotqa_slime/dev.jsonl
```

输出 JSONL 字段：`prompt`（问题）、`label`（答案）、`metadata.context`（长文档）。

## 5. 启动训练

MemAgent 训练使用 SGLang 的**共置模式（colocate）**，无需单独启动推理服务，训练脚本一键搞定：

```bash
bash agentic/memagent/run_memagent_7b.sh
```

模型路径通过脚本顶部的 `MODEL_PATH`、`REF_PATH`、`SAVE_PATH` 变量配置。权重默认保存至 `/data/MemAgent_Qwen25-7B-RL/`。

**通过环境变量覆盖路径：**
```bash
MODEL_PATH=/data/my_model \
SAVE_PATH=/data/my_output \
bash agentic/memagent/run_memagent_7b.sh
```

**覆盖记忆超参数：**
```bash
MEM_CHUNK_TOKENS=3000 MEM_MAX_MEMORY=512 bash agentic/memagent/run_memagent_7b.sh
```

## 6. 关键训练参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--advantage-estimator` | `grpo` | 优势估计算法 |
| `--lr` | `1e-6` | 学习率 |
| `--n-samples-per-prompt` | `16` | 每条 prompt 采样数 |
| `--rollout-batch-size` | `16` | rollout batch size |
| `--rollout-temperature` | `1.0` | 采样温度 |
| `--kl-loss-coef` | `0.001` | KL 散度系数 |
| `--eps-clip` / `--eps-clip-high` | `0.2` / `0.3` | PPO clip 范围 |
| `--sglang-context-length` | `131072` | SGLang 上下文长度（启用 YaRN） |
| `MEM_CHUNK_TOKENS` | `5000` | 每个文档 chunk 的 token 数 |
| `MEM_MAX_MEMORY` | `1024` | 每轮记忆输出的最大 token 数 |
| `MEM_MAX_FINAL` | `256` | 最终答案的最大 token 数 |
| `MEM_MAX_CHUNKS` | `512` | 每篇文档的最大 chunk 数 |

## 7. 评测

### 7.1 转换 Checkpoint 格式

训练保存的是 Megatron 分布式格式，评测前需先转换为 HuggingFace 格式。

#### 批量转换（推荐）

使用 `convert_memagent_to_hf.sh` 一次性转换所有 `iter_*` checkpoint，已转换的自动跳过：

```bash
bash agentic/memagent/convert_memagent_to_hf.sh
```

脚本顶部可配置的路径变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CHECKPOINT_DIR` | `/data/MemAgent_Qwen25-7B-RL` | 存放 `iter_*` Megatron checkpoint 的源目录 |
| `OUTPUT_BASE` | `/data/MemAgent_Qwen25-7B-RL-HF` | 输出根目录，每个 checkpoint 保存为 `OUTPUT_BASE/iter_xxxxx/` |
| `ORIGIN_HF_DIR` | `/data/models/qwen25_7b` | 原始 HuggingFace 模型路径（用于补全配置文件） |

脚本结束后会打印汇总信息，列出失败的 checkpoint。

#### 单个转换

只转换某一个 checkpoint：

```bash
SINGLE_ITER=iter_0000299 bash agentic/memagent/convert_memagent_to_hf.sh
```

或手动转换：

```bash
cd /path/to/slime-agentic
source scripts/models/qwen2.5-7B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir  /data/MemAgent_Qwen25-7B-RL/iter_0000299 \
    --output-dir /data/MemAgent_Qwen25-7B-RL-HF/iter_0000299 \
    --origin-hf-dir /data/models/qwen25_7b
```

- `--input-dir`：训练产出的 Megatron checkpoint 路径
- `--output-dir`：转换后的 HuggingFace 格式保存路径
- `--origin-hf-dir`：原始 HuggingFace 模型路径（用于补全配置文件）

### 7.2 一键评测（推荐）

使用 `run_eval.sh` 自动启动 SGLang 服务、执行评测、退出时自动关闭服务：

```bash
MODEL_PATH=/data/MemAgent_Qwen25-7B-RL-HF/iter_0000299 bash agentic/memagent/run_eval.sh
```

`MODEL_PATH` 为必填项，其余参数均有默认值，可通过环境变量覆盖：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_PATH` | *（必填）* | 待评测的 HF checkpoint 路径 |
| `TP` | `1` | SGLang tensor parallel size |
| `SERVE_PORT` | `8000` | SGLang 服务端口 |
| `TASKS` | `hqa` | 评测套件：`hqa` \| `general` \| `all` |
| `LENGTH` | `50 100 200 400 800 1600 3200 6400` | RULER-HQA 的文档数列表 |
| `DATA_ROOT` | `/data/hotpotqa` | 存放 `eval_<length>.json` 的目录 |
| `RULER_DATA_ROOT` | `/data/ruler` | 存放 RULER general split 文件的目录 |
| `SAVE_DIR` | `results/` | 评测结果输出目录 |
| `N_PROC` | `64` | 并发请求数 |
| `API` | `recurrent` | 推理模式：`recurrent` \| `openai` |
| `FORCE` | `0` | 设为 `1` 则忽略缓存重新评测 |

**只跑特定长度：**
```bash
MODEL_PATH=/data/my_ckpt TASKS=hqa LENGTH="50 200 800" bash agentic/memagent/run_eval.sh
```

**跑完整 RULER benchmark：**
```bash
MODEL_PATH=/data/my_ckpt TASKS=all bash agentic/memagent/run_eval.sh
```

结果保存至 `SAVE_DIR/ruler_hqa_<length>/`（HQA 任务）和 `SAVE_DIR/ruler_<split>_<length>/`（general 任务）。

### 7.3 手动评测

如果 SGLang 服务已在运行，可直接调用评测脚本：

**RULER-HQA：**
```bash
python agentic/memagent/eval_ruler_hqa.py \
    --model     /data/MemAgent_Qwen25-7B-RL-HF/iter_0000299 \
    --tokenizer /data/MemAgent_Qwen25-7B-RL-HF/iter_0000299 \
    --length    200 \
    --data-root /data/hotpotqa \
    --save-dir  results/ruler_hqa_200 \
    --save-file iter_0000299 \
    --api       recurrent \
    --n-proc    64
```

**RULER General：**
```bash
python agentic/memagent/eval_ruler_general.py \
    --model     /data/MemAgent_Qwen25-7B-RL-HF/iter_0000299 \
    --tokenizer /data/MemAgent_Qwen25-7B-RL-HF/iter_0000299 \
    --split     niah_single_1 \
    --length    32768 \
    --data-root /data/ruler \
    --save-dir  results/ruler_niah_single_1_32768 \
    --save-file iter_0000299 \
    --api       recurrent \
    --n-proc    64
```

## 8. 批量评测所有 Checkpoint

将所有 checkpoint 批量转换为 HF 格式后（见 §7.1），使用 `eval_all_checkpoints.sh` 自动遍历所有 checkpoint，每个 checkpoint 依次重启 SGLang 服务并跑 `RUNS` 次评测。已评测过的 checkpoint 再次运行时自动跳过。

```bash
bash agentic/memagent/eval_all_checkpoints.sh
```

**可配置的环境变量：**

| 变量 | 默认值 | 说明 |
|---|---|---|
| `CKPT_DIR` | `/data/MemAgent_Qwen25-7B-RL-HF` | 存放转换后 `iter_*` HF checkpoint 的目录 |
| `LENGTH` | `50` | 要评测的文档数列表（空格分隔） |
| `RUNS` | `5` | 每个 checkpoint 每个 length 的重复评测次数 |
| `DATA_ROOT` | `/data/hotpotqa_dataset/files` | 存放 `eval_<length>.json` 的目录 |
| `SAVE_BASE` | `results/checkpoint_sweep` | 结果输出根目录 |
| `TP` | `1` | SGLang tensor parallel size |
| `N_PROC` | `64` | 并发请求数 |

每个 checkpoint 的结果写入 `SAVE_BASE/<checkpoint>/result_length<N>.txt`：

```
checkpoint: iter_0000299
length:     200
runs:       75.78,72.66,75.78,75.0,71.88
max:        75.78
avg:        74.22
```

每个 checkpoint 评测完成后，汇总行追加至 `SAVE_BASE/summary.tsv`。
