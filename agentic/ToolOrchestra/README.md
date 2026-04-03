# ToolOrchestra — Agentic RL with Slime

Reproduction of [ToolOrchestra](https://arxiv.org/abs/2511.21689) based on the slime framework. An **Orchestrator-Expert** multi-agent framework for RL training. A central Orchestrator LLM learns to route tasks to the best specialized expert model and the corresponding tools through multi-turn tool calls. GRPO is applied to the Orchestrator's decision trajectory, enabling it to improve tool-use and routing capabilities without manually annotated intermediate steps.

[中文文档](./README_zh.md)

---

## Results

| Model | Dataset | Baseline (Qwen3-8B) | ToolOrchestra (Ours) | Improvement |
|---|---|---|---|---|
| Qwen3-8B | τ²-Bench | 0.278 | 0.388 | +0.110 |

---

## 0. Prerequisites

### LLM API Key

The τ² User Simulator and the QA reward judge both call LLMs via the DashScope (Alibaba Cloud Bailian) API. Set the following environment variable before running:

```bash
export DASHSCOPE_API_KEY=<your-dashscope-api-key>

# Optional overrides (defaults are set, usually no need to change)
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export DASHSCOPE_MODEL=qwen-turbo-latest        # model used by the user simulator
export QA_REWARD_JUDGE_MODEL=qwen-turbo-latest  # model used for QA reward judging
```

> You can obtain a DashScope API key from the [Bailian Console](https://bailian.console.aliyun.com/).

### Environment Setup

Two conda environment requirement files are provided in this directory:

- `orche_requirement.txt` — dependencies for the **Orchestrator** environment (`orche`)
- `sglang_requirement.txt` — dependencies for the **SGLang** expert-serving environment (`sglang`)

Create and set up the environments:

```bash
# Create and activate the orche environment
conda create -n orche python=3.10 -y
conda activate orche
pip install -r agentic/ToolOrchestra/orche_requirement.txt

# Create and activate the sglang environment
conda create -n sglang python=3.10 -y
conda activate sglang
pip install -r agentic/ToolOrchestra/sglang_requirement.txt
```

> Use the `orche` environment for retrieval service, and the `sglang` environment for launching expert SGLang services.

---

## 1. Download Model

```bash
huggingface-cli download Qwen/Qwen3-8B \
  --local-dir /data/models/qwen3_8b
```

### Retrieval service (FAISS dense retrieval)

The HTTP retrieval service (`retrieval_general_thought.py`, port 8000) encodes queries with **[Qwen/Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)** and searches a FAISS index over a JSONL corpus.

**Default paths in code** ([`retrieval_general_thought.py`](./retrieval_general_thought.py)):

| Item | Default |
|---|---|
| Embedding checkpoint | `/data/models/qwen3_8b_emb` (same weights as `Qwen/Qwen3-Embedding-8B`; the code also accepts the HF id string directly) |
| Index directory (`INDEX_DIR` env) | `/data/dataset/index` (set by [`launch.sh`](./launch.sh) and [`eval_orchestra.sh`](./eval_orchestra.sh)) |
| FAISS index file | `{INDEX_DIR}/train.index` |
| Passage corpus | `{INDEX_DIR}/train.jsonl` |

**Corpus source:** download `train.index` and `train.jsonl` from the Hugging Face dataset **[multi-train/index](https://huggingface.co/datasets/multi-train/index)** and place them under your `INDEX_DIR` (defaults to `/data/dataset/index`).

Example:

```bash
# Embedding model (match the default path expected by retrieval_general_thought.py)
huggingface-cli download Qwen/Qwen3-Embedding-8B \
  --local-dir /data/models/qwen3_8b_emb

# FAISS index + passages (training retrieval uses train.*; avoid downloading the full dataset — wiki.* is very large)
mkdir -p /data/dataset/index
huggingface-cli download multi-train/index \
  --repo-type dataset \
  --local-dir /data/dataset/index \
  train.index train.jsonl
```

Override the index location with `INDEX_DIR` when starting the retrieval service; override the embedding path by editing `retrieval_model_path` in `retrieval_general_thought.py` (or extend the script to read from an environment variable).

## 2. Convert Model Format

slime training requires converting the HuggingFace checkpoint to Megatron distributed format:

```bash
cd /path/to/slime-agentic
source scripts/models/qwen3-8B.sh

python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /data/models/qwen3_8b \
  --save /data/qwen3_8b_dist/
```

Convert back to HuggingFace format after training:

```bash
cd /path/to/slime-agentic
source scripts/models/qwen3-8B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  ${MODEL_ARGS[@]} \
  --load /data/checkpoints/orchestra_qwen3_8b_rl/ \
  --hf-checkpoint /data/models/qwen3_8b \
  --save /data/orchestra_qwen3_8b_hf/
```

- `--load`: Path to the torch_dist checkpoint saved during training
- `--hf-checkpoint`: Original HuggingFace model path (used to fill in config files)
- `--save`: Output path for the converted HuggingFace checkpoint

## 3. Training

### 3.1 Start Expert Services

ToolOrchestra requires a retrieval service and multiple expert SGLang services running before training. The recommended setup (8 GPUs) uses `launch.sh` to manage everything:

```bash
bash agentic/ToolOrchestra/launch.sh
```

`launch.sh` starts all expert services on GPUs 0–3 and then launches training on GPUs 4–7 automatically.

| GPU | Service | Port |
|---|---|---|
| 0 | Retrieval service (FAISS) | 8000 |
| 0 | Qwen3-32B-FP8 | 30001 |
| 0 | Qwen2.5-Math-7B | 30003 |
| 1 | DeepSeek-R1-Distill-32B | 30005 |
| 1 | Qwen3-30B-A3B | 30006 |
| 2 | Qwen2.5-Coder-32B | 30002 |
| 2 | Qwen3-14B | 30007 |
| 3 | Qwen2.5-Math-72B | 30004 |
| 4–7 | Training: Qwen3-8B Orchestrator (TP=2, DP=2) | — |

### 3.2 Launch Training Only

If expert services are already running, start training directly:

```bash
cd /path/to/slime-agentic
CUDA_VISIBLE_DEVICES=4,5,6,7 SKIP_PROCESS_KILL=1 \
  bash agentic/ToolOrchestra/train_orchestra.sh
```

Checkpoints are saved to `/data/checkpoints/orchestra_qwen3_8b_rl/` by default. This can be changed in the `CKPT_ARGS` section of the script.

## 4. Key Training Parameters

| Parameter | Value | Description |
|---|---|---|
| `--advantage-estimator` | `grpo` | Advantage estimation algorithm |
| `--lr` | `1e-6` | Learning rate |
| `--n-samples-per-prompt` | `8` | Number of samples per prompt |
| `--rollout-batch-size` | `32` | Rollout batch size |
| `--global-batch-size` | `128` | Global batch size |
| `--rollout-temperature` | `0.7` | Sampling temperature |
| `--rollout-max-response-len` | `16384` | Max tokens per Orchestrator response |
| `--kl-loss-coef` | `0.001` | KL divergence coefficient |
| `--eps-clip` / `--eps-clip-high` | `0.2` / `0.3` | PPO clip range |
| `--sglang-context-length` | `131072` | SGLang context length (128K) |

## 5. Evaluation

### 5.1 Convert Checkpoint to HF Format

Before evaluation, convert the saved torch_dist checkpoint to HuggingFace format using the one-click script:

```bash
# Convert the latest checkpoint
bash agentic/ToolOrchestra/convert_to_hf.sh

# Convert a specific iteration
SINGLE_ITER=iter_0000129 bash agentic/ToolOrchestra/convert_to_hf.sh

# Convert all checkpoints
CONVERT_ALL=1 bash agentic/ToolOrchestra/convert_to_hf.sh

# Convert and immediately run evaluation
bash agentic/ToolOrchestra/convert_to_hf.sh && \
  ORCH_CKPT=/data/checkpoints/orchestra_qwen3_8b_rl_hf/iter_0000129 bash agentic/ToolOrchestra/eval_orchestra.sh
```

The converted checkpoints are saved to `/data/checkpoints/orchestra_qwen3_8b_rl_hf/` by default. Already-converted iterations are automatically skipped.

### 5.2 Start All Services

Use the eval script to automatically start all required services and run evaluation:

```bash
# Evaluate on tau2 benchmark (default)
ORCH_CKPT=/data/checkpoints/orchestra_qwen3_8b_rl_hf/iter_0000129 bash agentic/ToolOrchestra/eval_orchestra.sh

# Evaluate on FRAMES benchmark
ORCH_CKPT=/data/checkpoints/orchestra_qwen3_8b_rl_hf/iter_0000129 BENCHMARK=frames bash agentic/ToolOrchestra/eval_orchestra.sh

# Evaluate on HLE benchmark
ORCH_CKPT=/data/checkpoints/orchestra_qwen3_8b_rl_hf/iter_0000129 BENCHMARK=hle bash agentic/ToolOrchestra/eval_orchestra.sh

# Quick smoke test (5 samples)
ORCH_CKPT=/data/checkpoints/orchestra_qwen3_8b_rl_hf/iter_0000129 MAX_EXAMPLES=5 bash agentic/ToolOrchestra/eval_orchestra.sh
```

Results are saved to `/data/eval_results/{benchmark}_{timestamp}/`.

### 5.2 Skip Service Startup (Services Already Running)

```bash
# Expert services already running, only start Orchestrator
SKIP_EXPERT_SERVICES=1 ORCH_CKPT=/data/checkpoints/orchestra_qwen3_8b_rl bash eval_orchestra.sh

# All services already running
SKIP_SERVICES=1 ORCH_URL=http://127.0.0.1:30000/v1 \
  ORCH_MODEL=Qwen/Qwen3-8B bash eval_orchestra.sh
```

## 6. Architecture

```
Input question
  │
  ▼
Orchestrator LLM                        ← Decide which tool to call (loss_mask=1)
  │
  └─► for turn in range(max_turns):
        │
        ├─ parse_tool_call()            ← Parse <tool_call> from model output
        │
        ├─ tool call                    ← Call retrieval / external tool (loss_mask=0)
        │    └─ FAISS retrieval service (port 8000)
        │
        ├─ call_expert ──────────────► Expert LLM routing (loss_mask=0)
        │                               ├─ Qwen3-32B-FP8      (port 30001)
        │                               ├─ Qwen2.5-Coder-32B  (port 30002)
        │                               ├─ Qwen2.5-Math-7B    (port 30003)
        │                               ├─ Qwen2.5-Math-72B   (port 30004)
        │                               ├─ DeepSeek-R1-32B    (port 30005)
        │                               ├─ Qwen3-30B-A3B      (port 30006)
        │                               └─ Qwen3-14B          (port 30007)
        │
        └─ answer ──────────────────► Final answer → stop loop
  │
  ▼
GenerationOutput
  - token_ids + log_probs  (all turns concatenated)
  - loss_mask: Orchestrator output = 1 / tool result = 0
```

**Two task modes:**
- **QA**: Orchestrator drives the loop autonomously until `answer` is called
- **Func_call**: Loop is driven by the tau2 simulation environment (file-based IPC)

## 7. Training Configuration

- **Algorithm**: GRPO with KL divergence constraint (`low_var_kl`)
- **Orchestrator model**: Qwen3-8B (replaceable)
- **Expert models**: Fixed specialist LLMs running on separate SGLang ports
- **Retrieval**: FAISS-based dense retrieval service
- **Task types**: QA reasoning + Function-call simulation (tau2)
- **Inference engine**: SGLang
