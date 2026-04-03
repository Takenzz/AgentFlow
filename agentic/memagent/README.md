# MemAgent — Agentic RL with Slime

Reproduction of MemAgent based on the slime framework, applying GRPO to chunk-by-chunk memory-update trajectories for RL training on long-context question answering tasks.

[中文文档](./README_zh.md)

---

> **Troubleshooting — script fails to start?**
> The most common cause is a model path mismatch. All scripts use the following default paths:
> - Base model: `/data/models/qwen25_7b`
> - Reference model: `/data/models/qwen2.5_7b_dist/`
>
> If your models are stored elsewhere, update the paths at the top of the relevant script, or pass them via environment variables (e.g. `MODEL_PATH=/your/path bash run_memagent_7b.sh`).

---

## 1. Download Datasets

```bash
# Training dataset (HotpotQA — via HuggingFace)
huggingface-cli download --repo-type dataset BytedTsinghua-SIA/hotpotqa \
  --local-dir /data/hotpotqa_hf

# Evaluation dataset (RULER-HQA)
# Place eval_<length>.json files under /data/hotpotqa (see prepare_data.py for format)
```

## 2. Download Models

```bash
# Base model (training + evaluation)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir /data/models/qwen25_7b
```

## 3. Convert Model Format

slime training requires converting HuggingFace checkpoints to Megatron distributed format:

```bash
cd /path/to/slime-agentic
source scripts/models/qwen2.5-7B.sh

python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /data/models/qwen25_7b \
  --save /data/qwen2.5_7b_dist/
```

## 4. Prepare Training Data

Use `prepare_data.py` to convert the HotpotQA dataset to slime-compatible JSONL format.

**From a local parquet file:**
```bash
python agentic/memagent/prepare_data.py \
    --input  /data/hotpotqa_hf/hotpotqa_train_process.parquet \
    --output /data/hotpotqa_slime/train.jsonl
```

**Directly from HuggingFace:**
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

Output JSONL fields: `prompt` (question), `label` (answer), `metadata.context` (long document).

## 5. Training

MemAgent training uses SGLang in **colocate mode** — no separate inference server needs to be started. The training script handles everything in one command:

```bash
bash agentic/memagent/run_memagent_7b.sh
```

Model paths are configured at the top of the script via `MODEL_PATH`, `REF_PATH`, and `SAVE_PATH`. Checkpoints are saved to `/data/MemAgent_Qwen25-7B-RL/` by default.

**Override paths via environment variables:**
```bash
MODEL_PATH=/data/my_model \
SAVE_PATH=/data/my_output \
bash agentic/memagent/run_memagent_7b.sh
```

**Override memory hyperparameters:**
```bash
MEM_CHUNK_TOKENS=3000 MEM_MAX_MEMORY=512 bash agentic/memagent/run_memagent_7b.sh
```

## 6. Key Training Parameters

| Parameter | Value | Description |
|---|---|---|
| `--advantage-estimator` | `grpo` | Advantage estimation algorithm |
| `--lr` | `1e-6` | Learning rate |
| `--n-samples-per-prompt` | `16` | Number of samples per prompt |
| `--rollout-batch-size` | `16` | Rollout batch size |
| `--rollout-temperature` | `1.0` | Sampling temperature |
| `--kl-loss-coef` | `0.001` | KL divergence coefficient |
| `--eps-clip` / `--eps-clip-high` | `0.2` / `0.3` | PPO clip range |
| `--sglang-context-length` | `131072` | SGLang context length (YaRN enabled) |
| `MEM_CHUNK_TOKENS` | `5000` | Tokens per document chunk |
| `MEM_MAX_MEMORY` | `1024` | Max tokens for memory output per turn |
| `MEM_MAX_FINAL` | `256` | Max tokens for the final answer |
| `MEM_MAX_CHUNKS` | `512` | Max number of chunks per document |

## 7. Evaluation

### 7.1 Convert Checkpoints to HF Format

Training saves checkpoints in Megatron distributed format. Convert them to HuggingFace format before evaluation.

#### Batch Conversion (Recommended)

Use `convert_memagent_to_hf.sh` to convert **all** `iter_*` checkpoints at once. Already-converted checkpoints are automatically skipped:

```bash
bash agentic/memagent/convert_memagent_to_hf.sh
```

Key paths configured at the top of the script:

| Variable | Default | Description |
|---|---|---|
| `CHECKPOINT_DIR` | `/data/MemAgent_Qwen25-7B-RL` | Source directory containing `iter_*` Megatron checkpoints |
| `OUTPUT_BASE` | `/data/MemAgent_Qwen25-7B-RL-HF` | Output root; each checkpoint saved as `OUTPUT_BASE/iter_xxxxx/` |
| `ORIGIN_HF_DIR` | `/data/models/qwen25_7b` | Original HuggingFace model (used to fill in config files) |

A summary is printed at the end listing any failed conversions.

#### Single Checkpoint

To convert a specific checkpoint:

```bash
SINGLE_ITER=iter_0000299 bash agentic/memagent/convert_memagent_to_hf.sh
```

Or manually:

```bash
cd /path/to/slime-agentic
source scripts/models/qwen2.5-7B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir  /data/MemAgent_Qwen25-7B-RL/iter_0000299 \
    --output-dir /data/MemAgent_Qwen25-7B-RL-HF/iter_0000299 \
    --origin-hf-dir /data/models/qwen25_7b
```

- `--input-dir`: Path to the Megatron checkpoint saved during training
- `--output-dir`: Output path for the converted HuggingFace checkpoint
- `--origin-hf-dir`: Original HuggingFace model path (used to fill in config files)

### 7.2 One-Click Evaluation (Recommended)

Use `run_eval.sh` to automatically start a SGLang server, run evaluation, and shut down the server on exit:

```bash
MODEL_PATH=/data/MemAgent_Qwen25-7B-RL-HF/iter_0000299 bash agentic/memagent/run_eval.sh
```

`MODEL_PATH` is required. All other parameters have defaults and can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | *(required)* | Path to the HF checkpoint to evaluate |
| `TP` | `1` | SGLang tensor parallel size |
| `SERVE_PORT` | `8000` | SGLang server port |
| `TASKS` | `hqa` | Evaluation suite: `hqa` \| `general` \| `all` |
| `LENGTH` | `50 100 200 400 800 1600 3200 6400` | Number of documents for RULER-HQA |
| `DATA_ROOT` | `/data/hotpotqa` | Directory containing `eval_<length>.json` files |
| `RULER_DATA_ROOT` | `/data/ruler` | Directory containing RULER general split files |
| `SAVE_DIR` | `results/` | Output directory for evaluation results |
| `N_PROC` | `64` | Number of concurrent requests |
| `API` | `recurrent` | Inference mode: `recurrent` \| `openai` |
| `FORCE` | `0` | Set to `1` to re-evaluate ignoring cached results |

**Run only specific lengths:**
```bash
MODEL_PATH=/data/my_ckpt TASKS=hqa LENGTH="50 200 800" bash agentic/memagent/run_eval.sh
```

**Run the full RULER benchmark:**
```bash
MODEL_PATH=/data/my_ckpt TASKS=all bash agentic/memagent/run_eval.sh
```

Results are saved under `SAVE_DIR/ruler_hqa_<length>/` for HQA tasks and `SAVE_DIR/ruler_<split>_<length>/` for general tasks.

### 7.3 Manual Evaluation

If the SGLang server is already running, call the evaluation scripts directly:

**RULER-HQA:**
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

**RULER General:**
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

## 8. Batch Evaluation Across All Checkpoints

After batch-converting checkpoints to HF format (see §7.1), use `eval_all_checkpoints.sh` to evaluate every checkpoint automatically. For each checkpoint, the SGLang server is restarted and `RUNS` independent evaluations are performed per length. Already-evaluated checkpoints are skipped on re-runs.

```bash
bash agentic/memagent/eval_all_checkpoints.sh
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `CKPT_DIR` | `/data/MemAgent_Qwen25-7B-RL-HF` | Directory containing converted `iter_*` HF checkpoints |
| `LENGTH` | `50` | Space-separated list of doc counts to evaluate |
| `RUNS` | `5` | Number of evaluation runs per checkpoint per length |
| `DATA_ROOT` | `/data/hotpotqa_dataset/files` | Directory containing `eval_<length>.json` files |
| `SAVE_BASE` | `results/checkpoint_sweep` | Output root directory |
| `TP` | `1` | SGLang tensor parallel size |
| `N_PROC` | `64` | Number of concurrent requests |

Per-checkpoint results are written to `SAVE_BASE/<checkpoint>/result_length<N>.txt`:

```
checkpoint: iter_0000299
length:     200
runs:       75.78,72.66,75.78,75.0,71.88
max:        75.78
avg:        74.22
```

A summary TSV is appended to `SAVE_BASE/summary.tsv` after each checkpoint finishes.
