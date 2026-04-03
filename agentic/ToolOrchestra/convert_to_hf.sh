#!/bin/bash
# 将 ToolOrchestra Orchestrator Megatron checkpoint 转换为 HuggingFace 格式
# 转换后的模型可直接用于 SGLang 推理（eval_orchestra.sh 的 ORCH_CKPT 参数）
#
# 用法：
#   # 转换最新 checkpoint
#   bash convert_to_hf.sh
#
#   # 转换所有 checkpoint
#   CONVERT_ALL=1 bash convert_to_hf.sh
#
#   # 指定特定 iter
#   SINGLE_ITER=iter_0000129 bash convert_to_hf.sh
#
#   # 转换后直接运行评测
#   bash convert_to_hf.sh && ORCH_CKPT=/data/checkpoints/orchestra_qwen3_8b_rl_hf/iter_0000129 bash eval_orchestra.sh

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/checkpoints/orchestra_qwen3_8b_rl}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/checkpoints/orchestra_qwen3_8b_rl_hf}"
ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-/data/models/qwen3_8b}"
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-/root/Megatron-LM}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-/data/slime-agentic/tools/convert_torch_dist_to_hf.py}"
SLIME_DIR="${SLIME_DIR:-/data/slime-agentic}"

mkdir -p "$OUTPUT_BASE"

# 确定要转换的 iter 列表
if [[ -n "${SINGLE_ITER:-}" ]]; then
    ITERS=("$CHECKPOINT_DIR/$SINGLE_ITER")
elif [[ "${CONVERT_ALL:-0}" == "1" ]]; then
    mapfile -t ITERS < <(ls -d "$CHECKPOINT_DIR"/iter_* 2>/dev/null | sort)
else
    # 默认只转最新的 checkpoint
    LATEST=$(cat "$CHECKPOINT_DIR/latest_checkpointed_iteration.txt" 2>/dev/null | tr -d '[:space:]')
    if [[ -z "$LATEST" ]]; then
        echo "[ERROR] 无法读取 latest_checkpointed_iteration.txt"
        exit 1
    fi
    iter_num=$(printf "%07d" "$LATEST")
    ITERS=("$CHECKPOINT_DIR/iter_${iter_num}")
fi

if [ ${#ITERS[@]} -eq 0 ]; then
    echo "No iter_* checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Found ${#ITERS[@]} checkpoint(s) to convert:"
for iter_path in "${ITERS[@]}"; do
    echo "  $iter_path"
done
echo ""

FAILED=()

for iter_path in "${ITERS[@]}"; do
    iter_name=$(basename "$iter_path")
    output_dir="$OUTPUT_BASE/$iter_name"

    if [ -d "$output_dir" ] && [ -f "$output_dir/config.json" ]; then
        echo "[SKIP] $iter_name — already converted at $output_dir"
        continue
    fi

    echo "[CONVERTING] $iter_name -> $output_dir"

    PYTHONPATH="$MEGATRON_LM_DIR:$SLIME_DIR" \
        /usr/bin/python3 "$CONVERT_SCRIPT" \
        --input-dir    "$iter_path" \
        --output-dir   "$output_dir" \
        --origin-hf-dir "$ORIGIN_HF_DIR"

    if [ $? -eq 0 ]; then
        echo "[DONE] $iter_name -> $output_dir"
    else
        echo "[FAILED] $iter_name"
        FAILED+=("$iter_name")
        rm -rf "$output_dir" 2>/dev/null || true
    fi
    echo ""
done

echo "===== Summary ====="
echo "Total: ${#ITERS[@]}, Failed: ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Failed checkpoints:"
    for f in "${FAILED[@]}"; do
        echo "  $f"
    done
    exit 1
fi

echo "All conversions completed successfully."
echo "Output directory: $OUTPUT_BASE"
echo ""
echo "Run eval with:"
LATEST_HF=$(ls -d "$OUTPUT_BASE"/iter_* 2>/dev/null | sort | tail -1)
echo "  ORCH_CKPT=$LATEST_HF bash eval_orchestra.sh"
