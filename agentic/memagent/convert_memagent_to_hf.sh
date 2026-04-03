#!/bin/bash
# Convert all MemAgent Megatron checkpoints to HF format
#
# 用法：
#   bash convert_memagent_to_hf.sh                  # 用默认路径转换全部
#   CHECKPOINT_DIR=/data/my_ckpt bash convert_memagent_to_hf.sh
#   SINGLE_ITER=iter_0000399 bash convert_memagent_to_hf.sh  # 只转一个

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/MemAgent_Qwen25-7B-RL}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/MemAgent_Qwen25-7B-RL-HF}"
ORIGIN_HF_DIR="${ORIGIN_HF_DIR:-/data/models/qwen25_7b}"
MEGATRON_LM_DIR="${MEGATRON_LM_DIR:-/root/Megatron-LM}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-/data/slime-agentic/tools/convert_torch_dist_to_hf.py}"

mkdir -p "$OUTPUT_BASE"

# 支持只转指定 iter（SINGLE_ITER=iter_0000399）
if [[ -n "${SINGLE_ITER:-}" ]]; then
    ITERS=("$CHECKPOINT_DIR/$SINGLE_ITER")
else
    mapfile -t ITERS < <(ls -d "$CHECKPOINT_DIR"/iter_* 2>/dev/null | sort)
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

    PYTHONPATH="$MEGATRON_LM_DIR" python "$CONVERT_SCRIPT" \
        --input-dir  "$iter_path" \
        --output-dir "$output_dir" \
        --origin-hf-dir "$ORIGIN_HF_DIR"

    if [ $? -eq 0 ]; then
        echo "[DONE] $iter_name"
    else
        echo "[FAILED] $iter_name"
        FAILED+=("$iter_name")
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
