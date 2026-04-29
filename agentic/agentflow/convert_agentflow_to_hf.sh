#!/bin/bash
# Convert all AgentFlow Megatron checkpoints to HF format

CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/data/AgentFlow_Qwen25-1.5B-RL"}
OUTPUT_BASE=${OUTPUT_BASE:-"/data/AgentFlow_Qwen25-1.5B-RL-HF"}
ORIGIN_HF_DIR=${ORIGIN_HF_DIR:-"/data/models/qwen25_1.5b"}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-"/mnt/models/scipt/framework/Megatron-LM"}
CONVERT_SCRIPT=${CONVERT_SCRIPT:-"/data/slime-agentic/tools/convert_torch_dist_to_hf.py"}

mkdir -p "$OUTPUT_BASE"

# Collect all iter_* directories
ITERS=($(ls -d "$CHECKPOINT_DIR"/iter_* 2>/dev/null | sort))

if [ ${#ITERS[@]} -eq 0 ]; then
    echo "No iter_* checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Found ${#ITERS[@]} checkpoints to convert:"
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
        --input-dir "$iter_path" \
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
