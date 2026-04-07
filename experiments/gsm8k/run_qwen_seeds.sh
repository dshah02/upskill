#!/bin/bash

# Run Qwen control and experimental GSM8K training across multiple seeds.
# Each (config, seed) pair is run sequentially since each run occupies the GPU.
#
# Usage:
#   bash run_qwen_seeds.sh              # run all 6 jobs sequentially
#   bash run_qwen_seeds.sh --dry-run    # print commands without executing

set -e

# ---- Configuration ----
SEEDS=(42 123 7)
CONFIGS=(
    "config_qwen_control.yaml"
    "config_qwen_experimental.yaml"
)
MODEL="qwen"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# ------------------------

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

TOTAL=$(( ${#CONFIGS[@]} * ${#SEEDS[@]} ))
RUN_NUM=0

echo "=============================================="
echo " Qwen multi-seed GSM8K experiments"
echo " Configs: ${CONFIGS[*]}"
echo " Seeds:   ${SEEDS[*]}"
echo " Total runs: $TOTAL"
echo "=============================================="
echo ""

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        RUN_NUM=$((RUN_NUM + 1))
        echo "----------------------------------------------"
        echo "[$RUN_NUM/$TOTAL] config=$config  seed=$seed"
        echo "----------------------------------------------"

        CMD="python ${SCRIPT_DIR}/full_train.py \
            --config ${SCRIPT_DIR}/${config} \
            --model ${MODEL} \
            --seed ${seed}"

        if $DRY_RUN; then
            echo "[dry-run] $CMD"
        else
            echo "Starting: $(date)"
            eval $CMD
            echo "Finished: $(date)"
        fi
        echo ""
    done
done

echo "=============================================="
echo " All $TOTAL runs complete."
echo "=============================================="
