#!/bin/bash

# GSM8K Training and Evaluation Script for MISL Paper
# Reproduces the GSM8K experiments with GRPO + MISL

set -e

echo "Running GSM8K experiments with MISL..."

# Check if required files exist
if [[ ! -f "../experiments/gsm8k/full_train.py" ]]; then
    echo "Error: full_train.py not found in experiments/gsm8k"
    exit 1
fi

if [[ ! -f "../experiments/gsm8k/benchmark.py" ]]; then
    echo "Error: benchmark.py not found in experiments/gsm8k"
    exit 1
fi

# Parameters from the paper
MAX_Z=5
ALPHA_MI=5.0
MODELS=("qwen" "llama" "r1-distilled")

# Create results directory
RESULTS_DIR="../results/gsm8k_results"
mkdir -p "$RESULTS_DIR"

echo "Running experiments with parameters: max_z=$MAX_Z, alpha_mi=$ALPHA_MI"

# Run experiments for each model
for model in "${MODELS[@]}"; do
    echo ""
    echo "================================================"
    echo "Training and evaluating model: $model"
    echo "================================================"

    # Create model-specific results directory
    MODEL_DIR="$RESULTS_DIR/$model"
    mkdir -p "$MODEL_DIR"

    # Run training
    echo "Training $model with MISL..."
    python ../experiments/gsm8k/full_train.py \
        --model "$model" \
        --max_z $MAX_Z \
        --alpha_mi $ALPHA_MI \
        > "$MODEL_DIR/training.log" 2>&1

    if [[ $? -eq 0 ]]; then
        echo "Training completed for $model"

        # Run evaluation on first 100 problems
        echo "Evaluating $model on first 100 GSM8K problems..."
        python ../experiments/gsm8k/benchmark.py \
            --model "$model" \
            --max_z $MAX_Z \
            --num_problems 100 \
            > "$MODEL_DIR/evaluation.log" 2>&1

        if [[ $? -eq 0 ]]; then
            echo "Evaluation completed for $model"
        else
            echo "Warning: Evaluation failed for $model"
        fi
    else
        echo "Warning: Training failed for $model"
    fi
done

# Optional: Run semantic mutual information experiments
echo ""
echo "================================================"
echo "Optional: Testing semantic mutual information"
echo "================================================"

read -p "Run semantic MI experiments? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ALPHA_SMI_VALUES=(0.1 0.5 1.0)

    for alpha_smi in "${ALPHA_SMI_VALUES[@]}"; do
        echo "Running with alpha_smi=$alpha_smi..."
        python ../experiments/gsm8k/full_train.py \
            --model "qwen" \
            --max_z $MAX_Z \
            --alpha_mi $ALPHA_MI \
            --alpha_smi $alpha_smi \
            > "$RESULTS_DIR/semantic_mi_$alpha_smi.log" 2>&1
    done
fi

echo ""
echo "GSM8K experiments completed!"
echo "Results saved in $RESULTS_DIR/"
echo ""
echo "Directory structure:"
for model in "${MODELS[@]}"; do
    if [[ -d "$RESULTS_DIR/$model" ]]; then
        echo "  $RESULTS_DIR/$model/"
        echo "    ├── training.log"
        echo "    └── evaluation.log"
    fi
done
echo ""
echo "To analyze results, check the log files for:"
echo "  - pass@1 accuracy"
echo "  - pass@k improvements"
echo "  - consensus@k improvements"