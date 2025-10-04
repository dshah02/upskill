#!/bin/bash

# Arithmetic Experiments Script for MISL Paper
# Reproduces experiments from Section 5.1 and Appendix E.6, F

set -e

echo "Setting up environment for arithmetic experiments..."

# Check if PyTorch is installed
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null || {
    echo "Error: PyTorch not found. Please install PyTorch first:"
    echo "pip install torch"
    exit 1
}

# Create output directory
OUTPUT_DIR="../results/arithmetic_results"
mkdir -p "$OUTPUT_DIR"

echo "Running arithmetic experiments..."
echo "Note: Results may vary slightly between CPU and GPU"

# Figure 4 experiments
echo "Running Figure 4 experiments..."

echo "Running with MI training (alpha=0.5, beta=0.5)..."
python ../experiments/arithmetic/arithmetic_llm.py --alpha 0.5 --beta 0.5 --N 5 --warmup 100 --steps 2000 --C 5 --cap 1. \
    > "$OUTPUT_DIR/figure4_with_mi.log" 2>&1

echo "Running without MI training (alpha=0.0, beta=0.0)..."
python ../experiments/arithmetic/arithmetic_llm.py --alpha 0.0 --beta 0.0 --N 5 --warmup 100 --steps 2000 --C 5 --cap 1. \
    > "$OUTPUT_DIR/figure4_without_mi.log" 2>&1

# Appendix E.6 Sensitivity experiments
echo "Running Appendix E.6 sensitivity experiments..."

echo "Running with alpha=1.0, beta=1.0..."
python ../experiments/arithmetic/arithmetic_llm.py --alpha 1.0 --beta 1.0 --N 5 --warmup 100 --steps 2000 --C 5 --cap 1. \
    > "$OUTPUT_DIR/sensitivity_alpha1.0.log" 2>&1

echo "Running with cap=1.5..."
python ../experiments/arithmetic/arithmetic_llm.py --alpha 0.5 --beta 0.5 --N 5 --warmup 100 --steps 2000 --C 5 --cap 1.5 \
    > "$OUTPUT_DIR/sensitivity_cap1.5.log" 2>&1

# Appendix F: Ablated Arithmetic Environment
echo "Running Appendix F ablated experiments..."

echo "Running with KL coefficient ablation..."
python ../experiments/arithmetic/arithmetic_llm.py --alpha 0.5 --beta 0.5 --N 5 --warmup 100 --steps 2000 --C 5 --cap 1. --kl_coef 0.05 \
    > "$OUTPUT_DIR/ablated_kl0.05.log" 2>&1

echo "Arithmetic experiments completed!"
echo "Results saved in $OUTPUT_DIR/"
echo ""
echo "Log files created:"
echo "  - figure4_with_mi.log: Figure 4 with MI training"
echo "  - figure4_without_mi.log: Figure 4 without MI training"
echo "  - sensitivity_alpha1.0.log: Sensitivity to alpha=1.0"
echo "  - sensitivity_cap1.5.log: Sensitivity to cap=1.5"
echo "  - ablated_kl0.05.log: Ablated experiment with KL coefficient"