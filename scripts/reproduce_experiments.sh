#!/bin/bash

# Main script to reproduce all MISL paper experiments
# Usage: ./reproduce_experiments.sh [arithmetic|gsm8k|all]

set -e

EXPERIMENT_TYPE="${1:-all}"

echo "MISL Paper - Experiment Reproduction Script"
echo "=========================================="
echo ""

case "$EXPERIMENT_TYPE" in
    "arithmetic")
        echo "Running only arithmetic experiments..."
        ./run_arithmetic_experiments.sh
        ;;
    "gsm8k")
        echo "Running only GSM8K experiments..."
        echo "Step 1: Setting up environment..."
        ./setup_gsm8k_environment.sh
        echo ""
        echo "Step 2: Caching models..."
        ./cache_models.sh
        echo ""
        echo "Step 3: Running training and evaluation..."
        ./run_gsm8k_experiments.sh
        ;;
    "all")
        echo "Running all experiments..."
        echo ""
        echo "=== ARITHMETIC EXPERIMENTS ==="
        ./run_arithmetic_experiments.sh
        echo ""
        echo "=== GSM8K EXPERIMENTS ==="
        echo "Step 1: Setting up environment..."
        ./setup_gsm8k_environment.sh
        echo ""
        echo "Step 2: Caching models..."
        ./cache_models.sh
        echo ""
        echo "Step 3: Running training and evaluation..."
        ./run_gsm8k_experiments.sh
        ;;
    *)
        echo "Usage: $0 [arithmetic|gsm8k|all]"
        echo ""
        echo "Options:"
        echo "  arithmetic  - Run only arithmetic experiments (Section 5.1)"
        echo "  gsm8k      - Run only GSM8K experiments (main results)"
        echo "  all        - Run all experiments (default)"
        echo ""
        echo "Individual scripts:"
        echo "  ./run_arithmetic_experiments.sh     - Arithmetic experiments"
        echo "  ./setup_gsm8k_environment.sh       - Setup GSM8K environment"
        echo "  ./cache_models.sh                  - Cache models offline"
        echo "  ./run_gsm8k_experiments.sh         - GSM8K training & evaluation"
        exit 1
        ;;
esac

echo ""
echo "Experiments completed!"
echo ""
echo "Results can be found in:"
echo "  - results/arithmetic_results/ (arithmetic experiments)"
echo "  - results/gsm8k_results/ (GSM8K experiments)"