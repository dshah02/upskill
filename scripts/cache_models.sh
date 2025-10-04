#!/bin/bash

# Model Caching Script for GSM8K Experiments
# Downloads and caches models used in the MISL paper

set -e

echo "Caching models for GSM8K experiments..."

# Check if cache_for_offline.py exists
if [[ ! -f "../experiments/gsm8k/cache_for_offline.py" ]]; then
    echo "Error: cache_for_offline.py not found in experiments/gsm8k directory"
    echo "Please ensure you have the cache_for_offline.py script"
    exit 1
fi

# Create cache directory if it doesn't exist
CACHE_DIR="../data/cached_models"
mkdir -p "$CACHE_DIR"

echo "Downloading models mentioned in the paper..."

# Qwen 2.5-7B-Instruct
echo "Downloading Qwen 2.5-7B-Instruct..."
python ../experiments/gsm8k/cache_for_offline.py --model qwen

# Llama 3.1-8B-Instruct
echo "Downloading Llama 3.1-8B-Instruct..."
python ../experiments/gsm8k/cache_for_offline.py --model llama

# R1-Distilled-Qwen2.5-Math-1.5B
echo "Downloading R1-Distilled-Qwen2.5-Math-1.5B..."
python ../experiments/gsm8k/cache_for_offline.py --model r1-distilled

# Download GSM8K dataset
echo "Downloading GSM8K dataset..."
python -c "
from datasets import load_dataset
dataset = load_dataset('gsm8k', 'main')
print('GSM8K dataset cached successfully')
print(f'Train set size: {len(dataset[\"train\"])}')
print(f'Test set size: {len(dataset[\"test\"])}')
"

echo "Model caching completed!"
echo ""
echo "Cached models:"
echo "  - Qwen/Qwen2.5-7B-Instruct"
echo "  - meta-llama/Llama-3.1-8B-Instruct"
echo "  - R1-Distilled-Qwen2.5-Math-1.5B"
echo "  - GSM8K dataset"
echo ""
echo "You can now run GSM8K experiments with run_gsm8k_experiments.sh"