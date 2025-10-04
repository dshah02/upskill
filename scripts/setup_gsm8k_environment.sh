#!/bin/bash

# GSM8K Environment Setup Script for MISL Paper
# Sets up the environment based on the GRPO notebook

set -e

echo "Setting up GSM8K experiment environment..."

# Check if we're in a virtual environment or conda environment
if [[ -z "$VIRTUAL_ENV" && -z "$CONDA_DEFAULT_ENV" ]]; then
    echo "Warning: No virtual environment detected. Consider using a virtual environment."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Installing required packages..."

# Core dependencies from GRPO notebook
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Unsloth for efficient training
pip install unsloth

# vLLM for fast inference (specific version required)
pip install vllm==0.8.2

# Transformers and related packages
pip install transformers>=4.44.0
pip install datasets
pip install accelerate
pip install peft  # for LoRA fine-tuning

# Additional dependencies for GRPO
pip install deepspeed
pip install wandb  # for experiment tracking

# Math and utility packages
pip install numpy
pip install matplotlib
pip install seaborn
pip install pandas
pip install scipy

# Hugging Face packages
pip install huggingface_hub

echo "Verifying installation..."

# Check core packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import datasets; print(f'Datasets: {datasets.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"

echo "Environment setup completed!"
echo ""
echo "Next steps:"
echo "1. Run cache_models.sh to download required models"
echo "2. Run run_gsm8k_experiments.sh to train and evaluate"
echo ""
echo "Note: You may need to configure wandb for experiment tracking:"
echo "wandb login"