# UpSkill

This repository contains scripts to reproduce the experiments from the paper "UpSkill: Mutual Information Skill Learning for Large Language Models".

## Quick Start

Run all experiments:
```bash
./reproduce_experiments.sh
```

Or run specific experiments:
```bash
./reproduce_experiments.sh arithmetic  # Only arithmetic experiments
./reproduce_experiments.sh gsm8k      # Only GSM8K experiments
```



## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for GSM8K experiments)
- ~20GB disk space for models and data
- Macbook M3 Pro CPU to exactly replicate Figure 4 Results. GPU computed results will differ slightly.

### Required Packages

Install these packages before running experiments:

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Unsloth for efficient training
pip install unsloth

# vLLM for fast inference (specific version)
pip install vllm==0.8.2

# Additional requirements
pip install transformers>=4.44.0
pip install datasets
pip install accelerate
pip install peft
pip install deepspeed
pip install wandb
pip install numpy
pip install matplotlib
pip install seaborn
pip install pandas
pip install scipy
pip install huggingface_hub
```

Or run the setup script which installs all required packages:
```bash
./setup_gsm8k_environment.sh
```

## Scripts Overview

### 1. Arithmetic Experiments (`run_arithmetic_experiments.sh`)
- Reproduces Section 5.1 and Appendix E.6, F
- Requires only PyTorch
- Fast execution (~30 minutes)
- Generates results for Figure 4 and sensitivity analyses

**Command:**
```bash
./run_arithmetic_experiments.sh
```

**Outputs:**
- `arithmetic_results/figure4_with_mi.log` - Main results with MI
- `arithmetic_results/figure4_without_mi.log` - Baseline without MI
- `arithmetic_results/sensitivity_*.log` - Parameter sensitivity tests
- `arithmetic_results/ablated_*.log` - Ablation studies

### 2. GSM8K Experiments

#### Setup (`setup_gsm8k_environment.sh`)
Installs required packages including:
- PyTorch with CUDA
- Transformers, PEFT, DeepSpeed
- Datasets and evaluation tools

**Command:**
```bash
./setup_gsm8k_environment.sh
```

#### Model Caching (`cache_models.sh`)
Downloads and caches the three models used in the paper:
- Qwen/Qwen2.5-7B-Instruct
- meta-llama/Llama-3.1-8B-Instruct
- R1-Distilled-Qwen2.5-Math-1.5B

Also downloads the GSM8K dataset.

**Command:**
```bash
./cache_models.sh
```

#### Training & Evaluation (`run_gsm8k_experiments.sh`)
Runs GRPO + MISL training and evaluation with paper parameters:
- max_z = 5
- alpha_mi = 5.0
- Evaluates on first 100 GSM8K problems

**Command:**
```bash
./run_gsm8k_experiments.sh
```

**Outputs:**
- `gsm8k_results/[model]/training.log` - Training logs
- `gsm8k_results/[model]/evaluation.log` - Evaluation metrics

## Expected Results

### Arithmetic Experiments
- ~10% improvement in pass@5 with MI training
- Diverse strategies learned across different z values
- Results match Figure 4 in the paper

### GSM8K Experiments
- Median +4% improvement in pass@k
- Median +7% improvement in consensus@k
- Preserved pass@1 accuracy
- Results for Qwen, Llama, and R1-Distilled models

## Computational Requirements

### Arithmetic Experiments
- CPU: Any modern processor
- Memory: 4GB RAM
- Time: ~30 minutes
- Storage: <1GB

### GSM8K Experiments
- GPU: 16GB+ VRAM recommended (A100, V100, etc.)
- CPU: 8+ cores recommended
- Memory: 32GB+ RAM
- Time: ~4-8 hours per model
- Storage: ~50GB for models and data

## Troubleshooting

### Common Issues

1. **PyTorch Installation**: Ensure CUDA-compatible PyTorch is installed
2. **Model Download Failures**: Check internet connection and HuggingFace access
3. **GPU Memory Issues**: Reduce batch size in full_train.py
4. **Permission Denied**: Run `chmod +x *.sh` to make scripts executable

### Environment Setup

We recommend using conda or virtualenv:
```bash
conda create -n misl python=3.9
conda activate misl
./setup_gsm8k_environment.sh
```

### Weights & Biases (Optional)

For experiment tracking:
```bash
wandb login
```

## File Structure After Running

```
upskill_code/
├── arithmetic_results/          # Arithmetic experiment outputs
│   ├── figure4_with_mi.log
│   ├── figure4_without_mi.log
│   └── ...
├── gsm8k_results/              # GSM8K experiment outputs
│   ├── qwen/
│   ├── llama/
│   └── r1-distilled/
├── model_cache/                # Cached models
└── *.sh                       # Reproduction scripts
```

## Citation

If you use these scripts, please cite:

```bibtex
@article{upskills2025,
  title={UpSkill: Mutual Information Skill Learning for Structured Response Diversity in LLMs},
  author={[Devan Shah, Owen Yang, Daniel Yang, Chongyi Zheng, Ben Eysenbach]},
  year={2025}
}
```
