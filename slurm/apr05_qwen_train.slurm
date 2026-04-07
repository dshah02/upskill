#!/bin/sh
#SBATCH --job-name=apr05_train      # create a short name for your job
#SBATCH --output=slurm/slurm_output/apr05/exp/%x_%A_%a.out
#SBATCH --error=slurm/slurm_output/apr05/exp/%x_%A_%a.err
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=2       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G                 # total memory per CPU core (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80       # Use for enforcing GPUs with 80GB memory, without it will get either 40GB or 80GB
#SBATCH --time=34:00:00          # total run time limit (HH:MM:SS)
#SBATCH --account=eladgroup
#SBATCH --partition=pli
#SBATCH --mail-type=BEGIN,END,FAIL          # send email on job start, end and on fail
#SBATCH --mail-user=oy3975@princeton.edu #--> ADD YOUR EMAIL
#SBATCH --array=0-2              # 3 jobs: control config x 3 seeds

# ---- Array mapping ----
# Index | Config                          | Seed
#   0   | config_qwen_control.yaml        |   5
#   1   | config_qwen_control.yaml        |   6
#   2   | config_qwen_control.yaml        |   7

CONFIGS=(
    "config_qwen_control.yaml"
    "config_qwen_control.yaml"
    "config_qwen_control.yaml"
)
SEEDS=(5 6 7)

CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

module purge
module load anaconda3/2023.9
module load cudatoolkit/12.6
module load gcc-toolset/14

conda activate grpo #--> REPLACE WITH YOUR CONDA ENV

log_info "Python version: $(python --version 2>&1)"
log_info "Array task: $SLURM_ARRAY_TASK_ID | Config: $CONFIG | Seed: $SEED"

python -c "import torch; print(f'PyTorch version: {torch.version.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Print the GPU information
if command -v nvidia-smi &>/dev/null; then
    log_info "GPU Information: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
else
    log_info "CUDA not installed or GPUs not available."
fi

echo "Running nvidia-smi to check GPU status:"
nvidia-smi

export HF_HOME=/scratch/gpfs/EHAZAN/oy3975/.cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export VLLM_CACHE_DIR=$HF_HOME/vllm
export PIP_CACHE_DIR=$HF_HOME/pip
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UNSLOTH_PREFER_VLLM=0

python experiments/gsm8k/full_train.py \
    --model "qwen" \
    --config ./experiments/gsm8k/${CONFIG} \
    --seed ${SEED}
