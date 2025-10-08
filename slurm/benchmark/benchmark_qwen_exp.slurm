#!/bin/sh
#SBATCH --job-name=qwen_exp_benchmark # create a short name for your job
#SBATCH --output=slurm/slurm_output/oct6/exp/%x_%j.out
#SBATCH --error=slurm/slurm_output/oct6/exp/%x_%j.err
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=2       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G                 # total memory per CPU core (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80       # Use for enforcing GPUs with 80GB memory, without it will get either 40GB or 80GB
#SBATCH --time=6:00:00          # total run time limit (HH:MM:SS)
#SBATCH --account=eladgroup
#SBATCH --partition=pli
#SBATCH --mail-type=BEGIN,END,FAIL          # send email on job start, end and on fail
#SBATCH --mail-user=ds6237@princeton.edu #--> ADD YOUR EMAIL

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

module purge
module load anaconda3/2023.9
module load cudatoolkit/12.6
module load gcc-toolset/14

conda activate grpo #--> REPLACE WITH YOUR CONDA ENV

log_info "Python version: $(python --version 2>&1)"

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

# python experiments/gsm8k/benchmark.py --model qwen --model_id 5855
python experiments/gsm8k/benchmark.py --model qwen --model_id 5855 --mode control