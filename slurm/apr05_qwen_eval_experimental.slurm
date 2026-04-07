#!/bin/sh
#SBATCH --job-name=apr05_exp_eval
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
#SBATCH --array=0-2              # 3 jobs: 3 experimental seeds

# ---- Array mapping ----
# Index | Config type   | Seed | Mode         | Run ID
#   0   | experimental  |  5   | experimental | 2923
#   1   | experimental  |  6   | experimental | 9189
#   2   | experimental  |  7   | experimental | 2087

STORE=/scratch/gpfs/EHAZAN/oy3975/.cache/hub
BASE_MODEL=${STORE}/models--unsloth--qwen2.5-7b-instruct-unsloth-bnb-4bit
QWEN_MODELS=${STORE}/models/models--unsloth--qwen2.5-7b-instruct-unsloth-bnb-4bit/gsm8k

MODEL_PATHS=(
    "${QWEN_MODELS}/5_5.0_0_0_1_0_seed5_2923/final_model"
    "${QWEN_MODELS}/5_5.0_0_0_1_0_seed6_9189/final_model"
    "${QWEN_MODELS}/5_5.0_0_0_1_0_seed7_2087/final_model"
)
MODES=(experimental experimental experimental)
LABELS=(exp_seed5 exp_seed6 exp_seed7)

MODEL_PATH=${MODEL_PATHS[$SLURM_ARRAY_TASK_ID]}
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
LABEL=${LABELS[$SLURM_ARRAY_TASK_ID]}

if [ ! -d "$MODEL_PATH" ]; then
    FALLBACK=$(echo "$MODEL_PATH" | sed 's|/final_model$||')/checkpoint-2000
    echo "final_model not found, trying fallback: $FALLBACK"
    if [ -d "$FALLBACK" ]; then
        MODEL_PATH=$FALLBACK
    else
        echo "ERROR: Neither final_model nor checkpoint-2000 found. Aborting."
        exit 1
    fi
fi

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

module purge
module load anaconda3/2023.9
module load cudatoolkit/12.6
module load gcc-toolset/14

conda activate grpo #--> REPLACE WITH YOUR CONDA ENV

log_info "Python version: $(python --version 2>&1)"
log_info "Array task: $SLURM_ARRAY_TASK_ID | Label: $LABEL | Mode: $MODE"
log_info "Model path: $MODEL_PATH"

# FlashInfer JIT needs curand.h — add the targets include path
export CPATH=/usr/local/cuda-12.6/targets/x86_64-linux/include:${CPATH:-}
export CUDA_HOME=/usr/local/cuda-12.6

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

OUTPUT_DIR="./eval_outputs/apr05/qwen_seeds/${LABEL}"
mkdir -p "$OUTPUT_DIR"

python experiments/gsm8k/run_evals.py \
    --base_model_path "${BASE_MODEL}" \
    --model_path "${MODEL_PATH}" \
    --evals gsm8k math humaneval \
    --num_problems -1 \
    --max_z 5 \
    --mode "${MODE}" \
    --numeric_only \
    --output_dir "${OUTPUT_DIR}"
