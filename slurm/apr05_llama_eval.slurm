#!/bin/sh
#SBATCH --job-name=apr05_llama_eval
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
#SBATCH --array=0-5              # 6 jobs: 3 control seeds + 3 experimental seeds

# ---- Array mapping ----
# Index | Config type   | Seed | Mode
#   0   | control       |  5   | control
#   1   | control       |  6   | control
#   2   | control       |  7   | control
#   3   | experimental  |  5   | experimental
#   4   | experimental  |  6   | experimental
#   5   | experimental  |  7   | experimental

STORE=/scratch/gpfs/EHAZAN/oy3975/.cache/hub
BASE_MODEL=${STORE}/models--unsloth--llama-3-8b-Instruct-bnb-4bit
LLAMA_MODELS=${STORE}/models/models--unsloth--llama-3-8b-Instruct-bnb-4bit/gsm8k

MODES=(control control control experimental experimental experimental)
SEEDS=(5 6 7 5 6 7)
ALPHA_MIS=(0.0 0.0 0.0 1.0 1.0 1.0) # From config_llama_control.yaml and config_llama_experimental.yaml
PREFIXES=("ctrl" "ctrl" "ctrl" "exp" "exp" "exp")

MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
ALPHA_MI=${ALPHA_MIS[$SLURM_ARRAY_TASK_ID]}
LABEL="${PREFIXES[$SLURM_ARRAY_TASK_ID]}_seed${SEED}"

# Find the latest model directory for the given seed and alpha_mi
MODEL_DIR=$(ls -td ${LLAMA_MODELS}/5_${ALPHA_MI}_0_0_1_0_seed${SEED}_* 2>/dev/null | head -1)

if [ -z "$MODEL_DIR" ]; then
    echo "ERROR: Could not find any model directory matching 5_${ALPHA_MI}_0_0_1_0_seed${SEED}_*"
    exit 1
fi

MODEL_PATH="${MODEL_DIR}/final_model"

if [ ! -d "$MODEL_PATH" ]; then
    FALLBACK=$(echo "$MODEL_PATH" | sed 's|/final_model$||')/checkpoint-500
    echo "final_model not found, trying fallback: $FALLBACK"
    if [ -d "$FALLBACK" ]; then
        MODEL_PATH=$FALLBACK
    else
        echo "ERROR: Neither final_model nor checkpoint-500 found. Aborting."
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
log_info "Model path: ${MODEL_PATH}"

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

OUTPUT_DIR="./eval_outputs/apr05/llama_seeds/${LABEL}"
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
