#!/bin/bash
for SEED in {0..4}
do
  echo "Running with seed $SEED..."
  uv run python experiments/gsm8k/full_train.py \
    --model "qwen" \
    --config ./experiments/gsm8k/config_qwen_experimental.yaml \
    --model_dir /home/devuser/.cache/huggingface/hub \
    --seed $SEED
done
