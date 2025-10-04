### CODE MODIFIED FROM UNSLOTH NOTEBOOK: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=hnbEBoBcCWOc

from unsloth import FastLanguageModel
import torch
import re
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import json
import os
import argparse
from utils import load_model

parser = argparse.ArgumentParser(
    description="Cache model and tokenizer for offline use"
)
parser.add_argument(
    "--model",
    type=str,
    default="llama",
    required=False,
    help="Name of the model to cache",
)

args = parser.parse_args()
model_name = args.model

store_dir = f"{os.environ['USER']}"

if "qwen" == model_name:
    model_name = "Qwen/Qwen2.5-7B"
    cache_dir = f"{store_dir}/cache/qwen-2-5-7b"
elif "r1-qwen" == model_name:
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # works but says 5b
    cache_dir = f"{store_dir}/cache/r1-qwen"
else:
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct"
    cache_dir = f"{store_dir}/cache/llama-3-1-8b"

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# should be from config
max_seq_length = 1024
lora_rank = 32

model, tokenizer = load_model(cache_dir, max_seq_length, lora_rank)

# setup LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


with open("./dataset_cache/gsm8k_train.json", "r") as f:
    dataset_data = json.load(f)
dataset = Dataset.from_list(dataset_data)

from UNSLOTH_rewards import (
    SYSTEM_PROMPT,
    extract_hash_answer,
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
)


max_prompt_length = 256


# saving code
import random

random.seed()  # some randomness issue
run_id = random.randint(1000, 9999)
output_dir = f"outputs_{run_id}_{model_name}_base"

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=6,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=2000,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir=f"{store_dir}/models/{output_dir}",
    hub_model_id=None,
    push_to_hub=False,
)

try:
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()

    trainer.model.save_pretrained(
        output_dir=f"{store_dir}/models/{output_dir}/final_model"
    )
    print(f"Model finished training and saved to {output_dir}")

except Exception as e:
    print(f"Error during training: {e}")
