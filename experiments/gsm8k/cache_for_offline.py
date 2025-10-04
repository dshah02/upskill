### CODE MODIFIED FROM UNSLOTH NOTEBOOK: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=hnbEBoBcCWOc

from unsloth import FastLanguageModel
import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama", required=False)

args = parser.parse_args()
model_name = args.model

if "qwen" in model_name:
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    cache_dir = f"{os.environ['USER']}/cache/qwen-2-5-7b"
elif "r1" in model_name:
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    cache_dir = f"{os.environ['USER']}/cache/r1-qwen"
else:
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct"
    cache_dir = f"{os.environ['USER']}/cache/llama-3-1-8b"

max_seq_length = 1024
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,
    cache_dir=cache_dir,
)

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

model.save_pretrained(cache_dir)
tokenizer.save_pretrained(cache_dir)

print("saved in", cache_dir)
