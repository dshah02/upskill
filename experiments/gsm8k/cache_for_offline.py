### CODE MODIFIED FROM UNSLOTH NOTEBOOK: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=hnbEBoBcCWOc

from unsloth import FastLanguageModel
import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama", required=False)
parser.add_argument(
    "--model_dir",
    type=str,
    default=None,
    help="Directory to use as HF hub cache root for downloaded models"
)

args = parser.parse_args()
model_name = args.model

# Set HF hub cache so models are stored in the expected hub format under model_dir
if args.model_dir:
    os.environ["HF_HUB_CACHE"] = args.model_dir
    os.makedirs(args.model_dir, exist_ok=True)

if "qwen" in model_name:
    model_name = "Qwen/Qwen2.5-7B-Instruct"
elif "r1" in model_name:
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
else:
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct"

max_seq_length = 1024
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=False,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,
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

hub_cache = args.model_dir or os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
print(f"Model cached in HF hub format under: {hub_cache}")
import glob
cached_dirs = glob.glob(f"{hub_cache}/models--*/snapshots/*/")
for d in cached_dirs:
    print(f"  Found: {d}")
