
from pathlib import Path
from datasets import load_dataset
import json


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from UNSLOTH_rewards import extract_hash_answer, SYSTEM_PROMPT

dataset = load_dataset("openai/gsm8k", "main")

processed_gsm_train = []
processed_gsm_test = []

for item in dataset["train"]:
    processed_item = {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ],
        "answer": extract_hash_answer(item["answer"]),
    }
    processed_gsm_train.append(processed_item)

for item in dataset["test"]:
    processed_item = {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ],
        "answer": extract_hash_answer(item["answer"]),
    }
    processed_gsm_test.append(processed_item)

# Ensure the directory exists before writing files
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
_cache_dir = os.path.join(_project_root, "dataset_cache")
os.makedirs(_cache_dir, exist_ok=True)

with open(os.path.join(_cache_dir, "gsm8k_train.json"), "w") as f:
    json.dump(processed_gsm_train, f)

with open(os.path.join(_cache_dir, "gsm8k_test.json"), "w") as f:
    json.dump(processed_gsm_test, f)
