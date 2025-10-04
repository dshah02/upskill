from UNSLOTH_rewards import extract_hash_answer, SYSTEM_PROMPT
from pathlib import Path
from datasets import load_dataset
import json

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

with open("./dataset_cache/gsm8k_train.json", "w") as f:
    json.dump(processed_gsm_train, f)

with open("./dataset_cache/gsm8k_test.json", "w") as f:
    json.dump(processed_gsm_test, f)
