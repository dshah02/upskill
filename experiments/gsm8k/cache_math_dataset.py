"""Cache the competition_math dataset in the same format as gsm8k_train.json.

Loads from: https://huggingface.co/datasets/qwedsacf/competition_math
Writes to:  dataset_cache/math_train.json  and  dataset_cache/math_test.json

Each entry has the form:
  {"prompt": [{"role":"system","content":...}, {"role":"user","content":...}],
   "answer": "<ground-truth answer string>"}

The ground-truth answer is extracted from the \\boxed{} in the solution field.
"""

import json
import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from datasets import load_dataset
from UNSLOTH_rewards import SYSTEM_PROMPT
from flex_extract import last_boxed_only_string, remove_boxed


def extract_answer(solution: str) -> str | None:
    """Extract the answer from \\boxed{} in the solution string."""
    boxed = last_boxed_only_string(solution)
    if boxed is None:
        return None
    return remove_boxed(boxed)


ds = load_dataset("qwedsacf/competition_math")

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
_cache_dir = os.path.join(_project_root, "dataset_cache")
os.makedirs(_cache_dir, exist_ok=True)

for split, out_name in [("train", "math_train.json"), ("test", "math_test.json")]:
    processed = []
    skipped = 0
    for item in ds[split]:
        answer = extract_answer(item["solution"])
        if answer is None:
            skipped += 1
            continue
        processed.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["problem"]},
            ],
            "answer": answer,
        })

    path = os.path.join(_cache_dir, out_name)
    with open(path, "w") as f:
        json.dump(processed, f)
    print(f"{out_name}: {len(processed)} examples ({skipped} skipped, no \\boxed{{}})")
