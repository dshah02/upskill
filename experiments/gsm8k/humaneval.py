#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
import random
import yaml

import torch
from tqdm import tqdm

from human_eval.data import write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

from utils import run_model_coding, load_model_alt

os.environ["TOKENIZERS_PARALLELISM"] = "false"
SYSTEM_PROMPT = "You are a helpful assistant who writes correct, well-tested Python code."
RANDOM_EVAL_NUM = random.SystemRandom().randint(1000, 9999)
print("Eval run number:", RANDOM_EVAL_NUM)

def set_seed(seed):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_cached_humaneval(path):
    with open(path, "r") as f:
        return json.load(f)

def load_model2(base_model_path, model_path):
    # Force offline behavior just like your GSM8K script
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    return load_model_alt(base_model_path, model_path, 2048)

def generate_samples(base_model_path, model_path, k, seed, base):
    set_seed(seed)

    print("Loading model...")
    model, tokenizer = load_model2(base_model_path, model_path)

    dataset = load_cached_humaneval("/home/oy3975/EntropicReasoners/dataset_cache/humaneval.json")
    num_problems = len(dataset)

    samples = []

    for item in tqdm(dataset):
        task_id = item["task_id"]

        # Convert chat messages into a single prompt string
        # so utils.run_model can use it
        chat = item["prompt"][1]["content"]

        # Generate k completions using your sampling logic
        completions = run_model_coding(model, tokenizer, {"max_strategy": 5, "num_times": k // 5}, chat, base=base)

        # Store completions in HumanEval harness format
        for comp in completions:
            samples.append({
                "task_id": task_id,
                "completion": comp,   # MUST be only the model's generated code
            })
        # break

    samples_filename = f"/home/oy3975/EntropicReasoners/outputs/humaneval_samples_{RANDOM_EVAL_NUM}.jsonl"
    write_jsonl(samples_filename, samples)
    return samples_filename

def extract_first_function(text):
    """
    Extracts the first function definition from raw LLM output.
    Strips markdown, explanation text, and everything else.
    """

    # Remove markdown fences
    text = text.replace("```python", "").replace("```py", "").replace("```", "")
    text = text.strip()
    lines = text.splitlines()

    # ---- Find the first "def" ----
    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            start = i
            break
    if start is None:
        return ""

    func_lines = [lines[start]]
    indent = len(lines[start]) - len(lines[start].lstrip())

    # ---- Collect function block ----
    for line in lines[start + 1:]:
        stripped = line.strip()

        # Blank lines allowed inside function
        if stripped == "":
            func_lines.append(line)
            continue

        curr_indent = len(line) - len(line.lstrip())

        # Stop at a new function
        if curr_indent <= indent and line.lstrip().startswith("def "):
            break

        # Stop at any new top-level non-comment code (indent 0)
        if curr_indent <= indent and not stripped.startswith("#"):
            break

        func_lines.append(line)

    return "\n".join(func_lines).strip()

def clean_humaneval_jsonl(input_path):
    """
    Reads a HumanEval samples JSONL file, cleans each completion using
    extract_first_function(), and writes a new file with suffix _cleaned.jsonl.
    Returns the cleaned_path.
    """
    # Determine cleaned output path
    base, ext = os.path.splitext(input_path)
    cleaned_path = f"{base}_cleaned{ext}"

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(cleaned_path, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)

            # Clean the completion field only
            cleaned = extract_first_function(entry["completion"])
            entry["completion"] = cleaned

            # Write cleaned entry as JSONL
            fout.write(json.dumps(entry) + "\n")

    return cleaned_path

def evaluate_samples(cleaned_path, k):
    results = evaluate_functional_correctness(cleaned_path, k=[1, k])
    print(results)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)
    print(configs)

    base_model_path  = configs["base_model_path"]
    model_path  = configs["model_path"]
    k           = configs.get("k", 5)
    seed        = configs.get("seed", 42)
    base        = configs.get("base", True)

    samples_filename = generate_samples(
        base_model_path=base_model_path,
        model_path=model_path,
        k=k,
        seed=seed,
        base=base
    )
    # samples_filename = "/home/oy3975/EntropicReasoners/outputs/humaneval_samples_9108_cleaned.jsonl"  # <-- YOUR existing samples file
    cleaned_filename = clean_humaneval_jsonl(samples_filename)
    evaluate_samples(
        cleaned_filename,
        k=5,
    )