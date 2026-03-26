"""
Unified evaluation script for GSM8K, MATH, and HumanEval.
Adapts the existing benchmark.py and humaneval.py for RunPod.

Usage:
    python run_evals.py \
        --base_model_path /workspace/huggingface_cache/models--unsloth--meta-llama-3.1-8b-instruct-unsloth-bnb-4bit \
        --model_path /workspace/upskill/root/models/.../final_model \
        --evals gsm8k math humaneval \
        --num_problems 100 \
        --max_z 5 \
        --mode control
"""

import sys
import os
import argparse
import json
import random
import time

import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from flex_extract import ExtractConfig, extract_numeric_answer, normalize_answer
from utils import load_model_alt

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SYSTEM_PROMPT_MATH = "You are a helpful math assistant that solves problems step by step."
SYSTEM_PROMPT_CODE = (
    "You are a helpful assistant who writes correct, well-tested Python code. "
    "Generate only the Python code to complete the function described in the docstring. "
    "Do not include any explanations, introductory text, or natural language comments "
    "outside the function body."
)

_EXTRACT_CFG = ExtractConfig(
    trim_chars=2000,
    accept_xml=True,
    accept_boxed=True,
    accept_tags=True,
    accept_fallback_number=True,
    strip_units_before_pick=True,
)

# ---- Seed ----
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---- Answer helpers ----
def extract_pred(text):
    raw, _ = extract_numeric_answer(text, _EXTRACT_CFG)
    return normalize_answer(raw)

def normalize_gt(ans):
    s = str(ans)
    raw, _ = extract_numeric_answer(s, _EXTRACT_CFG)
    if raw != "[INVALID]":
        return normalize_answer(raw)
    return normalize_answer(s)

def check_answer(pred, gt):
    if pred == gt:
        return True
    try:
        fp, fg = float(pred), float(gt)
        return fp == fg or abs(fp - fg) <= 1e-9 * max(1.0, abs(fg))
    except Exception:
        return False

# ---- Generation ----
def generate_response(model, tokenizer, system_prompt, user_text):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        response = response.split("assistant")[-1].strip()
    except Exception:
        pass
    return response

# ---- Math eval (GSM8K / MATH) ----
def run_math_eval(model, tokenizer, dataset_path, num_problems, max_z, mode, out_path):
    with open(dataset_path) as f:
        data = json.load(f)
    if num_problems > 0:
        data = data[:num_problems]

    stats = {"num_problems": len(data), "pass@1": 0, "pass@k": 0, "plurality@k": 0, "consensus@k": 0}

    for i, problem in tqdm(enumerate(data), total=len(data)):
        problem_text = problem["prompt"][1]["content"]
        expected = normalize_gt(problem["answer"])
        results = []

        for strat in range(1, max_z + 1):
            if mode == "experimental":
                prompt = f"Strategy {strat} | {problem_text}"
            else:
                prompt = problem_text
            response = generate_response(model, tokenizer, SYSTEM_PROMPT_MATH, prompt)
            results.append(extract_pred(response))

        correct = sum(1 for r in results if check_answer(r, expected))
        counts = {a: results.count(a) for a in set(results)}
        max_count = max(counts.values()) if counts else 0
        max_answers = [a for a, c in counts.items() if c == max_count]

        stats["pass@1"] += correct / len(results) if results else 0
        stats["pass@k"] += 1 if correct > 0 else 0
        stats["plurality@k"] += 1 if (len(max_answers) == 1 and check_answer(max_answers[0], expected)) else 0
        stats["consensus@k"] += 1 if correct > len(results) / 2 else 0

    n = len(data)
    summary = {k: (v / n if k != "num_problems" else v) for k, v in stats.items()}
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Results ({out_path}) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary

# ---- HumanEval ----
def extract_first_function(text):
    text = text.replace("```python", "").replace("```py", "").replace("```", "").strip()
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def "):
            start = i
            break
    if start is None:
        return ""
    func_lines = [lines[start]]
    indent = len(lines[start]) - len(lines[start].lstrip())
    for line in lines[start + 1:]:
        stripped = line.strip()
        if stripped == "":
            func_lines.append(line)
            continue
        curr_indent = len(line) - len(line.lstrip())
        if curr_indent <= indent and line.lstrip().startswith("def "):
            break
        if curr_indent <= indent and not stripped.startswith("#"):
            break
        func_lines.append(line)
    return "\n".join(func_lines).strip()

def run_humaneval(model, tokenizer, dataset_path, num_problems, max_z, mode, out_dir):
    from human_eval.data import write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness

    with open(dataset_path) as f:
        data = json.load(f)
    if num_problems > 0:
        data = data[:num_problems]

    samples = []
    for item in tqdm(data, total=len(data)):
        task_id = item["task_id"]
        problem_text = item["prompt"][1]["content"]

        for strat in range(1, max_z + 1):
            if mode == "experimental":
                prompt = f"Strategy {strat} | {problem_text}"
            else:
                prompt = problem_text

            ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT_CODE},
                 {"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt",
            )
            if isinstance(ids, torch.Tensor):
                model_inputs = {"input_ids": ids.to(model.device),
                                "attention_mask": torch.ones_like(ids).to(model.device)}
            else:
                model_inputs = {k: v.to(model.device) for k, v in ids.items()}

            outputs = model.generate(**model_inputs, max_new_tokens=1024,
                                     temperature=0.7, top_p=0.9, do_sample=True)
            generated_ids = outputs[0, model_inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            cleaned = extract_first_function(response)
            samples.append({"task_id": task_id, "completion": cleaned})

    run_id = random.randint(1000, 9999)
    samples_path = os.path.join(out_dir, f"humaneval_samples_{run_id}.jsonl")
    write_jsonl(samples_path, samples)

    results = evaluate_functional_correctness(samples_path, k=[1, max_z])
    result_path = os.path.join(out_dir, f"humaneval_results_{run_id}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== HumanEval Results ({result_path}) ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results

# ---- Main ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--evals", nargs="+", choices=["gsm8k", "math", "humaneval"], required=True)
    parser.add_argument("--num_problems", type=int, default=100)
    parser.add_argument("--max_z", type=int, default=5)
    parser.add_argument("--mode", choices=["control", "experimental"], default="control")
    parser.add_argument("--output_dir", default="./eval_outputs")
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model, tokenizer = load_model_alt(args.base_model_path, args.model_path, 2048)
    print("Model loaded.")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    run_tag = f"{os.path.basename(args.model_path)}_{args.mode}_{int(time.time())}"

    for eval_name in args.evals:
        print(f"\n{'='*60}\nRunning {eval_name} eval ({args.num_problems} problems, mode={args.mode})\n{'='*60}")
        if eval_name == "gsm8k":
            ds_path = os.path.join(project_root, "dataset_cache", "gsm8k_test.json")
            out = os.path.join(args.output_dir, f"gsm8k_{run_tag}.json")
            run_math_eval(model, tokenizer, ds_path, args.num_problems, args.max_z, args.mode, out)
        elif eval_name == "math":
            ds_path = os.path.join(project_root, "dataset_cache", "math_test.json")
            out = os.path.join(args.output_dir, f"math_{run_tag}.json")
            run_math_eval(model, tokenizer, ds_path, args.num_problems, args.max_z, args.mode, out)
        elif eval_name == "humaneval":
            ds_path = os.path.join(project_root, "dataset_cache", "humaneval.json")
            if not os.path.exists(ds_path):
                print(f"  WARNING: {ds_path} not found, skipping humaneval")
                continue
            run_humaneval(model, tokenizer, ds_path, args.num_problems, args.max_z, args.mode, args.output_dir)
