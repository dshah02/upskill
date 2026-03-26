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
from unsloth import FastLanguageModel

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
EVAL_TEMPERATURE = 0.7  # overridden from CLI in main

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
        temperature=EVAL_TEMPERATURE,
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
    from human_eval.data import write_jsonl, read_problems
    from human_eval.evaluation import evaluate_functional_correctness

    # Load from cache if available, otherwise from library directly
    if dataset_path and os.path.exists(dataset_path):
        with open(dataset_path) as f:
            data = json.load(f)
    else:
        raw = read_problems()
        data = [{"task_id": tid, "prompt_text": prob["prompt"]} for tid, prob in raw.items()]
    if num_problems > 0:
        data = data[:num_problems]

    samples = []
    for item in tqdm(data, total=len(data)):
        task_id = item["task_id"]
        # Support both cached format (chat messages) and raw format (code string)
        if "prompt_text" in item:
            problem_text = item["prompt_text"]
        else:
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
                                     temperature=EVAL_TEMPERATURE, top_p=0.9, do_sample=True)
            generated_ids = outputs[0, model_inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            cleaned = extract_first_function(response)
            samples.append({"task_id": task_id, "completion": cleaned})

    run_id = random.randint(1000, 9999)
    samples_path = os.path.join(out_dir, f"humaneval_samples_{run_id}.jsonl")
    write_jsonl(samples_path, samples)

    # evaluate_functional_correctness requires all 164 problems by default.
    # When evaluating a subset, write a custom problem file with only the attempted tasks.
    attempted_tasks = set(s["task_id"] for s in samples)
    all_problems = read_problems()
    subset_problems = {tid: all_problems[tid] for tid in attempted_tasks}

    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as pf:
        for tid, prob in subset_problems.items():
            pf.write(json.dumps(prob) + "\n")
        problem_file = pf.name

    results = evaluate_functional_correctness(samples_path, k=[1, max_z], problem_file=problem_file)
    os.unlink(problem_file)
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
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    globals()["EVAL_TEMPERATURE"] = args.temperature
    print(f"Sampling temperature: {args.temperature}")

    print("Loading model...")
    # Detect if base model is pre-quantized (bnb-4bit in name)
    load_4bit = "bnb-4bit" in args.base_model_path
    import glob as _glob
    snapshot_dirs = _glob.glob(f"{args.base_model_path}/snapshots/*/")
    base_dir = snapshot_dirs[0] if snapshot_dirs else args.base_model_path

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_dir,
        max_seq_length=2048,
        load_in_4bit=load_4bit,
        fast_inference=True,
        local_files_only=True,
        device_map="auto",
    )
    # Load LoRA adapter if present
    if os.path.exists(os.path.join(args.model_path, "adapter_config.json")):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model_path, is_trainable=False)
        print(f"Loaded LoRA adapter from {args.model_path}")
    print("Model loaded.")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    # Extract run_id from model path (e.g. ...5_0.0_0_0_1_0_5044/final_model -> 5044)
    # Fall back to timestamp if not found
    _path_parts = args.model_path.rstrip("/").split("/")
    _run_id = "base"
    for part in reversed(_path_parts):
        if "_" in part:
            candidate = part.split("_")[-1]
            if candidate.isdigit() and len(candidate) == 4:
                _run_id = candidate
                break
    run_tag = f"{_run_id}_{args.mode}_{int(time.time())}"

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
                ds_path = None  # will load from human_eval library directly
            run_humaneval(model, tokenizer, ds_path, args.num_problems, args.max_z, args.mode, args.output_dir)
