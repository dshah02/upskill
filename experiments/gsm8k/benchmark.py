# benchmark.py

import sys
import os
import argparse
import json
import random
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from flex_extract import ExtractConfig, extract_numeric_answer, normalize_answer
from utils import load_model_alt

# ----------------- Seeding -----------------
FIXED_SEED = 0
random.seed(FIXED_SEED)
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
np.random.seed(FIXED_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------- Extractor config -----------------
_EXTRACT_CFG = ExtractConfig(
    trim_chars=2000,
    accept_xml=True,
    accept_boxed=True,
    accept_tags=True,
    accept_fallback_number=True,
    strip_units_before_pick=True,
)

def _extract_pred_from_solution(text: str) -> str:
    raw, _ = extract_numeric_answer(text, _EXTRACT_CFG)
    return normalize_answer(raw)

def _normalize_ground_truth(ans_any) -> str:
    s = str(ans_any)
    raw, _ = extract_numeric_answer(s, _EXTRACT_CFG)
    if raw != "[INVALID]":
        return normalize_answer(raw)
    return normalize_answer(s)

def check_answer(pred: str, gt: str) -> bool:
    # String equality first; numeric tolerance fallback
    if pred == gt:
        return True
    try:
        fp, fg = float(pred), float(gt)
        if fp == fg:
            return True
        return abs(fp - fg) <= 1e-9 * max(1.0, abs(fg))
    except Exception:
        return False

# ----------------- Model + data loaders -----------------
def load_model2(base_model_path, model_path):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    return load_model_alt(base_model_path, model_path, 2048)

def load_dataset(dataset_path, num_problems=-1):
    with open(dataset_path, "r") as f:
        dataset_data = json.load(f)
    if num_problems > 0:
        dataset_data = dataset_data[:num_problems]
    return dataset_data

# ----------------- Prompt builder -----------------
def _make_prompt(problem_text: str, strat: int, mode: str) -> str:
    if mode == "experimental":
        return f"Strategy {strat} | {problem_text}"
    return problem_text

# ----------------------- Generation loop -----------------------
def run_model(model, tokenizer, args, problem_text):
    answers = []
    for strat in range(1, int(args["max_z"])+1):
        strategy_prompt = _make_prompt(problem_text, strat, args["mode"])
        messages = [
            {
                "role": "system",
                "content": "You are a helpful math assistant that solves problems step by step.",
            },
            {"role": "user", "content": strategy_prompt},
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

        # Save the raw solution for debugging
        with open(args["solution_path"], "a") as solution_file:
            solution_file.write(
                f"MODE={args['mode']} STRAT={strat}\nSOLUTION:\n{response}\n"
            )

        extracted_answer = _extract_pred_from_solution(response)
        answers.append(extracted_answer)
    return answers

# ----------------- Benchmark loop -----------------
def run_benchmark(configs):
    model, tokenizer = load_model2(configs["base_model_path"], configs["model_path"])
    dataset = load_dataset(configs["dataset_path"], configs["num_problems"])
    print("DATASET LEN", len(dataset))
    stats = {
        "num_problems": len(dataset),
        "processed_problems": 0,
        "pass@1": 0,
        "pass@k": 0,
        "plurality@k": 0,
        "consensus@k": 0,
        "mode": configs["mode"],
        "max_z": configs["max_z"],
    }

    for i, problem in tqdm(enumerate(dataset), total=len(dataset)):
        problem_text, expected_answer = problem["prompt"][1]["content"], problem["answer"]
        expected_answer = _normalize_ground_truth(expected_answer)

        results = run_model(model, tokenizer, configs, problem_text)
        stats["processed_problems"] += 1

        correct = sum(1 for r in results if check_answer(r, expected_answer))
        answer_counts = {answer: results.count(answer) for answer in set(results)}
        max_count = max(answer_counts.values()) if answer_counts else 0
        max_answers = [ans for ans, count in answer_counts.items() if count == max_count]

        # same behavior as your original script
        pass_at_1 = correct / len(results) if results else 0.0
        pass_at_k = 1 if correct > 0 else 0
        plurality_at_k = 1 if (len(max_answers) == 1 and check_answer(max_answers[0], expected_answer)) else 0
        consensus_at_k = 1 if correct > len(results) / 2 else 0

        stats["pass@1"] += pass_at_1
        stats["pass@k"] += pass_at_k
        stats["plurality@k"] += plurality_at_k
        stats["consensus@k"] += consensus_at_k

        with open(configs["output_path"], "w") as f:
            json.dump({"statistics": stats}, f)

# ----------------- CLI -----------------
# parser = argparse.ArgumentParser(description="GSM8K Benchmarking Script for MISL Paper")
# parser.add_argument("--model", type=str, required=True, choices=["qwen", "llama", "r1"], help="Model type used in MISL paper: qwen, llama, or r1-distilled")
# parser.add_argument("--max_z", type=int, default=5, help="Number of strategies to evaluate (default: 5)")
# parser.add_argument("--num_problems", type=int, default=500, help="How many GSM8K problems to evaluate")
# parser.add_argument("--mode", type=str, default="experimental", choices=["control", "experimental"], help="Prompt mode: `experimental` prepends `Strategy z |`, `control` has no prefix")
# args = parser.parse_args()

# dataset_path = "./dataset_cache/gsm8k_test.json"

# formal_name_map = {
#     "llama": "models--unsloth--meta-llama-3.1-8b-instruct-unsloth-bnb-4bit",
#     "qwen": "models--unsloth--qwen2.5-7b-instruct-unsloth-bnb-4bit",
#     "r1-distilled": "models--unsloth--deepseek-r1-distill-qwen-1.5b-bnb-4bit",
# }
# formal_name = formal_name_map[args.model]

# import glob

# snapshot_glob = f"./models/{formal_name}/snapshots/*/"
# snapshot_dirs = glob.glob(snapshot_glob)
# if not snapshot_dirs:
#     # fallback: check for cached finetune result
#     snapshot_glob2 = f"./models/{formal_name}/gsm8k/*/checkpoint-2000/"
#     snapshot_dirs = glob.glob(snapshot_glob2)
# assert snapshot_dirs, f"No model found at pattern(s): {snapshot_glob}"
# model_path = snapshot_dirs[0]
# print("Using model snapshot:", model_path)

import argparse
parser = argparse.ArgumentParser(description="GSM8K Benchmarking Script")
parser.add_argument("--model_id", type=str, required=True, help="4 digit model id or 'base'")
parser.add_argument("--model", type=str, required=True, choices=["qwen", "llama", "r1"], help="Model type: qwen, llama, or r1")
parser.add_argument("--mode", type=str, default="experimental", choices=["control", "experimental"],
                    help="Prompt mode: 'experimental' prepends 'Strategy z |', 'control' does not.")
args = parser.parse_args()
model_id = args.model_id

dataset_path = "./dataset_cache/gsm8k_test.json"

formal_name_map = {
    "llama": "models--unsloth--meta-llama-3.1-8b-instruct-unsloth-bnb-4bit",
    "qwen": "models--unsloth--qwen2.5-7b-instruct-unsloth-bnb-4bit",
    "r1": "models--unsloth--deepseek-r1-distill-qwen-1.5b-bnb-4bit",
}
formal_name = formal_name_map[args.model]

BASE_PATH = "/scratch/gpfs/ds6237" #os.environ['USER']}

import glob
if model_id == "base":
    snapshot_glob = f"{BASE_PATH}/{formal_name}/snapshots/*/"
else:
    snapshot_glob = f"{BASE_PATH}/models/{formal_name}/gsm8k/*_{model_id}/checkpoint-2000/"

snapshot_dirs = glob.glob(snapshot_glob)
assert snapshot_dirs, f"No model found at pattern: {snapshot_glob}"
model_path = snapshot_dirs[0]
print(model_path)


os.makedirs("./outputs", exist_ok=True)
configs = {
    "num_problems": 500,
    "max_z": 5,
    "mode": args.mode,  # <-- control vs experimental
    "base_model_path": f"/scratch/gpfs/ds6237/{formal_name}",
    "model_path": model_path,
    "dataset_path": dataset_path,
    "output_path": f"./outputs/{args.model}_{model_id}_{args.mode}_data.json",
    "solution_path": f"./outputs/{args.model}_{model_id}_{args.mode}_solutions.json",
}

run_benchmark(configs)
