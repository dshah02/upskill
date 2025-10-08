import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from datetime import datetime
from unsloth import FastLanguageModel
import random
import json
from utils import run_model, load_model_alt, extract_answer
from tqdm import tqdm

def run_benchmark(configs):
    model, tokenizer = load_model2(configs["base_model_path"],configs["model_path"])
    dataset = load_dataset(configs["dataset_path"], configs["num_problems"])
    statistics = {
        "num_problems": len(dataset),
        "processed_problems": 0,
        "pass@1": 0,
        "pass@k": 0,
        "plurality@k": 0,
        "consensus@k": 0,
    }

    for i, problem in tqdm(enumerate(dataset)):
        problem_text, expected_answer = problem["question"], problem["answer"]
        expected_answer = extract_answer(str(expected_answer))
        results = run_model(model, tokenizer, configs, problem_text)

        statistics["processed_problems"] += 1
        pass_at_1, pass_at_k, plurality_at_k, consensus_at_k = 0, 0, 0, 0

        correct = sum(1 for r in results if check_answer(r, expected_answer))
        answer_counts = {answer: results.count(answer) for answer in set(results)}
        max_count = max(answer_counts.values())
        max_answers = [
            ans for ans, count in answer_counts.items() if count == max_count
        ]

        pass_at_1 = correct / len(results)
        pass_at_k = 1 if correct > 0 else 0
        if len(max_answers) == 1 and check_answer(max_answers[0], expected_answer):
            plurality_at_k = 1
        consensus_at_k = 1 if correct > len(results) / 2 else 0

        statistics["pass@1"] += pass_at_1
        statistics["pass@k"] += pass_at_k
        statistics["plurality@k"] += plurality_at_k
        statistics["consensus@k"] += consensus_at_k

        with open(configs["output_path"], "w") as output_file:
            json.dump({"statistics": statistics}, output_file)


def check_answer(pred, gt):
    try:
        if float(pred) == float(gt):
            return True
    except ValueError:
        pass
    return pred == gt


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


# all this gets run
# ----------------- Seeding -----------------

import torch
import random
import numpy as np 

FIXED_SEED = 0
random.seed(FIXED_SEED)
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
np.random.seed(FIXED_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import argparse
parser = argparse.ArgumentParser(description="GSM8K Benchmarking Script")
parser.add_argument("--model_id", type=str, required=True, help="4 digit model id")
parser.add_argument("--model", type=str, required=True, choices=["qwen", "llama", "r1"], help="Model type: qwen, llama, or r1")
args = parser.parse_args()
model_id = args.model_id

output_path = f"./outputs/{model_id}_data.json"
dataset_path = "./data/GSM8K/test.json"
# model_path = (
#     f"/scratch/gpfs/ds6237/models/models--unsloth--qwen2.5-7b-instruct-unsloth-bnb-4bit/gsm8k/5_0.5_0_0_1_0_6645/checkpoint-2000/"
# )

formal_name_map = {
    "llama": "models--unsloth--meta-llama-3.1-8b-instruct-unsloth-bnb-4bit",
    "qwen": "models--unsloth--qwen2.5-7b-instruct-unsloth-bnb-4bit",
    "r1": "models--unsloth--deepseek-r1-distill-qwen-1.5b-bnb-4bit",
}

formal_name = formal_name_map[args.model]
import glob

if model_id == 'base':
    snapshot_glob = f"/scratch/gpfs/ds6237/{formal_name}/snapshots/*/"
else:
    snapshot_glob = f"/scratch/gpfs/ds6237/models/{formal_name}/gsm8k/*_{model_id}/checkpoint-2000/"

snapshot_dirs = glob.glob(snapshot_glob)
model_path = snapshot_dirs[0]
print(model_path)

configs = {
    "num_problems": 500,
    "max_strategy": 5,
    "base_model_path": f"/scratch/gpfs/ds6237/{formal_name}",
    "model_path": model_path,
    "dataset_path": dataset_path,
    "output_path": f"./outputs/{args.model}_{model_id}_data.json",
    "solution_path": f"./outputs/{args.model}_{model_id}_solutions.json"
}

os.makedirs(os.path.dirname(output_path), exist_ok=True)
run_benchmark(configs)
