from unsloth import FastLanguageModel
import torch
import re
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import json
import os
import random
import math
import numpy as np
import yaml
import argparse
import contextlib

if False:
    #Only needed for semantic tests
    from google import genai
    from semantic_determinant import (
        get_embeddings_by_question,
        analyze_embedding_determinants,
    )
    from semantic_similarity import compute_embedding_label_mi
else:
    get_embedding_by_question = None
    analyze_embedding_determinants = None
    compute_embedding_label_mi = None

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from UNSLOTH_rewards import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    # math_correctness_func,
)

from flex_rewards_adapter import math_correctness_func

from utils import load_model
from utils import (
    extract_strategy_idx,
    replace_strategy_idx,
    add_strategy_string,
    remove_strategy_string,
)
from dotenv import load_dotenv
import os
import glob
import time

# NEW: HF/Transformers callback for JSONL logging
from transformers.trainer_callback import TrainerCallback

# ----------------- CLI -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config_control.yaml")
parser.add_argument("--model", type=str, default="llama", required=False)
parser.add_argument(
    "--store_dir",
    type=str,
    default=f"{os.environ['USER']}",
    help="Base directory to store model outputs and logs"
)
parser.add_argument(
    "--model_dir",
    type=str,
    default=None,
    help="Directory containing cached HF hub models (hub cache root)"
)
parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Fixed seed for reproducibility"
)

args = parser.parse_args()
store_dir = args.store_dir
model_dir = args.model_dir
# ----------------- Seeding -----------------
FIXED_SEED = args.seed
random.seed(FIXED_SEED)
torch.manual_seed(FIXED_SEED)
torch.cuda.manual_seed_all(FIXED_SEED)
np.random.seed(FIXED_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch._dynamo.config.cache_size_limit = 128

# ----------------- Offline env -----------------
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# ----------------- Config -----------------
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

print(config["max_seq_length"])
print("lora_rank", config["lora_rank"])
print("alpha_mi", config.get("alpha_mi", 0))
print("alpha_mixkl", config.get("alpha_mixkl", 0))
print("alpha_det", config.get("alpha_det", 0))
print("alpha_smi", config.get("alpha_smi", 0))
print("individual_reward_factor", config.get("individual_reward_factor", 1))
print("pass_reward_factor", config.get("pass_reward_factor", 0))
print("max_z", config["max_z"])
print("steps", config.get("steps", 250))
print("lr", config.get("lr", 1e-4))
print("temperature", config.get("temperature", 1.0))
print("dataset", config.get("dataset", "gsm8k"))

alpha_mi = config.get("alpha_mi", 0)
alpha_det = config.get("alpha_det", 0)
alpha_mixkl = config.get("alpha_mixkl", 0)
alpha_smi = config.get("alpha_smi", 0)
use_mi = alpha_mi != 0
use_det = alpha_det != 0
use_mixkl = alpha_mixkl != 0
use_smi = alpha_smi != 0
cache_dir = "./cache"
max_seq_length = config["max_seq_length"]
lora_rank = config["lora_rank"]
individual_reward_factor = config.get("individual_reward_factor", 1)
pass_reward_factor = config.get("pass_reward_factor", 0)
Z = list(range(1, config["max_z"] + 1))
steps = config.get("steps", 250)
lr = float(config.get("lr", 1e-4))
temperature = float(config.get("temperature", 1.0))
model_name = config.get("model", args.model)
dataset = config.get("dataset", "gsm8k")
if dataset not in ["gsm8k", "math"]:
    raise ValueError("Dataset must be either 'gsm8k' or 'math'")
dataset_text = dataset

# ----------------- Snapshots -----------------
# Use model_dir as the HF hub cache root; fall back to store_dir if model_dir not provided
_hub_root = model_dir if model_dir else store_dir

if "qwen" == model_name:
    model_name = "models--unsloth--qwen2.5-7b-instruct-unsloth-bnb-4bit"
    snapshot_glob = f"{_hub_root}/{model_name}/snapshots/*/"
    snapshot_dirs = glob.glob(snapshot_glob)
    if not snapshot_dirs:
        # Try flexible glob for variant naming
        for pattern in [
            f"{_hub_root}/models--unsloth--*qwen*2.5*7b*4bit*/snapshots/*/",
            f"{_hub_root}/models--unsloth--*Qwen*2.5*7B*4bit*/snapshots/*/",
        ]:
            snapshot_dirs = glob.glob(pattern)
            if snapshot_dirs:
                break
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot directory found for {model_name} in {_hub_root}")
    cache_dir = snapshot_dirs[0]
elif "r1-qwen" == model_name:
    model_name = "models--unsloth--deepseek-r1-distill-qwen-1.5b-bnb-4bit"  # works but says 5b
    snapshot_glob = f"{_hub_root}/{model_name}/snapshots/*/"
    snapshot_dirs = glob.glob(snapshot_glob)
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot directory found for {model_name} in {snapshot_glob}")
    cache_dir = snapshot_dirs[0]
else:
    model_name = "models--unsloth--meta-llama-3.1-8b-instruct-unsloth-bnb-4bit"
    snapshot_glob = f"{_hub_root}/{model_name}/snapshots/*/"
    snapshot_dirs = glob.glob(snapshot_glob)
    if not snapshot_dirs:
        raise FileNotFoundError(f"No snapshot directory found for {model_name} in {snapshot_glob}")
    cache_dir = snapshot_dirs[0]

model, tokenizer = load_model(cache_dir, max_seq_length, lora_rank, peft_apply=True)
ref_model, ref_tokenizer = load_model(cache_dir, max_seq_length, lora_rank, peft_apply=True)

# ----------------- Dataset -----------------
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if dataset == "gsm8k":
    filepath = os.path.join(_project_root, "dataset_cache", "gsm8k_train.json")
else:
    filepath = os.path.join(_project_root, "dataset_cache", "math_train.json")

with open(filepath, "r") as f:
    dataset_data = json.load(f)

if use_mi:
    for item in dataset_data:
        idx = random.choice(Z)
        item["prompt"][1]["content"] = add_strategy_string(
            item["prompt"][1]["content"], idx
        )

dataset = Dataset.from_list(dataset_data)
print(dataset[0])

# ----------------- MI helpers -----------------
def get_log_probability(prompt, completion, ref=False):
    if ref:
        curr_model = ref_model
        curr_tokenizer = ref_tokenizer
        ctx = torch.no_grad()
    else:
        curr_model = model
        curr_tokenizer = tokenizer
        ctx = contextlib.nullcontext()

    prompt_tokens = curr_tokenizer.encode(prompt, add_special_tokens=False)
    combined_input = prompt + "\n" + completion
    combined_tokens = curr_tokenizer.encode(combined_input, add_special_tokens=False)
    input_ids = torch.tensor([combined_tokens]).to(curr_model.device)

    with ctx:
        os_set = False
        if "UNSLOTH_RETURN_HIDDEN_STATES" in os.environ:
            os_set = True
            del os.environ["UNSLOTH_RETURN_HIDDEN_STATES"]  # unsloth sets this to 1 during GRPO
        outputs = curr_model(input_ids)
        logits = outputs.logits
        if os_set:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

    log_probs = torch.log_softmax(logits, dim=-1)
    completion_log_prob = 0.0
    output_len = log_probs.shape[1]
    for i in range(len(prompt_tokens), min(len(combined_tokens), output_len)):
        token_id = combined_tokens[i]
        token_log_prob = log_probs[0, i - 1, token_id].item()  # -1 cause next token
        completion_log_prob += token_log_prob
    return completion_log_prob

def _compute_skill_logprobs(question_with_z, completion):
    """
    For a given question (with some strategy index already inserted)
    and a completion, compute log P_z(y | x,z) for all z in Z.
    Returns a list of logprobs aligned with Z.
    """
    logps = []
    for z in Z:
        _, q_with_z_variant = replace_strategy_idx(question_with_z, z)
        log_p_z = get_log_probability(q_with_z_variant, completion)
        logps.append(log_p_z)
    return logps  # len == len(Z)


def _log_mixture_from_logps(logps):
    """
    Given log P_z(y|x,z) for all z, compute log \bar P(y|x)
    where \bar P is the uniform mixture over skills:
        \bar P(y|x) = (1/|Z|) sum_z P_z(y|x,z)
    """
    M = max(logps)
    # log (1/|Z| sum_z exp(logps[z])) = logsumexp(logps) - log|Z|
    return M + math.log(sum(math.exp(lp - M) for lp in logps)) - math.log(len(logps))


def mi_reward(completions, prompts, answer, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    questions = [prompt[1]["content"] for prompt in prompts]

    rewards = []
    with torch.no_grad():
        for i in range(len(contents)):  # TODO: parallelize
            y = contents[i]
            q_with_z = questions[i]

            # Which strategy z generated this prompt?
            z_idx = extract_strategy_idx(q_with_z)

            # Compute log P_z(y|x,z) for all z
            logps = _compute_skill_logprobs(q_with_z, y)
            # Map Z -> log P_z
            skill_to_logp = {z: lp for z, lp in zip(Z, logps)}

            # log P_{z_idx}(y|x,z_idx)
            log_p_z = skill_to_logp[z_idx]
            # log \bar P(y|x)
            log_mix = _log_mixture_from_logps(logps)

            reward = alpha_mi * (log_p_z - log_mix)
            rewards.append(reward)

    return rewards

def mixture_kl_reward(completions, prompts, answer, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    questions = [prompt[1]["content"] for prompt in prompts]

    rewards = []
    with torch.no_grad():
        for i in range(len(contents)):  # TODO: parallelize
            y = contents[i]
            q_with_z = questions[i]

            # log P_z(y|x,z) for all z
            logps = _compute_skill_logprobs(q_with_z, y)
            # log \bar P(y|x)
            log_mix = _log_mixture_from_logps(logps)

            # Reference prompt: question with strategy string removed
            q_ref = remove_strategy_string(q_with_z)
            log_Q = get_log_probability(q_ref, y, ref=True)

            # KL(\bar P || Q) ≈ log_mix - log_Q (for this sampled y)
            kl_estimate = log_mix - log_Q

            # Reward is negative KL so that maximizing reward => minimizing KL
            reward = -alpha_mixkl * kl_estimate
            rewards.append(reward)
    return rewards

def semantic_det_reward(completions, prompts, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    questions = [prompt[1]["content"] for prompt in prompts]

    # Group questions by incrementing the index, cut off when a question repeats
    groups = []
    current_group = []
    seen_questions = set()
    for i, q in enumerate(questions):
        if q in seen_questions:
            if current_group:
                groups.append(current_group)
            current_group = []
            seen_questions = set()
        current_group.append(contents[i])
        seen_questions.add(q)
    if current_group:
        groups.append(current_group)

    rewards = []
    embeddings = get_embeddings_by_question(groups, client)
    _, dets = analyze_embedding_determinants(embeddings)
    for group, det in zip(groups, dets):
        neg_log_det = -math.log(det) if det > 0 else float("inf")
        rewards.extend([alpha_det * neg_log_det] * len(group))
    return rewards

def batch_correctness_reward_func(completions, prompts, answer, **kwargs):
    questions = [prompt[1]["content"] for prompt in prompts]
    individual_rewards = math_correctness_func(prompts, completions, answer, **kwargs)

    # Group rewards by the same logic as semantic_det_reward
    groups = []
    current_group = []
    seen_questions = set()
    for i, q in enumerate(questions):
        if q in seen_questions:
            if current_group:
                groups.append(current_group)
            current_group = []
            seen_questions = set()
        current_group.append(individual_rewards[i])
        seen_questions.add(q)
    if current_group:
        groups.append(current_group)

    rewards = []
    for group in groups:
        max_reward = max(group)
        rewards.extend([max_reward] * len(group))

    result = [
        pass_reward_factor * x + individual_reward_factor * y
        for x, y in zip(rewards, individual_rewards)
    ]
    return result

def semantic_mi_reward(completions, prompts, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    questions = [prompt[1]["content"] for prompt in prompts]
    try:
        labels = [extract_strategy_idx(i) for i in questions]
        print(labels)
        with torch.no_grad():
            miest = compute_embedding_label_mi(contents, labels, compute_control=False)
        scaled_mi = alpha_smi * miest["total_mi"]
    except Exception as e:
        print(f"Error in semantic MI calculation: {e}")
        scaled_mi = 0.0
    return [scaled_mi] * len(contents)

class StrategyGroupedGRPOTrainer(GRPOTrainer):
    def _prepare_inputs(self, inputs):
        expanded = []
        for ex in inputs:
            base_prompt = ex["prompt"]
            for z in Z:
                new_ex = ex.copy()
                new_prompt = [msg.copy() for msg in base_prompt]
                new_prompt[1]["content"] = replace_strategy_idx(base_prompt[1]["content"], z)[1] if use_mi else base_prompt[1]["content"]
                new_ex["prompt"] = new_prompt
                expanded.append(new_ex)
        return super()._prepare_inputs(expanded)

max_prompt_length = 256

# ----------------- Run id -----------------
random.seed(int(time.time() * 1000000) % 2**32)
run_id = random.randint(1000, 9999)
random.seed(FIXED_SEED)

# ----------------- Output dirs -----------------
output_dir = f"{model_name}/{dataset_text}/{config['max_z']}_{alpha_mi}_{alpha_det}_{alpha_smi}_{individual_reward_factor}_{pass_reward_factor}_{run_id}"
print("RUN ID: ", run_id)
print("MODEL: ", model_name)

base_out_dir = f"{store_dir}/models/{output_dir}"
os.makedirs(base_out_dir, exist_ok=True)

# Save config & run meta for reproducibility
with open(os.path.join(base_out_dir, "used_config.yaml"), "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
with open(os.path.join(base_out_dir, "run_meta.json"), "w") as f:
    json.dump({
        "run_id": run_id,
        "model_name": model_name,
        # "dataset": dataset,
        "timestamp": int(time.time()),
        "args": {"config": args.config, "model": args.model}
    }, f, indent=2)

# ------------- Tee stdout/stderr to files -------------
class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

stdout_f = open(os.path.join(base_out_dir, "stdout.log"), "a", buffering=1)
stderr_f = open(os.path.join(base_out_dir, "stderr.log"), "a", buffering=1)
sys.stdout = _Tee(sys.stdout, stdout_f)
sys.stderr = _Tee(sys.stderr, stderr_f)

# ------------- Trainer JSONL logger callback -------------
class JsonlLoggerCallback(TrainerCallback):
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        rec = {"step": int(state.global_step) if state.global_step is not None else None,
               "epoch": float(state.epoch) if state.epoch is not None else None,
               "time": int(time.time())}
        # coerce numbers to float for JSON consistency
        for k, v in logs.items():
            if isinstance(v, (int, float, np.floating)):
                rec[k] = float(v)
            else:
                rec[k] = v
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

trainer_log_path = os.path.join(base_out_dir, "trainer_log.jsonl")

training_args = GRPOConfig(
    learning_rate=lr,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=1,
    num_generations=6,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=steps,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="none",  # we log via callback + tee
    output_dir=base_out_dir,
    hub_model_id=None,
    push_to_hub=False,
    temperature=temperature,
)

print("starting training")

standard_reward_funcs = [
    # xmlcount_reward_func,
    # soft_format_reward_func,
    # strict_format_reward_func,
    batch_correctness_reward_func,
]

trainer = StrategyGroupedGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=standard_reward_funcs
    + [
        *([] if not use_mi else [mi_reward]),
        *([] if not use_det else [semantic_det_reward]),
        *([] if not use_smi else [semantic_mi_reward]),
        *([] if not use_mixkl else [mixture_kl_reward]),
    ],
    args=training_args,
    train_dataset=dataset,
    callbacks=[JsonlLoggerCallback(trainer_log_path)],  # <-- save metrics here
)

trainer.train()

trainer.model.save_pretrained(os.path.join(base_out_dir, "final_model"))
print(f"saved to {run_id}")
