from unsloth import FastLanguageModel
import torch
import re
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import json
import os
import random
import math
import yaml
import argparse
from google import genai
from semantic_determinant import (
    get_embeddings_by_question,
    analyze_embedding_determinants,
)
from semantic_similarity import compute_embedding_label_mi
from UNSLOTH_rewards import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    math_correctness_func,
)
from utils import load_model
from utils import (
    extract_strategy_idx,
    replace_strategy_idx,
    add_strategy_string,
    remove_strategy_string,
)
from dotenv import load_dotenv
import os

 #only needed for Semantic tests
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# client = genai.Client(api_key=GEMINI_API_KEY)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config_1.yaml")
parser.add_argument("--model", type=str, default="llama", required=False)

args = parser.parse_args()

# required for offline
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

print(config["max_seq_length"])
print("lora_rank", config["lora_rank"])
print("alpha_mi", config.get("alpha_mi", 0))
print("alpha_det", config.get("alpha_det", 0))
print("alpha_smi", config.get("alpha_smi", 0))
print("individual_reward_factor", config.get("individual_reward_factor", 1))
print("pass_reward_factor", config.get("pass_reward_factor", 0))
print("max_z", config["max_z"])
print("steps", config.get("steps", 250))
print("dataset", config.get("dataset", "gsm8k"))

alpha_mi = config.get("alpha_mi", 0)
alpha_det = config.get("alpha_det", 0)
alpha_smi = config.get("alpha_smi", 0)
use_mi = alpha_mi != 0
use_det = alpha_det != 0
use_smi = alpha_smi != 0
cache_dir = "./cache"
max_seq_length = config["max_seq_length"]
lora_rank = config["lora_rank"]
individual_reward_factor = config.get("individual_reward_factor", 1)
pass_reward_factor = config.get("pass_reward_factor", 0)
Z = list(range(1, config["max_z"] + 1))
steps = config.get("steps", 250)
model_name = config.get("model", args.model)
dataset = config.get("dataset", "gsm8k")
if dataset not in ["gsm8k", "math"]:
    raise ValueError("Dataset must be either 'gsm8k' or 'math'")
dataset_text = dataset

store_dir = f"{os.environ['USER']}"

if "qwen" == model_name:
    model_name = "Qwen2.5-7B"
    cache_dir = f"{store_dir}/cache/qwen-2-5-7b"
elif "r1-qwen" == model_name:
    model_name = "DeepSeek-R1-Distill-Qwen-1.5B"  # works but says 5b
    cache_dir = f"{store_dir}/cache/r1-qwen"
else:
    model_name = "meta-Llama-3.1-8B-Instruct"
    cache_dir = f"{store_dir}/cache/llama-3-1-8b"

model, tokenizer = load_model(cache_dir, max_seq_length, lora_rank, peft_apply=True)

if dataset == "gsm8k":
    filepath = "./dataset_cache/gsm8k_train.json"
else:
    filepath = "./dataset_cache/math_train.json"

with open(filepath, "r") as f:
    dataset_data = json.load(f)

random.shuffle(dataset_data)

for item in dataset_data:
    idx = random.choice(Z)
    item["prompt"][1]["content"] = add_strategy_string(
        item["prompt"][1]["content"], idx
    )

dataset = Dataset.from_list(dataset_data)

def get_log_probability(prompt, completion):
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

    combined_input = prompt + "\n" + completion
    combined_tokens = tokenizer.encode(combined_input, add_special_tokens=False)

    input_ids = torch.tensor([combined_tokens]).to(model.device)

    with torch.no_grad():
        os_set = False
        if "UNSLOTH_RETURN_HIDDEN_STATES" in os.environ:
            os_set = True
            del os.environ[
                "UNSLOTH_RETURN_HIDDEN_STATES"
            ]  # internally unsloth sets this to 1 after the first GRPO step
        outputs = model(input_ids)
        logits = outputs.logits
        if os_set:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

        # if logits.shape[-1] == 4096: #alternative way to get around this
        #     #Internally, UnslothGRPOTrainer.py runs os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        #     lm_head = trainer.model.get_output_embeddings().weight
        #     logits = torch.matmul(outputs.logits, lm_head.t())

    log_probs = torch.log_softmax(logits, dim=-1)

    completion_log_prob = 0.0

    output_len = log_probs.shape[1]
    for i in range(len(prompt_tokens), min(len(combined_tokens), output_len)):
        token_id = combined_tokens[i]
        # print(token_id, tokenizer.decode([token_id]))
        token_log_prob = log_probs[0, i - 1, token_id].item()  # -1 cause next token
        completion_log_prob += token_log_prob

    return completion_log_prob


# need the dataset to incldue the Group [1,.., len(Z)], then we need to compute log p(y|z) - log sum_z(p(y | z))
def mi_reward(completions, prompts, answer, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    questions = [prompt[1]["content"] for prompt in prompts]

    rewards = []
    for i in range(
        len(contents)
    ):  # need to parallelize this, i.e. stack before get probs
        idx = extract_strategy_idx(questions[i])
        log_p_z = get_log_probability(questions[i], contents[i])

        log_values = []
        for z in Z:
            _, new_question = replace_strategy_idx(questions[i], z)
            log_val = get_log_probability(new_question, contents[i]) - math.log(len(Z))
            log_values.append(log_val)

        M = max(log_values)  # to avoid underflow
        log_p = M + math.log(sum(math.exp(lv - M) for lv in log_values))

        reward = alpha_mi * (log_p_z - log_p)
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
    # embeddings is a list of embedding groups, one per group
    # analyze_embedding_determinants expects a list of embedding groups
    _, dets = analyze_embedding_determinants(embeddings)
    for group, det in zip(groups, dets):
        neg_log_det = -math.log(det) if det > 0 else float("inf")
        rewards.extend([alpha_det * neg_log_det] * len(group))

    return rewards


def batch_correctness_reward_func(completions, prompts, answer, **kwargs):
    questions = [prompt[1]["content"] for prompt in prompts]
    # print(f"Questions: {questions}")

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

    # print(f"Batch reward: {rewards}")
    # print(f"Individual reward: {individual_rewards}")

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

        # Scale MI estimate similar to mi_reward
        scaled_mi = alpha_smi * miest["total_mi"]  # Access the total_mi value from dict
    except Exception as e:
        print(f"Error in semantic MI calculation: {e}")
        scaled_mi = 0.0

    return [scaled_mi] * len(contents)


class StrategyGroupedGRPOTrainer(GRPOTrainer):
    def _prepare_inputs(self, inputs):
        expanded = []
        for ex in inputs:
            base_prompt = ex["prompt"]  # this is your list of messages
            for z in Z:
                # 1) copy the entire example dict
                new_ex = ex.copy()

                # 2) make a fresh copy of the prompt list + its dicts
                new_prompt = [msg.copy() for msg in base_prompt]

                # 3) tweak the 2nd message’s content
                #    (replace_strategy_idx returns a tuple, so [1] is the modified message)
                new_prompt[1]["content"] = replace_strategy_idx(
                    base_prompt[1]["content"], z
                )[1]

                # 4) stick it back into the new example dict
                new_ex["prompt"] = new_prompt

                expanded.append(new_ex)

        # now every element of `expanded` is a dict with a "prompt" key
        return super()._prepare_inputs(expanded)


max_prompt_length = 256


random.seed()
run_id = random.randint(1000, 9999)
output_dir = f"{model_name}/{dataset_text}/{config['max_z']}_{alpha_mi}_{alpha_det}_{alpha_smi}_{individual_reward_factor}_{pass_reward_factor}_{run_id}"

training_args = GRPOConfig(
    learning_rate=5e-6,
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
    save_steps=steps,
    max_grad_norm=0.1,
    report_to="none",
    output_dir=f"{store_dir}/models/{output_dir}",
    hub_model_id=None,
    push_to_hub=False,
)

print("starting training")

standard_reward_funcs = [
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    batch_correctness_reward_func,
]

trainer = StrategyGroupedGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=standard_reward_funcs
    + [
        *([] if not use_mi else [mi_reward]),  # Conditionally include mi_reward
        *([] if not use_det else [semantic_det_reward]),
        *([] if not use_smi else [semantic_mi_reward]),
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

trainer.model.save_pretrained(f"{store_dir}/models/{output_dir}/final_model")
print(f"saved to {run_id}")
