#!/usr/bin/env python3
import argparse, glob, json, os, re
from pathlib import Path
import matplotlib.pyplot as plt

def find_model_dir(model: str, model_id: str) -> Path:
    formal_name_map = {
        "llama": "models--unsloth--meta-llama-3.1-8b-instruct-unsloth-bnb-4bit",
        "qwen":  "models--unsloth--qwen2.5-7b-instruct-unsloth-bnb-4bit",
        "r1":    "models--unsloth--deepseek-r1-distill-qwen-1.5b-bnb-4bit",
    }
    formal_name = formal_name_map[model]
    if model_id == "base":
        snapshot_glob = f"/scratch/gpfs/ds6237/{formal_name}/snapshots/*/"
    else:
        snapshot_glob = f"/scratch/gpfs/ds6237/models/{formal_name}/gsm8k/*_{model_id}/checkpoint-2000/"
    dirs = sorted(glob.glob(snapshot_glob))
    if not dirs:
        raise SystemExit(f"No match for {snapshot_glob}")
    return Path(dirs[0])

def find_trainer_log(model_path: Path) -> Path:
    candidates = [model_path.parent / "trainer_log.jsonl", model_path / "trainer_log.jsonl"]
    for c in candidates:
        if c.is_file():
            return c
    for up in model_path.parents:
        for p in up.glob("**/trainer_log.jsonl"):
            return p
    raise SystemExit("trainer_log.jsonl not found")

def iter_jsonl(log_path: Path):
    brace_re = re.compile(r"\{.*\}")
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = brace_re.search(line)
            if not m: continue
            try:
                yield json.loads(m.group(0))
            except Exception:
                continue

def exp_moving_average(values, alpha=0.9):
    """Return EMA series."""
    if not values:
        return []
    ema = [values[0]]
    for v in values[1:]:
        if v is None:
            ema.append(ema[-1])
        else:
            ema.append(alpha * ema[-1] + (1 - alpha) * v)
    return ema

def plot_with_ema(x, y, ylabel, title, filename, ema_alpha=0.9):
    plt.figure()
    plt.plot(x, y, label="raw", alpha=0.5)
    ema_y = exp_moving_average([v if v is not None else 0 for v in y], ema_alpha)
    plt.plot(x, ema_y, label=f"EMA (α={ema_alpha})", linewidth=2)
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    print(f"Saved {filename}")

def main():
    parser = argparse.ArgumentParser(description="Plot rewards with EMA smoothing")
    parser.add_argument("--model", required=True, choices=["qwen", "llama", "r1"])
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--outdir", default="./outputs")
    parser.add_argument("--ema", type=float, default=0.9, help="EMA smoothing alpha (0-1)")
    args = parser.parse_args()

    model_path = find_model_dir(args.model, args.model_id)
    log_path = find_trainer_log(model_path)
    print(f"Using log: {log_path}")

    xs, reward, batch_correct, mi = [], [], [], []
    for rec in iter_jsonl(log_path):
        step = rec.get("step")
        if step is None:
            continue
        xs.append(step)
        reward.append(rec.get("reward"))
        batch_correct.append(rec.get("rewards/batch_correctness_reward_func"))
        mi.append(rec.get("rewards/mi_reward"))

    os.makedirs(args.outdir, exist_ok=True)
    stem = f"{args.model}_{args.model_id}"

    plot_with_ema(xs, reward, "reward", f"{stem}: reward", f"{args.outdir}/{stem}_reward_ema.png", args.ema)
    plot_with_ema(xs, batch_correct, "batch_correctness_reward_func",
                  f"{stem}: batch_correctness_reward_func", f"{args.outdir}/{stem}_batch_correctness_ema.png", args.ema)
    plot_with_ema(xs, mi, "mi_reward", f"{stem}: mi_reward", f"{args.outdir}/{stem}_mi_reward_ema.png", args.ema)

if __name__ == "__main__":
    main()
