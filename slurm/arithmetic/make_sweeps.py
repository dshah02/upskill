#!/usr/bin/env python3
"""
Generate SLURM scripts for arithmetic 24-game sweeps.

Usage:
  python tools/make_sweeps.py

This creates one script per config under slurm/jobs/, and prints sbatch commands.
"""

import os
from pathlib import Path
from itertools import product
from datetime import datetime

# ---------- YOU CAN TWEAK THESE ----------
PY_BIN = "python"
ENTRYPOINT = "experiments/arithmetic/arithmetic_testing.py"
CONDA_ENV = "grpo"  # -> your environment name
EMAIL = "ds6237@princeton.edu"

# Base (your control) configuration
BASE = dict(
    alpha=0.1, beta=0.1, N=5, warmup=1000, steps=2000, C=5, cap=0.5,
    kl_coef=0.05, temp=0.9, eval_every=200, eval_size=300
)

# ---------- SLURM HEADER TEMPLATE ----------
SLURM_HEADER = """#!/bin/sh
#SBATCH --job-name={job_name}              # create a short name for your job
#SBATCH --output=slurm/slurm_output/oct6/exp/%x_%j.out
#SBATCH --error=slurm/slurm_output/oct6/exp/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=12:00:00
#SBATCH --account=eladgroup
#SBATCH --partition=pli
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user={email}

log_info() {{
    echo "[ $(date '+%Y-%m-%d %H:%M:%S') ] $1"
}}

module purge
module load anaconda3/2023.9
module load cudatoolkit/12.6
module load gcc-toolset/14

conda activate {conda_env}

log_info "Python version: $(python --version 2>&1)"
python -c "import torch; print(f'PyTorch version: {{torch.version.__version__}}')"
python -c "import torch; print(f'CUDA available: {{torch.cuda.is_available()}}')"
python -c "import torch; print(f'CUDA version: {{torch.version.cuda}}')"

if command -v nvidia-smi &>/dev/null; then
    log_info "GPU: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
else
    log_info "CUDA not installed or GPUs not available."
fi

echo "Running nvidia-smi:"
nvidia-smi

set -euo pipefail
"""

# ---------- HELPER: turn a config dict into an argv string ----------
def build_cmd(cfg: dict) -> str:
    args = []
    for k, v in cfg.items():
        if k == "mi_ratio":
            # Only include if not None
            if v is None:
                continue
            args.append(f"--mi_ratio {v}")
        else:
            # print floats compactly; keep ints as ints
            if isinstance(v, float):
                args.append(f"--{k} {v:g}")
            else:
                args.append(f"--{k} {v}")
    return f"{PY_BIN} {ENTRYPOINT} " + " ".join(args)

# ---------- Define the sweeps ----------
def build_sweeps():
    sweeps = []

    # 0) CONTROL (no mi_ratio)
    for kl in [0.0, BASE["kl_coef"]]:
        cfg = dict(BASE)
        cfg["kl_coef"] = kl
        cfg["mi_ratio"] = None  # OFF => not printed
        name = f"ctrl_kl{kl:g}"
        sweeps.append((name, cfg))

    # 1) MI sweep (mi on), with tighter cap (avoid MI dominance)
    for mi in [0.1, 0.3, 0.5]:
        for kl in [0.02, 0.05]:
            cfg = dict(BASE)
            cfg.update(dict(cap=0.1, kl_coef=kl, mi_ratio=mi))
            name = f"mi{mi:g}_kl{kl:g}_cap0.1"
            sweeps.append((name, cfg))

    # 2) KL sweep with MI=0.3 (common sweet-spot), cap=0.1
    for kl in [0.0, 0.02, 0.05, 0.1]:
        cfg = dict(BASE)
        cfg.update(dict(cap=0.1, kl_coef=kl, mi_ratio=0.3))
        name = f"mi0.3_kl{kl:g}_cap0.1"
        sweeps.append((name, cfg))

    # 3) Cap sensitivity at MI=0.3, kl=0.05
    for cap in [0.05, 0.1, 0.5]:
        cfg = dict(BASE)
        cfg.update(dict(cap=cap, kl_coef=0.05, mi_ratio=0.3))
        name = f"mi0.3_kl0.05_cap{cap:g}"
        sweeps.append((name, cfg))

    # 4) Alpha/Beta scan at MI=0.3, kl=0.05, cap=0.1
    for ab in [(0.05, 0.05), (0.1, 0.1), (0.2, 0.2)]:
        a, b = ab
        cfg = dict(BASE)
        cfg.update(dict(alpha=a, beta=b, cap=0.1, kl_coef=0.05, mi_ratio=0.3))
        name = f"mi0.3_kl0.05_cap0.1_ab{a:g}"
        sweeps.append((name, cfg))

    # 5) Temperature & N (strategy count) robustness at MI=0.3
    for temp, N in product([0.7, 0.9, 1.1], [3, 5]):
        cfg = dict(BASE)
        cfg.update(dict(temp=temp, N=N, cap=0.1, kl_coef=0.05, mi_ratio=0.3))
        name = f"mi0.3_kl0.05_cap0.1_t{temp:g}_N{N}"
        sweeps.append((name, cfg))

    # 6) MISL off (mi_ratio=None), but KL scan & tighter cap (sanity)
    for kl in [0.0, 0.02, 0.05, 0.1]:
        cfg = dict(BASE)
        cfg.update(dict(cap=0.1, kl_coef=kl, mi_ratio=None))
        name = f"ctrl_kl{kl:g}_cap0.1"
        sweeps.append((name, cfg))

    return sweeps

# ---------- Write scripts ----------
def main():
    sweeps = build_sweeps()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs_dir = Path("slurm/jobs")
    jobs_dir.mkdir(parents=True, exist_ok=True)

    sbatch_cmds = []

    for name, cfg in sweeps:
        job_name = f"bench_{name}"
        script_path = jobs_dir / f"{job_name}.sh"

        header = SLURM_HEADER.format(
            job_name=job_name,
            email=EMAIL,
            conda_env=CONDA_ENV
        )

        cmd = build_cmd(cfg)

        body = f"""{header}
echo "Launching: {cmd}"
srun {cmd}
"""

        script_path.write_text(body)
        sbatch_cmds.append(f"sbatch {script_path}")

    print(f"\nGenerated {len(sweeps)} jobs in: {jobs_dir.resolve()}\n")
    print("Submit with:")
    for c in sbatch_cmds:
        print(c)

    # Also dump a compact CSV summary for tracking
    csv_path = jobs_dir / f"sweep_{ts}.csv"
    with open(csv_path, "w") as f:
        keys = sorted({k for _, cfg in sweeps for k in cfg.keys()})
        f.write("job_name," + ",".join(keys) + "\n")
        for name, cfg in sweeps:
            job_name = f"bench_{name}"
            row = [job_name] + [str(cfg.get(k, "")) for k in keys]
            f.write(",".join(row) + "\n")
    print(f"\nWrote summary CSV: {csv_path}")

if __name__ == "__main__":
    main()
