#!/usr/bin/env python3
# Example: python 24_llm.py --alpha 0.5 --beta 0.5 --N 5 --warmup 100 --steps 2000 --C 5 --cap 1.0 --kl_coef 0.05

import math, random, argparse, copy
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# CLI
# -----------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000, help="RL updates")
    p.add_argument("--warmup", type=int, default=300, help="supervised pretrain steps (0 to disable)")
    p.add_argument("--batch", type=int, default=32, help="prompts per update")
    p.add_argument("--C", type=int, default=3, help="completions per (x,z) group")
    p.add_argument("--N", type=int, default=5, help="number of strategies z∈{0..N-1}")
    p.add_argument("--alpha", type=float, default=0.0,
                   help="weight on H(tok|s) (specificity term). If --beta not set, beta=alpha.")
    p.add_argument("--beta", type=float, default=None,
                   help="weight on H(tok) (marginal term). If omitted, beta=alpha.")
    p.add_argument("--cap", type=float, default=0.5,
                   help="symmetric clamp for per-sample MISL reward (|r_misl|<=cap).")
    p.add_argument("--maxv", type=int, default=9, help="value range [0..maxv] (use 9 for single-digit)")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--temp", type=float, default=0.9, help="sampling temperature")
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--eval_size", type=int, default=300)
    p.add_argument("--n_embed", type=int, default=128, help="residual dimension (embedding size)")
    p.add_argument("--num_heads", type=int, default=4, help="number of attention heads")
    p.add_argument("--num_layers", type=int, default=2, help="number of transformer layers")
    # ==== BEGIN EDIT: add after other add_argument calls ====

    # ==== BEGIN EDIT ====
    p.add_argument("--mi_ratio", type=float, default=None,
                help="If set, mixes correctness and MI rewards in standardized form. "
                        "Example: 0.3 means MI contributes 30%% of total reward. "
                        "If None, uses the old additive reward (r_total = r_corr + r_misl).")
    # ==== END EDIT ====


    # ==== END EDIT ====

    p.add_argument("--mi_eval_prompts", type=int, default=100,
                   help="#prompts for MI estimation at eval/print")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--kl_coef", type=float, default=0.0,
                   help="Coefficient for KL(pi || pi_ref) penalty during RL. pi_ref is the frozen model after warmup.")
    return p.parse_args()

args = get_args()
if args.beta is None:
    args.beta = args.alpha
DEVICE = torch.device(args.device)
torch.manual_seed(0); random.seed(0)

# -----------------------------
# Vocab / tokens
# -----------------------------
# Standard 24-game operators only
OPS = ['+','-','*','/']
TOKENS = list("0123456789+-*/")  # 10 digits + 4 ops
stoi = {ch:i for i,ch in enumerate(TOKENS)}
itos = {i:ch for ch,i in stoi.items()}

def decode_ids(ids: List[int]) -> str:
    return "".join(itos[i] for i in ids)

# -----------------------------
# Helpers / safe eval
# -----------------------------
def safe_eval_binop(L: float, op: str, R: float) -> Optional[float]:
    if op == '/' and R == 0:
        return None
    try:
        if op == '+': return L + R
        if op == '-': return L - R
        if op == '*': return L * R
        if op == '/': return L / R
    except Exception:
        return None
    return None

def eval_chain_left_assoc(d1:int, o1:str, d2:int, o2:str, d3:int, o3:str, d4:int) -> Optional[float]:
    a = safe_eval_binop(float(d1), o1, float(d2))
    if a is None: return None
    b = safe_eval_binop(a, o2, float(d3))
    if b is None: return None
    c = safe_eval_binop(b, o3, float(d4))
    return c

def close_to_24(x: float) -> bool:
    # exact if integer, otherwise a tight tolerance for floats
    return abs(x - 24.0) < 1e-9

# -----------------------------
# Data generator (4 inputs → solvable to 24)
# -----------------------------
from itertools import permutations, product

def find_solution_for_quad(quad: Tuple[int,int,int,int]) -> Optional[Tuple[int,str,int,str,int,str,int]]:
    nums = list(quad)
    for a,b,c,d in permutations(nums, 4):
        for o1,o2,o3 in product(OPS, repeat=3):
            val = eval_chain_left_assoc(a, o1, b, o2, c, o3, d)
            if val is None: 
                continue
            if close_to_24(val):
                return (a,o1,b,o2,c,o3,d)
    return None

def sample_problem_24(maxv: int) -> Tuple[int,int,int,int]:
    # Sample until we get a solvable 24 instance (with left-assoc solution)
    while True:
        quad = tuple(random.randint(0, maxv) for _ in range(4))
        sol = find_solution_for_quad(quad)
        if sol is not None:
            return quad

def prompt_ids_from_quad(quad: Tuple[int,int,int,int], z:int) -> List[int]:
    # Keep the same prompt structure as before: "z * n * n * n * n *"
    ids = [stoi[str(z)], stoi['*']]
    for n in quad:
        ids += [stoi[str(n)], stoi['*']]
    return ids

# -----------------------------
# Parse / reward for 7-token chain
# -----------------------------
def parse_chain_7(expr: str) -> Optional[Tuple[int,str,int,str,int,str,int]]:
    # Expect digit op digit op digit op digit, no spaces
    if len(expr) < 7: return None
    # All tokens are single characters from our vocab
    kinds = ['d','o','d','o','d','o','d']
    items = []
    i = 0
    for k in kinds:
        if i >= len(expr): return None
        ch = expr[i]
        if k == 'd':
            if ch.isdigit():
                items.append(int(ch))
            else:
                return None
        else:
            if ch in OPS:
                items.append(ch)
            else:
                return None
        i += 1
    if i != len(expr):  # must be exactly 7 tokens, no extras
        return None
    d1,o1,d2,o2,d3,o3,d4 = items
    return (d1,o1,d2,o2,d3,o3,d4)

def reward_correct_24(quad: Tuple[int,int,int,int], out_ids: List[int]) -> int:
    expr = decode_ids(out_ids)
    parsed = parse_chain_7(expr)
    if parsed is None:
        return 0
    d1,o1,d2,o2,d3,o3,d4 = parsed
    # multiset check: digits must be exactly the prompt's digits
    want = sorted(list(quad))
    got  = sorted([d1,d2,d3,d4])
    if want != got:
        return 0
    val = eval_chain_left_assoc(d1,o1,d2,o2,d3,o3,d4)
    if val is None:
        return 0
    return int(close_to_24(val))

# -----------------------------
# Tiny causal Transformer (unchanged)
# -----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
    def forward(self, x, mask):
        B,T,C = x.size()
        k = self.key(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = self.query(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = self.value(x).view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(mask==0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(y)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.GELU(),
            nn.Linear(4*n_embd, n_embd),
        )
    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_layer=2, n_head=4, n_embd=128, max_len=64):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(max_len, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        # cache a mask buffer per length to avoid realloc
        self.register_buffer("_mask_cache_len", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_mask_cache", torch.ones(1,1,1,1))

    def _attn_mask(self, T: int, device):
        if int(self._mask_cache_len.item()) != T or self._mask_cache.device != device:
            mask = torch.tril(torch.ones(T,T,device=device)).unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
            self._mask_cache = mask
            self._mask_cache_len = torch.tensor(T, dtype=torch.long, device=device)
        return self._mask_cache

    def forward(self, idx):
        B,T = idx.shape
        device = idx.device
        pos = torch.arange(0, T, device=device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        mask = self._attn_mask(T, device)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        return self.head(x)

# -----------------------------
# Strict 7-token sampling & logprob (d o d o d o d)
# -----------------------------
@dataclass
class SampleResult:
    out_ids: List[int]
    logp_sum: torch.Tensor  # sum over the 7 emitted tokens

def _mask_logits_for_position(logits: torch.Tensor, pos_kind: str):
    """
    logits: [1, V]; pos_kind: 'digit' or 'op'
    """
    keep = [stoi[str(d)] for d in range(10)] if pos_kind == 'digit' else [stoi[o] for o in OPS]
    mask = torch.full_like(logits, -1e9)
    mask[:, keep] = 0.0
    return logits + mask

def sample_output_chain7(model: TinyGPT, prompt_ids: List[int], temp: float) -> SampleResult:
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    logp_sum = torch.zeros((), device=DEVICE)
    kinds = ['digit','op','digit','op','digit','op','digit']

    for kind in kinds:
        logits = model(ids)[:, -1, :] / max(1e-6, temp)
        logits = _mask_logits_for_position(logits, kind)
        probs = F.softmax(logits, dim=-1)
        m = torch.distributions.Categorical(probs)
        tok = m.sample()
        logp_sum += m.log_prob(tok).squeeze(0)
        ids = torch.cat([ids, tok.view(1,1)], dim=1)

    out = ids[0, -7:].tolist()
    return SampleResult(out, logp_sum)

def logprobs_chain7(model: TinyGPT, full_seq: List[int], prompt_len: int) -> torch.Tensor:
    """
    Correct teacher-forced per-step log-probs for the 7 output tokens.
    """
    device = next(model.parameters()).device
    full = torch.tensor(full_seq, dtype=torch.long, device=device).unsqueeze(0)

    ids = full[:, :prompt_len]
    targets = full[0, prompt_len:prompt_len+7]
    kinds = ['digit','op','digit','op','digit','op','digit']

    lp = []
    for j, kind in enumerate(kinds):
        logits = model(ids)[:, -1, :]
        logits = _mask_logits_for_position(logits, kind)
        logp = F.log_softmax(logits, dim=-1)
        tgt = targets[j]
        lp.append(logp[0, tgt])
        ids = torch.cat([ids, tgt.view(1, 1)], dim=1)

    return torch.stack(lp)  # [7]

# -----------------------------
# MI / entropy estimation (updated to 7 tokens)
# -----------------------------
@torch.no_grad()
def estimate_mi_and_entropies(model: TinyGPT, N: int, temp: float, num_prompts: int, maxv: int):
    seq_count = 0
    tok_count = 0
    sum_seq_logratio = 0.0
    sum_tok_logratio = 0.0
    sum_Hc_tok = 0.0
    sum_Hm_tok = 0.0

    for _ in range(num_prompts):
        quad = sample_problem_24(maxv)
        for z in range(N):
            prompt_ids = prompt_ids_from_quad(quad, z)
            s = sample_output_chain7(model, prompt_ids, temp=temp)
            full_seq_z = prompt_ids + s.out_ids

            lpz_vec = logprobs_chain7(model, full_seq_z, prompt_len=len(prompt_ids))  # [7]

            p_mix = None
            for zp in range(N):
                prompt_ids_zp = prompt_ids.copy()
                prompt_ids_zp[0] = stoi[str(zp)]
                lp_zp_vec = logprobs_chain7(model, prompt_ids_zp + s.out_ids,
                                            prompt_len=len(prompt_ids_zp))
                p = torch.exp(lp_zp_vec)  # [7]
                p_mix = p if p_mix is None else (p_mix + p)
            p_mix = (p_mix / N).clamp_min(1e-12)
            lpmix_vec = torch.log(p_mix)  # [7]

            seq_logratio = (lpz_vec - lpmix_vec).sum().item()
            sum_seq_logratio += seq_logratio
            sum_tok_logratio += (lpz_vec - lpmix_vec).sum().item()
            sum_Hc_tok += (-lpz_vec).sum().item()
            sum_Hm_tok += (-lpmix_vec).sum().item()
            seq_count += 1
            tok_count += 7

    MISL_seq = (sum_seq_logratio / max(1, seq_count))
    MISL_tok = (sum_tok_logratio / max(1, tok_count))
    Hc_tok  = (sum_Hc_tok / max(1, tok_count))
    Hm_tok  = (sum_Hm_tok / max(1, tok_count))
    return MISL_seq, MISL_tok, Hc_tok, Hm_tok

# -----------------------------
# Post-training analysis helpers (still digit/op for first step)
# -----------------------------
def digit_op_joint_distribution(model: TinyGPT, prompt_ids, temp: float = 1.0):
    model.eval()
    with torch.no_grad():
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        logits1 = model(ids)[:, -1, :] / max(1e-6, temp)
        logits1 = _mask_logits_for_position(logits1, 'digit')
        p_digit = F.softmax(logits1, dim=-1)[0]
        digit_idx = [stoi[str(d)] for d in range(10)]
        p_d = p_digit[digit_idx]

        joint = torch.zeros(10, len(OPS), device=DEVICE)
        for d_i, tok_id in enumerate(digit_idx):
            ids2 = torch.cat([ids, torch.tensor([[tok_id]], device=DEVICE)], dim=1)
            logits2 = model(ids2)[:, -1, :] / max(1e-6, temp)
            logits2 = _mask_logits_for_position(logits2, 'op')
            p_op = F.softmax(logits2, dim=-1)[0]
            op_idx = [stoi[o] for o in OPS]
            joint[d_i, :] = p_d[d_i] * p_op[op_idx]
        return joint

def _fmt_table_10x4(joint_10x4: torch.Tensor) -> str:
    arr = joint_10x4.detach().cpu().numpy()
    header = "digit \\ op |  " + "  ".join([f"{op:>3s}" for op in OPS])
    line   = "-" * len(header)
    rows = [header, line]
    for d in range(10):
        cells = "  ".join([f"{arr[d, j]:.3f}" for j in range(len(OPS))])
        rows.append(f"{d:>10d} |  {cells}")
    rows.append(f"sum={arr.sum():.6f}")
    return "\n".join(rows)

def _topk_pairs(joint_10x4: torch.Tensor, k=8):
    arr = joint_10x4.detach().cpu().numpy()
    flat = []
    for d in range(10):
        for j,op in enumerate(OPS):
            flat.append(((d, op), float(arr[d, j])))
    flat.sort(key=lambda x: x[1], reverse=True)
    return flat[:k]

def save_heatmap(joint_10x4: torch.Tensor, title: str, fname: str):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,3.5))
        plt.imshow(joint_10x4.detach().cpu().numpy(), aspect='auto')
        plt.colorbar(label="P(d,op)")
        plt.yticks(range(10), [str(d) for d in range(10)])
        plt.xticks(range(len(OPS)), OPS)
        plt.title(title)
        plt.xlabel("op")
        plt.ylabel("digit")
        plt.tight_layout()
        plt.savefig(fname, dpi=130)
        plt.close()
    except Exception as e:
        print(f"[warn] could not save heatmap {fname}: {e}")

# -----------------------------
# FAST KL helpers (batched, small support)
# -----------------------------
def _keep_indices(kind: str, device):
    if kind == 'digit':
        return torch.tensor([stoi[str(d)] for d in range(10)], dtype=torch.long, device=device)
    else:
        return torch.tensor([stoi[o] for o in OPS], dtype=torch.long, device=device)

def _subset_logprobs_last(model: TinyGPT, ids: torch.Tensor, kind: str) -> torch.Tensor:
    """
    Returns log-probs over the small allowed subset at the next position.
    ids: [B, T] int64
    kind: 'digit' or 'op'
    Output: [B, K] (K=10 for 'digit', K=4 for 'op')
    """
    logits = model(ids)[:, -1, :]  # [B, V]
    keep = _keep_indices(kind, ids.device)
    logits_small = logits.index_select(dim=1, index=keep)  # [B, K]
    return F.log_softmax(logits_small, dim=-1)

def kl_seq_current_vs_ref_batched(
    model: TinyGPT, ref_model: TinyGPT,
    prompt_ids: List[int],
    out_ids_list: List[List[int]]
) -> torch.Tensor:
    """
    KL(π_current || π_ref) over the 7 output positions for C samples, in parallel.
    Returns: [C] per-sample KL sums.
    """
    C = len(out_ids_list)
    base = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # [1, T]
    ids_curr = base.repeat(C, 1)  # [C, T]
    ids_ref  = ids_curr.clone()

    kl_total = torch.zeros(C, device=DEVICE)  # [C]
    kinds = ['digit','op','digit','op','digit','op','digit']
    out_tok_tensor = torch.tensor(out_ids_list, dtype=torch.long, device=DEVICE)  # [C, 7]

    for j, kind in enumerate(kinds):
        # Current (grad) and ref (no grad) log-probs on small support
        logp_curr_small = _subset_logprobs_last(model, ids_curr, kind)       # [C, K]
        with torch.no_grad():
            logp_ref_small  = _subset_logprobs_last(ref_model, ids_ref, kind)  # [C, K]

        # KL per sample on small support
        p_curr_small = logp_curr_small.exp()                                  # [C, K]
        kl = (p_curr_small * (logp_curr_small - logp_ref_small)).sum(dim=1)   # [C]
        kl_total = kl_total + kl

        # Advance with the actually sampled tokens from current policy
        next_tok = out_tok_tensor[:, j].view(C, 1)                            # [C,1]
        ids_curr = torch.cat([ids_curr, next_tok], dim=1)
        ids_ref  = torch.cat([ids_ref,  next_tok], dim=1)

    return kl_total  # [C]

# ==== BEGIN EDIT: add helper for standardized reward mixing ====
def _stdize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # zero-mean, unit-std (protect against zero-variance)
    return (x - x.mean()) / x.std(unbiased=False).clamp_min(eps)

def mix_rewards_fixed_ratio(r_corr: torch.Tensor,
                            r_mi: torch.Tensor,
                            mi_ratio: float) -> torch.Tensor:
    """
    Per-batch standardize both components, then mix:
        mixed = (1 - mi_ratio) * r_corr_z + mi_ratio * r_mi_z
    This ensures MI accounts for exactly `mi_ratio` of the combined signal magnitude.
    """
    rc = _stdize(r_corr)
    rm = _stdize(r_mi)
    mixed = (1.0 - mi_ratio) * rc + mi_ratio * rm
    # Already mean ~ 0; no further centering needed for advantage
    return mixed
# ==== END EDIT ====


# -----------------------------
# Training (warmup + RL)
# -----------------------------
def train():
    model = TinyGPT(
        vocab_size=len(TOKENS),
        n_layer=args.num_layers,
        n_head=args.num_heads,
        n_embd=args.n_embed,
        max_len=64
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def evaluate(model, size=200, k=None):
        k = k or min(5, args.N)
        succ1 = succk = 0
        for _ in range(size):
            quad = sample_problem_24(args.maxv)
            ok_any = False
            num_correct = 0
            for zi in range(k):
                prompt_ids = prompt_ids_from_quad(quad, zi)
                s = sample_output_chain7(model, prompt_ids, temp=args.temp)
                ok = reward_correct_24(quad, s.out_ids)
                num_correct += int(ok)
                ok_any = ok_any or ok
            succ1 += num_correct/k
            succk += int(ok_any)
        return succ1/size, succk/size

    # ---- Warmup: teacher-forced CE ----
    if args.warmup > 0:
        for step in range(1, args.warmup+1):
            quads = [sample_problem_24(args.maxv) for _ in range(args.batch)]
            loss = 0.0
            for quad in quads:
                sol = find_solution_for_quad(quad)
                if sol is None:
                    continue
                d1,o1,d2,o2,d3,o3,d4 = sol
                z = random.randrange(args.N)
                prompt_ids = prompt_ids_from_quad(quad, z)
                ids = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
                ce = 0.0
                targets = [stoi[str(d1)], stoi[o1], stoi[str(d2)], stoi[o2],
                           stoi[str(d3)], stoi[o3], stoi[str(d4)]]
                kinds   = ['digit','op','digit','op','digit','op','digit']
                cur = ids
                for tgt, kind in zip(targets, kinds):
                    logits = model(cur)[:, -1, :]
                    logits = _mask_logits_for_position(logits, kind)
                    ce += F.cross_entropy(logits, torch.tensor([tgt], device=DEVICE))
                    cur = torch.cat([cur, torch.tensor([[tgt]], device=DEVICE)], dim=1)
                loss = loss + ce
            loss = loss / max(1, len(quads))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 100 == 0:
                p1, pk = evaluate(model, size=200, k=min(5,args.N))
                MISL_seq, MISL_tok, Hc_tok, Hm_tok = estimate_mi_and_entropies(
                    model, N=args.N, temp=args.temp, num_prompts=args.mi_eval_prompts, maxv=args.maxv
                )
                print(f"[warmup {step}] loss={loss.item():.3f} "
                      f"pass@1={p1:.3f} pass@{min(5,args.N)}={pk:.3f} | "
                      f"MISL seq={MISL_seq:.3f} tok={MISL_tok:.3f} Hc(tok)={Hc_tok:.3f} Hm(tok)={Hm_tok:.3f}")

    # ---- Freeze a reference policy after warmup (for KL) ----
    ref_model = copy.deepcopy(model).to(DEVICE)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()

    # ---- RL (GRPO +/- MISL +/- KL) ----
    for step in range(1, args.steps+1):
        total_loss = torch.zeros((), device=DEVICE)
        kl_running = 0.0
        kl_count = 0

        for _ in range(args.batch):
            quad = sample_problem_24(args.maxv)
            z = random.randrange(args.N)
            prompt_ids = prompt_ids_from_quad(quad, z)

            # Sample C completions (7 tokens)
            samples = [sample_output_chain7(model, prompt_ids, temp=args.temp) for _ in range(args.C)]
            r_corr = torch.tensor([reward_correct_24(quad, s.out_ids) for s in samples],
                                  dtype=torch.float32, device=DEVICE)  # [C]

            # MISL reward with separate alpha/beta weights
            use_misl = (args.alpha != 0.0) or (args.beta != 0.0)
            if use_misl:
                r_misl = []
                for s in samples:
                    full_seq_z = prompt_ids + s.out_ids
                    lp_z = logprobs_chain7(model, full_seq_z, prompt_len=len(prompt_ids)).sum()  # Lz

                    # mixture over z' (exact for small N)
                    with torch.no_grad():
                        p_mix = None
                        for zp in range(args.N):
                            prompt_ids_zp = prompt_ids.copy()
                            prompt_ids_zp[0] = stoi[str(zp)]
                            full_seq_zp = prompt_ids_zp + s.out_ids
                            lp_zp = logprobs_chain7(model, full_seq_zp,
                                                    prompt_len=len(prompt_ids_zp)).sum()
                            p = torch.exp(lp_zp)
                            p_mix = p if p_mix is None else (p_mix + p)
                        lp_mix = torch.log((p_mix / args.N).clamp_min(1e-12))  # Lmix

                    misl_val = args.alpha * lp_z - args.beta * lp_mix
                    if args.cap is not None:
                        misl_val = torch.clamp(misl_val, -args.cap, args.cap)
                    r_misl.append(misl_val)
                r_misl = torch.stack(r_misl)  # [C]
            else:
                r_misl = torch.zeros(args.C, device=DEVICE)

        # ==== BEGIN EDIT (reward mixing with optional off-switch) ====
        # We now have r_corr (shape [C]) and r_misl ([C]) for the sampled completions.

        if args.mi_ratio is not None:
            # Optional: tiny jitter if either reward is constant to avoid zero-std
            if r_corr.std(unbiased=False) < 1e-8:
                r_corr = r_corr + 1e-6 * torch.randn_like(r_corr)
            if r_misl.std(unbiased=False) < 1e-8:
                r_misl = r_misl + 1e-6 * torch.randn_like(r_misl)

            # Fixed-ratio mixture after per-batch standardization
            adv = mix_rewards_fixed_ratio(r_corr, r_misl, mi_ratio=args.mi_ratio)

        else:
            # === Old behavior for reproducibility ===
            # Total reward is simple sum; then use group-relative baseline
            r_total = r_corr + r_misl
            adv = r_total - r_total.mean()

        # Policy-gradient term (+ optional KL penalty), batched
        losses = []
        for s, a in zip(samples, adv):
            losses.append(- s.logp_sum * a.detach())

        if args.kl_coef > 0.0:
            out_ids_list = [s.out_ids for s in samples]
            kl_vals = kl_seq_current_vs_ref_batched(model, ref_model, prompt_ids, out_ids_list)  # [C]
            for i in range(args.C):
                losses[i] = losses[i] + args.kl_coef * kl_vals[i]
            kl_running += float(kl_vals.sum().detach().cpu())
            kl_count += args.C

        total_loss += torch.stack(losses).sum()
        # ==== END EDIT ====


        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_every == 0:
            p1, pk = evaluate(model, size=args.eval_size, k=min(5,args.N))
            MISL_seq, MISL_tok, Hc_tok, Hm_tok = estimate_mi_and_entropies(
                model, N=args.N, temp=args.temp, num_prompts=args.mi_eval_prompts, maxv=args.maxv
            )
            kl_mean = (kl_running / max(1, kl_count)) if args.kl_coef > 0 else 0.0
            print(f"[step {step}] loss={total_loss.item():.3f} "
                  f"pass@1={p1:.3f} pass@{min(5,args.N)}={pk:.3f} | "
                  f"MISL seq={MISL_seq:.3f} tok={MISL_tok:.3f} Hc(tok)={Hc_tok:.3f} Hm(tok)={Hm_tok:.3f}"
                  f"{' | KL(seq)=' + str(round(kl_mean,3)) if args.kl_coef>0 else ''}")

    # Final eval
    p1, pk = evaluate(model, size=args.eval_size, k=min(5,args.N))
    MISL_seq, MISL_tok, Hc_tok, Hm_tok = estimate_mi_and_entropies(
        model, N=args.N, temp=args.temp, num_prompts=args.mi_eval_prompts, maxv=args.maxv
    )
    print(f"[FINAL] pass@1={p1:.3f} pass@{min(5,args.N)}={pk:.3f} | "
          f"MISL seq={MISL_seq:.3f} tok={MISL_tok:.3f} Hc(tok)={Hc_tok:.3f} Hm(tok)={Hm_tok:.3f}")

    # ---------- Post-training analysis ----------
    print("\n[ANALYZE] Joint distributions over (digit, op) for 5 fixed problems per strategy")
    rnd_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    random.seed(1234); torch.manual_seed(1234)

    fixed = [sample_problem_24(args.maxv) for _ in range(200)]
    random.setstate(rnd_state); torch.random.set_rng_state(torch_state)

    joint_sums = {z: torch.zeros(10, len(OPS), device=DEVICE) for z in range(args.N)}

    for idx, quad in enumerate(fixed, 1):
        for z in range(args.N):
            prompt_ids = prompt_ids_from_quad(quad, z)
            joint = digit_op_joint_distribution(model, prompt_ids, temp=args.temp)
            joint_sums[z] += joint

    for z in range(args.N):
        joint_avg = joint_sums[z] / len(fixed)
        print(f"\n[z={z}] Average joint distribution over {len(fixed)} problems:")
        print(f"[z={z}] top pairs (digit,op):prob → "
              + ", ".join([f"({d},{op})={p:.3f}" for (d,op),p in _topk_pairs(joint_avg, k=8)]))
        print(_fmt_table_10x4(joint_avg))
        save_heatmap(joint_avg, title=f"Average prob(d,op) | z={z}",
                     fname=f"digit_op_prob_avg_z{z}.png")

    model_save_path = f"toy_llm24_model_alpha{args.alpha}_beta{args.beta}_cap{args.cap}_kl{args.kl_coef}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'args': args,
        'final_metrics': {
            'pass_at_1': p1,
            'pass_at_k': pk,
            'MISL_seq': MISL_seq,
            'MISL_tok': MISL_tok,
            'Hc_tok': Hc_tok,
            'Hm_tok': Hm_tok
        }
    }, model_save_path)
    print(f"\n[SAVE] Model saved to: {model_save_path}")

if __name__ == "__main__":
    train()
