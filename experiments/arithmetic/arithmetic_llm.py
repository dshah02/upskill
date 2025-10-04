#!/usr/bin/env python3
# Example: python arithmetic_llm.py --alpha 0.5 --beta 0.5 --N 5 --warmup 100 --steps 2000 --C 5 --cap 1.0 --kl_coef 0.05

import math, random, argparse, copy
from dataclasses import dataclass
from typing import List, Tuple

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
    p.add_argument("--n_embed", type=int, default=128)
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
TOKENS = list("0123456789+-*/%")  # 10 digits + 5 ops (including '*')
stoi = {ch:i for i,ch in enumerate(TOKENS)}
itos = {i:ch for ch,i in stoi.items()}
OPS = ['+','-','*','/','%']

def decode_ids(ids: List[int]) -> str:
    return "".join(itos[i] for i in ids)

# -----------------------------
# Data generator per spec
# -----------------------------
def safe_eval_expr(L: int, op: str, R: int):
    if op in ('/','%') and R == 0:
        return None
    try:
        val = eval(f"{L}{op}{R}")
    except Exception:
        return None
    return val

def sample_problem(maxv: int) -> Tuple[int,int,int]:
    while True:
        x = random.randint(0, maxv)
        y = random.randint(0, maxv)
        op = random.choice(OPS)
        if random.random() < 0.5:
            x, y = y, x
        val = safe_eval_expr(x, op, y)
        if val is None:
            continue
        if isinstance(val, float):
            if not val.is_integer():
                continue
            val = int(val)
        if 0 <= val <= maxv:
            triple = [x, y, int(val)]
            random.shuffle(triple)
            return tuple(triple)

def find_solution_for_triple(triple: Tuple[int,int,int]):
    a,b,c = triple
    nums = [a,b,c]
    for i in range(3):
        target = nums[i]
        others = [nums[j] for j in range(3) if j!=i]
        for L,R in [(others[0], others[1]), (others[1], others[0])]:
            for op in OPS:
                val = safe_eval_expr(L, op, R)
                if val is None: continue
                if isinstance(val, float) and not val.is_integer(): continue
                if int(val) == target:
                    return L, op, R
    return None

def prompt_ids_from_triple(triple: Tuple[int,int,int], z:int) -> List[int]:
    ids = [stoi[str(z)], stoi['*']]
    for n in triple:
        ids += [stoi[str(n)], stoi['*']]
    return ids

# -----------------------------
# Reward / parsing
# -----------------------------
def parse_expression(expr: str):
    pos = [(expr.find(op), op) for op in OPS]
    pos = [(i,op) for i,op in pos if i!=-1]
    if not pos: return None
    i, op = min(pos, key=lambda x:x[0])
    left = expr[:i]; right = expr[i+1:]
    if left=="" or right=="": return None
    if not left.isdigit() or not right.isdigit(): return None
    return int(left), op, int(right)

def reward_correct(triple: Tuple[int,int,int], out_ids: List[int]) -> int:
    expr = decode_ids(out_ids)
    parsed = parse_expression(expr)
    if parsed is None: return 0
    L, op, R = parsed
    val = safe_eval_expr(L, op, R)
    if val is None: return 0
    tri = list(triple)
    for i in range(3):
        if int(val) == tri[i]:
            others = [tri[j] for j in range(3) if j!=i]
            if sorted([L,R]) == sorted(others):
                return 1
    return 0

# -----------------------------
# Tiny causal Transformer
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
    def forward(self, idx):
        B,T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        mask = torch.tril(torch.ones(T,T,device=idx.device)).unsqueeze(0).unsqueeze(0)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        return self.head(x)

# -----------------------------
# Strict 3-token sampling & logprob (digit, op, digit)
# -----------------------------
@dataclass
class SampleResult:
    out_ids: List[int]
    logp_sum: torch.Tensor  # sum over the 3 emitted tokens

def _mask_logits_for_position(logits: torch.Tensor, pos_kind: str):
    """
    logits: [1, V]; pos_kind: 'digit' or 'op'
    """
    keep = [stoi[str(d)] for d in range(10)] if pos_kind == 'digit' else [stoi[o] for o in OPS]
    mask = torch.full_like(logits, -1e9)
    mask[:, keep] = 0.0
    return logits + mask

def sample_output_digit_op_digit(model: TinyGPT, prompt_ids: List[int], temp: float) -> SampleResult:
    ids = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    logp_sum = torch.zeros((), device=DEVICE)

    # step 1: digit
    logits = model(ids)[:, -1, :] / max(1e-6, temp)
    logits = _mask_logits_for_position(logits, 'digit')
    probs = F.softmax(logits, dim=-1)
    m = torch.distributions.Categorical(probs)
    tok = m.sample()
    logp_sum += m.log_prob(tok).squeeze(0)
    ids = torch.cat([ids, tok.view(1,1)], dim=1)

    # step 2: op
    logits = model(ids)[:, -1, :] / max(1e-6, temp)
    logits = _mask_logits_for_position(logits, 'op')
    probs = F.softmax(logits, dim=-1)
    m = torch.distributions.Categorical(probs)
    tok = m.sample()
    logp_sum += m.log_prob(tok).squeeze(0)
    ids = torch.cat([ids, tok.view(1,1)], dim=1)

    # step 3: digit
    logits = model(ids)[:, -1, :] / max(1e-6, temp)
    logits = _mask_logits_for_position(logits, 'digit')
    probs = F.softmax(logits, dim=-1)
    m = torch.distributions.Categorical(probs)
    tok = m.sample()
    logp_sum += m.log_prob(tok).squeeze(0)
    ids = torch.cat([ids, tok.view(1,1)], dim=1)

    out = ids[0, -3:].tolist()
    return SampleResult(out, logp_sum)

def logprobs_digit_op_digit(model: TinyGPT, full_seq: List[int], prompt_len: int) -> torch.Tensor:
    """
    Correct teacher-forced per-step log-probs for the 3 output tokens.
    """
    device = next(model.parameters()).device
    full = torch.tensor(full_seq, dtype=torch.long, device=device).unsqueeze(0)

    ids = full[:, :prompt_len]
    targets = full[0, prompt_len:prompt_len+3]

    lp = []
    for j in range(3):
        logits = model(ids)[:, -1, :]
        kind = 'digit' if j in (0, 2) else 'op'
        logits = _mask_logits_for_position(logits, kind)
        logp = F.log_softmax(logits, dim=-1)
        tgt = targets[j]
        lp.append(logp[0, tgt])
        ids = torch.cat([ids, tgt.view(1, 1)], dim=1)

    return torch.stack(lp)  # [3]

# -----------------------------
# MI / entropy estimation (unchanged)
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
        triple = sample_problem(maxv)
        for z in range(N):
            prompt_ids = prompt_ids_from_triple(triple, z)
            s = sample_output_digit_op_digit(model, prompt_ids, temp=temp)
            full_seq_z = prompt_ids + s.out_ids

            lpz_vec = logprobs_digit_op_digit(model, full_seq_z, prompt_len=len(prompt_ids))  # [3]

            p_mix = None
            for zp in range(N):
                prompt_ids_zp = prompt_ids.copy()
                prompt_ids_zp[0] = stoi[str(zp)]
                lp_zp_vec = logprobs_digit_op_digit(model, prompt_ids_zp + s.out_ids,
                                                    prompt_len=len(prompt_ids_zp))
                p = torch.exp(lp_zp_vec)  # [3]
                p_mix = p if p_mix is None else (p_mix + p)
            p_mix = (p_mix / N).clamp_min(1e-12)
            lpmix_vec = torch.log(p_mix)  # [3]

            seq_logratio = (lpz_vec - lpmix_vec).sum().item()
            sum_seq_logratio += seq_logratio
            sum_tok_logratio += (lpz_vec - lpmix_vec).sum().item()
            sum_Hc_tok += (-lpz_vec).sum().item()
            sum_Hm_tok += (-lpmix_vec).sum().item()
            seq_count += 1
            tok_count += 3

    MISL_seq = (sum_seq_logratio / max(1, seq_count))
    MISL_tok = (sum_tok_logratio / max(1, tok_count))
    Hc_tok  = (sum_Hc_tok / max(1, tok_count))
    Hm_tok  = (sum_Hm_tok / max(1, tok_count))
    return MISL_seq, MISL_tok, Hc_tok, Hm_tok

# -----------------------------
# Post-training analysis helpers
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

def _fmt_table_10x5(joint_10x5: torch.Tensor) -> str:
    arr = joint_10x5.detach().cpu().numpy()
    header = "digit \\ op |  " + "  ".join([f"{op:>3s}" for op in OPS])
    line   = "-" * len(header)
    rows = [header, line]
    for d in range(10):
        cells = "  ".join([f"{arr[d, j]:.3f}" for j in range(len(OPS))])
        rows.append(f"{d:>10d} |  {cells}")
    rows.append(f"sum={arr.sum():.6f}")
    return "\n".join(rows)

def _topk_pairs(joint_10x5: torch.Tensor, k=8):
    arr = joint_10x5.detach().cpu().numpy()
    flat = []
    for d in range(10):
        for j,op in enumerate(OPS):
            flat.append(((d, op), float(arr[d, j])))
    flat.sort(key=lambda x: x[1], reverse=True)
    return flat[:k]

def save_heatmap(joint_10x5: torch.Tensor, title: str, fname: str):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,3.5))
        plt.imshow(joint_10x5.detach().cpu().numpy(), aspect='auto')
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
# KL helpers
# -----------------------------
def _masked_probs_and_logprobs(model: TinyGPT, ids: torch.Tensor, kind: str):
    """Return p and log p at the next position with digit/op mask. Shapes: [V]."""
    logits = model(ids)[:, -1, :]
    logits = _mask_logits_for_position(logits, kind)
    logp = F.log_softmax(logits, dim=-1)[0]
    p = logp.exp()
    return p, logp

def kl_seq_current_vs_ref(model: TinyGPT, ref_model: TinyGPT, prompt_ids: List[int], out_ids: List[int]) -> torch.Tensor:
    """
    KL(π_current || π_ref) across the 3 output positions, conditioning on the
    sampled tokens out_ids emitted by the current policy.
    Returns a scalar tensor (sum over positions).
    """
    # Build the same context step-by-step using the sampled tokens
    ids_curr = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    ids_ref  = ids_curr.clone()

    total_kl = torch.zeros((), device=DEVICE)
    kinds = ['digit', 'op', 'digit']

    for j, kind in enumerate(kinds):
        p_curr, logp_curr = _masked_probs_and_logprobs(model, ids_curr, kind)
        with torch.no_grad():
            # ref_model is frozen; avoid grads for speed
            p_ref, logp_ref = _masked_probs_and_logprobs(ref_model, ids_ref, kind)
            # Numerical safety
            p_ref = p_ref.clamp_min(1e-12)
            logp_ref = torch.log(p_ref)

        # KL(p_curr || p_ref) = sum p_curr * (log p_curr - log p_ref)
        total_kl = total_kl + torch.sum(p_curr * (logp_curr - logp_ref))

        # advance both contexts with the actually sampled token from current
        tok = torch.tensor([[out_ids[j]]], dtype=torch.long, device=DEVICE)
        ids_curr = torch.cat([ids_curr, tok], dim=1)
        ids_ref  = torch.cat([ids_ref,  tok], dim=1)

    return total_kl

# -----------------------------
# Training (warmup + RL)
# -----------------------------
def train():
    model = TinyGPT(vocab_size=len(TOKENS), n_layer=2, n_head=4, n_embd=args.n_embed, max_len=64).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def evaluate(model, size=200, k=None):
        k = k or min(5, args.N)
        succ1 = succk = 0
        for _ in range(size):
            triple = sample_problem(args.maxv)
            ok_any = False
            num_correct = 0
            for zi in range(k):
                prompt_ids = prompt_ids_from_triple(triple, zi)
                s = sample_output_digit_op_digit(model, prompt_ids, temp=args.temp)
                ok = reward_correct(triple, s.out_ids)
                num_correct += int(ok)
                ok_any = ok_any or ok
            succ1 += num_correct/k
            succk += int(ok_any)
        return succ1/size, succk/size

    # ---- Warmup: teacher-forced CE ----
    if args.warmup > 0:
        for step in range(1, args.warmup+1):
            triples = [sample_problem(args.maxv) for _ in range(args.batch)]
            loss = 0.0
            for triple in triples:
                sol = find_solution_for_triple(triple)
                if sol is None:
                    continue
                L, op, R = sol
                z = random.randrange(args.N)
                prompt_ids = prompt_ids_from_triple(triple, z)
                x = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
                ids = x.clone()
                ce = 0.0
                targets = [stoi[str(L)], stoi[op], stoi[str(R)]]
                for j, tgt in enumerate(targets):
                    logits = model(ids)[:, -1, :]
                    kind = 'digit' if j in (0,2) else 'op'
                    logits = _mask_logits_for_position(logits, kind)
                    ce += F.cross_entropy(logits, torch.tensor([tgt], device=DEVICE))
                    ids = torch.cat([ids, torch.tensor([[tgt]], device=DEVICE)], dim=1)
                loss = loss + ce
            loss = loss / max(1, len(triples))
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
            triple = sample_problem(args.maxv)
            z = random.randrange(args.N)
            prompt_ids = prompt_ids_from_triple(triple, z)

            # Sample C completions (digit, op, digit)
            samples = [sample_output_digit_op_digit(model, prompt_ids, temp=args.temp) for _ in range(args.C)]
            r_corr = torch.tensor([reward_correct(triple, s.out_ids) for s in samples],
                                  dtype=torch.float32, device=DEVICE)  # [C]

            # MISL reward with separate alpha/beta weights
            use_misl = (args.alpha != 0.0) or (args.beta != 0.0)
            if use_misl:
                r_misl = []
                for s in samples:
                    full_seq_z = prompt_ids + s.out_ids
                    lp_z = logprobs_digit_op_digit(model, full_seq_z, prompt_len=len(prompt_ids)).sum()  # Lz

                    # mixture over z' (exact for small N)
                    with torch.no_grad():
                        p_mix = None
                        for zp in range(args.N):
                            prompt_ids_zp = prompt_ids.copy()
                            prompt_ids_zp[0] = stoi[str(zp)]
                            full_seq_zp = prompt_ids_zp + s.out_ids
                            lp_zp = logprobs_digit_op_digit(model, full_seq_zp,
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

            # Total reward
            r_total = r_corr + r_misl

            # Group-relative baseline (mean-center within the C samples)
            adv = r_total - r_total.mean()

            # Policy-gradient term (+ optional KL penalty)
            for s, a in zip(samples, adv):
                loss = - s.logp_sum * a.detach()

                if args.kl_coef > 0.0:
                    kl_val = kl_seq_current_vs_ref(model, ref_model, prompt_ids, s.out_ids)
                    loss = loss + args.kl_coef * kl_val
                    kl_running += float(kl_val.detach().cpu())
                    kl_count += 1

                total_loss += loss

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

    fixed = [sample_problem(args.maxv) for _ in range(200)]
    random.setstate(rnd_state); torch.random.set_rng_state(torch_state)

    joint_sums = {z: torch.zeros(10, 5) for z in range(args.N)}
    correct_joint_sum = torch.zeros(10, 5)

    for idx, triple in enumerate(fixed, 1):
        for z in range(args.N):
            prompt_ids = prompt_ids_from_triple(triple, z)
            joint = digit_op_joint_distribution(model, prompt_ids, temp=args.temp)
            joint_sums[z] += joint
        sol = find_solution_for_triple(triple)
        if sol is not None:
            L, op, R = sol
            correct_joint_sum[L, OPS.index(op)] += 1.0

    correct_joint_avg = correct_joint_sum / correct_joint_sum.sum()
    print(f"\n[CORRECT] Distribution of correct answers over {len(fixed)} problems:")
    print(f"[CORRECT] top pairs (digit,op):prob → "
          + ", ".join([f"({d},{op})={p:.3f}" for (d,op),p in _topk_pairs(correct_joint_avg, k=8)]))
    print(_fmt_table_10x5(correct_joint_avg))
    save_heatmap(correct_joint_avg, title="Distribution of correct answers",
                 fname="digit_op_prob_correct_answers.png")

    for z in range(args.N):
        joint_avg = joint_sums[z] / len(fixed)
        print(f"\n[z={z}] Average joint distribution over {len(fixed)} problems:")
        print(f"[z={z}] top pairs (digit,op):prob → "
              + ", ".join([f"({d},{op})={p:.3f}" for (d,op),p in _topk_pairs(joint_avg, k=8)]))
        print(_fmt_table_10x5(joint_avg))
        save_heatmap(joint_avg, title=f"Average prob(d,op) | z={z}",
                     fname=f"digit_op_prob_avg_z{z}.png")

    model_save_path = f"toy_llm2_model_alpha{args.alpha}_beta{args.beta}_cap{args.cap}_kl{args.kl_coef}.pt"
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

