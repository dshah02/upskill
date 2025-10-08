# flex_rewards_adapter.py
# Adapter: exposes math_correctness_func (same signature),
# but logs sample extractions for sanity checks.

from __future__ import annotations
from typing import List, Any, Dict
import os, random, json

from flex_extract import compute_score, ExtractConfig

# ---- Config knobs (env overridable) ----
_LOG_SAMPLES = os.environ.get("FLEX_LOG_SAMPLES", "1") != "0"
_MAX_LOG = int(os.environ.get("FLEX_MAX_LOG", "3"))         # examples per call
_TRIM_Q = int(os.environ.get("FLEX_TRIM_Q", "160"))         # question tail shown
_TRIM_C = int(os.environ.get("FLEX_TRIM_C", "220"))         # completion tail shown

_DEFAULT_CFG = ExtractConfig(
    trim_chars=int(os.environ.get("FLEX_TRIM_CHARS", "2000")),  # None to disable
    accept_boxed=True,
    accept_tags=True,
    accept_fallback_number=True,
    strip_units_before_pick=True,
)

# simple global call counter
_CALL_IDX = 0

def _pairwise_contents(completions) -> List[str]:
    return [c[0]["content"] for c in completions]

def _pairwise_answers(answer) -> List[str]:
    return list(answer)

def _tail(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else ("…" + s[-n:])

def _sample_indices(n: int, k: int) -> List[int]:
    if k >= n:
        return list(range(n))
    # evenly spaced picks for stability
    step = max(n // k, 1)
    idxs = list(range(0, n, step))[:k]
    # ensure exactly k items
    while len(idxs) < k:
        r = random.randrange(n)
        if r not in idxs:
            idxs.append(r)
    return sorted(idxs)

def math_correctness_func(prompts, completions, answer, **kwargs) -> List[float]:
    global _CALL_IDX
    _CALL_IDX += 1

    contents = _pairwise_contents(completions)
    answers  = _pairwise_answers(answer)

    results = [
        compute_score(sol, gt, strict_box=False, cfg=_DEFAULT_CFG)
        for sol, gt in zip(contents, answers)
    ]
    scores = [r["score"] for r in results]

    # -------- optional logging --------
    if _LOG_SAMPLES and len(contents) > 0:
        # pick a few stable examples to print
        k = min(_MAX_LOG, len(contents))
        idxs = _sample_indices(len(contents), k)

        # pull questions (user messages at index 1)
        qs = [p[1]["content"] if len(p) > 1 and "content" in p[1] else "" for p in prompts]

        # batch metrics
        batch_acc = sum(1 for r in results if r["acc"]) / len(results)
        batch_sum = sum(scores)
        log_block: Dict[str, Any] = {
            "reward_call": _CALL_IDX,
            "batch_size": len(contents),
            "batch_acc": round(batch_acc, 4),
            "batch_score_sum": round(batch_sum, 4),
            "examples": []
        }

        for i in idxs:
            dbg = results[i].get("debug", {})
            ex = {
                "i": i,
                "gt": answers[i],
                "pred": results[i]["pred"],
                "acc": results[i]["acc"],
                "method": dbg.get("method"),
                "pred_raw": dbg.get("pred_raw"),
                "gt_norm": dbg.get("gt_norm"),
                "question_tail": _tail(qs[i], _TRIM_Q),
                "completion_tail": _tail(contents[i], _TRIM_C),
            }
            log_block["examples"].append(ex)

        # pretty print one JSON object to keep logs parseable
        print("[flex_examples]", json.dumps(log_block, ensure_ascii=False))

    return scores
