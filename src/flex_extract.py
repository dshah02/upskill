# flex_extract.py
# A robust, drop-in extraction + normalization + verification toolkit
# for math-style short answers (numbers, fractions, boxed LaTeX, etc.)

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

# ---------- Low-level helpers ----------

def last_boxed_only_string(string: str) -> Optional[str]:
    """Return the last '\\boxed{...}' segment including braces, or None."""
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left = 0
    while i < len(string):
        if string[i] == "{":
            num_left += 1
        if string[i] == "}":
            num_left -= 1
            if num_left == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None

def remove_boxed(s: str) -> str:
    """Strip the '\\boxed{...}' wrapper and return the inner content."""
    left = "\\boxed{"
    assert s.startswith(left), f"box error: {s}"
    assert s.endswith("}"), f"box error: {s}"
    return s[len(left):-1]

# Numeric finder: integers, decimals, comma-grouped, simple fractions.
NUM_PAT = re.compile(
    r"""
    (?<![\w/])                             # not preceded by letter or slash
    (?:-?\d{1,3}(?:,\d{3})*(?:\.\d+)?      # 1,234 or 1,234.56
      | -?\d+(?:\.\d+)?                    # 123 or 123.45
      | -?\d+\s*/\s*\d+                    # simple fraction a/b
    )
    (?![\w/])                              # not followed by letter or slash
    """,
    re.VERBOSE
)

# Common “final answer” tags and XML answer span
TAG_PAT = re.compile(r"(?i)\b(?:answer|final|result|output)\s*[:=]\s*([^\n]+)")
XML_ANSWER_PAT = re.compile(r"(?is)<answer[^>]*>(.*?)</answer>")

# Strip LaTeX wrappers
def strip_latex_wrappers(s: str) -> str:
    s = re.sub(r"\$(.*?)\$", r"\1", s)                # $...$
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)          # \text{...}
    s = re.sub(r"\\boxed\{(.*?)\}", r"\1", s)         # \boxed{...}
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)  # keep as fraction string
    s = re.sub(r"\\overline\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{(.*?)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{(.*?)\}", r"\1", s)
    return s

# Optional unit list to remove (case-insensitive word-bound)
DEFAULT_UNITS = [
    "pages","page","pg","points","point","pts","min","mins","minute","minutes",
    "hr","hrs","hour","hours","sec","second","seconds",
    "cm","mm","m","km","ft","feet","yard","yards","in","inch","inches",
    "dollars","usd","$","kg","g","grams","pounds","lbs","units",
    "degree","degrees","°","percent","percentage","%"
]

def strip_units(s: str, units: List[str]) -> str:
    for u in units:
        s = re.sub(rf"(?i)\b{re.escape(u)}\b", "", s)
    return s

# ---------- Config ----------

@dataclass
class ExtractConfig:
    accept_xml: bool = True                # try <answer>...</answer> first
    accept_boxed: bool = True              # then try \boxed{...}
    accept_tags: bool = True               # then try Answer:/Final:/Result:/Output:
    accept_fallback_number: bool = True    # fallback to last numeric token
    accept_equals_tail: bool = True        # if '=' present, keep the rightmost side
    strip_units_before_pick: bool = True   # remove common units before numeric pick
    units: List[str] = None                # override default units
    trim_chars: Optional[int] = None       # if not None, keep only last N chars
    # If completions are very long, you can set trim_chars=2000 or similar.

    def __post_init__(self):
        if self.units is None:
            self.units = list(DEFAULT_UNITS)

# ---------- Core extractor/normalizer ----------

def extract_numeric_answer(text: str, cfg: ExtractConfig) -> Tuple[str, str]:
    """
    Returns (raw_candidate, method) where method in {"xmltag","boxed","tag","fallback","invalid"}.
    """
    s = text if cfg.trim_chars is None else text[-cfg.trim_chars:]

    # 0) XML <answer>…</answer>
    if cfg.accept_xml:
        m = XML_ANSWER_PAT.findall(s)
        if m:
            return m[-1].strip(), "xmltag"

    # 1) Boxed
    if cfg.accept_boxed:
        boxed = last_boxed_only_string(s)
        if boxed:
            return remove_boxed(boxed).strip(), "boxed"

    # 2) Tagged
    if cfg.accept_tags:
        m = TAG_PAT.findall(s)
        if m:
            return m[-1].strip(), "tag"

    # 3) Fallback: last number anywhere
    if cfg.accept_fallback_number:
        s2 = strip_latex_wrappers(s)
        if cfg.strip_units_before_pick:
            s2 = strip_units(s2, cfg.units)
        # if there's an equals, keep right side
        if cfg.accept_equals_tail and "=" in s2:
            s2 = s2.split("=")[-1]
        nums = list(NUM_PAT.finditer(s2))
        if nums:
            return nums[-1].group(0).strip(), "fallback"

    return "[INVALID]", "invalid"

def normalize_answer(raw: str) -> str:
    """
    Normalize a raw extracted answer to a canonical numeric-like string.
    Keeps integers, decimals, or simple fractions 'a/b'. Removes commas.
    For sentences like 'total is 1,234.5 pages', it returns '1234.5'.
    """
    s = strip_latex_wrappers(raw.strip())
    # If '=' exists, keep right-most side to reduce 'x = 3' noise
    if "=" in s:
        s = s.split("=")[-1].strip()

    # Find last numeric token inside this candidate
    m_all = list(NUM_PAT.finditer(s))
    if m_all:
        s = m_all[-1].group(0).strip()

    # Canonicalize fraction spacing (e.g., "3 / 4" -> "3/4")
    if "/" in s:
        s = re.sub(r"\s*/\s*", "/", s)

    # Remove commas in numbers
    s = s.replace(",", "")
    return s

# ---------- Verifiers & scoring ----------

def verify_minerva_style(solution_str: str, gt: str, cfg: ExtractConfig, gt_need_extract: bool = False) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Minerva-like check: extract from solution, normalize both, compare strings.
    Returns (is_correct, pred_norm, debug_info).
    """
    raw_pred, method = extract_numeric_answer(solution_str, cfg)
    pred = normalize_answer(raw_pred)

    if gt_need_extract:
        # Extract from GT if GT contains \boxed{...}, else fallback to numbers
        if "\\boxed{" in gt:
            gt_norm = normalize_answer(remove_boxed(last_boxed_only_string(gt) or ""))
        else:
            _raw_gt, _ = extract_numeric_answer(gt, cfg)
            gt_norm = normalize_answer(_raw_gt)
    else:
        gt_norm = normalize_answer(gt)

    return (pred == gt_norm), pred, {
        "method": method,
        "pred_raw": raw_pred,
        "gt_norm": gt_norm
    }

def verify_strict_box(solution_str: str, gt_norm: str, cfg: ExtractConfig) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Strict boxed: only accept if the last \\boxed{...} equals gt (after normalize).
    """
    s = solution_str if cfg.trim_chars is None else solution_str[-cfg.trim_chars:]
    boxed = last_boxed_only_string(s)
    if not boxed:
        return False, None, {"error": "no_boxed"}
    pred = normalize_answer(remove_boxed(boxed))
    return (pred == gt_norm), pred, {"method": "strict_box"}

def compute_score(solution_str: str, ground_truth: str, strict_box: bool = False, cfg: ExtractConfig = None, gt_need_extract: bool = False) -> Dict[str, Any]:
    """
    Returns dict: {"score": 1.0|-1.0, "acc": bool, "pred": str, "debug": {...}}
    """
    cfg = cfg or ExtractConfig()
    if strict_box:
        gt_norm = normalize_answer(ground_truth)
        ok, pred, dbg = verify_strict_box(solution_str, gt_norm, cfg)
    else:
        ok, pred, dbg = verify_minerva_style(solution_str, ground_truth, cfg, gt_need_extract=gt_need_extract)
    return {"score": 1.0 if ok else -1.0, "acc": ok, "pred": pred, "debug": dbg}

# ---------- Batch helpers (plug-in friendly) ----------

def batch_compute_scores(solutions: List[str], answers: List[str], strict_box: bool = False, cfg: ExtractConfig = None, gt_need_extract: bool = False) -> List[Dict[str, Any]]:
    """
    Vectorized scoring for a list of solutions vs answers (pairwise).
    """
    assert len(solutions) == len(answers), "solutions and answers must be same length"
    return [compute_score(sol, ans, strict_box=strict_box, cfg=cfg, gt_need_extract=gt_need_extract)
            for sol, ans in zip(solutions, answers)]

def batch_correctness_reward(solutions: List[str], prompts: List[str], answers: List[str], pass_reward_factor: float = 1.0, individual_reward_factor: float = 1.0, strict_box: bool = False, cfg: ExtractConfig = None) -> List[float]:
    """
    Example “group max” reward like your current code: for each group of identical
    prompts, everyone gets the max reward from that group; then a linear combo with
    individual.
    """
    cfg = cfg or ExtractConfig()
    # Individual scores
    indiv = [compute_score(sol, ans, strict_box=strict_box, cfg=cfg)["score"] for sol, ans in zip(solutions, answers)]

    # Group by repeated prompts (same behavior you had)
    rewards = []
    groups: List[List[float]] = []
    cur, seen = [], set()
    for i, q in enumerate(prompts):
        if q in seen:
            if cur:
                groups.append(cur)
            cur = []
            seen = set()
        cur.append(indiv[i])
        seen.add(q)
    if cur:
        groups.append(cur)

    # Build group-max list matching original order
    expanded = []
    idx = 0
    for g in groups:
        gmax = max(g) if g else -1.0
        for _ in g:
            expanded.append(gmax)
            idx += 1

    # Combine
    return [pass_reward_factor * gm + individual_reward_factor * iv for gm, iv in zip(expanded, indiv)]

# ---------- Tiny self-test (optional) ----------

if __name__ == "__main__":
    cfg = ExtractConfig(trim_chars=2000)

    samples = [
        ("<answer>328 pages</answer>", "328"),
        ("Therefore, the total is 328 pages.", "328"),
        ("Final: 29", "29"),
        ("We get \\boxed{42}.", "42"),
        ("Answer = 1,234.5 units", "1234.5"),
        ("The fraction is \\frac{3}{4} of the pie, so result = 3/4.", "3/4"),
        ("No tag, last number 17 occurs here 3 times -> 1, 5, 17.", "17"),
    ]

    for s, gt in samples:
        out = compute_score(s, gt, strict_box=False, cfg=cfg)
        print(s, "|| GT:", gt, "=>", out)
