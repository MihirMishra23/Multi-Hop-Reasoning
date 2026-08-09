"""Token-level F1 reward, byte-identical to KBevo's `src/reward_func.py:f1_reward`.

Reused for the Search-R1 baseline so both methods score answers with the
exact same function (KBevo F1 = token-overlap F1 over normalize_text-ed
tokens). Plugged into verl via `custom_reward_function.path`.
"""

import random
import re
from collections import Counter


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def token_f1(a_tokens, b_tokens) -> float:
    a = Counter(a_tokens)
    b = Counter(b_tokens)
    overlap = sum((a & b).values())
    if overlap == 0:
        return 0.0
    p = overlap / max(1, sum(a.values()))
    r = overlap / max(1, sum(b.values()))
    return 2 * p * r / (p + r)


def _extract_answer(solution_str: str) -> str:
    # Matches KBevo extract_answer_from_tags: first <answer>...</answer> block.
    try:
        return solution_str.split("<answer>")[1].split("</answer>")[0]
    except Exception:
        return ""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    answer = _extract_answer(solution_str)
    if not answer.strip():
        return 0.0

    targets = ground_truth["target"] if isinstance(ground_truth, dict) else ground_truth
    if isinstance(targets, str):
        targets = [targets]

    pred_tokens = normalize_text(answer).split()
    best = 0.0
    for g in targets:
        gold_tokens = normalize_text(str(g)).split()
        f1 = token_f1(pred_tokens, gold_tokens)
        if f1 > best:
            best = f1

    score = float(best)
    if random.randint(1, 64) == 1:
        print("--------------------------------")
        print(f"[F1] golden = {targets}")
        print(f"[F1] extracted = {answer!r}")
        print(f"[F1] f1 = {score:.3f}")
        print(f"[F1] solution_str = {solution_str!r}")
    return score
