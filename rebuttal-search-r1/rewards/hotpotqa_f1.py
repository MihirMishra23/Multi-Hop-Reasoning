"""Token-level F1 reward, byte-identical to KBevo's reward function."""

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
    precision = overlap / max(1, sum(a.values()))
    recall = overlap / max(1, sum(b.values()))
    return 2 * precision * recall / (precision + recall)


def _extract_answer(solution_str: str) -> str:
    try:
        return solution_str.split("<answer>")[1].split("</answer>")[0]
    except Exception:
        return ""


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    del data_source, extra_info
    answer = _extract_answer(solution_str)
    if not answer.strip():
        return 0.0

    targets = ground_truth["target"] if isinstance(ground_truth, dict) else ground_truth
    if isinstance(targets, str):
        targets = [targets]

    pred_tokens = normalize_text(answer).split()
    score = max(
        (token_f1(pred_tokens, normalize_text(str(target)).split()) for target in targets),
        default=0.0,
    )
    if random.randint(1, 64) == 1:
        print("--------------------------------")
        print(f"[F1] golden = {targets}")
        print(f"[F1] extracted = {answer!r}")
        print(f"[F1] f1 = {score:.3f}")
        print(f"[F1] solution_str = {solution_str!r}")
    return float(score)
