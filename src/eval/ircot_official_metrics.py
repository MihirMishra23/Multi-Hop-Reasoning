"""Answer scoring used by the released IRCoT development evaluation.

Ported from ``StonyBrookNLP/ircot`` at the revision recorded by
``agent.ircot_agent.OFFICIAL_IRCOT_COMMIT``.  This is the DROP answer metric
used by upstream IRCoT for its non-official development-set sweep.
"""

from __future__ import annotations

import re
import string
from typing import Iterable, List, Sequence, Set, Tuple, Union

import ftfy
import numpy as np
from scipy.optimize import linear_sum_assignment


Answer = Union[str, List[str], Tuple[str, ...]]
EXCLUDE = set(string.punctuation)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_answer(text: str) -> str:
    def normalize_token(token: str) -> str:
        token = token.lower()
        if not _is_number(token):
            token = "".join(character for character in token if character not in EXCLUDE)
        if _is_number(token):
            token = str(float(token))
        token = re.sub(r"\b(a|an|the)\b", " ", token, flags=re.UNICODE)
        return " ".join(token.split())

    return " ".join(part for part in (normalize_token(t) for t in re.split(" |-", text)) if part).strip()


def _answer_to_bags(answer: Answer) -> Tuple[List[str], List[Set[str]]]:
    raw_spans: Sequence[str] = answer if isinstance(answer, (list, tuple)) else [answer]
    normalized = [_normalize_answer(span) for span in raw_spans]
    return normalized, [set(span.split()) for span in normalized]


def _match_numbers_if_present(gold: Set[str], predicted: Set[str]) -> bool:
    gold_numbers = {word for word in gold if _is_number(word)}
    predicted_numbers = {word for word in predicted if _is_number(word)}
    return not gold_numbers or bool(gold_numbers.intersection(predicted_numbers))


def _compute_f1(predicted: Set[str], gold: Set[str]) -> Tuple[float, float, float]:
    intersection = len(gold.intersection(predicted))
    precision = intersection / float(len(predicted)) if predicted else 1.0
    recall = intersection / float(len(gold)) if gold else 1.0
    f1 = 0.0 if precision == recall == 0.0 else 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f1s = np.zeros([len(gold), len(predicted)])
    precisions = np.zeros([len(gold), len(predicted)])
    recalls = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for predicted_index, predicted_item in enumerate(predicted):
            if _match_numbers_if_present(gold_item, predicted_item):
                f1, precision, recall = _compute_f1(predicted_item, gold_item)
                f1s[gold_index, predicted_index] = f1
                precisions[gold_index, predicted_index] = precision
                recalls[gold_index, predicted_index] = recall
    row_indices, column_indices = linear_sum_assignment(-f1s)
    output_size = max(len(gold), len(predicted))
    max_f1s = np.zeros([output_size])
    max_precisions = np.zeros([output_size])
    max_recalls = np.zeros([output_size])
    for row, column in zip(row_indices, column_indices):
        max_f1s[row] = max(max_f1s[row], f1s[row, column])
        max_precisions[row] = max(max_precisions[row], precisions[row, column])
        max_recalls[row] = max(max_recalls[row], recalls[row, column])
    return max_f1s, max_precisions, max_recalls


def drop_metrics(predicted: Answer, gold: Answer) -> Tuple[float, float, float, float]:
    """Return upstream IRCoT's per-answer EM, F1, precision, and recall."""
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)
    exact_match = float(
        set(predicted_bags[0]) == set(gold_bags[0])
        and len(predicted_bags[0]) == len(gold_bags[0])
    )
    f1s, precisions, recalls = _align_bags(predicted_bags[1], gold_bags[1])
    return (
        exact_match,
        round(float(np.mean(f1s)), 2),
        round(float(np.mean(precisions)), 2),
        round(float(np.mean(recalls)), 2),
    )


def evaluate_predictions(results: Iterable[dict]) -> dict:
    totals = [0.0, 0.0, 0.0, 0.0]
    count = 0
    for result in results:
        prediction = ftfy.fix_text(str(result.get("pred") or ""))
        answer_objects = result.get("gold_answer_objects") or []
        ground_truths: List[List[str]] = []
        for answer_object in answer_objects:
            spans = answer_object.get("spans") or []
            if spans:
                ground_truths.append([ftfy.fix_text(str(span)) for span in spans])
            elif answer_object.get("number"):
                ground_truths.append([ftfy.fix_text(str(answer_object["number"]))])
            else:
                date = answer_object.get("date") or {}
                ground_truths.append(
                    [ftfy.fix_text("-".join(str(date.get(k, "")) for k in ("day", "month", "year")))]
                )
        if not ground_truths:
            ground_truths = [[ftfy.fix_text(str(value))] for value in result.get("gold_answer") or [""]]
        scores = max(drop_metrics([prediction], ground_truth) for ground_truth in ground_truths)
        totals = [total + score for total, score in zip(totals, scores)]
        count += 1
    return {
        "em": round(totals[0] / count, 3) if count else 0.0,
        "f1": round(totals[1] / count, 3) if count else 0.0,
        "precision": round(totals[2] / count, 3) if count else 0.0,
        "recall": round(totals[3] / count, 3) if count else 0.0,
        "count": count,
    }


__all__ = ["drop_metrics", "evaluate_predictions"]
