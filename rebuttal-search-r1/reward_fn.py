#!/usr/bin/env python3
"""
Custom reward function for Search-R1 using F1 score from Multi-Hop-Reasoning metrics.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.metrics import f1_score, normalize_answer


def compute_f1_reward(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score reward for a single prediction.

    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer

    Returns:
        F1 score (between 0 and 1)
    """
    f1, precision, recall = f1_score(prediction, ground_truth)
    return f1


def compute_f1_reward_batch(predictions: list, ground_truths: list) -> list:
    """
    Compute F1 score rewards for a batch of predictions.

    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers

    Returns:
        List of F1 scores
    """
    assert len(predictions) == len(ground_truths), \
        f"Batch size mismatch: {len(predictions)} != {len(ground_truths)}"

    rewards = []
    for pred, gt in zip(predictions, ground_truths):
        f1 = compute_f1_reward(pred, gt)
        rewards.append(f1)

    return rewards


# For compatibility with Search-R1's reward function interface
def reward_function(predictions, references, **kwargs):
    """
    Reward function interface compatible with Search-R1.

    Args:
        predictions: List of model predictions (can be strings or dicts with 'text' key)
        references: List of reference answers (can be strings or dicts with 'text' key)

    Returns:
        List of rewards (F1 scores)
    """
    # Handle different input formats
    preds = []
    for pred in predictions:
        if isinstance(pred, dict):
            preds.append(pred.get('text', pred.get('answer', '')))
        else:
            preds.append(str(pred))

    refs = []
    for ref in references:
        if isinstance(ref, dict):
            refs.append(ref.get('text', ref.get('answer', '')))
        else:
            refs.append(str(ref))

    return compute_f1_reward_batch(preds, refs)


if __name__ == "__main__":
    # Test the reward function
    test_cases = [
        ("Barack Obama", "Barack Obama", 1.0),
        ("Obama", "Barack Obama", 0.5),
        ("The answer is Barack Obama", "Barack Obama", 0.4),
        ("Wrong answer", "Barack Obama", 0.0),
    ]

    print("Testing F1 reward function:")
    print("-" * 60)
    for pred, gt, expected_approx in test_cases:
        reward = compute_f1_reward(pred, gt)
        print(f"Pred: '{pred}'")
        print(f"GT:   '{gt}'")
        print(f"F1:   {reward:.3f} (expected ~{expected_approx:.1f})")
        print()
