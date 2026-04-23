"""Metric computation functions for CAPTCHA solver evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def tile_accuracy(predictions, ground_truth):
    """Compute per-tile classification accuracy."""
    return accuracy_score(ground_truth, predictions)


def tile_precision_recall_f1(binary_preds, binary_truth):
    """Compute binary precision, recall, F1 for target detection."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        binary_truth, binary_preds, average="binary", zero_division=0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_confusion_matrix(predictions, ground_truth, num_classes):
    """Compute multi-class confusion matrix."""
    return confusion_matrix(ground_truth, predictions, labels=list(range(num_classes)))


def captcha_success_rate(captcha_results):
    """Compute the fraction of CAPTCHAs where ALL tiles were classified correctly.

    Args:
        captcha_results: List of bools, True if all tiles in that CAPTCHA
                         were correctly classified.

    Returns:
        Success rate as a float.
    """
    if not captcha_results:
        return 0.0
    return sum(captcha_results) / len(captcha_results)


def per_category_success_rate(category_results):
    """Compute success rate broken down by target category.

    Args:
        category_results: Dict mapping category_name -> list of bools.

    Returns:
        Dict mapping category_name -> success_rate float.
    """
    rates = {}
    for category, results in category_results.items():
        rates[category] = sum(results) / len(results) if results else 0.0
    return rates


def threshold_sweep(tile_probs, tile_targets, target_idx, thresholds=None):
    """Sweep confidence thresholds for binary target detection.

    Args:
        tile_probs: Array of shape (N, num_classes) with softmax probabilities.
        tile_targets: Array of shape (N,) with binary ground truth (1=target).
        target_idx: The class index that is the target category.
        thresholds: List of thresholds to evaluate. Defaults to 0.1-0.9.

    Returns:
        List of dicts with threshold, precision, recall, f1 for each threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1).tolist()

    results = []
    for t in thresholds:
        binary_preds = (tile_probs[:, target_idx] >= t).astype(int)
        metrics = tile_precision_recall_f1(binary_preds, tile_targets)
        metrics["threshold"] = t
        results.append(metrics)
    return results
