"""Tests for evaluation metrics."""

import numpy as np
import pytest

from captcha_solver.evaluation.metrics import (
    captcha_success_rate,
    per_category_success_rate,
    tile_accuracy,
    tile_precision_recall_f1,
)


def test_tile_accuracy_perfect():
    preds = [0, 1, 2, 3, 4]
    truth = [0, 1, 2, 3, 4]
    assert tile_accuracy(preds, truth) == 1.0


def test_tile_accuracy_partial():
    preds = [0, 1, 2, 3, 4]
    truth = [0, 1, 2, 0, 0]
    assert tile_accuracy(preds, truth) == 0.6


def test_captcha_success_rate_all_solved():
    results = [True, True, True]
    assert captcha_success_rate(results) == 1.0


def test_captcha_success_rate_none_solved():
    results = [False, False, False]
    assert captcha_success_rate(results) == 0.0


def test_captcha_success_rate_partial():
    results = [True, False, True, False]
    assert captcha_success_rate(results) == 0.5


def test_captcha_success_rate_empty():
    assert captcha_success_rate([]) == 0.0


def test_per_category_success_rate():
    category_results = {
        "cat": [True, True, False],
        "dog": [False, False],
        "bird": [True, True, True, True],
    }
    rates = per_category_success_rate(category_results)
    assert abs(rates["cat"] - 2 / 3) < 1e-9
    assert rates["dog"] == 0.0
    assert rates["bird"] == 1.0


def test_tile_precision_recall_f1():
    preds = np.array([1, 1, 0, 0, 1])
    truth = np.array([1, 0, 0, 1, 1])
    metrics = tile_precision_recall_f1(preds, truth)
    # TP=2, FP=1, FN=1, TN=1
    assert abs(metrics["precision"] - 2 / 3) < 1e-9
    assert abs(metrics["recall"] - 2 / 3) < 1e-9
