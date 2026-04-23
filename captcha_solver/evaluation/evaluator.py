"""CAPTCHA-level evaluation: per-tile accuracy and full-grid success rate."""

from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from captcha_solver.config import Config
from captcha_solver.evaluation.metrics import (
    captcha_success_rate,
    compute_confusion_matrix,
    per_category_success_rate,
    threshold_sweep,
    tile_accuracy,
    tile_precision_recall_f1,
)


class CaptchaEvaluator:
    """Evaluates a tile classifier on full CAPTCHA challenges."""

    def __init__(self, model, captcha_dataset, config: Config, device):
        self.model = model.to(device)
        self.dataset = captcha_dataset
        self.config = config
        self.device = device

    @torch.no_grad()
    def evaluate(self, sweep_thresholds: bool = False) -> dict:
        """Run evaluation over all CAPTCHAs in the dataset.

        Returns a dict with all computed metrics.
        """
        self.model.eval()

        all_tile_preds = []      # Predicted class index per tile
        all_tile_labels = []     # True class index per tile
        all_tile_probs = []      # Softmax probabilities per tile
        all_binary_preds = []    # Binary: predicted as target?
        all_binary_truth = []    # Binary: actually target?
        captcha_solved = []      # Per-CAPTCHA: all tiles correct?
        category_results = defaultdict(list)

        for i in tqdm(range(len(self.dataset)), desc="Evaluating CAPTCHAs"):
            sample = self.dataset[i]
            tiles = sample["tiles"].to(self.device)
            ground_truth = sample["ground_truth"]  # Binary: 1=target, 0=not
            target_idx = sample["target_category_idx"]
            target_name = sample["target_category"]

            # Classify each tile
            logits = self.model(tiles)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            predicted = logits.argmax(dim=1).cpu()

            # Multi-class tile predictions
            # We need true class labels for the confusion matrix.
            # The captcha_dataset only provides binary ground truth,
            # so we use the predicted class vs target_idx for binary evaluation.
            all_tile_probs.append(probs)

            # Binary evaluation: does predicted class match target?
            binary_preds = (predicted == target_idx).long()
            binary_truth = ground_truth

            all_binary_preds.extend(binary_preds.tolist())
            all_binary_truth.extend(binary_truth.tolist())

            # CAPTCHA-level: solved if all binary predictions match ground truth
            solved = (binary_preds == binary_truth).all().item()
            captcha_solved.append(solved)
            category_results[target_name].append(solved)

        # Aggregate metrics
        results = {}

        # Binary tile metrics
        results["tile_accuracy"] = tile_accuracy(all_binary_preds, all_binary_truth)
        results["tile_metrics"] = tile_precision_recall_f1(
            np.array(all_binary_preds), np.array(all_binary_truth)
        )

        # CAPTCHA-level success rate
        results["captcha_success_rate"] = captcha_success_rate(captcha_solved)
        results["num_captchas"] = len(captcha_solved)
        results["num_solved"] = sum(captcha_solved)

        # Per-category breakdown
        results["per_category"] = per_category_success_rate(category_results)

        # Grid info
        results["grid_size"] = f"{self.config.grid_rows}x{self.config.grid_cols}"

        # Threshold sweep (optional)
        if sweep_thresholds:
            all_probs = np.vstack(all_tile_probs)
            # For threshold sweep, we need per-target-category probs.
            # Since CAPTCHAs have different targets, we collect probs
            # for each tile at the correct target index.
            target_probs = []
            idx = 0
            for i in range(len(self.dataset)):
                sample = self.dataset[i]
                num_tiles = sample["tiles"].shape[0]
                target_idx = sample["target_category_idx"]
                for j in range(num_tiles):
                    target_probs.append(all_probs[idx + j, target_idx])
                idx += num_tiles
            target_probs = np.array(target_probs)

            # Build array of shape (N, 1) with target-class probability
            # and pass to threshold_sweep with target_idx=0
            prob_array = np.column_stack([target_probs])
            results["threshold_sweep"] = threshold_sweep(
                prob_array,
                np.array(all_binary_truth),
                target_idx=0,
            )

        return results
