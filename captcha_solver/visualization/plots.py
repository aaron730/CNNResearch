"""Visualization functions for training curves, confusion matrices, and results."""

import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_curves(history: dict, save_dir: str):
    """Plot training and validation loss/accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curves
    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curves to {path}")


def plot_confusion_matrix(cm, class_names: list, save_dir: str):
    """Plot a confusion matrix heatmap."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Tile Classification Confusion Matrix")

    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {path}")


def plot_captcha_success_rate(per_category_rates: dict, save_dir: str):
    """Plot bar chart of CAPTCHA success rate per target category."""
    os.makedirs(save_dir, exist_ok=True)

    categories = sorted(per_category_rates.keys())
    rates = [per_category_rates[c] for c in categories]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, rates, color=sns.color_palette("muted", len(categories)))
    ax.set_xlabel("Target Category")
    ax.set_ylabel("CAPTCHA Success Rate")
    ax.set_title("CAPTCHA Success Rate by Target Category")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{rate:.1%}", ha="center", va="bottom", fontsize=9,
        )

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(save_dir, "success_rate_by_category.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved success rate chart to {path}")


def plot_tile_vs_captcha_accuracy(tile_acc: float, captcha_rate: float,
                                   grid_sizes: list, save_dir: str):
    """Plot theoretical vs actual tile-accuracy-to-CAPTCHA-success-rate curve."""
    os.makedirs(save_dir, exist_ok=True)

    tile_accs = np.linspace(0.5, 1.0, 100)
    fig, ax = plt.subplots(figsize=(10, 6))

    for gs in grid_sizes:
        theoretical = tile_accs ** gs
        ax.plot(tile_accs, theoretical, label=f"{int(gs**0.5)}x{int(gs**0.5)} grid (n={gs})", alpha=0.7)

    # Mark actual measured point
    actual_grid = grid_sizes[0] if grid_sizes else 9
    ax.scatter([tile_acc], [captcha_rate], color="red", s=100, zorder=5,
               label=f"Actual ({tile_acc:.1%} tile acc -> {captcha_rate:.1%} success)")

    ax.set_xlabel("Per-Tile Accuracy")
    ax.set_ylabel("CAPTCHA Success Rate")
    ax.set_title("Per-Tile Accuracy vs CAPTCHA Success Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "tile_vs_captcha_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved tile vs CAPTCHA accuracy chart to {path}")


def plot_threshold_sweep(sweep_results: list, save_dir: str):
    """Plot precision, recall, and F1 across confidence thresholds."""
    os.makedirs(save_dir, exist_ok=True)

    thresholds = [r["threshold"] for r in sweep_results]
    precisions = [r["precision"] for r in sweep_results]
    recalls = [r["recall"] for r in sweep_results]
    f1s = [r["f1"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, precisions, "o-", label="Precision")
    ax.plot(thresholds, recalls, "s-", label="Recall")
    ax.plot(thresholds, f1s, "^-", label="F1")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep: Precision / Recall / F1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(save_dir, "threshold_sweep.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved threshold sweep chart to {path}")


def plot_captcha_demo(tiles_images, predictions, ground_truth,
                      target_category: str, save_dir: str, captcha_id: str,
                      grid_rows: int = 3, grid_cols: int = 3):
    """Visualize a single CAPTCHA with model predictions overlaid.

    Green border = correct prediction, Red border = incorrect prediction.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(8, 8))
    fig.suptitle(f'Target: "Select all {target_category}"', fontsize=14)

    for idx in range(grid_rows * grid_cols):
        row = idx // grid_cols
        col = idx % grid_cols
        ax = axes[row][col] if grid_rows > 1 else axes[col]

        img = tiles_images[idx]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        pred = predictions[idx]
        truth = ground_truth[idx]
        is_correct = pred == truth

        # Green for correct, red for incorrect
        color = "green" if is_correct else "red"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)

        # Label
        pred_label = "YES" if pred else "NO"
        truth_label = "YES" if truth else "NO"
        ax.set_title(f"P:{pred_label} T:{truth_label}", fontsize=9,
                      color=color, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, f"captcha_demo_{captcha_id}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved CAPTCHA demo visualization to {path}")
