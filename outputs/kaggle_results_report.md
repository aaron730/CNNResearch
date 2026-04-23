# Kaggle reCAPTCHA v2 — CNN Solver Results

**Date:** 2026-04-16
**Dataset:** Google_Recaptcha_V2_Images_Dataset (Kaggle) — 2,179 raw images across 4 classes
**Model:** ResNet-18 (ImageNet-pretrained), 4-class tile head
**Hardware:** Apple MPS

## Dataset

| Class   | Raw images |
|---------|-----------:|
| Bicycle |        730 |
| Bridge  |        533 |
| Bus     |        236 |
| Car     |        680 |
| **Total** |  **2,179** |

Generated synthetic 3×3 CAPTCHAs by sampling tiles from a 70/15/15 per-class image split:
- Train: 3,000 CAPTCHAs (27,000 tiles)
- Val:   500 CAPTCHAs (4,500 tiles)
- Test:  500 CAPTCHAs (4,500 tiles)

Tile size 96×96, 2–7 positive tiles per CAPTCHA, brightness/contrast jitter + occasional JPEG compression.

## Training

15 epochs, batch 64, lr 1e-3 cosine-decayed, weight decay 1e-4, ResNet-18 pretrained.

| Epoch | Train acc | Val acc | Val loss |
|------:|----------:|--------:|---------:|
| 1     | 93.46%    | **89.09%** | 0.363 |
| 2     | 97.41%    | 80.80%  | 0.752 |
| 3     | 98.28%    | 86.42%  | 0.534 |
| 4     | 98.54%    | 88.58%  | 0.402 |
| 5     | 98.98%    | 84.02%  | 0.678 |
| 6     | 98.99%    | 87.29%  | 0.582 |
| 7     | 99.32%    | 87.20%  | 0.522 |
| 8     | 99.49%    | 85.84%  | 0.578 |
| 9     | 99.56%    | 85.71%  | 0.628 |
| 10    | 99.69%    | 86.16%  | 0.632 |
| 11    | 99.71%    | 87.07%  | 0.614 |
| 12    | 99.81%    | 88.64%  | 0.510 |
| 13    | 99.85%    | 88.13%  | 0.558 |
| 14    | 99.86%    | 88.49%  | 0.549 |
| 15    | 99.87%    | 88.40%  | 0.554 |

**Best checkpoint: epoch 1 (val acc 89.09%).**

Train saturates near 100% while val plateaus in the 85–89% band — a clear overfitting signature. The model learns the class discrimination almost immediately (1 epoch) and subsequent epochs add no generalization.

## Test results (500 CAPTCHAs, best checkpoint)

| Metric                   | Value |
|--------------------------|------:|
| Per-tile accuracy        | 93.04% |
| Tile precision           | 96.28% |
| Tile recall              | 89.41% |
| Tile F1                  | 92.72% |
| **Full-CAPTCHA success rate** | **55.80%** (279 / 500) |

### Per-category success rate

| Category | Success rate |
|----------|-------------:|
| Bicycle  | 74.80% |
| Bus      | 59.68% |
| Bridge   | 58.12% |
| Car      | 33.09% |

**Car** is the weakest class by a wide margin — roughly half the success rate of Bicycle despite having the second-most training images. This is likely where the tile recall gap is concentrated; worth inspecting misclassified Car tiles to see whether they're being confused with Bus (plausible) or Bridge (a signal the model latches onto background).

## Takeaways

- **Per-tile accuracy 93%, CAPTCHA success 56%** — the tile→CAPTCHA gap is the 9-tile compounding effect: 0.93^9 ≈ 0.52, so the solver is essentially hitting the theoretical ceiling of its per-tile accuracy.
- Lifting CAPTCHA success rate requires lifting **per-tile accuracy**, not post-processing. Promising levers:
  1. Early stopping — training past epoch 1 is actively hurting. Try 3–5 epochs with a lower LR or stronger regularization.
  2. Stronger augmentation — random crops, horizontal flips, color jitter, cutout. The current augmentation (brightness, contrast, JPEG) is mild.
  3. Class rebalancing for Bus (only 236 raw images) and targeted Car error analysis.
- Precision (96%) >> recall (89%) — the model is conservative. A lower decision threshold would trade some precision for recall and may improve captcha success if recall is the bottleneck; the threshold sweep figure shows this tradeoff.

## Artifacts

- Checkpoint: `outputs/checkpoints/best_model.pt`
- Figures: `outputs/figures/`
  - `training_curves.png` — train/val loss and accuracy over epochs
  - `success_rate_by_category.png` — per-category success bars
  - `tile_vs_captcha_accuracy.png` — tile accuracy vs. observed full-CAPTCHA success, with theoretical 3×3/4×4/5×5 curves
  - `threshold_sweep.png` — precision/recall/F1 vs. confidence threshold
  - `captcha_demo_00003.png` — qualitative solved example
- Logs: `/tmp/captcha_gen.log`, `/tmp/captcha_train.log`, `/tmp/captcha_eval.log`