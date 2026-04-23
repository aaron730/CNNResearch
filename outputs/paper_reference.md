# Paper Reference: Terms and Results

A reference document for writing the research paper. Section 1 defines every term. Section 2 walks through every number produced by the Kaggle reCAPTCHA v2 experiment and explains what it means and how to interpret it.

---

## 1. Glossary of Terms

### 1.1 Dataset and Task Terms

**CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart).** A challenge designed to distinguish automated scripts from human users. The specific type studied here is an *image-classification CAPTCHA*: a grid of image tiles with a prompt such as "Select all images containing Car."

**Image-classification CAPTCHA / grid CAPTCHA.** The specific CAPTCHA variant that presents N×M image tiles and requires the solver to mark every tile containing the target object. Historically the most common type deployed by Google reCAPTCHA v2.

**Tile.** One image cell in the CAPTCHA grid. In this experiment each tile is a 96×96 pixel RGB image.

**Target category.** The class named in the CAPTCHA prompt. For each generated challenge the target is sampled uniformly at random from the available classes.

**Positive tile / negative tile.** A positive tile belongs to the target category (the user should click it). A negative tile does not. Real CAPTCHAs always mix both; our generator enforces this with a constraint of 2 ≤ positives ≤ grid_size − 2, i.e. 2–7 positive tiles in a 3×3 grid.

**Grid size.** The number of tiles in one CAPTCHA (rows × columns). 3×3 = 9 tiles in this experiment. Grid size directly controls difficulty — see §1.5.

**Synthetic CAPTCHA.** A CAPTCHA we generate ourselves by sampling tiles from a labeled image dataset. Using synthetic CAPTCHAs lets us (a) produce unlimited training and test data and (b) know the ground truth for every tile, which is impossible for real CAPTCHAs served by live systems.

**Source dataset.** The pool of labeled images the generator draws from. In this run: the Kaggle *Google reCAPTCHA V2 Images Dataset* (2,179 real reCAPTCHA tiles across 4 categories — Bicycle, Bridge, Bus, Car).

**ImageFolder.** The PyTorch/torchvision convention for labeled image datasets, where each subdirectory name is a class label and the images inside belong to that class.

**Train / validation / test split.** Standard ML partitioning.
- *Train* — images the model's weights are updated on.
- *Validation* — held-out images the model never trains on, used to pick the best epoch and tune hyperparameters.
- *Test* — further held-out images used only for the final, reported performance number.
We use a 70/15/15 per-class split of the source images, then generate CAPTCHAs independently from each split so tiles in a test CAPTCHA never appeared in any training CAPTCHA.

**Data leakage.** The failure mode where information about the test set contaminates training (e.g., same image in both splits), which would make test accuracy unreliable. Our generator prevents this by splitting at the source-image level before CAPTCHA assembly.

### 1.2 Model and Training Terms

**CNN (Convolutional Neural Network).** A neural-network architecture class that uses convolution layers — weight-sharing filters that slide across the image — which makes them efficient and effective for visual tasks.

**ResNet-18.** An 18-layer CNN introduced by He et al. (2015) that uses *residual (skip) connections* around every pair of convolutional layers. The skip connection lets gradients flow directly through the network, which made it possible to reliably train much deeper CNNs. ResNet-18 has roughly 11.7 million parameters and is standard as a strong, small backbone.

**Residual block / skip connection.** A small subnetwork whose output is `F(x) + x` rather than just `F(x)`. Enables deep networks to be trained without the vanishing-gradient problem.

**Backbone.** The shared feature-extraction body of the model (all but the final classification head). We keep the ImageNet backbone and replace only the final fully-connected layer.

**Fully-connected layer / classification head.** The final linear layer that maps the backbone's feature vector (512-dimensional in ResNet-18) to class scores. We replace the ImageNet 1000-class head with a 4-class head for Kaggle (Bicycle / Bridge / Bus / Car).

**Transfer learning.** Initializing a model's weights from training on a different (usually larger) task and then fine-tuning on the task of interest. We initialize from ImageNet-pretrained weights, so the network starts with strong general-purpose visual features.

**ImageNet.** A 1.2-million-image, 1,000-class dataset that is the standard pretraining corpus for vision models. The pretrained weights encode general visual features (edges, textures, object parts) that transfer broadly.

**Fine-tuning.** Continuing to update all weights of a pretrained model on the new task, as opposed to freezing the backbone and training only the head.

**Parameters / weights.** The learnable floating-point numbers that define the model's behavior. ResNet-18 has ~11.7 M of them.

**Epoch.** One complete pass through the training data. We run 15 epochs.

**Batch / batch size.** The number of training examples the model processes per weight update. We use batch size 64 (each batch is 64 tiles).

**Iteration / step.** One batch forward-pass + backward-pass + weight update. Training 27,000 tiles at batch 64 = 422 iterations per epoch.

**Optimizer.** The algorithm that updates weights using gradients. We use *Adam*, an adaptive-learning-rate method that works well as a default.

**Learning rate (LR).** The step size for weight updates. We start at 1e-3 (0.001).

**Weight decay.** An L2 regularization term (1e-4 in our runs) that penalizes large weights to discourage overfitting.

**Cosine annealing / cosine learning-rate schedule.** The learning rate is smoothly decayed from its starting value to near zero over the 15 epochs following the first half of a cosine curve. This gives large early steps for coarse learning and tiny late steps for fine-tuning.

**Loss function / cross-entropy loss.** The scalar quantity the optimizer minimizes. For multi-class classification, *cross-entropy* compares the model's predicted class probabilities against the true one-hot label — it is large when the model is confidently wrong and small when it is confidently right.

**Training loss / validation loss.** Loss averaged over the train split (updated during training) vs. the val split (evaluated after each epoch with no weight updates).

**Training accuracy / validation accuracy.** Fraction of tiles whose top-scoring class matches the ground-truth label, measured on train and val respectively.

**Checkpoint.** A saved snapshot of model weights (and optimizer state). We save the checkpoint with the best validation accuracy across all epochs and use it for final test evaluation.

**Early stopping.** The practice of stopping training (or selecting the checkpoint from) the epoch with best validation performance, rather than the last epoch. We select by best val accuracy; in this run that was epoch 1.

**Overfitting.** When a model memorizes the training set rather than learning generalizable features. The diagnostic signature is training accuracy continuing to improve while validation accuracy plateaus or degrades. Our training run shows this clearly (§2.3).

**Generalization gap.** The difference between training-set performance and held-out performance. Large gap ⇒ overfitting.

**Augmentation.** Random perturbations applied to training images (brightness, contrast, JPEG compression in our pipeline) to synthetically expand the dataset and make the model robust to distribution shift.

**Device (CPU / CUDA / MPS).** Where the model runs.
- *CPU* — the general processor.
- *CUDA* — NVIDIA GPUs.
- *MPS* — Apple Silicon's Metal Performance Shaders GPU backend. We used MPS for this experiment.

### 1.3 Evaluation Metrics

**Ground truth.** The known-correct labels. For synthetic CAPTCHAs we record ground truth for every tile during generation.

**Per-tile accuracy.** The fraction of individual tiles whose predicted class matches the true class. This is the standard single-image classification metric, computed across all 4,500 tiles in the test split (500 CAPTCHAs × 9 tiles).

**Full-CAPTCHA success rate.** The fraction of CAPTCHAs where *every* tile's binary prediction (target vs not-target) matches ground truth. A single wrong tile fails the CAPTCHA. This is the metric that matters operationally.

**Binary prediction (match / no-match).** Derived from the multiclass prediction by comparing the predicted class index to the target-category index. The model predicts a class; the CAPTCHA logic turns that into "is this the target? yes/no."

**Confusion matrix.** A table of true vs. predicted classes; diagonal entries are correct, off-diagonal are errors. Not explicitly plotted here but implicit in the per-category breakdown.

**Precision.** Of the tiles the model flagged as "target," what fraction were actually the target.
`precision = true_positives / (true_positives + false_positives)`.

**Recall (sensitivity).** Of the tiles that actually are the target, what fraction did the model flag.
`recall = true_positives / (true_positives + false_negatives)`.

**F1 score.** The harmonic mean of precision and recall: `F1 = 2·P·R / (P+R)`. A balanced single-number summary when both matter.

**True / false positive, true / false negative.** Standard confusion-matrix cells for the binary "target vs not-target" reduction.

**Confidence threshold.** The minimum predicted probability for the target class required to flag a tile positive. Default 0.5. Lowering the threshold raises recall and lowers precision.

**Threshold sweep.** Plotting precision, recall, and F1 across all possible thresholds, revealing the tradeoff curve. Used to pick an operating point that favors the metric that matters most for the deployment.

**Per-category success rate.** The CAPTCHA success rate conditioned on the target category. Surfaces which categories the model handles well vs. poorly. Computed by bucketing the 500 test CAPTCHAs by their target category and reporting success within each bucket.

**Theoretical success curve (p^n).** The ideal success rate if tile errors were independent and identically distributed: `success = (per-tile accuracy)^(grid size)`. For 3×3, that's p⁹. Real numbers usually track this closely; large deviations indicate that tile errors are correlated (e.g., systematically harder CAPTCHAs).

### 1.4 Implementation / Tooling Terms

**PyTorch.** The deep-learning framework we use.

**torchvision.** The companion library for image datasets, transforms, and prebuilt model architectures (ResNet-18 lives here).

**DataLoader.** The PyTorch utility that batches a Dataset, shuffles, and parallelizes data loading. We use batch size 64 with 2 worker processes.

**`pin_memory`.** A DataLoader flag that pins RAM pages so GPU transfers are faster. It is a no-op on Apple MPS — we saw the corresponding warning in the training log.

**Dataset (PyTorch).** An object that returns a single example on demand. We have two: `TileDataset` (individual tiles, used during training) and `CaptchaDataset` (full CAPTCHA grids, used for evaluation).

**ImageNet normalization (mean/std).** Standard per-channel normalization applied to match the pretrained weights' input distribution: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.

**Pretrained weights.** Specifically the `ResNet18_Weights.IMAGENET1K_V1` weights shipped with torchvision.

### 1.5 Why the Grid Compounds Errors (Key Security Argument)

If individual tile errors were independent with probability `(1 − p)`, the probability of getting *all* `n` tiles right is `p^n`. This exponential decay is the core defense of grid CAPTCHAs:

| Per-tile accuracy | 3×3 (n=9) | 4×4 (n=16) | 5×5 (n=25) |
|------------------:|----------:|-----------:|-----------:|
| 90%               |   38.7%   |   18.5%    |    7.2%    |
| 93%               |   52.0%   |   31.1%    |   16.3%    |
| 95%               |   63.0%   |   44.0%    |   27.7%    |
| 98%               |   83.4%   |   72.4%    |   60.3%    |
| 99%               |   91.4%   |   85.1%    |   77.8%    |

Holding tile accuracy fixed, enlarging the grid is a cheap way for CAPTCHA designers to buy exponentially more security. Our result sits almost exactly on the 93% row at 3×3 (see §2.4).

---

## 2. Results Walkthrough

All numbers below come from the Kaggle-data run on 2026-04-16 and are reproduced verbatim from the generation / training / evaluation logs.

### 2.1 Dataset Statistics

**Source:** Google reCAPTCHA V2 Images Dataset (Kaggle, user `mikhailma`).

| Class   | Raw images | % of source |
|---------|-----------:|------------:|
| Bicycle |        730 |      33.5%  |
| Bridge  |        533 |      24.5%  |
| Bus     |        236 |      10.8%  |
| Car     |        680 |      31.2%  |
| **Total** |  **2,179** |   100%     |

Observations to cite:
- The dataset is **imbalanced**; Bus has roughly one-third as many images as Bicycle. This affects training diversity for Bus and should be mentioned as a limitation.
- 2,179 images is small by modern CV standards — ImageNet pretraining is doing most of the feature-learning work.

**Generated CAPTCHAs:**

| Split | CAPTCHAs | Tiles | Source-image pool |
|-------|---------:|------:|-------------------|
| Train | 3,000    | 27,000 | 70% of each class |
| Val   | 500      | 4,500  | 15% of each class |
| Test  | 500      | 4,500  | 15% of each class |

Grid: 3×3, tile size 96×96. Positive tile count per CAPTCHA uniform in [2, 7]. Augmentation on each tile: brightness ×U(0.8, 1.2), contrast ×U(0.8, 1.2), JPEG compression at quality U(70, 95) with 50% probability.

### 2.2 Training Configuration

| Hyperparameter      | Value               |
|---------------------|---------------------|
| Architecture        | ResNet-18           |
| Initialization      | ImageNet-pretrained |
| Output classes      | 4                   |
| Epochs              | 15                  |
| Batch size          | 64                  |
| Optimizer           | Adam                |
| Learning rate       | 1e-3 (cosine to 0)  |
| Weight decay        | 1e-4                |
| Loss                | Cross-entropy       |
| Device              | Apple MPS           |
| Train tiles         | 27,000              |
| Val tiles           | 4,500               |
| Iterations / epoch  | 422                 |

### 2.3 Training Dynamics (per epoch)

| Epoch | Train loss | Train acc | Val loss | Val acc | LR     |
|------:|-----------:|----------:|---------:|--------:|-------:|
| 1     | 0.1908     | 93.46%    | 0.3625   | **89.09%** (best) | 0.000989 |
| 2     | 0.0789     | 97.41%    | 0.7516   | 80.80%  | 0.000957 |
| 3     | 0.0531     | 98.28%    | 0.5338   | 86.42%  | 0.000905 |
| 4     | 0.0456     | 98.54%    | 0.4022   | 88.58%  | 0.000835 |
| 5     | 0.0325     | 98.98%    | 0.6785   | 84.02%  | 0.000750 |
| 6     | 0.0284     | 98.99%    | 0.5819   | 87.29%  | 0.000655 |
| 7     | 0.0210     | 99.32%    | 0.5222   | 87.20%  | 0.000552 |
| 8     | 0.0163     | 99.49%    | 0.5781   | 85.84%  | 0.000448 |
| 9     | 0.0137     | 99.56%    | 0.6283   | 85.71%  | 0.000345 |
| 10    | 0.0086     | 99.69%    | 0.6320   | 86.16%  | 0.000250 |
| 11    | 0.0071     | 99.71%    | 0.6144   | 87.07%  | 0.000165 |
| 12    | 0.0045     | 99.81%    | 0.5103   | 88.64%  | 0.000095 |
| 13    | 0.0031     | 99.85%    | 0.5584   | 88.13%  | 0.000043 |
| 14    | 0.0028     | 99.86%    | 0.5493   | 88.49%  | 0.000011 |
| 15    | 0.0026     | 99.87%    | 0.5545   | 88.40%  | 0.000000 |

**How to read this table / discussion points:**

- *Epoch 1 already matches best val acc.* Because ResNet-18 is ImageNet-pretrained, the 4-class reCAPTCHA task is effectively solved in one epoch. This is the expected behavior for transfer learning on a small, coarse-grained task.
- *Train vs val divergence from epoch 2 onward is overfitting.* Training loss keeps falling (0.19 → 0.003, ~73×) and train accuracy approaches 100%, while val accuracy oscillates in the 85–89% band. The **generalization gap** widens from ~4 points at epoch 1 to ~11 points by epoch 15.
- *Val loss increases while val accuracy stays roughly flat.* This is a signal that the model is becoming more *confidently* wrong on the val mistakes — predictions push harder into one class, so when they're wrong the cross-entropy penalty is larger. Accuracy is a coarser metric and so doesn't swing as much.
- *Best-checkpoint selection correctly picks epoch 1.* Training longer did not help. For the paper, this motivates either (a) fewer epochs, (b) stronger regularization, or (c) more aggressive augmentation.
- *The epoch 2 dip to 80.8% val accuracy* is most likely the optimizer briefly diverging after the initial large gradient updates; by epoch 3 it recovers. Worth noting but not load-bearing for conclusions.

### 2.4 Test Set Results

Checkpoint used: **epoch 1** (val acc 89.09%). Evaluated on 500 held-out test CAPTCHAs (4,500 tiles).

**Aggregate metrics:**

| Metric                          | Value              | Interpretation |
|---------------------------------|-------------------:|----------------|
| Per-tile accuracy               | **93.04%**         | 4,187 of 4,500 tiles classified correctly. The raw image-classification performance. |
| Tile precision                  | 96.28%             | When the model says "target," it is right 96% of the time. |
| Tile recall                     | 89.41%             | The model catches 89% of the true target tiles. |
| Tile F1                         | 92.72%             | Balanced summary of precision and recall. |
| **Full-CAPTCHA success rate**   | **55.80%** (279/500) | The model fully solves 56% of CAPTCHAs. |

**Sanity check against the theoretical curve.** With tile accuracy p = 0.9304 and n = 9, the independent-errors prediction is p⁹ = 0.5226 (52.3%). The measured success rate is 55.8% — **3.5 points above** the theoretical line. Tile errors are therefore slightly *positively correlated within a CAPTCHA* (a CAPTCHA is either easy-for-the-model or hard-for-the-model, rather than every tile being independent), but the effect is small. The p⁹ model is an excellent first-order fit.

**Precision-recall asymmetry (96.28% vs 89.41%).** The model is *conservative*: when it says "target," it is usually right, but it misses ~11% of real targets. Because CAPTCHA success requires *every* tile to be right, a missed target (false negative) is equally fatal as a wrongly-flagged non-target (false positive). However, these errors are not equal in quantity — the 6.28% / 10.59% asymmetry says recall failures dominate the error budget. Lowering the confidence threshold below 0.5 would trade some precision for recall; the threshold sweep (`outputs/figures/threshold_sweep.png`) visualizes the whole curve.

**Per-category success:**

| Category | Success | Relative to overall |
|----------|--------:|--------------------:|
| Bicycle  | 74.80%  | +19.0 |
| Bus      | 59.68%  |  +3.9 |
| Bridge   | 58.12%  |  +2.3 |
| Car      | 33.09%  | −22.7 |
| Overall  | 55.80%  |    —  |

**How to read the per-category table:**
- *Bicycle is the easiest class by a large margin.* Bicycles have distinctive thin-line silhouettes against varied backgrounds — visually separable from the other three classes. It also has the most training images (730), though Car has nearly as many (680), so raw count doesn't fully explain the gap.
- *Car is the hardest by a wide margin (33%, ~half the overall rate).* Car is the most visually confusable class here: in a dataset that also contains Bus and Bicycle, a "car" tile can share framing (street, parking, urban background) with Bus tiles and framing (small wheeled vehicle) with Bicycle. Confusion matrix analysis (future work) would pinpoint whether Car→Bus is the dominant mistake.
- *Bridge and Bus perform similarly* despite Bus having only 236 source images. Suggests that the Bus class is not data-starved for this level of discrimination — probably because Bus tiles are visually quite distinctive (large vehicle, windows, front-facing view).
- **Effective per-tile accuracy by class.** Given p_class such that p_class⁹ ≈ observed_success:
  - Bicycle: (0.748)^(1/9) ≈ **96.8%** tile acc
  - Bus:     (0.597)^(1/9) ≈ **94.4%** tile acc
  - Bridge:  (0.581)^(1/9) ≈ **94.1%** tile acc
  - Car:     (0.331)^(1/9) ≈ **88.4%** tile acc
  Car's tile accuracy is ~8 points below Bicycle — that small-looking gap amplifies to a 42-point gap at the CAPTCHA level. **This is the headline example of the grid's exponential amplification working as a defense against an imperfect classifier.**

### 2.5 Figures (and how to describe them in the paper)

All in `outputs/figures/`:

- **`training_curves.png`** — four curves (train loss, val loss, train acc, val acc) vs. epoch. Use this to show the overfitting signature: train loss/acc and val loss/acc separating after epoch 1.
- **`success_rate_by_category.png`** — bar chart of the four per-category success rates. Use to highlight the Car weakness.
- **`tile_vs_captcha_accuracy.png`** — plots the theoretical p^n curves for n = 9, 16, 25 and overlays our measured (tile, CAPTCHA) point. Use to visually ground the "grid size = security" argument and show how close the measured point is to the 3×3 curve.
- **`threshold_sweep.png`** — precision, recall, F1 vs. confidence threshold. Use to argue about the precision/recall operating point and motivate tuning the threshold for deployment realism.
- **`captcha_demo_00003.png`** — one example CAPTCHA visualized with model predictions vs. ground truth. Useful as a qualitative Figure 1.

### 2.6 Summary Statement for the Abstract / Conclusion

> A ResNet-18 classifier initialized from ImageNet weights and fine-tuned on 3,000 synthetic 3×3 CAPTCHAs generated from the Google reCAPTCHA V2 image dataset (4 classes, 2,179 real reCAPTCHA tiles) achieves 93.04% per-tile accuracy and a 55.80% full-CAPTCHA success rate on a held-out test set of 500 CAPTCHAs. The full-CAPTCHA rate tracks the p⁹ independent-errors prediction to within 3.5 percentage points, empirically confirming the exponential amplification of tile errors as the grid's core security property. Per-category success rates range from 74.8% (Bicycle) to 33.1% (Car), and an ~8 point per-tile accuracy gap between classes produces a ~42 point full-CAPTCHA gap — the same amplification that protects users from imperfect classifiers also produces disproportionate variance across categories.

### 2.7 Threats to Validity / Limitations to Acknowledge

- **Only 4 classes.** Real reCAPTCHA v2 uses 12+ categories, reducing chance-level and increasing visual-confusion opportunities. A 12-class run is the natural next experiment.
- **Small source-image pool (2,179).** Training tiles are resampled with replacement, so the same underlying image appears in many CAPTCHAs. This bounds achievable generalization.
- **Class imbalance (Bus: 236).** Minority-class representation may cap how well we can evaluate that class.
- **Synthetic CAPTCHAs, not live CAPTCHAs.** Our 3×3 assembly mimics the grid but does not reproduce deployment-time adversarial conditions (dynamic tile replacement, behavioral signals, rate limits).
- **Overfitting after epoch 1.** Our chosen checkpoint is empirically correct, but the training recipe has headroom — stronger augmentation and/or fewer epochs would likely improve validation accuracy by ~1–3 points.
- **Single random seed.** Results in this run are from one training seed; reporting mean ± std over 3+ seeds is standard for a paper.

---

## 3. Reproducibility Commands

Exact commands to regenerate every number in §2:

```bash
# Data (3000/500/500 captchas from Kaggle reCAPTCHA v2, 4 classes)
python3 -m captcha_solver.cli generate-data \
  --source custom \
  --image-dir data/recaptcha_raw/Google_Recaptcha_V2_Images_Dataset/images \
  --num-train 3000 --num-val 500 --num-test 500 \
  --grid-rows 3 --grid-cols 3 --tile-size 96

# Training (15 epochs, ImageNet-pretrained ResNet-18)
python3 -m captcha_solver.cli train \
  --epochs 15 --batch-size 64 --lr 1e-3 --tile-size 96 --pretrained

# Evaluation on the test split, with threshold sweep
python3 -m captcha_solver.cli evaluate \
  --split test --sweep-thresholds
```

Logs:
- `/tmp/captcha_gen.log` — generation
- `/tmp/captcha_train.log` — training (per-epoch train/val loss and accuracy)
- `/tmp/captcha_eval.log` — evaluation
