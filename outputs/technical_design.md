# Technical Design: How the CNN CAPTCHA Solver Works

A reference describing the solver end-to-end — the model, the data pipeline, the training loop, and the evaluation pipeline — at a level of detail suitable for a methods section. Every claim in this document is grounded in the source code under `captcha_solver/`; file and line references are provided.

---

## 1. System Overview

The solver reduces grid-based image-classification CAPTCHAs ("select all tiles containing X") to an **N-way single-tile classification problem**, solved by a CNN. It then reassembles per-tile predictions into a single CAPTCHA-level "solve / fail" decision. There are four stages:

```
 ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │ 1. Generate  │ →  │ 2. Train     │ →  │ 3. Evaluate  │ →  │ 4. Report    │
 │  synthetic   │    │ ResNet-18 on │    │ on held-out  │    │ tile acc +   │
 │  CAPTCHAs    │    │ single tiles │    │ CAPTCHAs     │    │ CAPTCHA rate │
 └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

**Key design choice: the model never sees a full CAPTCHA grid during training.** It only sees one tile at a time with a multi-class label (e.g., "Bus"). The CAPTCHA-solving logic — "did the model classify every tile correctly relative to the prompt?" — is applied only at evaluation time. This decoupling means the classifier is independent of grid size and prompt phrasing, and one trained model can be evaluated on any grid configuration.

**Source code layout:**

```
captcha_solver/
  cli.py                          # 4 subcommands: generate-data, train, evaluate, solve
  config.py                       # Config dataclass + dataset_info.json persistence
  data/
    generator.py                  # CAPTCHA synthesis (stage 1)
    tile_dataset.py               # Per-tile PyTorch Dataset (stage 2)
    captcha_dataset.py            # Per-grid PyTorch Dataset (stage 3)
  models/
    tile_classifier.py            # ResNet-18 wrapper (stage 2 & 3)
    utils.py                      # Device autodetect, checkpoint I/O
  training/
    trainer.py                    # Training loop, validation, checkpointing (stage 2)
  evaluation/
    evaluator.py                  # CAPTCHA-level evaluation (stage 3)
    metrics.py                    # accuracy / precision / recall / F1 / threshold sweep
  visualization/
    plots.py                      # Training curves, per-category bars, threshold sweep
```

---

## 2. The Model: ResNet-18 with a Swapped Head

**File:** `captcha_solver/models/tile_classifier.py`

```python
class TileClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features   # 512
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
```

### 2.1 Architecture

Input: a batch of RGB tiles, shape `(B, 3, 96, 96)`, where `B` is batch size.

The stock torchvision ResNet-18 pipeline applies:

| Stage        | Layer                                   | Output shape (per sample)     |
|--------------|-----------------------------------------|-------------------------------|
| Stem         | 7×7 conv (stride 2) → BN → ReLU → 3×3 maxpool (stride 2) | `(64, 24, 24)` |
| Stage 1      | 2× BasicBlock (64 ch, stride 1)         | `(64, 24, 24)`                |
| Stage 2      | 2× BasicBlock (128 ch, stride 2)        | `(128, 12, 12)`               |
| Stage 3      | 2× BasicBlock (256 ch, stride 2)        | `(256, 6, 6)`                 |
| Stage 4      | 2× BasicBlock (512 ch, stride 2)        | `(512, 3, 3)`                 |
| Head         | Global average pool → 512-dim vector    | `(512,)`                      |
| Classifier   | **Linear(512 → num_classes)** *(we replace this)* | `(num_classes,)`    |

Each `BasicBlock` is two 3×3 convolutions wrapped in a residual (skip) connection: `y = ReLU(BN(conv3x3(ReLU(BN(conv3x3(x))))) + shortcut(x))`. Skip connections let gradients flow past the convolution stack, which is what makes deep residual networks trainable in practice.

Total learnable parameters: **~11.7 M**. For our 4-class Kaggle run the final layer has `512 × 4 + 4 = 2,052` parameters; the rest come from the backbone.

### 2.2 Output: Logits, Not Probabilities

`forward(x)` returns raw **logits** of shape `(B, num_classes)`. These are *not* probabilities — they are unbounded real numbers. Two places convert them:

- **Training** — `nn.CrossEntropyLoss` applies log-softmax + negative-log-likelihood internally, so it accepts raw logits.
- **Evaluation** — `torch.softmax(logits, dim=1)` in `evaluator.py:54` converts logits to probabilities when we need a confidence score (used only by the threshold sweep).

For the default "pick the highest-scoring class" decision, we take `logits.argmax(dim=1)`; since softmax is monotonic this gives the same answer as taking argmax over probabilities but skips one operation.

### 2.3 Transfer Learning via Head Replacement

When `pretrained=True`, torchvision downloads `ResNet18_Weights.IMAGENET1K_V1` — weights from supervised training on ImageNet (1.2 M images, 1000 classes). All 11.7 M parameters are initialized from those weights.

We then **replace the final layer**: `self.backbone.fc = nn.Linear(in_features, num_classes)` overwrites the pretrained 1000-way head with a freshly-initialized N-way head (e.g., 4 for Kaggle, 10 for CIFAR-10). The replacement layer is initialized with PyTorch's default Kaiming-uniform scheme — *not* pretrained weights.

All parameters — both the pretrained backbone and the new head — remain trainable. We do not freeze the backbone; this is *full fine-tuning*, not *linear probing*. Empirically this gives the best val accuracy when the downstream data (tens of thousands of tiles) is plentiful enough to safely update the backbone.

---

## 3. Data Pipeline

Three distinct representations of the data live at different stages:

1. **Source images** on disk in ImageFolder layout (the Kaggle dataset).
2. **Generated CAPTCHA artifacts** on disk: nested directories of tile PNGs + metadata JSON.
3. **In-memory PyTorch Datasets** that feed the model: `TileDataset` (training) and `CaptchaDataset` (evaluation).

### 3.1 Source → Generated CAPTCHAs

**File:** `captcha_solver/data/generator.py`

#### 3.1.1 Source indexing (`_index_by_class`, lines 136–142)

```python
def _index_by_class(self, dataset):
    by_class = defaultdict(list)
    for idx in range(len(dataset)):
        img, label = dataset[idx]          # loads PIL image from disk
        by_class[label].append(img)        # holds decoded PIL image in RAM
    return by_class
```

This iterates the torchvision `ImageFolder` once, opening and decoding every image into RAM. For the Kaggle dataset (2,179 images, ~60 MB), this takes ~5 minutes on a Mac due to PIL decode overhead but produces a dict keyed by class index with lists of PIL images.

#### 3.1.2 Source-level 70/15/15 split (`_split_custom`, lines 113–134)

Each class's image list is shuffled and partitioned with ratios 0.70 / 0.15 / 0.15 into train / val / test pools **before any CAPTCHA is assembled**. This is the critical step that prevents data leakage — no source image appears in CAPTCHAs across different splits.

#### 3.1.3 CAPTCHA assembly (`_generate_split`, lines 144–222)

For each of the requested `num_captchas` in a split:

1. Sample a target class uniformly at random from the `num_classes` available.
2. Sample **number of positive tiles** uniformly in `[2, grid_size − 2]` (so on a 3×3 grid, between 2 and 7 positives).
3. Sample `num_positive` images **with replacement** from the target class's pool.
4. Sample `num_negative` images by first picking a non-target class uniformly, then picking an image from that class — this spreads negatives across all other classes.
5. Shuffle a flat index list of length `grid_size` to assign the collected images to random tile positions.
6. For each tile, apply `_process_tile` (augmentation — see next section) and save as `tile_{row}_{col}.png`.
7. Write a `metadata.json` recording the target category, grid dimensions, and per-tile `{row, col, label, label_idx, is_target}`.

Because positives are sampled **with replacement** and `num_negative ≥ 2`, on a 3×3 grid the same source image can appear in multiple tiles of the same CAPTCHA — this is a minor source of within-CAPTCHA error correlation.

#### 3.1.4 Tile augmentation during generation (`_process_tile`, lines 224–248)

Applied once per tile at save time, so the augmentation is *baked into the PNG files on disk*:

1. Resize to `tile_size × tile_size` (default 96×96) using bilinear interpolation.
2. Brightness multiplier sampled `U(0.8, 1.2)` via `PIL.ImageEnhance.Brightness`.
3. Contrast multiplier sampled `U(0.8, 1.2)` via `PIL.ImageEnhance.Contrast`.
4. With probability 0.5, round-trip through an in-memory JPEG encode/decode at quality `U(70, 95)` to inject compression artifacts.

This simulates the quality degradation characteristic of real CAPTCHA tiles. Note this augmentation is independent of the *training-time* augmentation described in §3.2.1 — a generated tile has fixed baked-in degradation, and additional random augmentation is applied each training epoch.

#### 3.1.5 On-disk artifacts

```
data/captchas/
  train/00000/metadata.json
         00000/tile_0_0.png … tile_2_2.png
         00001/...
         ...
  val/   ...
  test/  ...
data/dataset_info.json      # { source_dataset, num_classes, class_names, custom_image_dir }
```

`dataset_info.json` is written by `Config.save_dataset_info` (`config.py:52–63`) and read back by `train` and `evaluate` so they see the correct `num_classes` and class names without needing CLI flags.

### 3.2 PyTorch Datasets

#### 3.2.1 `TileDataset` (training)

**File:** `captcha_solver/data/tile_dataset.py`

`__init__` walks the captcha directory tree once, reading every `metadata.json` and appending `(tile_path, label_idx)` tuples to `self.tiles`. For 3,000 train CAPTCHAs that yields 27,000 tuples — all paths, no decoded images.

`__getitem__(idx)`:
1. Open the tile PNG with PIL and convert to RGB.
2. Apply the `transform` pipeline.
3. Return `(image_tensor, label_idx)`.

Two transform pipelines are defined:

**`get_train_transforms`** (applied during training — adds stochastic augmentation on top of the baked-in generation-time augmentation):

```python
Compose([
    Resize((tile_size, tile_size)),
    RandomHorizontalFlip(),                               # 50% left-right flip
    RandomAffine(degrees=5, translate=(0.05, 0.05)),      # ±5° rotation + ±5% shift
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # ±10% each
    ToTensor(),                                           # → (3, H, W), [0,1]
    Normalize(IMAGENET_MEAN, IMAGENET_STD),               # per-channel zero-mean/unit-std
])
```

**`get_eval_transforms`** (applied during validation and test — deterministic):

```python
Compose([
    Resize((tile_size, tile_size)),
    ToTensor(),
    Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

ImageNet normalization uses `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` — the same statistics used when training the pretrained weights. Matching them means the backbone sees inputs in the distribution it was trained on.

#### 3.2.2 `CaptchaDataset` (evaluation)

**File:** `captcha_solver/data/captcha_dataset.py`

One item = one full CAPTCHA grid. `__getitem__` opens all 9 tile PNGs, stacks them into a `(9, 3, 96, 96)` tensor, and returns a dict:

```python
{
    "tiles":              Tensor (grid_size, 3, H, W),  # all tiles, preprocessed
    "ground_truth":       LongTensor (grid_size,),      # 1 if target, 0 otherwise
    "target_category":    str,                          # e.g. "Car"
    "target_category_idx": int,                         # 0..num_classes-1
    "captcha_id":         str,                          # e.g. "00042"
}
```

This lets the evaluator run one CAPTCHA at a time as a batch of 9 tiles, score each tile, and aggregate a per-CAPTCHA solve/fail decision.

---

## 4. Training Loop

**File:** `captcha_solver/training/trainer.py`

### 4.1 Configuration

Set in `captcha_solver/config.py` `Config` dataclass (defaults can be overridden via CLI):

| Hyperparameter  | Default  | Notes                                       |
|-----------------|---------:|---------------------------------------------|
| `batch_size`    | 64       | 64 tiles per gradient step                  |
| `learning_rate` | 1e-3     | Adam initial LR, cosine-decayed to 0        |
| `weight_decay`  | 1e-4     | L2 regularization                           |
| `num_epochs`    | 20       | Full passes over the training tile set      |
| `num_workers`   | 2        | DataLoader prefetch workers                 |

### 4.2 Setup (one-time)

1. **Device autodetect** — `get_device()` in `models/utils.py`: returns `cuda` if available, else `mps` on Apple Silicon, else `cpu`. The Kaggle run used MPS.
2. **Data loading** — two `DataLoader`s wrap the train and val `TileDataset` instances. The train loader uses `shuffle=True`; the val loader does not.
3. **Model** — instantiated as `TileClassifier(num_classes=config.num_classes, pretrained=True)` and moved to the device.
4. **Loss** — `nn.CrossEntropyLoss()`, which combines log-softmax + negative log-likelihood in a numerically stable single op.
5. **Optimizer** — `Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)`. Adam is an adaptive-moment method that tracks per-parameter running means of gradient and squared gradient.
6. **LR schedule** — `CosineAnnealingLR(optimizer, T_max=num_epochs)`. The LR follows `lr(t) = 0.5·lr_init·(1 + cos(π·t/T_max))`, decaying from `lr_init` at epoch 1 to ~0 at epoch `T_max`.

### 4.3 Per-Epoch Training (`_train_epoch`, lines 79–108)

For each batch `(images, labels)` in the train loader:

```python
images = images.to(device)
labels = labels.to(device)

optimizer.zero_grad()              # clear gradients from previous step
outputs = model(images)            # forward pass → logits (B, num_classes)
loss = criterion(outputs, labels)  # cross-entropy loss (scalar)
loss.backward()                    # compute gradients via autograd
optimizer.step()                   # update all parameters
```

After each batch, running totals are updated for loss (weighted by batch size to produce a correct epoch mean at the end) and top-1 accuracy.

### 4.4 Per-Epoch Validation (`_validate`, lines 110–130)

Wrapped in `@torch.no_grad()` — disables autograd bookkeeping so validation is faster and uses less memory. Switches the model to `eval()` mode (disables dropout and uses running BatchNorm statistics instead of batch statistics). Streams through the val loader computing the same loss and accuracy; no weight updates.

### 4.5 Checkpointing (`save_checkpoint` in `models/utils.py`)

Each epoch, if `val_acc > best_val_acc`, we write `outputs/checkpoints/best_model.pt` with:

```python
{
    "epoch":                 current_epoch,
    "model_state_dict":      model.state_dict(),
    "optimizer_state_dict":  optimizer.state_dict(),
    "metrics":               {"val_acc": ..., "val_loss": ...},
}
```

This implements **implicit early stopping by checkpoint selection**: training runs for the full `num_epochs`, but the file on disk always holds the best-val-accuracy weights seen so far. No explicit patience counter — we rely on the fact that reloading the best checkpoint gives us the effect of early stopping after the fact.

### 4.6 LR Step

`self.scheduler.step()` is called **once per epoch**, after both training and validation. The cosine schedule's value at each epoch is what gets logged in the per-epoch LR column.

### 4.7 Training History

The returned `history` dict contains per-epoch lists of train loss, val loss, train acc, val acc, plus `best_epoch` and `best_val_acc`. `visualization/plots.py::plot_training_curves` renders these as `outputs/figures/training_curves.png`.

---

## 5. Evaluation Pipeline

**File:** `captcha_solver/evaluation/evaluator.py`

### 5.1 Setup

1. Load `Config` and call `load_dataset_info()` to read the stored `num_classes` and `class_names`.
2. Instantiate `TileClassifier(num_classes=config.num_classes, pretrained=False)` — the shape must match the checkpoint's final layer; pretraining doesn't matter since we immediately overwrite all weights from the checkpoint.
3. `load_checkpoint(path, model)` restores the saved weights.
4. Wrap the test split with `CaptchaDataset` and iterate.

### 5.2 Per-CAPTCHA Inference Loop (`evaluate`, lines 29–121)

For each of the 500 test CAPTCHAs:

```python
sample = self.dataset[i]               # one CAPTCHA worth of tiles
tiles = sample["tiles"].to(device)     # (grid_size, 3, 96, 96)
target_idx = sample["target_category_idx"]

logits = self.model(tiles)             # (grid_size, num_classes)
probs  = torch.softmax(logits, dim=1)  # kept for threshold sweep
predicted = logits.argmax(dim=1)       # (grid_size,) multi-class prediction
```

The multi-class prediction is then collapsed to a **binary target / not-target** prediction for each tile:

```python
binary_preds = (predicted == target_idx).long()   # 1 if predicted == target, else 0
binary_truth = sample["ground_truth"]             # 1 if actually target, else 0
```

And the CAPTCHA is marked solved if and only if every tile's binary prediction matches ground truth:

```python
solved = (binary_preds == binary_truth).all().item()
```

Across the full test set we accumulate:

- `all_binary_preds`, `all_binary_truth` — flat lists across all tiles for precision/recall/F1.
- `captcha_solved` — per-CAPTCHA booleans for the overall success rate.
- `category_results[target_name]` — per-category booleans for the per-category breakdown.
- `all_tile_probs` — per-tile softmax probabilities, used only if `--sweep-thresholds`.

### 5.3 Metrics Computation (`evaluation/metrics.py`)

- **`tile_accuracy`** — calls `sklearn.metrics.accuracy_score`: fraction of tile binary predictions equal to ground truth.
- **`tile_precision_recall_f1`** — `sklearn.metrics.precision_recall_fscore_support(average="binary")`. Precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = 2PR/(P+R).
- **`captcha_success_rate`** — `sum(captcha_solved) / len(captcha_solved)`.
- **`per_category_success_rate`** — per-category `sum(hits) / count` across the 500 CAPTCHAs.

### 5.4 Threshold Sweep (`evaluation/metrics.py::threshold_sweep`)

Invoked by `--sweep-thresholds`. For each threshold `t ∈ {0.1, 0.2, ..., 0.9}`:

```python
binary_preds = (prob_of_target_class >= t).astype(int)
```

and recompute precision / recall / F1. This answers "how does performance change if we move off the argmax decision rule and instead require the target class probability to exceed `t`?" The resulting curve is rendered as `outputs/figures/threshold_sweep.png` and lets you pick an operating point biased toward precision or recall.

The code path in `evaluator.py` threads the per-tile softmax probability at the correct target index through a flattened array, because each CAPTCHA has a different target class.

### 5.5 Why Evaluation Iterates One CAPTCHA at a Time

A larger batch would be faster, but per-CAPTCHA iteration gives us three things cleanly:

1. Different CAPTCHAs have different target classes, and we need to reduce the multi-class prediction against the *correct* target per CAPTCHA.
2. The solve/fail decision is per-CAPTCHA — a batch reduction would just re-introduce the same bookkeeping.
3. On MPS/GPU, the 9-tile forward pass is small enough that kernel launch overhead already dominates, so batching across CAPTCHAs yields modest speedup.

For 500 CAPTCHAs the full evaluation takes ~11 seconds.

---

## 6. Reducing a Multi-Class Model to a Binary CAPTCHA Decision

Worth calling out explicitly because it is the conceptual crux of the system:

```
multi-class prediction  →  binary per-tile decision  →  per-CAPTCHA solve/fail
   (N-way argmax)            (== target_class?)          (all tiles correct?)
```

- **Why N-way classification and not binary?** A binary classifier ("target or not?") would need to be retrained for each possible target class. An N-way classifier is trained once on all classes, and at inference time the prompt's target class index selects which output column to compare against. A single model can be evaluated against any prompt.
- **Why argmax and not a threshold?** The default decision rule is "predicted class = argmax of logits → is that class the target?" This is equivalent to putting the threshold at the probability value that would make the target class the argmax, which is usually tighter than a fixed 0.5 probability threshold. The `--sweep-thresholds` option exists to explore the fixed-threshold alternative and make the precision/recall tradeoff explicit.

---

## 7. What the Classifier Is Actually Learning

The network is trained with cross-entropy over N class labels, treating each tile independently. Two consequences worth stating plainly:

1. **There is no structured prediction.** The model has no explicit knowledge of the grid layout, no message passing between tiles, no conditional random field over the 9 predictions. Each tile is classified in isolation, and the only interaction between tiles happens in the evaluator's `.all()` reduction.
2. **There is no prompt conditioning.** The model does not see the target category. It outputs the same N-dim vector regardless of what the CAPTCHA asks for. The prompt only enters at the binary-reduction step.

These simplifications are deliberate — they make the experiment clean and the result a pure measurement of **per-tile classification quality under realistic-looking tile degradation**, independent of prompt engineering or structured-prediction overhead. A future extension could explicitly condition the model on the target class (e.g., by concatenating a class embedding to the feature vector before the final layer), which might improve recall on hard classes.

---

## 8. End-to-End Call Graph (Kaggle Run)

To make the file/function interactions concrete, here is the actual call graph for one complete run:

```
captcha-solver generate-data --source custom --image-dir …
  └─ cli.cmd_generate_data
     └─ Config(…)
     └─ CaptchaGenerator.generate                       (data/generator.py)
        └─ _generate_from_custom
           ├─ datasets.ImageFolder(root=image_dir)      (torchvision)
           ├─ _index_by_class                           (decode all PIL images)
           ├─ _split_custom                             (per-class 70/15/15)
           └─ _generate_split × {train, val, test}
              └─ for each CAPTCHA:
                 _process_tile → PNG write; metadata.json write
        └─ Config.save_dataset_info                     (writes dataset_info.json)

captcha-solver train --epochs 15 --pretrained
  └─ cli.cmd_train
     └─ Config.load_dataset_info                        (reads dataset_info.json)
     └─ TileDataset(train) / TileDataset(val)           (data/tile_dataset.py)
     └─ DataLoader × 2
     └─ TileClassifier(num_classes=4, pretrained=True)  (models/tile_classifier.py)
     └─ Trainer.train                                   (training/trainer.py)
        └─ for epoch in 1..15:
           ├─ _train_epoch  ← forward, CE loss, backward, Adam step, tqdm
           ├─ _validate     ← forward in eval(), no_grad
           ├─ scheduler.step()                           (cosine LR)
           └─ if best: save_checkpoint                  (models/utils.py)
        └─ plot_training_curves                         (visualization/plots.py)

captcha-solver evaluate --split test --sweep-thresholds
  └─ cli.cmd_evaluate
     └─ Config.load_dataset_info
     └─ TileClassifier(num_classes=4, pretrained=False)
     └─ load_checkpoint                                 (models/utils.py)
     └─ CaptchaDataset(test)                            (data/captcha_dataset.py)
     └─ CaptchaEvaluator.evaluate                       (evaluation/evaluator.py)
        └─ for each CAPTCHA: forward → argmax → binary reduction → per-captcha solved?
        └─ tile_accuracy / precision_recall_f1 / captcha_success_rate
          / per_category_success_rate / threshold_sweep         (evaluation/metrics.py)
     └─ plot_captcha_success_rate / plot_tile_vs_captcha_accuracy
       / plot_threshold_sweep                           (visualization/plots.py)
```

---

## 9. Design Decisions and Their Tradeoffs

A short inventory for the paper's "Methodology" / "Design choices" section:

| Decision                                         | Alternative                                    | Why we picked it                                                                 |
|--------------------------------------------------|------------------------------------------------|----------------------------------------------------------------------------------|
| ResNet-18 backbone                               | ViT / ResNet-50 / EfficientNet                  | Small, well-understood, strong ImageNet baseline; fits our small dataset         |
| Full fine-tuning                                 | Linear probing (frozen backbone)               | Tile distribution differs from ImageNet enough that updating the backbone helps  |
| N-way classifier + binary reduction              | Binary target-present classifier per class     | One model trains faster and generalizes across prompts                           |
| CAPTCHA generation baked to PNG                  | Augment in-memory each epoch                   | Reproducibility — same CAPTCHAs in every training run                            |
| Augmentation split across generation + training  | All at generation / all at training            | Generation-time simulates CAPTCHA quality; training-time simulates view variance |
| Cross-entropy + Adam + cosine                    | SGD + momentum + step decay                    | Standard, fast, and robust defaults for fine-tuning                              |
| Checkpoint on best val accuracy                  | Explicit early stopping with patience           | Simpler; same effect in practice                                                 |
| Per-CAPTCHA evaluation loop (no batching)        | Batched across CAPTCHAs                         | Clean per-CAPTCHA solve/fail accounting; 500 CAPTCHAs takes ~11 seconds anyway   |
| Confidence threshold = argmax (i.e., ~implicit)  | Fixed 0.5 threshold                             | Argmax is the natural decision rule for a multi-class model; sweep available via flag |
