# CNN CAPTCHA Solver Research Tool

A research tool for measuring the effectiveness of Convolutional Neural Networks (CNNs) at solving image-classification CAPTCHAs -- the type that presents a grid of images and asks users to "select all images containing X."

## Table of Contents

- [Methodology](#methodology)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Using Custom Images](#using-custom-images)
- [Using a Real reCAPTCHA Dataset](#using-a-real-recaptcha-dataset)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)
- [Understanding the Results](#understanding-the-results)

## Methodology

### Problem Formulation

Image-classification CAPTCHAs present a grid of image tiles (typically 3x3 or 4x4) alongside a prompt such as "Select all images containing traffic lights." The user must correctly identify every tile that matches the target category. A CAPTCHA is only considered solved if **all tiles** are classified correctly -- a single mistake means failure.

This tool reframes the problem as a multi-class image classification task: a CNN classifies each tile independently into one of N categories, then compares its prediction against the target category to decide "match" or "no match."

### CNN Architecture

The classifier uses **ResNet-18** as a backbone, a well-established convolutional architecture with 11.7 million parameters organized into residual blocks with skip connections. The final fully-connected layer is replaced to output N classes (matching the number of categories in the dataset).

**Transfer learning** is applied by initializing the network with weights pretrained on ImageNet (1.2 million images across 1,000 categories). This provides strong general-purpose feature extraction that transfers well to smaller datasets, enabling the model to converge faster and achieve higher accuracy than training from scratch.

### Synthetic Data Generation

Since real CAPTCHA images cannot be programmatically obtained at scale with ground truth labels, this tool generates synthetic CAPTCHA challenges:

1. **Source images** are drawn from CIFAR-10 (60,000 images across 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) or from a custom image directory you provide.

2. **CAPTCHA assembly**: For each challenge, a target category is randomly selected. Tiles are sampled such that 2 to (grid_size - 2) tiles match the target, with the remainder drawn from other categories. This constraint mirrors real CAPTCHAs, which always contain a mix of matching and non-matching tiles.

3. **Tile augmentation**: Source images are resized to 96x96 pixels and augmented with random brightness/contrast variation and optional JPEG compression artifacts to simulate the degraded quality typical of real CAPTCHA tiles.

4. **Data leakage prevention**: When using CIFAR-10, the train/validation CAPTCHAs draw from CIFAR-10's training split, while test CAPTCHAs draw exclusively from CIFAR-10's test split. When using custom images, the source images are split 70/15/15 into train/val/test before CAPTCHA generation.

### Training Pipeline

- **Optimizer**: Adam with learning rate 1e-3 and weight decay 1e-4
- **Loss function**: Cross-entropy loss (standard multi-class classification)
- **Learning rate schedule**: Cosine annealing over the total number of epochs
- **Checkpointing**: The model with the best validation accuracy is saved automatically

The model is trained as an N-class classifier on individual tiles. It does not see full CAPTCHA grids during training -- only individual tiles with their category labels. The CAPTCHA-solving logic (comparing the predicted class against the target prompt) is applied at evaluation time.

### Evaluation Metrics

The tool reports two levels of accuracy:

- **Per-tile accuracy**: What fraction of individual tiles are classified correctly. This is the standard image classification metric.

- **CAPTCHA success rate**: What fraction of full CAPTCHAs are solved perfectly (every tile correct). This is the metric that matters in practice.

The relationship between these two metrics is exponential. For a 3x3 grid (9 tiles), if per-tile accuracy is `p`, the expected CAPTCHA success rate is approximately `p^9`. For example:

| Per-Tile Accuracy | 3x3 Success Rate | 4x4 Success Rate |
|-------------------|-------------------|-------------------|
| 90%               | 39%               | 19%               |
| 93%               | 52%               | 30%               |
| 95%               | 63%               | 44%               |
| 98%               | 83%               | 72%               |

This exponential drop-off is the core security property of grid-based image CAPTCHAs: even a highly accurate classifier fails the majority of challenges when the grid is large enough.

Additional metrics include per-category success rates (some categories are harder to classify than others), precision/recall/F1 for binary target detection, and an optional confidence threshold sweep.

## Installation

Requires Python 3.8+ and pip.

```bash
cd CNNResearch
pip install -e .
```

This installs the `captcha-solver` command and all dependencies (PyTorch, torchvision, matplotlib, scikit-learn, etc.).

## Quick Start

Run a complete experiment with default settings:

```bash
# Step 1: Generate synthetic CAPTCHAs from CIFAR-10
captcha-solver generate-data

# Step 2: Train the ResNet-18 classifier (20 epochs)
captcha-solver train

# Step 3: Evaluate on test CAPTCHAs
captcha-solver evaluate

# Step 4: Visualize a single CAPTCHA solve attempt
captcha-solver solve --random
```

On a Mac with Apple Silicon, training takes approximately 13 minutes. On CPU, expect roughly 40 minutes.

## Usage

### generate-data

Downloads the source dataset (if needed) and creates synthetic CAPTCHA challenges with known ground truth labels.

```bash
captcha-solver generate-data [OPTIONS]
```

| Flag           | Default   | Description                                  |
|----------------|-----------|----------------------------------------------|
| `--source`     | `cifar10` | Image source: `cifar10` or `custom`          |
| `--image-dir`  | None      | Path to custom image folder (required when `--source=custom`) |
| `--num-train`  | 5000      | Number of training CAPTCHAs to generate      |
| `--num-val`    | 1000      | Number of validation CAPTCHAs                |
| `--num-test`   | 1000      | Number of test CAPTCHAs                      |
| `--grid-rows`  | 3         | Number of rows in the CAPTCHA grid           |
| `--grid-cols`  | 3         | Number of columns in the CAPTCHA grid        |
| `--tile-size`  | 96        | Tile resolution in pixels (width and height) |
| `--data-dir`   | `data`    | Output directory for generated data          |

### train

Trains the ResNet-18 tile classifier on the generated tile images.

```bash
captcha-solver train [OPTIONS]
```

| Flag              | Default                    | Description                          |
|-------------------|----------------------------|--------------------------------------|
| `--epochs`        | 20                         | Number of training epochs            |
| `--batch-size`    | 64                         | Batch size for training              |
| `--lr`            | 0.001                      | Learning rate                        |
| `--pretrained`    | True                       | Use ImageNet-pretrained weights      |
| `--no-pretrained` | --                         | Train from scratch (no pretraining)  |
| `--checkpoint-dir`| `outputs/checkpoints`      | Where to save model checkpoints      |
| `--figure-dir`    | `outputs/figures`          | Where to save training curve plots   |
| `--device`        | `auto`                     | Device: `auto`, `cpu`, `cuda`, `mps` |
| `--data-dir`      | `data`                     | Data directory from generate-data    |

Outputs:
- `outputs/checkpoints/best_model.pt` -- best model by validation accuracy
- `outputs/figures/training_curves.png` -- loss and accuracy over epochs

### evaluate

Evaluates the trained model on full CAPTCHA challenges, reporting both per-tile accuracy and CAPTCHA-level success rate.

```bash
captcha-solver evaluate [OPTIONS]
```

| Flag                 | Default                              | Description                        |
|----------------------|--------------------------------------|------------------------------------|
| `--checkpoint`       | `outputs/checkpoints/best_model.pt`  | Path to model checkpoint           |
| `--split`            | `test`                               | Evaluation split: `val` or `test`  |
| `--sweep-thresholds` | False                                | Run precision/recall threshold sweep |
| `--figure-dir`       | `outputs/figures`                    | Where to save evaluation plots     |
| `--device`           | `auto`                               | Device: `auto`, `cpu`, `cuda`, `mps` |
| `--data-dir`         | `data`                               | Data directory from generate-data  |

Outputs:
- Printed metrics table (per-tile accuracy, CAPTCHA success rate, per-category breakdown)
- `outputs/figures/success_rate_by_category.png` -- bar chart by target category
- `outputs/figures/tile_vs_captcha_accuracy.png` -- theoretical vs actual success curves
- `outputs/figures/threshold_sweep.png` -- precision/recall/F1 across thresholds (if `--sweep-thresholds`)

### solve

Demo mode. Picks a single CAPTCHA, runs inference, and shows the model's tile-by-tile predictions compared to ground truth.

```bash
captcha-solver solve [OPTIONS]
```

| Flag            | Default                              | Description                      |
|-----------------|--------------------------------------|----------------------------------|
| `--checkpoint`  | `outputs/checkpoints/best_model.pt`  | Path to model checkpoint         |
| `--captcha-id`  | None                                 | Specific CAPTCHA ID to solve     |
| `--split`       | `test`                               | Which split to pick from         |
| `--device`      | `auto`                               | Device: `auto`, `cpu`, `cuda`, `mps` |
| `--data-dir`    | `data`                               | Data directory from generate-data |

If `--captcha-id` is not provided, a random CAPTCHA is selected. Outputs a visualization to `outputs/figures/captcha_demo_<id>.png` with green borders on correct predictions and red borders on incorrect ones.

## Using Custom Images

You can supply your own images instead of CIFAR-10. Organize them in an ImageFolder structure where each subfolder name is a class label:

```
my_images/
    traffic_lights/
        img001.jpg
        img002.jpg
        ...
    crosswalks/
        img003.jpg
        img004.jpg
        ...
    buses/
        img005.jpg
        img006.jpg
        ...
```

Then generate, train, and evaluate:

```bash
captcha-solver generate-data --source custom --image-dir ./my_images
captcha-solver train --epochs 20
captcha-solver evaluate
```

The tool automatically discovers class names from subfolder names, splits the images into train/val/test (70/15/15), and adjusts the model's output layer to match the number of classes. Any image format supported by PIL works (JPEG, PNG, BMP, WebP, etc.).

Requirements for custom datasets:
- At least 2 classes (subfolder names become category labels)
- At least 5 images per class recommended (more is better)
- Images can be any resolution (they are resized to `--tile-size` during generation)

## Using a Real reCAPTCHA Dataset

A helper script is provided to pull the [Google reCAPTCHA V2 image dataset](https://www.kaggle.com/datasets/mikhailma/test-dataset) from Kaggle and reshape it into the ImageFolder layout the tool expects. The dataset contains real reCAPTCHA tiles across 12 categories (bicycle, bridge, bus, car, chimney, crosswalk, hydrant, motorcycle, other, palm, stair, traffic_light).

One-time setup (requires a free Kaggle account):

```bash
pip install kaggle
# Kaggle -> Account -> Settings -> Create New API Token
# Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Then download and prepare:

```bash
python scripts/prepare_recaptcha.py --output-dir data/recaptcha_images
captcha-solver generate-data --source custom --image-dir data/recaptcha_images
captcha-solver train
captcha-solver evaluate
```

The script downloads to `data/recaptcha_raw/`, then flattens the class folders into `data/recaptcha_images/` (merging any duplicate class names across splits and dropping classes with fewer than 5 images).

## Configuration Reference

All defaults can be overridden via CLI flags. There is no configuration file -- the tool is designed to be driven entirely from the command line for reproducibility.

Key defaults:

| Parameter       | Value  | Rationale                                                    |
|-----------------|--------|--------------------------------------------------------------|
| Grid size       | 3x3    | Standard CAPTCHA grid; increase for harder challenges        |
| Tile size       | 96x96  | Balances ResNet-18 effectiveness with memory usage           |
| Learning rate   | 1e-3   | Standard for Adam with pretrained ResNet fine-tuning         |
| Epochs          | 20     | Sufficient for convergence with cosine annealing             |
| Pretrained      | True   | ImageNet pretraining significantly boosts accuracy           |

## Project Structure

```
CNNResearch/
    captcha_solver/
        cli.py                       # CLI entry point (4 subcommands)
        config.py                    # Config dataclass and dataset info persistence
        data/
            generator.py             # CAPTCHA generation from CIFAR-10 or custom images
            tile_dataset.py          # PyTorch Dataset for individual tiles (training)
            captcha_dataset.py       # PyTorch Dataset for full CAPTCHA grids (evaluation)
        models/
            tile_classifier.py       # ResNet-18 tile classifier
            utils.py                 # Device detection, checkpoint save/load
        training/
            trainer.py               # Training loop with validation and checkpointing
        evaluation/
            evaluator.py             # CAPTCHA-level evaluation logic
            metrics.py               # Metric computation (accuracy, success rate, F1)
        visualization/
            plots.py                 # Training curves, confusion matrix, bar charts
    data/                            # Generated at runtime (not committed)
    outputs/                         # Model checkpoints and figures
    tests/                           # Unit tests
    setup.py                         # Package installation
    requirements.txt                 # Python dependencies
```

## Understanding the Results

After running `captcha-solver evaluate`, you will see output like:

```
==================================================
  EVALUATION RESULTS (3x3 grid)
==================================================
  Per-tile accuracy:       93.2%
  Tile precision:          87.5%
  Tile recall:             91.3%
  Tile F1:                 89.4%
  CAPTCHA success rate:    54.1%  (541/1000)

  Per-category success rates:
    airplane         62.3%
    automobile       48.7%
    bird             58.1%
    ...
==================================================
```

- **Per-tile accuracy** reflects raw classification performance on individual tiles.
- **CAPTCHA success rate** is the practically relevant metric -- it tells you what percentage of CAPTCHAs the model can fully solve. Note how 93% tile accuracy translates to only ~54% CAPTCHA success on a 3x3 grid.
- **Per-category breakdown** reveals which categories are hardest. Visually similar categories (e.g., automobile vs truck) tend to have lower success rates because misclassification between them causes failures.
- The **tile vs CAPTCHA accuracy chart** (`outputs/figures/tile_vs_captcha_accuracy.png`) plots the theoretical exponential curves for different grid sizes alongside the actual measured data point, visualizing the security margin that grid-based CAPTCHAs provide.
