"""CLI entry point for the CAPTCHA solver research tool."""

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from captcha_solver.config import Config


def cmd_generate_data(args):
    """Generate synthetic CAPTCHA data."""
    from captcha_solver.data.generator import CaptchaGenerator

    config = Config(
        data_root=args.data_dir,
        source_dataset=args.source,
        custom_image_dir=args.image_dir,
        num_train_captchas=args.num_train,
        num_val_captchas=args.num_val,
        num_test_captchas=args.num_test,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        tile_size=args.tile_size,
    )
    generator = CaptchaGenerator(config)
    generator.generate()


def cmd_train(args):
    """Train the tile classifier."""
    from captcha_solver.data.tile_dataset import (
        TileDataset, get_train_transforms, get_eval_transforms,
    )
    from captcha_solver.models.tile_classifier import TileClassifier
    from captcha_solver.models.utils import get_device
    from captcha_solver.training.trainer import Trainer
    from captcha_solver.visualization.plots import plot_training_curves

    config = Config(
        data_root=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        pretrained=args.pretrained,
        tile_size=args.tile_size,
        checkpoint_dir=args.checkpoint_dir,
        figure_dir=args.figure_dir,
    )
    config.load_dataset_info()

    device = get_device(force_cpu=args.device == "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {config.source_dataset} ({config.num_classes} classes)")

    captcha_dir = os.path.join(config.data_root, "captchas")

    print("Loading training data...")
    train_dataset = TileDataset(
        captcha_dir, "train",
        transform=get_train_transforms(config.tile_size),
    )
    val_dataset = TileDataset(
        captcha_dir, "val",
        transform=get_eval_transforms(config.tile_size),
    )

    print(f"  Train tiles: {len(train_dataset)}")
    print(f"  Val tiles: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )

    model = TileClassifier(
        num_classes=config.num_classes, pretrained=config.pretrained,
    )
    print(f"Model: ResNet-18 (pretrained={config.pretrained})")

    trainer = Trainer(model, train_loader, val_loader, config, device)
    history = trainer.train()

    plot_training_curves(history, config.figure_dir)
    print("\nTraining complete.")


def cmd_evaluate(args):
    """Evaluate the model on CAPTCHA challenges."""
    from captcha_solver.data.captcha_dataset import CaptchaDataset
    from captcha_solver.evaluation.evaluator import CaptchaEvaluator
    from captcha_solver.models.tile_classifier import TileClassifier
    from captcha_solver.models.utils import get_device, load_checkpoint
    from captcha_solver.visualization.plots import (
        plot_captcha_success_rate,
        plot_threshold_sweep,
        plot_tile_vs_captcha_accuracy,
    )

    config = Config(
        data_root=args.data_dir,
        tile_size=args.tile_size,
        figure_dir=args.figure_dir,
    )
    config.load_dataset_info()

    device = get_device(force_cpu=args.device == "cpu")
    print(f"Using device: {device}")
    print(f"Dataset: {config.source_dataset} ({config.num_classes} classes)")

    # Load model
    model = TileClassifier(num_classes=config.num_classes, pretrained=False)
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    epoch, metrics = load_checkpoint(checkpoint_path, model)
    print(f"Loaded checkpoint from epoch {epoch} (val_acc={metrics.get('val_acc', 'N/A')})")

    # Load test CAPTCHAs
    captcha_dir = os.path.join(config.data_root, "captchas")
    dataset = CaptchaDataset(captcha_dir, args.split, tile_size=config.tile_size)
    print(f"Evaluating on {len(dataset)} {args.split} CAPTCHAs...")

    evaluator = CaptchaEvaluator(model, dataset, config, device)
    results = evaluator.evaluate(sweep_thresholds=args.sweep_thresholds)

    # Print results
    print(f"\n{'=' * 50}")
    print(f"  EVALUATION RESULTS ({results['grid_size']} grid)")
    print(f"{'=' * 50}")
    print(f"  Per-tile accuracy:       {results['tile_accuracy']:.2%}")
    print(f"  Tile precision:          {results['tile_metrics']['precision']:.2%}")
    print(f"  Tile recall:             {results['tile_metrics']['recall']:.2%}")
    print(f"  Tile F1:                 {results['tile_metrics']['f1']:.2%}")
    print(f"  CAPTCHA success rate:    {results['captcha_success_rate']:.2%}"
          f"  ({results['num_solved']}/{results['num_captchas']})")
    print(f"\n  Per-category success rates:")
    for category, rate in sorted(results["per_category"].items()):
        print(f"    {category:15s}  {rate:.2%}")
    print(f"{'=' * 50}")

    # Generate plots
    plot_captcha_success_rate(results["per_category"], config.figure_dir)

    grid_size = config.grid_rows * config.grid_cols
    plot_tile_vs_captcha_accuracy(
        results["tile_accuracy"],
        results["captcha_success_rate"],
        [9, 16, 25],  # 3x3, 4x4, 5x5 theoretical curves
        config.figure_dir,
    )

    if "threshold_sweep" in results:
        plot_threshold_sweep(results["threshold_sweep"], config.figure_dir)

    print("\nEvaluation complete.")


def cmd_solve(args):
    """Demo: solve a single CAPTCHA and visualize the result."""
    from PIL import Image
    from captcha_solver.data.captcha_dataset import CaptchaDataset
    from captcha_solver.data.tile_dataset import IMAGENET_MEAN, IMAGENET_STD
    from captcha_solver.models.tile_classifier import TileClassifier
    from captcha_solver.models.utils import get_device, load_checkpoint
    from captcha_solver.visualization.plots import plot_captcha_demo

    config = Config(
        data_root=args.data_dir,
        tile_size=args.tile_size,
        figure_dir=args.figure_dir,
    )
    config.load_dataset_info()

    device = get_device(force_cpu=args.device == "cpu")

    # Load model
    model = TileClassifier(num_classes=config.num_classes, pretrained=False)
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    load_checkpoint(checkpoint_path, model)
    model.to(device)
    model.eval()

    # Load dataset
    captcha_dir = os.path.join(config.data_root, "captchas")
    dataset = CaptchaDataset(captcha_dir, args.split, tile_size=config.tile_size)

    # Pick a CAPTCHA
    if args.captcha_id is not None:
        idx = None
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample["captcha_id"] == args.captcha_id:
                idx = i
                break
        if idx is None:
            print(f"Error: CAPTCHA ID {args.captcha_id} not found")
            sys.exit(1)
    else:
        idx = random.randint(0, len(dataset) - 1)

    sample = dataset[idx]
    tiles = sample["tiles"].to(device)
    ground_truth = sample["ground_truth"].numpy()
    target_idx = sample["target_category_idx"]
    target_name = sample["target_category"]
    captcha_id = sample["captcha_id"]

    # Classify
    with torch.no_grad():
        logits = model(tiles)
        predicted_classes = logits.argmax(dim=1).cpu()
        binary_preds = (predicted_classes == target_idx).numpy()

    # Denormalize tiles for visualization
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tiles_cpu = tiles.cpu()
    tile_images = []
    for i in range(tiles_cpu.shape[0]):
        img = tiles_cpu[i] * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        tile_images.append(img)

    solved = (binary_preds == ground_truth).all()
    print(f"\nCAPTCHA {captcha_id}: Select all '{target_name}'")
    print(f"Result: {'SOLVED' if solved else 'FAILED'}")

    num_tiles = config.grid_rows * config.grid_cols
    for i in range(num_tiles):
        row = i // config.grid_cols
        col = i % config.grid_cols
        pred = "YES" if binary_preds[i] else "NO"
        truth = "YES" if ground_truth[i] else "NO"
        correct = "OK" if binary_preds[i] == ground_truth[i] else "WRONG"
        print(f"  Tile [{row},{col}]: Predicted={pred}, Truth={truth} [{correct}]")

    plot_captcha_demo(
        tile_images, binary_preds.tolist(), ground_truth.tolist(),
        target_name, config.figure_dir, captcha_id,
        config.grid_rows, config.grid_cols,
    )


def main():
    parser = argparse.ArgumentParser(
        description="CNN-based CAPTCHA solver research tool",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- generate-data ---
    gen_parser = subparsers.add_parser(
        "generate-data", help="Generate synthetic CAPTCHA data",
    )
    gen_parser.add_argument("--data-dir", default="data", help="Data root directory")
    gen_parser.add_argument("--source", default="cifar10", choices=["cifar10", "custom"],
                            help="Image source: cifar10 or custom ImageFolder directory")
    gen_parser.add_argument("--image-dir", default=None,
                            help="Path to custom image directory (ImageFolder format: "
                                 "image_dir/class_name/image.jpg). Required when --source=custom")
    gen_parser.add_argument("--num-train", type=int, default=5000)
    gen_parser.add_argument("--num-val", type=int, default=1000)
    gen_parser.add_argument("--num-test", type=int, default=1000)
    gen_parser.add_argument("--grid-rows", type=int, default=3)
    gen_parser.add_argument("--grid-cols", type=int, default=3)
    gen_parser.add_argument("--tile-size", type=int, default=96)

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train the tile classifier")
    train_parser.add_argument("--data-dir", default="data")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--tile-size", type=int, default=96)
    train_parser.add_argument("--pretrained", action="store_true", default=True)
    train_parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    train_parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    train_parser.add_argument("--figure-dir", default="outputs/figures")
    train_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    # --- evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on CAPTCHA challenges")
    eval_parser.add_argument("--data-dir", default="data")
    eval_parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    eval_parser.add_argument("--split", default="test", choices=["val", "test"])
    eval_parser.add_argument("--tile-size", type=int, default=96)
    eval_parser.add_argument("--sweep-thresholds", action="store_true")
    eval_parser.add_argument("--figure-dir", default="outputs/figures")
    eval_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    # --- solve ---
    solve_parser = subparsers.add_parser("solve", help="Demo: solve a single CAPTCHA")
    solve_parser.add_argument("--data-dir", default="data")
    solve_parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    solve_parser.add_argument("--split", default="test", choices=["val", "test"])
    solve_parser.add_argument("--captcha-id", default=None)
    solve_parser.add_argument("--tile-size", type=int, default=96)
    solve_parser.add_argument("--figure-dir", default="outputs/figures")
    solve_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")

    args = parser.parse_args()

    commands = {
        "generate-data": cmd_generate_data,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "solve": cmd_solve,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
