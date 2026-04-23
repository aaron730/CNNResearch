"""Download the mikhailma Google reCAPTCHA V2 dataset from Kaggle and reshape
it into the ImageFolder layout that `captcha-solver generate-data --source
custom` expects.

Kaggle dataset: https://www.kaggle.com/datasets/mikhailma/test-dataset

Usage:
    python scripts/prepare_recaptcha.py --output-dir data/recaptcha_images

Requirements:
    pip install kaggle
    # Place your Kaggle API token at ~/.kaggle/kaggle.json
    # (Account -> Settings -> Create New API Token on kaggle.com)

After running, feed the output directory to the tool:
    captcha-solver generate-data --source custom --image-dir data/recaptcha_images
    captcha-solver train
    captcha-solver evaluate
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "mikhailma/test-dataset"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def run_kaggle_download(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
        DATASET_SLUG,
        "-p",
        str(dest),
        "--unzip",
    ]
    print(f"$ {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit(
            "Kaggle CLI not found. Install with: pip install kaggle\n"
            "Then place your API token at ~/.kaggle/kaggle.json."
        )
    except subprocess.CalledProcessError as e:
        # Fall back: try zip download and extract manually
        print(f"Kaggle --unzip failed ({e}); attempting manual zip extract.")
        zip_cmd = cmd[:-1]  # drop --unzip
        subprocess.run(zip_cmd, check=True)
        for zip_path in dest.glob("*.zip"):
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(dest)
            zip_path.unlink()


def find_class_dirs(root: Path) -> list[Path]:
    """Find directories whose immediate children are mostly image files.

    The Kaggle dataset nests class folders a few levels deep (the exact
    structure varies by upload), so we walk and pick leaf-ish folders that
    hold images directly.
    """
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        children = list(path.iterdir())
        if not children:
            continue
        image_children = [
            c for c in children if c.is_file() and c.suffix.lower() in IMAGE_EXTS
        ]
        if len(image_children) >= 5 and len(image_children) >= len(children) // 2:
            candidates.append(path)
    return candidates


def consolidate(raw_dir: Path, output_dir: Path) -> dict[str, int]:
    """Copy class folders into a flat ImageFolder layout under output_dir."""
    class_dirs = find_class_dirs(raw_dir)
    if not class_dirs:
        sys.exit(
            f"No class folders found under {raw_dir}. "
            "The Kaggle archive layout may have changed; inspect it manually."
        )

    # If multiple candidates share the same class name (e.g. 'bus' appears in
    # train/ and test/ splits), merge them.
    grouped: dict[str, list[Path]] = {}
    for d in class_dirs:
        grouped.setdefault(d.name.lower(), []).append(d)

    counts: dict[str, int] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name, sources in grouped.items():
        target = output_dir / class_name
        target.mkdir(parents=True, exist_ok=True)
        n = 0
        for src in sources:
            for img in src.iterdir():
                if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
                    continue
                dest_name = f"{src.parent.name}_{img.name}" if (target / img.name).exists() else img.name
                shutil.copy2(img, target / dest_name)
                n += 1
        counts[class_name] = n
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/recaptcha_images"),
        help="Final ImageFolder directory to feed to captcha-solver.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/recaptcha_raw"),
        help="Scratch directory for the Kaggle download.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse an existing raw-dir instead of re-downloading.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=5,
        help="Drop classes with fewer than this many images (default 5).",
    )
    args = parser.parse_args()

    if not args.skip_download:
        run_kaggle_download(args.raw_dir)
    elif not args.raw_dir.exists():
        sys.exit(f"--skip-download given but {args.raw_dir} does not exist.")

    counts = consolidate(args.raw_dir, args.output_dir)

    dropped = [c for c, n in counts.items() if n < args.min_images]
    for c in dropped:
        shutil.rmtree(args.output_dir / c)
        del counts[c]

    print("\nFinal class counts:")
    for cls in sorted(counts):
        print(f"  {cls:<20} {counts[cls]}")
    print(f"\nTotal classes: {len(counts)}")
    print(f"Total images:  {sum(counts.values())}")
    if dropped:
        print(f"Dropped (< {args.min_images} images): {', '.join(dropped)}")
    print(f"\nReady. Next:")
    print(
        f"  captcha-solver generate-data --source custom --image-dir {args.output_dir}"
    )
    print("  captcha-solver train")
    print("  captcha-solver evaluate")


if __name__ == "__main__":
    main()
