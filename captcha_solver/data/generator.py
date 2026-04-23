"""Synthetic CAPTCHA grid generator using CIFAR-10 or custom images."""

import json
import os
import random
from collections import defaultdict

from PIL import Image, ImageEnhance
import numpy as np
import torchvision.datasets as datasets

from captcha_solver.config import Config


class CaptchaGenerator:
    """Generates synthetic image-classification CAPTCHA challenges.

    Each CAPTCHA is a grid of tiles where some match a target category.
    Supports CIFAR-10 or a custom ImageFolder directory as the image source.
    """

    def __init__(self, config: Config):
        self.config = config
        self.raw_dir = os.path.join(config.data_root, "raw")
        self.captcha_dir = os.path.join(config.data_root, "captchas")

    def generate(self):
        """Generate train, val, and test CAPTCHA datasets."""
        if self.config.source_dataset == "custom":
            self._generate_from_custom()
        else:
            self._generate_from_cifar10()

        self.config.save_dataset_info()
        print("Data generation complete.")

    def _generate_from_cifar10(self):
        """Load CIFAR-10 and generate CAPTCHAs."""
        print("Downloading/loading CIFAR-10...")
        cifar_train = datasets.CIFAR10(
            root=self.raw_dir, train=True, download=True
        )
        cifar_test = datasets.CIFAR10(
            root=self.raw_dir, train=False, download=True
        )

        train_by_class = self._index_by_class(cifar_train)
        test_by_class = self._index_by_class(cifar_test)

        # Train/val use CIFAR-10 train split; test uses CIFAR-10 test split
        splits = [
            ("train", self.config.num_train_captchas, train_by_class),
            ("val", self.config.num_val_captchas, train_by_class),
            ("test", self.config.num_test_captchas, test_by_class),
        ]

        for split_name, num_captchas, class_index in splits:
            self._generate_split(split_name, num_captchas, class_index)

    def _generate_from_custom(self):
        """Load a custom ImageFolder directory and generate CAPTCHAs."""
        image_dir = self.config.custom_image_dir
        if not image_dir or not os.path.isdir(image_dir):
            raise FileNotFoundError(
                f"Custom image directory not found: {image_dir}\n"
                f"Expected an ImageFolder structure:\n"
                f"  {image_dir}/\n"
                f"    class_a/\n"
                f"      img1.jpg\n"
                f"      img2.jpg\n"
                f"    class_b/\n"
                f"      img3.jpg\n"
                f"      ..."
            )

        print(f"Loading custom images from {image_dir}...")
        custom_dataset = datasets.ImageFolder(root=image_dir)

        # Update config with discovered classes
        class_names = custom_dataset.classes
        num_classes = len(class_names)
        if num_classes < 2:
            raise ValueError(
                f"Need at least 2 classes, found {num_classes} in {image_dir}"
            )
        self.config.class_names = list(class_names)
        self.config.num_classes = num_classes

        print(f"  Found {num_classes} classes: {', '.join(class_names)}")
        print(f"  Total images: {len(custom_dataset)}")

        # Check minimum images per class
        by_class = self._index_by_class(custom_dataset)
        for cls_idx, cls_name in enumerate(class_names):
            count = len(by_class[cls_idx])
            if count < 5:
                print(f"  Warning: class '{cls_name}' has only {count} images")

        # Split into train (70%), val (15%), test (15%)
        train_by_class, val_by_class, test_by_class = self._split_custom(
            by_class, num_classes, train_ratio=0.7, val_ratio=0.15
        )

        splits = [
            ("train", self.config.num_train_captchas, train_by_class),
            ("val", self.config.num_val_captchas, val_by_class),
            ("test", self.config.num_test_captchas, test_by_class),
        ]

        for split_name, num_captchas, class_index in splits:
            self._generate_split(split_name, num_captchas, class_index)

    def _split_custom(self, by_class, num_classes, train_ratio=0.7, val_ratio=0.15):
        """Split custom images into train/val/test by class, preserving balance."""
        train_by_class = defaultdict(list)
        val_by_class = defaultdict(list)
        test_by_class = defaultdict(list)

        for cls_idx in range(num_classes):
            images = list(by_class[cls_idx])
            random.shuffle(images)
            n = len(images)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))

            train_by_class[cls_idx] = images[:n_train]
            val_by_class[cls_idx] = images[n_train:n_train + n_val]
            test_by_class[cls_idx] = images[n_train + n_val:]

            # If test is empty, duplicate from val
            if not test_by_class[cls_idx]:
                test_by_class[cls_idx] = list(val_by_class[cls_idx])

        return train_by_class, val_by_class, test_by_class

    def _index_by_class(self, dataset):
        """Group dataset images by their class label."""
        by_class = defaultdict(list)
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            by_class[label].append(img)
        return by_class

    def _generate_split(self, split_name, num_captchas, class_index):
        """Generate CAPTCHAs for a single split."""
        split_dir = os.path.join(self.captcha_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        grid_size = self.config.grid_size
        class_names = self.config.class_names
        num_classes = len(class_names)

        print(f"Generating {num_captchas} {split_name} CAPTCHAs...")

        for i in range(num_captchas):
            captcha_id = f"{i:05d}"
            captcha_dir = os.path.join(split_dir, captcha_id)
            os.makedirs(captcha_dir, exist_ok=True)

            # Pick random target category
            target_idx = random.randint(0, num_classes - 1)
            target_name = class_names[target_idx]

            # Determine number of positive tiles (2 to grid_size-2)
            min_pos = 2
            max_pos = grid_size - 2
            num_positive = random.randint(min_pos, max_pos)
            num_negative = grid_size - num_positive

            # Sample positive tiles
            positive_imgs = random.choices(class_index[target_idx], k=num_positive)
            positive_labels = [target_idx] * num_positive

            # Sample negative tiles from diverse non-target classes
            non_target_classes = [c for c in range(num_classes) if c != target_idx]
            negative_imgs = []
            negative_labels = []
            for _ in range(num_negative):
                neg_class = random.choice(non_target_classes)
                neg_img = random.choice(class_index[neg_class])
                negative_imgs.append(neg_img)
                negative_labels.append(neg_class)

            # Combine and shuffle
            all_imgs = positive_imgs + negative_imgs
            all_labels = positive_labels + negative_labels
            indices = list(range(grid_size))
            random.shuffle(indices)

            # Build tile metadata and save tiles
            tiles_meta = []
            for pos, idx in enumerate(indices):
                row = pos // self.config.grid_cols
                col = pos % self.config.grid_cols
                img = all_imgs[idx]
                label_idx = all_labels[idx]

                # Process and save tile
                tile = self._process_tile(img)
                tile_filename = f"tile_{row}_{col}.png"
                tile.save(os.path.join(captcha_dir, tile_filename))

                tiles_meta.append({
                    "row": row,
                    "col": col,
                    "label": class_names[label_idx],
                    "label_idx": label_idx,
                    "is_target": label_idx == target_idx,
                })

            # Save metadata
            metadata = {
                "captcha_id": captcha_id,
                "target_category": target_name,
                "target_category_idx": target_idx,
                "grid_size": [self.config.grid_rows, self.config.grid_cols],
                "tiles": tiles_meta,
            }
            with open(os.path.join(captcha_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

        print(f"  {split_name}: {num_captchas} CAPTCHAs saved to {split_dir}")

    def _process_tile(self, img: Image.Image) -> Image.Image:
        """Upscale and augment a CIFAR-10 image to simulate CAPTCHA tile quality."""
        # Upscale from 32x32 to tile_size x tile_size
        tile = img.resize(
            (self.config.tile_size, self.config.tile_size),
            Image.BILINEAR,
        )

        # Random brightness adjustment (0.8 to 1.2)
        enhancer = ImageEnhance.Brightness(tile)
        tile = enhancer.enhance(random.uniform(0.8, 1.2))

        # Random contrast adjustment (0.8 to 1.2)
        enhancer = ImageEnhance.Contrast(tile)
        tile = enhancer.enhance(random.uniform(0.8, 1.2))

        # Optional JPEG compression artifacts (50% chance)
        if random.random() < 0.5:
            from io import BytesIO
            buffer = BytesIO()
            tile.save(buffer, format="JPEG", quality=random.randint(70, 95))
            buffer.seek(0)
            tile = Image.open(buffer).copy()

        return tile
