"""Centralized configuration for the CAPTCHA solver."""

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@dataclass
class Config:
    # Data source
    source_dataset: str = "cifar10"
    num_classes: int = 10
    class_names: List[str] = field(default_factory=lambda: list(CIFAR10_CLASSES))
    custom_image_dir: Optional[str] = None  # Path to ImageFolder-structured directory

    # Data generation
    data_root: str = "data"
    tile_size: int = 96
    grid_rows: int = 3
    grid_cols: int = 3
    num_train_captchas: int = 5000
    num_val_captchas: int = 1000
    num_test_captchas: int = 1000

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 20
    weight_decay: float = 1e-4
    pretrained: bool = True
    num_workers: int = 2

    # Evaluation
    confidence_threshold: float = 0.5

    # Paths
    checkpoint_dir: str = "outputs/checkpoints"
    figure_dir: str = "outputs/figures"
    log_dir: str = "outputs/logs"

    @property
    def grid_size(self) -> int:
        return self.grid_rows * self.grid_cols

    def save_dataset_info(self):
        """Save class names and count so downstream commands can load them."""
        info = {
            "source_dataset": self.source_dataset,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "custom_image_dir": self.custom_image_dir,
        }
        path = os.path.join(self.data_root, "dataset_info.json")
        os.makedirs(self.data_root, exist_ok=True)
        with open(path, "w") as f:
            json.dump(info, f, indent=2)

    def load_dataset_info(self):
        """Load class names and count saved during data generation."""
        path = os.path.join(self.data_root, "dataset_info.json")
        if not os.path.exists(path):
            return
        with open(path) as f:
            info = json.load(f)
        self.source_dataset = info["source_dataset"]
        self.num_classes = info["num_classes"]
        self.class_names = info["class_names"]
        self.custom_image_dir = info.get("custom_image_dir")
