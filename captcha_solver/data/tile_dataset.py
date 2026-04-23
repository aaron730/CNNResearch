"""PyTorch Dataset for individual CAPTCHA tiles (used for training)."""

import json
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ImageNet normalization (used with pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(tile_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((tile_size, tile_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transforms(tile_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((tile_size, tile_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class TileDataset(Dataset):
    """Dataset of individual tiles extracted from generated CAPTCHAs.

    Each item returns (image_tensor, class_label_int) for multi-class training.
    """

    def __init__(self, captcha_dir: str, split: str, transform=None):
        self.transform = transform
        self.tiles = []  # List of (tile_path, label_idx)

        split_dir = os.path.join(captcha_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Walk all CAPTCHA directories and collect tiles
        for captcha_id in sorted(os.listdir(split_dir)):
            captcha_path = os.path.join(split_dir, captcha_id)
            meta_path = os.path.join(captcha_path, "metadata.json")
            if not os.path.isfile(meta_path):
                continue

            with open(meta_path) as f:
                metadata = json.load(f)

            for tile_info in metadata["tiles"]:
                tile_filename = f"tile_{tile_info['row']}_{tile_info['col']}.png"
                tile_path = os.path.join(captcha_path, tile_filename)
                self.tiles.append((tile_path, tile_info["label_idx"]))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile_path, label_idx = self.tiles[idx]
        image = Image.open(tile_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label_idx
