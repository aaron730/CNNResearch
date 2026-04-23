"""PyTorch Dataset for full CAPTCHA grid challenges (used for evaluation)."""

import json
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from captcha_solver.data.tile_dataset import IMAGENET_MEAN, IMAGENET_STD


class CaptchaDataset(Dataset):
    """Dataset of full CAPTCHA challenges for evaluation.

    Each item returns a dict with all tiles, target category, and ground truth.
    """

    def __init__(self, captcha_dir: str, split: str, tile_size: int = 96):
        self.tile_size = tile_size
        self.transform = transforms.Compose([
            transforms.Resize((tile_size, tile_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.captchas = []  # List of captcha directory paths

        split_dir = os.path.join(captcha_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        for captcha_id in sorted(os.listdir(split_dir)):
            captcha_path = os.path.join(split_dir, captcha_id)
            meta_path = os.path.join(captcha_path, "metadata.json")
            if os.path.isfile(meta_path):
                self.captchas.append(captcha_path)

    def __len__(self):
        return len(self.captchas)

    def __getitem__(self, idx):
        captcha_path = self.captchas[idx]

        with open(os.path.join(captcha_path, "metadata.json")) as f:
            metadata = json.load(f)

        tiles = []
        ground_truth = []

        for tile_info in metadata["tiles"]:
            tile_filename = f"tile_{tile_info['row']}_{tile_info['col']}.png"
            tile_path = os.path.join(captcha_path, tile_filename)
            img = Image.open(tile_path).convert("RGB")
            tiles.append(self.transform(img))
            ground_truth.append(1 if tile_info["is_target"] else 0)

        return {
            "tiles": torch.stack(tiles),
            "ground_truth": torch.tensor(ground_truth, dtype=torch.long),
            "target_category": metadata["target_category"],
            "target_category_idx": metadata["target_category_idx"],
            "captcha_id": metadata["captcha_id"],
        }
