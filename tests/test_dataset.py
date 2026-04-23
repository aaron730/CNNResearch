"""Tests for the dataset classes."""

import os

import pytest
import torch

from captcha_solver.config import Config
from captcha_solver.data.captcha_dataset import CaptchaDataset
from captcha_solver.data.generator import CaptchaGenerator
from captcha_solver.data.tile_dataset import TileDataset, get_eval_transforms


@pytest.fixture
def generated_data(tmp_path):
    config = Config(
        data_root=str(tmp_path / "data"),
        num_train_captchas=3,
        num_val_captchas=2,
        num_test_captchas=2,
        grid_rows=3,
        grid_cols=3,
        tile_size=64,
    )
    generator = CaptchaGenerator(config)
    generator.generate()
    return config


def test_tile_dataset_length(generated_data):
    config = generated_data
    captcha_dir = os.path.join(config.data_root, "captchas")
    dataset = TileDataset(captcha_dir, "train", transform=get_eval_transforms(64))
    # 3 CAPTCHAs * 9 tiles = 27 tiles
    assert len(dataset) == 27


def test_tile_dataset_returns_correct_shape(generated_data):
    config = generated_data
    captcha_dir = os.path.join(config.data_root, "captchas")
    dataset = TileDataset(captcha_dir, "train", transform=get_eval_transforms(64))
    image, label = dataset[0]
    assert image.shape == (3, 64, 64)
    assert isinstance(label, int)
    assert 0 <= label < 10


def test_captcha_dataset_length(generated_data):
    config = generated_data
    captcha_dir = os.path.join(config.data_root, "captchas")
    dataset = CaptchaDataset(captcha_dir, "train", tile_size=64)
    assert len(dataset) == 3


def test_captcha_dataset_returns_correct_structure(generated_data):
    config = generated_data
    captcha_dir = os.path.join(config.data_root, "captchas")
    dataset = CaptchaDataset(captcha_dir, "train", tile_size=64)
    sample = dataset[0]

    assert sample["tiles"].shape == (9, 3, 64, 64)
    assert sample["ground_truth"].shape == (9,)
    assert isinstance(sample["target_category"], str)
    assert isinstance(sample["target_category_idx"], int)
    assert isinstance(sample["captcha_id"], str)
