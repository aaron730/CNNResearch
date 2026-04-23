"""Tests for the CAPTCHA generator."""

import json
import os
import tempfile

import pytest

from captcha_solver.config import Config
from captcha_solver.data.generator import CaptchaGenerator


@pytest.fixture
def config(tmp_path):
    return Config(
        data_root=str(tmp_path / "data"),
        num_train_captchas=5,
        num_val_captchas=2,
        num_test_captchas=2,
        grid_rows=3,
        grid_cols=3,
        tile_size=64,
    )


def test_generate_creates_correct_structure(config):
    generator = CaptchaGenerator(config)
    generator.generate()

    captcha_dir = os.path.join(config.data_root, "captchas")
    for split, expected_count in [("train", 5), ("val", 2), ("test", 2)]:
        split_dir = os.path.join(captcha_dir, split)
        assert os.path.isdir(split_dir)
        captcha_ids = [d for d in os.listdir(split_dir)
                       if os.path.isdir(os.path.join(split_dir, d))]
        assert len(captcha_ids) == expected_count


def test_captcha_metadata_is_valid(config):
    generator = CaptchaGenerator(config)
    generator.generate()

    captcha_dir = os.path.join(config.data_root, "captchas", "train", "00000")
    meta_path = os.path.join(captcha_dir, "metadata.json")
    assert os.path.isfile(meta_path)

    with open(meta_path) as f:
        metadata = json.load(f)

    assert metadata["captcha_id"] == "00000"
    assert metadata["target_category"] in config.class_names
    assert metadata["grid_size"] == [3, 3]
    assert len(metadata["tiles"]) == 9

    # Check that there are both target and non-target tiles
    target_count = sum(1 for t in metadata["tiles"] if t["is_target"])
    assert 2 <= target_count <= 7  # grid_size - 2


def test_tile_images_exist(config):
    generator = CaptchaGenerator(config)
    generator.generate()

    captcha_dir = os.path.join(config.data_root, "captchas", "train", "00000")
    for row in range(3):
        for col in range(3):
            tile_path = os.path.join(captcha_dir, f"tile_{row}_{col}.png")
            assert os.path.isfile(tile_path)
