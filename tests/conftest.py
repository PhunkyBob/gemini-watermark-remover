"""Fixtures for watermark remover tests."""

from pathlib import Path

import numpy as np
import pytest
import cv2


@pytest.fixture
def sample_watermark_48(tmp_path: Path) -> Path:
    """Create a sample 48x48 watermark image."""
    watermark_path = tmp_path / "watermark_48x48.png"
    # Create a simple gradient pattern
    watermark = np.zeros((48, 48, 3), dtype=np.uint8)
    for i in range(48):
        for j in range(48):
            watermark[i, j] = [i * 5, j * 5, (i + j) * 2]
    cv2.imwrite(str(watermark_path), watermark)
    return watermark_path


@pytest.fixture
def sample_watermark_96(tmp_path: Path) -> Path:
    """Create a sample 96x96 watermark image."""
    watermark_path = tmp_path / "watermark_96x96.png"
    # Create a simple gradient pattern
    watermark = np.zeros((96, 96, 3), dtype=np.uint8)
    for i in range(96):
        for j in range(96):
            watermark[i, j] = [i * 2, min(j * 2, 255), min((i + j), 255)]
    cv2.imwrite(str(watermark_path), watermark)
    return watermark_path


@pytest.fixture
def sample_image_512(tmp_path: Path) -> Path:
    """Create a sample 512x512 image."""
    image_path = tmp_path / "sample_512.png"
    image = np.ones((512, 512, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(image_path), image)
    return image_path


@pytest.fixture
def sample_image_1024(tmp_path: Path) -> Path:
    """Create a sample 1024x1024 image."""
    image_path = tmp_path / "sample_1024.png"
    image = np.ones((1024, 1024, 3), dtype=np.uint8) * 128
    cv2.imwrite(str(image_path), image)
    return image_path


@pytest.fixture
def sample_image_grayscale(tmp_path: Path) -> Path:
    """Create a sample grayscale image."""
    image_path = tmp_path / "sample_gray.png"
    image = np.ones((512, 512), dtype=np.uint8) * 128
    cv2.imwrite(str(image_path), image)
    return image_path


@pytest.fixture
def sample_image_bgra(tmp_path: Path) -> Path:
    """Create a sample BGRA image."""
    image_path = tmp_path / "sample_bgra.png"
    image = np.ones((512, 512, 4), dtype=np.uint8) * 128
    image[:, :, 3] = 255  # Full alpha
    cv2.imwrite(str(image_path), image)
    return image_path


@pytest.fixture
def watermark_remover(sample_watermark_48: Path, sample_watermark_96: Path):
    """Create a WatermarkRemover instance with sample watermarks."""
    from remove_watermark import WatermarkRemover
    return WatermarkRemover(sample_watermark_48, sample_watermark_96)
