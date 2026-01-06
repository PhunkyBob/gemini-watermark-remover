"""Tests for WatermarkRemover class."""

from pathlib import Path

import numpy as np
import cv2
import pytest

from remove_watermark import WatermarkRemover, WatermarkSize


class TestWatermarkRemoverInit:
    """Tests for WatermarkRemover initialization."""

    def test_init_with_valid_files(self, sample_watermark_48, sample_watermark_96):
        """Test initialization with valid watermark files."""
        remover = WatermarkRemover(sample_watermark_48, sample_watermark_96)
        assert remover.bg_48 is not None
        assert remover.bg_96 is not None
        assert remover.alpha_48 is not None
        assert remover.alpha_96 is not None
        assert remover.template_48 is not None
        assert remover.template_96 is not None

    def test_init_with_invalid_file(self, tmp_path):
        """Test initialization with non-existent watermark file."""
        fake_watermark = tmp_path / "nonexistent.bin"
        with pytest.raises(FileNotFoundError):
            WatermarkRemover(fake_watermark, fake_watermark)

    def test_templates_are_grayscale(self, watermark_remover):
        """Test that templates are converted to grayscale."""
        assert len(watermark_remover.template_48.shape) == 2
        assert len(watermark_remover.template_96.shape) == 2

    def test_alpha_maps_are_float32(self, watermark_remover):
        """Test that alpha maps are float32 type."""
        assert watermark_remover.alpha_48.dtype == np.float32
        assert watermark_remover.alpha_96.dtype == np.float32

    def test_alpha_maps_normalized(self, watermark_remover):
        """Test that alpha maps are normalized to [0, 1]."""
        assert np.all(watermark_remover.alpha_48 >= 0.0)
        assert np.all(watermark_remover.alpha_48 <= 1.0)
        assert np.all(watermark_remover.alpha_96 >= 0.0)
        assert np.all(watermark_remover.alpha_96 <= 1.0)


class TestWatermarkRemoverLoadWatermark:
    """Tests for _load_watermark method."""

    def test_load_watermark_valid(self, watermark_remover, tmp_path):
        """Test loading a valid watermark image."""
        test_path = tmp_path / "test_watermark.png"
        test_image = np.zeros((48, 48, 3), dtype=np.uint8)
        cv2.imwrite(str(test_path), test_image)

        result = watermark_remover._load_watermark(test_path)
        assert result is not None
        assert result.shape == (48, 48, 3)

    def test_load_watermark_invalid(self, watermark_remover, tmp_path):
        """Test loading a non-existent watermark."""
        fake_path = tmp_path / "fake.png"
        with pytest.raises(FileNotFoundError):
            watermark_remover._load_watermark(fake_path)


class TestWatermarkRemoverCalculateAlphaMap:
    """Tests for _calculate_alpha_map method."""

    def test_calculate_alpha_3d_array(self, watermark_remover):
        """Test alpha map calculation for 3D array (BGR image)."""
        bgr_image = np.array([
            [[100, 150, 200], [50, 75, 100]],
            [[25, 30, 35], [10, 20, 30]]
        ], dtype=np.uint8)
        alpha = watermark_remover._calculate_alpha_map(bgr_image)

        assert alpha.shape == (2, 2)
        assert alpha[0, 0] == pytest.approx(200.0 / 255.0)  # max(100, 150, 200)
        assert alpha[0, 1] == pytest.approx(100.0 / 255.0)
        assert alpha.dtype == np.float32

    def test_calculate_alpha_2d_array(self, watermark_remover):
        """Test alpha map calculation for 2D array (grayscale)."""
        gray_image = np.array([[100, 150], [50, 75]], dtype=np.uint8)
        alpha = watermark_remover._calculate_alpha_map(gray_image)

        assert alpha.shape == (2, 2)
        assert alpha[0, 0] == pytest.approx(100.0 / 255.0)
        assert alpha.dtype == np.float32


class TestWatermarkRemoverGetConfig:
    """Tests for _get_config method."""

    def test_get_config_force_small(self, watermark_remover):
        """Test config with forced small size."""
        config = watermark_remover._get_config(512, 512, WatermarkSize.SMALL)
        assert config.margin_right == 32
        assert config.margin_bottom == 32
        assert config.logo_size == 48

    def test_get_config_force_large(self, watermark_remover):
        """Test config with forced large size."""
        config = watermark_remover._get_config(512, 512, WatermarkSize.LARGE)
        assert config.margin_right == 64
        assert config.margin_bottom == 64
        assert config.logo_size == 96

    def test_get_config_auto_large(self, watermark_remover):
        """Test auto config for large images."""
        config = watermark_remover._get_config(1025, 1025, None)
        assert config.logo_size == 96
        assert config.margin_right == 64
        assert config.margin_bottom == 64

    def test_get_config_auto_small(self, watermark_remover):
        """Test auto config for small images."""
        config = watermark_remover._get_config(512, 512, None)
        assert config.logo_size == 48
        assert config.margin_right == 32
        assert config.margin_bottom == 32

    def test_get_config_boundary(self, watermark_remover):
        """Test config at size boundary (1024)."""
        # Both dimensions must be > 1024 for large watermark
        config1 = watermark_remover._get_config(1025, 1025, None)
        assert config1.logo_size == 96

        config2 = watermark_remover._get_config(1024, 1024, None)
        assert config2.logo_size == 48

        config3 = watermark_remover._get_config(1025, 1024, None)
        assert config3.logo_size == 48


class TestWatermarkRemoverGetPosition:
    """Tests for _get_watermark_position method."""

    def test_position_small_image(self, watermark_remover):
        """Test watermark position for small image."""
        config = watermark_remover._get_config(512, 512, WatermarkSize.SMALL)
        x, y = watermark_remover._get_watermark_position(512, 512, config)
        assert x == 512 - 32 - 48  # width - margin_right - logo_size
        assert y == 512 - 32 - 48  # height - margin_bottom - logo_size

    def test_position_large_image(self, watermark_remover):
        """Test watermark position for large image."""
        config = watermark_remover._get_config(2048, 2048, WatermarkSize.LARGE)
        x, y = watermark_remover._get_watermark_position(2048, 2048, config)
        assert x == 2048 - 64 - 96
        assert y == 2048 - 64 - 96


class TestWatermarkRemoverEnsureBGR:
    """Tests for _ensure_bgr_format method."""

    def test_ensure_bgr_from_bgr(self, watermark_remover):
        """Test BGR image is unchanged."""
        bgr = np.ones((100, 100, 3), dtype=np.uint8)
        result = watermark_remover._ensure_bgr_format(bgr)
        assert np.array_equal(result, bgr)
        assert result.shape == (100, 100, 3)

    def test_ensure_bgr_from_grayscale(self, watermark_remover):
        """Test grayscale is converted to BGR."""
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        result = watermark_remover._ensure_bgr_format(gray)
        assert result.shape == (100, 100, 3)
        # All channels should have the same value
        assert np.all(result[:, :, 0] == 128)
        assert np.all(result[:, :, 1] == 128)
        assert np.all(result[:, :, 2] == 128)

    def test_ensure_bgr_from_bgra(self, watermark_remover):
        """Test BGRA is converted to BGR."""
        bgra = np.ones((100, 100, 4), dtype=np.uint8) * 128
        result = watermark_remover._ensure_bgr_format(bgra)
        assert result.shape == (100, 100, 3)
        assert np.all(result == 128)


class TestWatermarkRemoverDetect:
    """Tests for detect_watermark method."""

    def test_detect_watermark_not_found(self, watermark_remover):
        """Test detection on image without watermark."""
        # Create a plain image without watermark
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        detected = watermark_remover.detect_watermark(image)
        # Should not detect watermark in plain image
        assert detected == False

    def test_detect_watermark_small(self, watermark_remover):
        """Test detection with forced small size."""
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        detected = watermark_remover.detect_watermark(image, WatermarkSize.SMALL)
        # Template matching threshold is 0.3
        assert isinstance(detected, bool)

    def test_detect_watermark_large(self, watermark_remover):
        """Test detection with forced large size."""
        image = np.ones((1024, 1024, 3), dtype=np.uint8) * 128
        detected = watermark_remover.detect_watermark(image, WatermarkSize.LARGE)
        assert isinstance(detected, bool)

    def test_detect_handles_grayscale(self, watermark_remover):
        """Test detection works with grayscale input."""
        gray = np.ones((512, 512), dtype=np.uint8) * 128
        detected = watermark_remover.detect_watermark(gray)
        assert isinstance(detected, bool)


class TestWatermarkRemoverRemove:
    """Tests for remove_watermark method."""

    def test_remove_watermark_small_image(self, watermark_remover):
        """Test watermark removal on small image."""
        # Create image with synthetic watermark at expected location
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        # Apply watermark blending at bottom-right corner
        x_start = 512 - 32 - 48  # width - margin_right - logo_size
        y_start = 512 - 32 - 48  # height - margin_bottom - logo_size
        image[y_start:y_start+48, x_start:x_start+48] = 200

        # Store original watermark region before processing
        original_region = image[y_start:y_start+48, x_start:x_start+48].copy()

        result = watermark_remover.remove_watermark(image, WatermarkSize.SMALL)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # The watermark region should be processed (different from original)
        watermark_region = result[y_start:y_start+48, x_start:x_start+48]
        assert not np.array_equal(watermark_region, original_region)

    def test_remove_watermark_large_image(self, watermark_remover):
        """Test watermark removal on large image."""
        # Create image with synthetic watermark at expected location
        image = np.ones((2048, 2048, 3), dtype=np.uint8) * 128
        # Apply watermark blending at bottom-right corner
        x_start = 2048 - 64 - 96  # width - margin_right - logo_size
        y_start = 2048 - 64 - 96  # height - margin_bottom - logo_size
        image[y_start:y_start+96, x_start:x_start+96] = 200

        # Store original watermark region before processing
        original_region = image[y_start:y_start+96, x_start:x_start+96].copy()

        result = watermark_remover.remove_watermark(image, WatermarkSize.LARGE)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # The watermark region should be processed (different from original)
        watermark_region = result[y_start:y_start+96, x_start:x_start+96]
        assert not np.array_equal(watermark_region, original_region)

    def test_remove_watermark_auto_size(self, watermark_remover):
        """Test watermark removal with auto size detection."""
        image = np.ones((512, 512, 3), dtype=np.uint8) * 200
        result = watermark_remover.remove_watermark(image)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_remove_watermark_very_small_image(self, watermark_remover):
        """Test watermark removal on very small image."""
        # Image smaller than watermark + margins
        image = np.ones((32, 32, 3), dtype=np.uint8) * 128
        result = watermark_remover.remove_watermark(image, WatermarkSize.SMALL)

        # Should return unchanged image
        assert np.array_equal(result, image)

    def test_remove_handles_grayscale(self, watermark_remover):
        """Test removal works with grayscale input."""
        gray = np.ones((512, 512), dtype=np.uint8) * 128
        result = watermark_remover.remove_watermark(gray, WatermarkSize.SMALL)

        assert result.shape == (512, 512, 3)
        assert result.dtype == np.uint8

    def test_remove_handles_bgra(self, watermark_remover):
        """Test removal works with BGRA input."""
        bgra = np.ones((512, 512, 4), dtype=np.uint8) * 128
        bgra[:, :, 3] = 255
        result = watermark_remover.remove_watermark(bgra, WatermarkSize.SMALL)

        assert result.shape == (512, 512, 3)
        assert result.dtype == np.uint8

    def test_remove_values_in_range(self, watermark_remover):
        """Test that removed pixel values stay in valid range."""
        image = np.ones((512, 512, 3), dtype=np.uint8) * 128
        result = watermark_remover.remove_watermark(image, WatermarkSize.SMALL)

        assert np.all(result >= 0)
        assert np.all(result <= 255)
