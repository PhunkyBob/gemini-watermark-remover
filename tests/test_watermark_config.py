"""Tests for WatermarkSize enum, WatermarkConfig dataclass, and ImageFormat enum."""

import numpy as np
import pytest
from remove_watermark import WatermarkSize, WatermarkConfig, ImageFormat


class TestWatermarkSize:
    """Tests for WatermarkSize enum."""

    def test_small_value(self):
        """Test SMALL enum value."""
        assert WatermarkSize.SMALL.value == "small"

    def test_large_value(self):
        """Test LARGE enum value."""
        assert WatermarkSize.LARGE.value == "large"

    def test_enum_members(self):
        """Test all enum members are present."""
        assert len(WatermarkSize) == 2
        assert WatermarkSize.SMALL in WatermarkSize
        assert WatermarkSize.LARGE in WatermarkSize


class TestWatermarkConfig:
    """Tests for WatermarkConfig dataclass."""

    def test_watermark_config_creation(self):
        """Test creating a WatermarkConfig instance."""
        alpha_map = np.array([[0.5, 0.6], [0.7, 0.8]])
        config = WatermarkConfig(
            margin_right=32,
            margin_bottom=32,
            logo_size=48,
            alpha_map=alpha_map
        )

        assert config.margin_right == 32
        assert config.margin_bottom == 32
        assert config.logo_size == 48
        assert np.array_equal(config.alpha_map, alpha_map)

    def test_watermark_config_immutable(self):
        """Test that WatermarkConfig is frozen (immutable)."""
        alpha_map = np.array([[0.5, 0.6]])
        config = WatermarkConfig(
            margin_right=32,
            margin_bottom=32,
            logo_size=48,
            alpha_map=alpha_map
        )

        with pytest.raises(AttributeError):
            config.margin_right = 64

    def test_watermark_config_slots(self):
        """Test that WatermarkConfig uses __slots__."""
        # Check that __slots__ is defined in the class
        assert hasattr(WatermarkConfig, '__slots__')

        # Create instance
        config = WatermarkConfig(
            margin_right=32,
            margin_bottom=32,
            logo_size=48,
            alpha_map=np.array([[0.5]])
        )

        # Slots should prevent adding new attributes (raises TypeError for slot-based classes)
        with pytest.raises((TypeError, AttributeError)):
            config.new_attr = "value"


class TestImageFormat:
    """Tests for ImageFormat enum."""

    def test_jpeg_extensions(self):
        """Test JPEG format extensions."""
        assert ImageFormat.JPEG.value == (".jpg", ".jpeg")

    def test_png_extensions(self):
        """Test PNG format extensions."""
        assert ImageFormat.PNG.value == (".png",)

    def test_webp_extensions(self):
        """Test WEBP format extensions."""
        assert ImageFormat.WEBP.value == (".webp",)

    def test_enum_members(self):
        """Test all enum members are present."""
        assert len(ImageFormat) == 3
        assert ImageFormat.JPEG in ImageFormat
        assert ImageFormat.PNG in ImageFormat
        assert ImageFormat.WEBP in ImageFormat

    @pytest.mark.parametrize("ext,expected_format", [
        (".jpg", ImageFormat.JPEG),
        (".jpeg", ImageFormat.JPEG),
        (".JPG", ImageFormat.JPEG),
        (".JPEG", ImageFormat.JPEG),
        (".png", ImageFormat.PNG),
        (".PNG", ImageFormat.PNG),
        (".webp", ImageFormat.WEBP),
        (".WEBP", ImageFormat.WEBP),
    ])
    def test_extension_detection(self, ext, expected_format):
        """Test that extensions match their expected formats."""
        ext_lower = ext.lower()
        for fmt in ImageFormat:
            if ext_lower in fmt.value:
                assert fmt == expected_format
                break
        else:
            pytest.fail(f"Extension {ext} not found in any format")
