"""Tests for utility functions."""

from pathlib import Path

import cv2
import pytest

from remove_watermark import get_image_writer_params, get_output_path


class TestGetImageWriterParams:
    """Tests for get_image_writer_params function."""

    def test_jpeg_params(self):
        """Test JPEG format parameters."""
        params = get_image_writer_params(".jpg")
        assert params == [cv2.IMWRITE_JPEG_QUALITY, 100]

        params = get_image_writer_params(".jpeg")
        assert params == [cv2.IMWRITE_JPEG_QUALITY, 100]

        params = get_image_writer_params(".JPG")
        assert params == [cv2.IMWRITE_JPEG_QUALITY, 100]

    def test_png_params(self):
        """Test PNG format parameters."""
        params = get_image_writer_params(".png")
        assert params == [cv2.IMWRITE_PNG_COMPRESSION, 6]

        params = get_image_writer_params(".PNG")
        assert params == [cv2.IMWRITE_PNG_COMPRESSION, 6]

    def test_webp_params(self):
        """Test WEBP format parameters."""
        params = get_image_writer_params(".webp")
        assert params == [cv2.IMWRITE_WEBP_QUALITY, 101]

        params = get_image_writer_params(".WEBP")
        assert params == [cv2.IMWRITE_WEBP_QUALITY, 101]

    def test_unknown_format(self):
        """Test unknown format returns empty list."""
        params = get_image_writer_params(".tiff")
        assert params == []

        params = get_image_writer_params(".bmp")
        assert params == []

    def test_empty_extension(self):
        """Test empty extension."""
        params = get_image_writer_params("")
        assert params == []


class TestGetOutputPath:
    """Tests for get_output_path function."""

    def test_no_output_specified(self):
        """Test output path when no output is specified."""
        input_path = Path("/path/to/image.png")
        output_path = get_output_path(input_path, None, False)

        expected = Path("/path/to/image_nowatermark.png")
        assert output_path == expected

    def test_output_is_file(self):
        """Test when output is specified as file path."""
        input_path = Path("/path/to/image.png")
        output_arg = "/custom/output/result.jpg"
        output_path = get_output_path(input_path, output_arg, False)

        expected = Path("/custom/output/result.jpg")
        assert output_path == expected

    def test_output_is_directory(self, tmp_path):
        """Test when output is specified as directory."""
        input_path = Path("/path/to/image.png")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        output_path = get_output_path(input_path, str(output_dir), False)

        expected = output_dir / "image_nowatermark.png"
        assert output_path == expected

    def test_output_is_directory_multiple_inputs(self, tmp_path):
        """Test output directory with multiple input files."""
        input_path = Path("/path/to/image.png")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        output_path = get_output_path(input_path, str(output_dir), True)

        expected = output_dir / "image_nowatermark.png"
        assert output_path == expected

    def test_preserves_extension(self):
        """Test that original extension is preserved."""
        test_cases = [
            ("image.png", "image_nowatermark.png"),
            ("image.JPG", "image_nowatermark.JPG"),
            ("image.webp", "image_nowatermark.webp"),
        ]

        for input_name, expected_name in test_cases:
            input_path = Path(f"/path/to/{input_name}")
            output_path = get_output_path(input_path, None, False)
            assert output_path.name == expected_name

    def test_nested_input_path(self):
        """Test with nested input path."""
        input_path = Path("/deep/nested/path/to/image.png")
        output_path = get_output_path(input_path, None, False)

        expected = Path("/deep/nested/path/to/image_nowatermark.png")
        assert output_path == expected

    def test_output_directory_preserves_parent(self):
        """Test that output directory preserves parent structure."""
        input_path = Path("/path/to/image.png")
        output_path = get_output_path(input_path, None, False)

        # Parent should be same as input parent
        assert output_path.parent == input_path.parent
