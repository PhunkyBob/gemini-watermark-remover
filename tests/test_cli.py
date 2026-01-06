"""Tests for CLI argument parsing."""

import pytest
from remove_watermark import parse_arguments


class TestParseArguments:
    """Tests for parse_arguments function."""

    def test_parse_single_input(self, monkeypatch):
        """Test parsing a single input file."""
        monkeypatch.setattr("sys.argv", ["remove_watermark.py", "image.png"])
        args = parse_arguments()

        assert args.input == ["image.png"]
        assert args.out is None
        assert args.force_small is False
        assert args.force_large is False
        assert args.force_remove is False

    def test_parse_multiple_inputs(self, monkeypatch):
        """Test parsing multiple input files."""
        monkeypatch.setattr("sys.argv", ["remove_watermark.py", "image1.png", "image2.png", "image3.png"])
        args = parse_arguments()

        assert args.input == ["image1.png", "image2.png", "image3.png"]
        assert len(args.input) == 3

    def test_parse_with_output_short(self, monkeypatch):
        """Test parsing with short output flag."""
        monkeypatch.setattr("sys.argv", ["remove_watermark.py", "image.png", "-O", "output.png"])
        args = parse_arguments()

        assert args.input == ["image.png"]
        assert args.out == "output.png"

    def test_parse_with_output_long(self, monkeypatch):
        """Test parsing with long output flag."""
        monkeypatch.setattr("sys.argv", ["remove_watermark.py", "image.png", "--out", "output.png"])
        args = parse_arguments()

        assert args.input == ["image.png"]
        assert args.out == "output.png"

    def test_parse_force_small(self, monkeypatch):
        """Test parsing --force-small flag."""
        monkeypatch.setattr("sys.argv", ["remove_watermark.py", "image.png", "--force-small"])
        args = parse_arguments()

        assert args.force_small is True
        assert args.force_large is False

    def test_parse_force_large(self, monkeypatch):
        """Test parsing --force-large flag."""
        monkeypatch.setattr("sys.argv", ["remove_watermark.py", "image.png", "--force-large"])
        args = parse_arguments()

        assert args.force_small is False
        assert args.force_large is True

    def test_parse_force_remove(self, monkeypatch):
        """Test parsing --force-remove flag."""
        monkeypatch.setattr("sys.argv", ["remove_watermark.py", "image.png", "--force-remove"])
        args = parse_arguments()

        assert args.force_remove is True

    def test_parse_all_flags(self, monkeypatch):
        """Test parsing with all flags."""
        monkeypatch.setattr("sys.argv", [
            "remove_watermark.py",
            "image.png",
            "--out", "output.png",
            "--force-large",
            "--force-remove"
        ])
        args = parse_arguments()

        assert args.input == ["image.png"]
        assert args.out == "output.png"
        assert args.force_large is True
        assert args.force_remove is True

    def test_parse_multiple_inputs_with_output_dir(self, monkeypatch):
        """Test parsing multiple inputs with output directory."""
        monkeypatch.setattr("sys.argv", [
            "remove_watermark.py",
            "image1.png",
            "image2.png",
            "--out", "/output/dir"
        ])
        args = parse_arguments()

        assert args.input == ["image1.png", "image2.png"]
        assert args.out == "/output/dir"

    def test_version_flag(self, monkeypatch):
        """Test --version flag triggers SystemExit."""
        monkeypatch.setattr("sys.argv", ["remove_watermark.py", "--version"])
        with pytest.raises(SystemExit):
            parse_arguments()

    def test_help_flag(self, monkeypatch):
        """Test --help flag triggers SystemExit."""
        monkeypatch.setattr("sys.argv", ["remove_watermark.py", "--help"])
        with pytest.raises(SystemExit):
            parse_arguments()

    @pytest.mark.parametrize("flag_combination", [
        (["--force-small"], True, False),
        (["--force-large"], False, True),
        ([], False, False),
    ])
    def test_force_flags_mutual_exclusive_values(self, monkeypatch, flag_combination):
        """Test that force flags have correct values."""
        flags, expected_small, expected_large = flag_combination
        monkeypatch.setattr("sys.argv", ["remove_watermark.py", "image.png"] + flags)
        args = parse_arguments()

        assert args.force_small == expected_small
        assert args.force_large == expected_large

    def test_complex_real_world_scenario(self, monkeypatch):
        """Test a complex real-world command line scenario."""
        monkeypatch.setattr("sys.argv", [
            "remove_watermark.py",
            "photo1.jpg",
            "photo2.png",
            "photo3.webp",
            "--out", "./processed",
            "--force-remove"
        ])
        args = parse_arguments()

        assert len(args.input) == 3
        assert "photo1.jpg" in args.input
        assert args.out == "./processed"
        assert args.force_remove is True
