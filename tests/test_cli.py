"""
Unit tests for CLI
==================

Tests the Click CLI command 'sample' under various usage patterns.
"""

from pathlib import Path
import tempfile
from click.testing import CliRunner
import pytest
from sulku.cli import main


@pytest.fixture
def temp_dataset():
    """Create a temporary dataset with dummy files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for i in range(5):
            f = tmp_path / f"item_{i}.txt"
            f.write_text(f"content {i}", encoding="utf-8")
        yield tmp_path


def test_cli_sample_success(temp_dataset):
    """Test successful sampling from dataset."""
    runner = CliRunner()
    result = runner.invoke(main, ["sample", str(temp_dataset), "-n", "3", "-s", "42"])

    assert result.exit_code == 0
    paths = result.output.strip().splitlines()
    assert len(paths) == 3
    for p in paths:
        path_obj = Path(p)
        assert path_obj.exists()
        assert path_obj.parent == temp_dataset


def test_cli_sample_count_too_large(temp_dataset):
    """Test when count is larger than dataset size."""
    runner = CliRunner()
    result = runner.invoke(main, ["sample", str(temp_dataset), "-n", "10"])

    assert result.exit_code != 0
    assert "Error: Requested count 10 is larger than dataset size 5." in result.output


def test_cli_sample_no_matching_files(temp_dataset):
    """Test when pattern matches no files."""
    runner = CliRunner()
    result = runner.invoke(main, ["sample", str(temp_dataset), "-n", "1", "-p", "*.md"])

    assert result.exit_code != 0
    assert "Error: No files found matching pattern" in result.output
