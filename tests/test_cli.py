"""
Unit tests for CLI
==================

Tests the Click CLI command 'sample' under various usage patterns.
"""

from pathlib import Path
import tempfile
import logging
from unittest.mock import MagicMock, patch
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


def test_cli_sample_with_filtering():
    """Test sampling with language and word count filters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # art1: fi, 60 words
        art1 = tmp_path / "art1.md"
        art1.write_text("---\nlanguage: fi\n---\n" + " ".join(["sana"] * 60), encoding="utf-8")

        # art2: sv, 120 words
        art2 = tmp_path / "art2.md"
        art2.write_text("---\nlang: sv\n---\n" + " ".join(["ord"] * 120), encoding="utf-8")

        # art3: fi, 5 words
        art3 = tmp_path / "art3.md"
        art3.write_text("---\nlanguage: fi\n---\n" + " ".join(["sana"] * 5), encoding="utf-8")

        runner = CliRunner()

        # 1. Filter by language 'sv'
        result = runner.invoke(main, ["sample", str(tmp_path), "-n", "1", "-l", "sv"])
        assert result.exit_code == 0
        paths = result.output.strip().splitlines()
        assert len(paths) == 1
        assert Path(paths[0]).name == "art2.md"

        # 2. Filter by language 'fi' and min-words 50
        result = runner.invoke(main, ["sample", str(tmp_path), "-n", "1", "-l", "fi", "-mw", "50"])
        assert result.exit_code == 0
        paths = result.output.strip().splitlines()
        assert len(paths) == 1
        assert Path(paths[0]).name == "art1.md"

        # 3. Filter with options that match no files
        result = runner.invoke(main, ["sample", str(tmp_path), "-n", "1", "-l", "en"])
        assert result.exit_code != 0
        assert "Error: No files found matching pattern" in result.output


@patch("sulku.cli.SyntheticDatasetGenerator")
def test_cli_generate_synthetic_success(mock_generator_class):
    """Test successful CLI execution for generate-synthetic command."""
    mock_generator = MagicMock()
    mock_generator.generate.return_value = [
        Path("/dummy/out1.md"),
        Path("/dummy/out2.md"),
    ]
    mock_generator_class.return_value = mock_generator

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate-synthetic", tmpdir, "-n", "2", "-m", "test-model", "-s", "100"],
        )

        assert result.exit_code == 0
        assert "Sampling 2 articles and generating synthetic articles using model 'test-model'..." in result.output
        assert "Successfully generated 2 synthetic articles:" in result.output
        assert "- /dummy/out1.md" in result.output
        assert "- /dummy/out2.md" in result.output

        mock_generator_class.assert_called_once_with(source_dir=Path(tmpdir), model_name="test-model")
        mock_generator.generate.assert_called_once_with(n_samples=2, seed=100, dest_dir=None, force=False, min_words=50)


@patch("sulku.cli.SyntheticDatasetGenerator")
def test_cli_generate_synthetic_force(mock_generator_class):
    """Test CLI execution for generate-synthetic command with --force flag."""
    mock_generator = MagicMock()
    mock_generator.generate.return_value = [
        Path("/dummy/out1.md"),
    ]
    mock_generator_class.return_value = mock_generator

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["generate-synthetic", tmpdir, "-n", "1", "-m", "test-model", "--force"],
        )

        assert result.exit_code == 0
        mock_generator.generate.assert_called_once_with(n_samples=1, seed=None, dest_dir=None, force=True, min_words=50)


@patch("logging.basicConfig")
def test_cli_logging_default(mock_basic_config, temp_dataset):
    """Test that default CLI invocation configures logging with INFO level."""
    runner = CliRunner()
    result = runner.invoke(main, ["sample", str(temp_dataset), "-n", "1"])
    assert result.exit_code == 0
    assert mock_basic_config.call_count == 1
    kwargs = mock_basic_config.call_args[1]
    assert kwargs["level"] == logging.INFO
    assert kwargs["force"] is True
    assert len(kwargs["handlers"]) == 1
    assert kwargs["handlers"][0].formatter.show_extra is False


@patch("logging.basicConfig")
def test_cli_logging_debug_shorthand(mock_basic_config, temp_dataset):
    """Test that --debug option configures logging with DEBUG level."""
    runner = CliRunner()
    result = runner.invoke(main, ["--debug", "sample", str(temp_dataset), "-n", "1"])
    assert result.exit_code == 0
    assert mock_basic_config.call_count == 1
    kwargs = mock_basic_config.call_args[1]
    assert kwargs["level"] == logging.DEBUG
    assert kwargs["force"] is True
    assert len(kwargs["handlers"]) == 1
    assert kwargs["handlers"][0].formatter.show_extra is True


@patch("logging.basicConfig")
def test_cli_logging_custom_level(mock_basic_config, temp_dataset):
    """Test that --log-level option configures logging with the requested level."""
    runner = CliRunner()
    result = runner.invoke(main, ["--log-level", "ERROR", "sample", str(temp_dataset), "-n", "1"])
    assert result.exit_code == 0
    assert mock_basic_config.call_count == 1
    kwargs = mock_basic_config.call_args[1]
    assert kwargs["level"] == logging.ERROR
    assert kwargs["force"] is True
    assert len(kwargs["handlers"]) == 1
    assert kwargs["handlers"][0].formatter.show_extra is False


def test_extra_formatter():
    """Test the custom ExtraFormatter formatting with/without show_extra."""
    from sulku.cli import ExtraFormatter

    record = logging.LogRecord(
        name="test_logger",
        level=logging.DEBUG,
        pathname="test.py",
        lineno=10,
        msg="A simple log message",
        args=(),
        exc_info=None,
    )
    # Add some extra attributes
    record.token_usage = {"prompt_tokens": 10, "completion_tokens": 20}
    record.simple_extra = "hello"

    # 1. Formatting with show_extra=False
    formatter_no_extra = ExtraFormatter(fmt="%(message)s", show_extra=False)
    output_no_extra = formatter_no_extra.format(record)
    assert output_no_extra == "A simple log message"

    # 2. Formatting with show_extra=True
    formatter_with_extra = ExtraFormatter(fmt="%(message)s", show_extra=True)
    output_with_extra = formatter_with_extra.format(record)

    # Check that extra attributes are vertically formatted and indented
    assert "A simple log message" in output_with_extra
    assert "simple_extra: hello" in output_with_extra
    assert "token_usage:" in output_with_extra
    assert "    completion_tokens: 20" in output_with_extra
    assert "    prompt_tokens: 10" in output_with_extra


def test_cli_generate_fasttext(temp_dataset):
    """Test generating FastText training data using the CLI command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "fasttext.txt"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "generate-fasttext",
                str(temp_dataset),
                "-o",
                str(output_file),
                "-l",
                "human",
                "-mw",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert "Successfully wrote FastText sentence data to" in result.output
        assert output_file.exists()
        lines = output_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) > 0
        assert all(line.startswith("__label__human ") for line in lines)


@patch("uvicorn.run")
def test_cli_serve_defaults(mock_uvicorn_run):
    """Test that the serve command invokes uvicorn with default parameters."""
    runner = CliRunner()
    result = runner.invoke(main, ["serve"])

    assert result.exit_code == 0
    assert "Starting server on 127.0.0.1:8000 (reload=False)..." in result.output
    mock_uvicorn_run.assert_called_once_with(
        "sulku.http:create_app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        factory=True,
    )


@patch("uvicorn.run")
def test_cli_serve_custom(mock_uvicorn_run):
    """Test that the serve command invokes uvicorn with custom host, port, and reload flags."""
    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--host", "0.0.0.0", "--port", "9000", "--reload"])

    assert result.exit_code == 0
    assert "Starting server on 0.0.0.0:9000 (reload=True)..." in result.output
    mock_uvicorn_run.assert_called_once_with(
        "sulku.http:create_app",
        host="0.0.0.0",
        port=9000,
        reload=True,
        factory=True,
    )


@patch("uvicorn.run")
def test_cli_serve_exception(mock_uvicorn_run):
    """Test serve command error handling when uvicorn fails to start."""
    mock_uvicorn_run.side_effect = RuntimeError("Could not bind to port")
    runner = CliRunner()
    result = runner.invoke(main, ["serve"])

    assert result.exit_code != 0
    assert "Error starting server: Could not bind to port" in result.output


@patch("httpx.post")
def test_cli_detect_success(mock_post):
    """Test successful execution of the detect command."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "is_ai": False,
        "ai_votes": 1,
        "total_models": 1,
        "predictions": {"gemini-3.1-flash-lite": 0.85},
    }
    mock_post.return_value = mock_response

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("This is some sample text to analyze.", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["detect", str(test_file)])

        assert result.exit_code == 0
        assert "Sending" in result.output
        assert "AI-Generated: False" in result.output
        assert "Votes: 1/1" in result.output
        mock_post.assert_called_once_with(
            "http://127.0.0.1:8000/api/v1/aidetect/",
            content="This is some sample text to analyze.",
            headers={"Content-Type": "text/plain"},
            timeout=15.0,
        )


@patch("httpx.post")
def test_cli_detect_markdown_success(mock_post):
    """Test successful execution of the detect command on a markdown file."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "is_ai": False,
        "ai_votes": 1,
        "total_models": 1,
        "predictions": {"gemini-3.1-flash-lite": 0.85},
    }
    mock_post.return_value = mock_response

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.md"
        test_file.write_text("This is some sample text to analyze.", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["detect", str(test_file)])

        assert result.exit_code == 0
        assert "Sending" in result.output
        mock_post.assert_called_once_with(
            "http://127.0.0.1:8000/api/v1/aidetect/",
            content="This is some sample text to analyze.",
            headers={"Content-Type": "text/markdown"},
            timeout=15.0,
        )
