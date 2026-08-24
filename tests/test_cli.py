"""
Unit tests for CLI
==================

Tests the Click CLI command 'sample' under various usage patterns.
"""

from pathlib import Path
import tempfile
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


def assert_aidetect_post(mock_post, content, content_type):
    """
    Assert the aidetect request, ignoring the W3C trace context headers that
    ``_inject_trace_headers`` adds when tracing is active.
    """
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args == ("http://127.0.0.1:8000/api/v1/aidetect/",)
    assert kwargs["content"] == content
    assert kwargs["timeout"] == 15.0
    assert kwargs["headers"]["Content-Type"] == content_type


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


@patch("sulku.cli.generate_synthetic.SyntheticDatasetGenerator")
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


@patch("sulku.cli.generate_synthetic.SyntheticDatasetGenerator")
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


@patch("sulku.bootstrap.setup_logging")
def test_cli_logging_default(mock_setup_logging, temp_dataset):
    """Test that default CLI invocation configures logging with INFO level."""
    runner = CliRunner()
    result = runner.invoke(main, ["sample", str(temp_dataset), "-n", "1"])
    assert result.exit_code == 0
    assert mock_setup_logging.call_count == 1
    settings = mock_setup_logging.call_args[0][0]
    assert settings.LOG_LEVEL == "INFO"


@patch("sulku.bootstrap.setup_logging")
def test_cli_logging_debug_shorthand(mock_setup_logging, temp_dataset):
    """Test that --debug option configures logging with DEBUG level."""
    runner = CliRunner()
    result = runner.invoke(main, ["--debug", "sample", str(temp_dataset), "-n", "1"])
    assert result.exit_code == 0
    assert mock_setup_logging.call_count == 1
    settings = mock_setup_logging.call_args[0][0]
    assert settings.LOG_LEVEL == "DEBUG"


@patch("sulku.bootstrap.setup_logging")
def test_cli_logging_custom_level(mock_setup_logging, temp_dataset):
    """Test that --log-level option configures logging with the requested level."""
    runner = CliRunner()
    result = runner.invoke(main, ["--log-level", "ERROR", "sample", str(temp_dataset), "-n", "1"])
    assert result.exit_code == 0
    assert mock_setup_logging.call_count == 1
    settings = mock_setup_logging.call_args[0][0]
    assert settings.LOG_LEVEL == "ERROR"





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
    assert "Starting server on 127.0.0.1:8000 (reload=False, preload=False)..." in result.output
    assert mock_uvicorn_run.call_count == 1
    call_args, call_kwargs = mock_uvicorn_run.call_args
    assert call_kwargs["host"] == "127.0.0.1"
    assert call_kwargs["port"] == 8000


@patch("uvicorn.run")
def test_cli_serve_custom(mock_uvicorn_run):
    """Test that the serve command invokes uvicorn with custom host, port, and reload flags."""
    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--host", "0.0.0.0", "--port", "9000", "--reload"])

    assert result.exit_code == 0
    assert "Starting server on 0.0.0.0:9000 (reload=True, preload=False)..." in result.output
    mock_uvicorn_run.assert_called_once_with(
        "sulku.http:create_app_from_env",
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
        "final_score": 0.85,
        "final_confidence": 0.70,
        "predictions": {"gemini-3.1-flash-lite": 0.85},
        "confidences": {"gemini-3.1-flash-lite": 0.70},
        "paragraphs": [
            {
                "text": "This is some sample text to analyze.",
                "sentences": ["This is some sample text to analyze."],
                "predictions": {"gemini-3.1-flash-lite": 0.85},
                "final_score": 0.85,
            }
        ],
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
        assert "Final Score: 0.8500" in result.output
        assert "Final Confidence: 0.7000" in result.output
        assert "gemini-3.1-flash-lite: 0.8500 (confidence: 0.7000)" in result.output
        assert "Paragraphs:" in result.output
        assert "Paragraph 1 (1 sentences): score: 0.8500 | \"This is some sample text to analyze.\"" in result.output
        assert_aidetect_post(mock_post, "This is some sample text to analyze.", "text/plain")


@patch("httpx.post")
def test_cli_detect_markdown_success(mock_post):
    """Test successful execution of the detect command on a markdown file."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "is_ai": False,
        "ai_votes": 1,
        "total_models": 1,
        "final_score": 0.85,
        "final_confidence": 0.70,
        "predictions": {"gemini-3.1-flash-lite": 0.85},
        "confidences": {"gemini-3.1-flash-lite": 0.70},
    }
    mock_post.return_value = mock_response

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.md"
        test_file.write_text("This is some sample text to analyze.", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["detect", str(test_file)])

        assert result.exit_code == 0
        assert "Sending" in result.output
        assert_aidetect_post(mock_post, "This is some sample text to analyze.", "text/markdown")


@patch("httpx.post")
@patch("trafilatura.extract")
def test_cli_detect_html_success(mock_extract, mock_post):
    """Test successful execution of the detect command on an HTML file using trafilatura."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "is_ai": False,
        "ai_votes": 1,
        "total_models": 1,
        "final_score": 0.85,
        "final_confidence": 0.70,
        "predictions": {"gemini-3.1-flash-lite": 0.85},
    }
    mock_post.return_value = mock_response
    mock_extract.return_value = "Extracted markdown from local HTML file."

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.html"
        test_file.write_text("<html><body><h1>Article</h1><p>Content</p></body></html>", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["detect", str(test_file)])

        assert result.exit_code == 0
        assert "Sending" in result.output
        mock_extract.assert_called_once_with("<html><body><h1>Article</h1><p>Content</p></body></html>", output_format="markdown")
        assert_aidetect_post(mock_post, "Extracted markdown from local HTML file.", "text/markdown")


@patch("trafilatura.extract")
def test_cli_detect_html_extract_failure(mock_extract):
    """Test error handling when trafilatura extraction fails on an HTML file."""
    mock_extract.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "empty.html"
        test_file.write_text("<html></html>", encoding="utf-8")

        runner = CliRunner()
        result = runner.invoke(main, ["detect", str(test_file)])

        assert result.exit_code != 0
        assert "Failed to extract markdown content from HTML file" in result.output



@patch("httpx.post")
@patch("trafilatura.extract")
@patch("trafilatura.fetch_url")
def test_cli_detect_url_success(mock_fetch_url, mock_extract, mock_post):
    """Test successful execution of the detect command when given a URL."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "is_ai": False,
        "ai_votes": 1,
        "total_models": 1,
        "final_score": 0.85,
        "final_confidence": 0.70,
        "predictions": {"gemini-3.1-flash-lite": 0.85},
        "confidences": {"gemini-3.1-flash-lite": 0.70},
    }
    mock_post.return_value = mock_response
    mock_fetch_url.return_value = "<html><body>Some content</body></html>"
    mock_extract.return_value = "Extracted content from URL"

    runner = CliRunner()
    result = runner.invoke(main, ["detect", "https://example.com/page"])

    assert result.exit_code == 0
    assert "Fetching content from https://example.com/page..." in result.output
    assert "Sending https://example.com/page to aidetect service" in result.output
    assert "AI-Generated: False" in result.output
    mock_fetch_url.assert_called_once_with("https://example.com/page")
    mock_extract.assert_called_once_with("<html><body>Some content</body></html>", output_format="markdown")
    assert_aidetect_post(mock_post, "Extracted content from URL", "text/markdown")


@patch("trafilatura.fetch_url")
def test_cli_detect_url_fetch_failure(mock_fetch_url):
    """Test detect command behavior when URL fetching fails."""
    mock_fetch_url.return_value = None

    runner = CliRunner()
    result = runner.invoke(main, ["detect", "https://example.com/bad-page"])

    assert result.exit_code != 0
    assert "Error: Failed to fetch content from URL https://example.com/bad-page" in result.output


@patch("trafilatura.extract")
@patch("trafilatura.fetch_url")
def test_cli_detect_url_extract_failure(mock_fetch_url, mock_extract):
    """Test detect command behavior when markdown extraction from fetched content fails."""
    mock_fetch_url.return_value = "<html></html>"
    mock_extract.return_value = None

    runner = CliRunner()
    result = runner.invoke(main, ["detect", "https://example.com/empty-page"])

    assert result.exit_code != 0
    assert "Error: Failed to extract markdown content from URL https://example.com/empty-page" in result.output


@patch("httpx.post")
def test_cli_test_detect_success(mock_post):
    """Test successful execution of the 'sulku test detect' command."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = [
        {"is_ai": False},  # First call (human file) -> correct (is_ai=False)
        {"is_ai": True},   # Second call (synthetic file) -> correct (is_ai=True)
    ]
    mock_post.return_value = mock_response

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        human_dir = tmp_path / "human"
        human_dir.mkdir()
        synthetic_dir = tmp_path / "synthetic"
        synthetic_dir.mkdir()
        
        (human_dir / "human1.md").write_text("This is some human written content.", encoding="utf-8")
        (synthetic_dir / "syn1.md").write_text("This is some synthetic written content.", encoding="utf-8")
        
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "test",
                "detect",
                "--count", "1",
                "--human-dir", str(human_dir),
                "--synthetic-dir", str(synthetic_dir),
                "--seed", "42",
            ],
        )

        assert result.exit_code == 0
        assert "Testing classification accuracy" in result.output
        assert "human1.md -> Classified as Correct (HUMAN)" in result.output
        assert "syn1.md -> Classified as Correct (AI)" in result.output
        assert "DETECTION TEST RESULTS" in result.output
        assert "Accuracy: 100.00%" in result.output
        assert mock_post.call_count == 2


