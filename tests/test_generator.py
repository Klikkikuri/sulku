"""
Unit tests for Synthetic Dataset Generator
=========================================

Tests the SyntheticDatasetGenerator class, including deterministic sampling,
caching logic (cache hit, cache miss, cache invalidation on content changes),
metadata serialization, and output file generation with front matter and
directory structure.
"""

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import pytest

from sulku.dataset.generator import SyntheticDatasetGenerator
from sulku.dataset.reader import DatasetItem
from sulku.summarize.models import ArticleSummary, StyleVector


@pytest.fixture
def dummy_dataset_dir():
    """Create a temporary directory with dummy article files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create dummy articles mirroring Yle structure
        art1_dir = tmp_path / "2021" / "01" / "0000"
        art1_dir.mkdir(parents=True, exist_ok=True)
        art1 = art1_dir / "art1.md"
        art1.write_text(
            "---\n"
            "id: yle-1\n"
            "language: fi\n"
            'title: "Artikkeli Yksi"\n'
            "---\n"
            "Tämä on ensimmäinen artikkeli testejä varten.",
            encoding="utf-8",
        )

        art2 = art1_dir / "art2.md"
        art2.write_text(
            "---\n"
            "id: yle-2\n"
            "language: fi\n"
            'title: "Artikkeli Kaksi"\n'
            "---\n"
            "Tämä on toinen artikkeli testejä varten. Siinä on vähän enemmän sanoja.",
            encoding="utf-8",
        )

        yield tmp_path


@pytest.fixture
def dummy_cache_dir():
    """Create a temporary directory for caches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dummy_dest_dir():
    """Create a temporary directory for output synthetic articles."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_summary():
    """Return a dummy ArticleSummary pydantic object."""
    return ArticleSummary(
        headline="Reconstructed Headline",
        summary=["This is bullet point 1", "This is bullet point 2"],
        style=StyleVector(
            tone="Clinical / Detached",
            perspective="Third-Person Objective",
            angle="Macroscopic/Analytical",
            audience=["General / Mass Public"],
            type="Hard News / Breaking News",
        ),
    )


def test_generator_initialization(dummy_dataset_dir, dummy_cache_dir):
    """Test that generator initializes correctly with proper directories."""
    generator = SyntheticDatasetGenerator(
        source_dir=dummy_dataset_dir, model_name="test-model", cache_dir=dummy_cache_dir
    )
    assert generator.source_dir == dummy_dataset_dir
    assert generator.model_name == "test-model"
    assert generator.cache_dir == dummy_cache_dir
    assert generator.cache_dir.exists()


def test_generator_initialization_invalid_source():
    """Test that generator raises FileNotFoundError if source directory doesn't exist."""
    with pytest.raises(FileNotFoundError):
        SyntheticDatasetGenerator(source_dir="/path/to/nonexistent/directory")


@patch("sulku.dataset.generator.summarize_text")
def test_caching_logic_miss_and_hit(
    mock_summarize, dummy_dataset_dir, dummy_cache_dir, mock_summary
):
    """Test that cache is written on miss, and read on hit."""
    generator = SyntheticDatasetGenerator(
        source_dir=dummy_dataset_dir, model_name="test-model", cache_dir=dummy_cache_dir
    )

    article_path = dummy_dataset_dir / "2021" / "01" / "0000" / "art1.md"
    article = DatasetItem(article_path)

    # Configure mock
    mock_summarize.return_value = mock_summary

    # First call - cache miss
    summary1 = generator.get_or_create_summary(article)
    assert summary1 == mock_summary
    assert mock_summarize.call_count == 1

    # Verify cache file was created
    cache_file = generator._get_cache_file(article)
    assert cache_file.exists()

    with open(cache_file, "r", encoding="utf-8") as f:
        cached_data = json.load(f)
    assert cached_data["relative_path"] == "2021/01/0000/art1.md"
    assert cached_data["content_hash"] == generator._get_content_hash(article.content)
    assert cached_data["summary"]["headline"] == "Reconstructed Headline"

    # Second call - cache hit (should load from file and not call summarize_text)
    summary2 = generator.get_or_create_summary(article)
    assert summary2 == mock_summary
    assert mock_summarize.call_count == 1  # Still 1 call


@patch("sulku.dataset.generator.summarize_text")
def test_caching_logic_invalidation_on_change(
    mock_summarize, dummy_dataset_dir, dummy_cache_dir, mock_summary
):
    """Test that changing article content invalidates cache."""
    generator = SyntheticDatasetGenerator(
        source_dir=dummy_dataset_dir, model_name="test-model", cache_dir=dummy_cache_dir
    )

    article_path = dummy_dataset_dir / "2021" / "01" / "0000" / "art1.md"
    article = DatasetItem(article_path)

    mock_summarize.return_value = mock_summary

    # Cache miss
    generator.get_or_create_summary(article)
    assert mock_summarize.call_count == 1

    # Change content of the article file
    article_path.write_text(
        "---\n"
        "id: yle-1\n"
        "language: fi\n"
        'title: "Artikkeli Yksi Muokattu"\n'
        "---\n"
        "Uutta ja erilaista sisältöä artikkelissa.",
        encoding="utf-8",
    )

    # Force lazy loading to reload
    new_article = DatasetItem(article_path)

    # Cache should miss again because content_hash is different
    generator.get_or_create_summary(new_article)
    assert mock_summarize.call_count == 2


@patch("sulku.dataset.generator.create_synthetic_article")
@patch("sulku.dataset.generator.summarize_text")
def test_generate_workflow(
    mock_summarize,
    mock_create_synthetic,
    dummy_dataset_dir,
    dummy_cache_dir,
    dummy_dest_dir,
    mock_summary,
):
    """Test the complete synthetic dataset generation workflow."""
    generator = SyntheticDatasetGenerator(
        source_dir=dummy_dataset_dir, model_name="test-model", cache_dir=dummy_cache_dir
    )

    mock_summarize.return_value = mock_summary

    def side_effect(article, summary, model=None, metadata_out=None):
        if metadata_out is not None:
            metadata_out["token_usage"] = {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        return "Tämä on tekoälyn luoma synteettinen artikkeli."

    mock_create_synthetic.side_effect = side_effect

    # Generate from dummy dataset with 2 samples
    generated_paths = generator.generate(n_samples=2, seed=100, dest_dir=dummy_dest_dir)

    assert len(generated_paths) == 2
    assert mock_summarize.call_count == 2
    assert mock_create_synthetic.call_count == 2

    # Check that output directories mirror the source directory structure
    for path in generated_paths:
        assert path.exists()
        assert path.relative_to(dummy_dest_dir) in [
            Path("2021/01/0000/art1.md"),
            Path("2021/01/0000/art2.md"),
        ]

        # Verify markdown content (front matter + body)
        content = path.read_text(encoding="utf-8")
        assert content.startswith("---")
        assert "language: fi" in content
        assert "Tämä on tekoälyn luoma synteettinen artikkeli." in content
        assert "generation_details:" in content
        assert "model: test-model" in content
        assert "date: " in content
        assert "headline: Reconstructed Headline" in content
        assert "This is bullet point 1" in content
        assert "tone: Clinical / Detached" in content
        assert "token_usage:" in content
        assert "prompt_tokens: 100" in content
        assert "completion_tokens: 50" in content
        assert "total_tokens: 150" in content


@patch("sulku.dataset.generator.create_synthetic_article")
@patch("sulku.dataset.generator.summarize_text")
def test_generate_skip_existing(
    mock_summarize,
    mock_create_synthetic,
    dummy_dataset_dir,
    dummy_cache_dir,
    dummy_dest_dir,
    mock_summary,
):
    """Test that existing synthetic articles are skipped unless force=True."""
    generator = SyntheticDatasetGenerator(
        source_dir=dummy_dataset_dir, model_name="test-model", cache_dir=dummy_cache_dir
    )

    mock_summarize.return_value = mock_summary
    mock_create_synthetic.return_value = "Tämä on tekoälyn luoma synteettinen artikkeli."

    # First run: generates 2 articles
    generated_paths = generator.generate(n_samples=2, seed=100, dest_dir=dummy_dest_dir)
    assert len(generated_paths) == 2
    assert mock_summarize.call_count == 2
    assert mock_create_synthetic.call_count == 2

    # Reset mock call counts
    mock_summarize.reset_mock()
    mock_create_synthetic.reset_mock()

    # Second run with force=False: should skip both since they already exist
    generated_paths_second = generator.generate(n_samples=2, seed=100, dest_dir=dummy_dest_dir, force=False)
    assert len(generated_paths_second) == 0
    assert mock_summarize.call_count == 0
    assert mock_create_synthetic.call_count == 0

    # Third run with force=True: should regenerate both (calling LLM generator, but summary is cached)
    generated_paths_third = generator.generate(n_samples=2, seed=100, dest_dir=dummy_dest_dir, force=True)
    assert len(generated_paths_third) == 2
    assert mock_summarize.call_count == 0
    assert mock_create_synthetic.call_count == 2

