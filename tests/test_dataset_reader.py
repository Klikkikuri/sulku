"""
Unit tests for Dataset Reader Utility
=====================================

Tests the FileDataset and DatasetItem functionalities, including lazy-loaded
metadata, content filtering, front matter stripping, and sampling.
"""

import json
from pathlib import Path
import tempfile
import pytest
from sulku.dataset.reader import (
    FileDataset,
    DatasetItem,
    json_metadata_loader,
    yaml_front_matter_loader,
)


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory with various dummy dataset files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # File 1: Text file with YAML front matter
        file1 = tmp_path / "article_1.md"
        file1.write_text(
            "---\n"
            "id: yle-1\n"
            "title: \"Yle News Item 1\"\n"
            "authors:\n"
            "  - name: Author A\n"
            "    organization: Yle\n"
            "subjects:\n"
            "  - news\n"
            "  - finland\n"
            "---\n"
            "Body content of article 1 starts here.\n"
            "This is the actual text.",
            encoding="utf-8"
        )

        # File 2: Simple text file (no front matter)
        file2 = tmp_path / "article_2.txt"
        file2.write_text("Just simple text without front matter.", encoding="utf-8")

        # File 3: JSON file
        file3 = tmp_path / "article_3.json"
        json_data = {
            "id": "yle-3",
            "title": "JSON Article",
            "content": "This is JSON text content."
        }
        file3.write_text(json.dumps(json_data), encoding="utf-8")

        # Nested Directory File 4
        nested_dir = tmp_path / "nested" / "dir"
        nested_dir.mkdir(parents=True, exist_ok=True)
        file4 = nested_dir / "article_4.md"
        file4.write_text(
            "---\n"
            "id: nested-4\n"
            "title: Nested Item\n"
            "---\n"
            "Nested body text.",
            encoding="utf-8"
        )

        yield tmp_path


def test_file_discovery_all(temp_dataset_dir):
    """Test that all files are correctly discovered recursively."""
    ds = FileDataset(temp_dataset_dir, pattern="*")
    # Should discover: article_1.md, article_2.txt, article_3.json, nested/dir/article_4.md
    assert len(ds) == 4
    paths = [item.path.name for item in ds]
    assert "article_1.md" in paths
    assert "article_2.txt" in paths
    assert "article_3.json" in paths
    assert "article_4.md" in paths


def test_file_discovery_non_recursive(temp_dataset_dir):
    """Test non-recursive file discovery."""
    ds = FileDataset(temp_dataset_dir, pattern="*", recursive=False)
    # Should not discover nested/dir/article_4.md
    assert len(ds) == 3
    paths = [item.path.name for item in ds]
    assert "article_4.md" not in paths


def test_pattern_filtering(temp_dataset_dir):
    """Test filtering by glob pattern."""
    ds = FileDataset(temp_dataset_dir, pattern="*.md")
    assert len(ds) == 2
    paths = [item.path.name for item in ds]
    assert "article_1.md" in paths
    assert "article_4.md" in paths


def test_custom_filtering(temp_dataset_dir):
    """Test custom filtering function."""
    def filter_no_txt(path: Path) -> bool:
        return path.suffix != ".txt"

    ds = FileDataset(temp_dataset_dir, pattern="*", filter_fn=filter_no_txt)
    assert len(ds) == 3
    paths = [item.path.name for item in ds]
    assert "article_2.txt" not in paths


def test_sequence_operations(temp_dataset_dir):
    """Test index-based access, slicing, and iteration."""
    ds = FileDataset(temp_dataset_dir, pattern="*", recursive=False)
    # Total 3 files in main dir. Let's index them.
    assert isinstance(ds[0], DatasetItem)
    
    # Slicing
    sliced = ds[0:2]
    assert isinstance(sliced, list)
    assert len(sliced) == 2
    assert isinstance(sliced[0], DatasetItem)

    # Iteration
    items = list(ds)
    assert len(items) == 3
    assert all(isinstance(item, DatasetItem) for item in items)


def test_content_lazy_loading_and_front_matter(temp_dataset_dir):
    """Test that text content is lazy loaded and front matter is skipped when requested."""
    # File 1: Has front matter
    item1 = DatasetItem(temp_dataset_dir / "article_1.md")
    expected_content = "Body content of article 1 starts here.\nThis is the actual text."
    assert item1.content == expected_content
    # __str__ should match content
    assert str(item1) == expected_content

    # File 2: No front matter
    item2 = DatasetItem(temp_dataset_dir / "article_2.txt")
    assert item2.content == "Just simple text without front matter."
    assert str(item2) == "Just simple text without front matter."


def test_metadata_lazy_loading(temp_dataset_dir):
    """Test that metadata is only loaded on demand (lazily)."""
    # By default, YAML front matter metadata should be loaded
    item = DatasetItem(temp_dataset_dir / "article_1.md")
    assert item.metadata["id"] == "yle-1"
    assert item.metadata["title"] == "Yle News Item 1"

    # A non-front-matter file still yields empty metadata by default
    item_no_fm = DatasetItem(temp_dataset_dir / "article_2.txt")
    assert item_no_fm.metadata == {}

    # With YAML front matter loader
    item_yaml = DatasetItem(temp_dataset_dir / "article_1.md", yaml_front_matter_loader)
    meta = item_yaml.metadata
    assert meta["id"] == "yle-1"
    assert meta["title"] == "Yle News Item 1"
    assert len(meta["authors"]) == 1
    assert meta["authors"][0]["name"] == "Author A"
    assert meta["authors"][0]["organization"] == "Yle"
    assert meta["subjects"] == ["news", "finland"]


def test_json_metadata_loader(temp_dataset_dir):
    """Test the JSON metadata loader."""
    item = DatasetItem(temp_dataset_dir / "article_3.json", json_metadata_loader)
    meta = item.metadata
    assert meta["id"] == "yle-3"
    assert meta["title"] == "JSON Article"


def test_sampling(temp_dataset_dir):
    """Test random sampling from dataset."""
    ds = FileDataset(temp_dataset_dir, pattern="*")
    
    # Check basic sampling
    sampled = ds.sample(2)
    assert len(sampled) == 2
    assert isinstance(sampled[0], DatasetItem)

    # Check reproducibility with seed
    sampled1 = ds.sample(2, seed=42)
    sampled2 = ds.sample(2, seed=42)
    assert [s.path for s in sampled1] == [s.path for s in sampled2]

    # Check too large sample size raises error
    with pytest.raises(ValueError):
        ds.sample(10)


def test_extend_with_metadata_loader(temp_dataset_dir):
    """Test extending an existing dataset using with_metadata_loader."""
    # Start with default YAML front matter loading
    ds = FileDataset(temp_dataset_dir, pattern="*.md")
    assert ds[0].metadata != {}

    # Extend it with the YAML loader
    ds_extended = ds.with_metadata_loader(yaml_front_matter_loader)
    assert ds_extended[0].metadata != {}
    assert ds_extended[0].metadata["title"] in ["Yle News Item 1", "Nested Item"]

    # Original dataset remains unchanged and still has metadata
    assert ds[0].metadata != {}
