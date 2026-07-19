"""
Unit tests for Paired Dataset Utility
====================================

Tests the PairedDataset and load_paired_dataset functionalities, including
paired file discovery, lazy-loaded sequence properties (.source, .synthetic),
indexing/slicing, random sampling, and filtering.
"""

from pathlib import Path
import tempfile
from unittest.mock import patch
import pytest

from sulku.dataset.paired import ItemPair, PairedDataset, load_paired_dataset, generate_fasttext_sentence_data
from sulku.dataset.reader import DatasetItem


@pytest.fixture
def temp_paired_dirs():
    """Create temporary source and synthetic directories with matching and non-matching files."""
    with tempfile.TemporaryDirectory() as tmp_src_dir, tempfile.TemporaryDirectory() as tmp_syn_dir:
        src_path = Path(tmp_src_dir)
        syn_path = Path(tmp_syn_dir)

        # Pair 1: Exists in both source and synthetic
        src_file1 = src_path / "2021" / "01" / "article1.md"
        src_file1.parent.mkdir(parents=True, exist_ok=True)
        src_file1.write_text("---\nid: yle-1\nlanguage: fi\n---\nSource content 1", encoding="utf-8")

        syn_file1 = syn_path / "2021" / "01" / "article1.md"
        syn_file1.parent.mkdir(parents=True, exist_ok=True)
        syn_file1.write_text("---\nid: yle-1\nlanguage: fi\n---\nSynthetic content 1", encoding="utf-8")

        # Pair 2: Exists in both source and synthetic
        src_file2 = src_path / "2021" / "02" / "article2.md"
        src_file2.parent.mkdir(parents=True, exist_ok=True)
        src_file2.write_text("---\nid: yle-2\nlanguage: sv\n---\nSource content 2", encoding="utf-8")

        syn_file2 = syn_path / "2021" / "02" / "article2.md"
        syn_file2.parent.mkdir(parents=True, exist_ok=True)
        syn_file2.write_text("---\nid: yle-2\nlanguage: sv\n---\nSynthetic content 2", encoding="utf-8")

        # File 3: Source only (no synthetic counterpart)
        src_file3 = src_path / "2021" / "03" / "article3.md"
        src_file3.parent.mkdir(parents=True, exist_ok=True)
        src_file3.write_text("---\nid: yle-3\nlanguage: fi\n---\nSource only content", encoding="utf-8")

        # File 4: Synthetic only (no source counterpart)
        syn_file4 = syn_path / "2021" / "04" / "article4.md"
        syn_file4.parent.mkdir(parents=True, exist_ok=True)
        syn_file4.write_text("---\nid: yle-4\nlanguage: fi\n---\nSynthetic only content", encoding="utf-8")

        yield src_path, syn_path


def test_paired_dataset_matching(temp_paired_dirs):
    """Test that PairedDataset correctly matches files present in both directories."""
    src_dir, syn_dir = temp_paired_dirs
    ds = PairedDataset(source_dir=src_dir, synthetic_dir=syn_dir)

    # Should only discover: article1.md and article2.md
    assert len(ds) == 2

    # Check that they match article1 and article2
    relative_paths = [p[1].relative_to(syn_dir) for p in ds.paired_paths]
    assert Path("2021/01/article1.md") in relative_paths
    assert Path("2021/02/article2.md") in relative_paths
    assert Path("2021/03/article3.md") not in relative_paths
    assert Path("2021/04/article4.md") not in relative_paths


def test_paired_dataset_sequence_operations(temp_paired_dirs):
    """Test index access, slicing, and iteration."""
    src_dir, syn_dir = temp_paired_dirs
    ds = PairedDataset(source_dir=src_dir, synthetic_dir=syn_dir)

    # Index access
    item = ds[0]
    assert isinstance(item, ItemPair)
    assert isinstance(item.source, DatasetItem)
    assert isinstance(item.synthetic, DatasetItem)
    assert item.source.content == "Source content 1"
    assert item.synthetic.content == "Synthetic content 1"

    # Slicing
    sliced = ds[0:2]
    assert len(sliced) == 2
    assert sliced[0].source.content == "Source content 1"
    assert sliced[1].source.content == "Source content 2"

    # Iteration
    items = list(ds)
    assert len(items) == 2
    assert items[0].source.content == "Source content 1"
    assert items[1].source.content == "Source content 2"


def test_source_and_synthetic_properties(temp_paired_dirs):
    """Test the .source and .synthetic sequence properties."""
    src_dir, syn_dir = temp_paired_dirs
    ds = PairedDataset(source_dir=src_dir, synthetic_dir=syn_dir)

    # Test .source property
    assert len(ds.source) == 2
    assert isinstance(ds.source[0], DatasetItem)
    assert ds.source[0].content == "Source content 1"
    assert ds.source[1].content == "Source content 2"

    # Test .synthetic property
    assert len(ds.synthetic) == 2
    assert isinstance(ds.synthetic[0], DatasetItem)
    assert ds.synthetic[0].content == "Synthetic content 1"
    assert ds.synthetic[1].content == "Synthetic content 2"


def test_sampling_and_filtering(temp_paired_dirs):
    """Test sampling and filtering logic."""
    src_dir, syn_dir = temp_paired_dirs
    ds = PairedDataset(source_dir=src_dir, synthetic_dir=syn_dir)

    # Test sampling
    sampled = ds.sample(1, seed=42)
    assert len(sampled) == 1
    assert sampled[0].source.content in ["Source content 1", "Source content 2"]

    # Test filtering (keep only Swedish articles)
    sv_ds = ds.filter(lambda item: item.source.metadata.get("language") == "sv")
    assert len(sv_ds) == 1
    assert sv_ds[0].source.content == "Source content 2"


def test_load_paired_dataset_directory(temp_paired_dirs):
    """Test load_paired_dataset with direct path inputs."""
    src_dir, syn_dir = temp_paired_dirs
    ds = load_paired_dataset(path_or_model_name=syn_dir, source_dir=src_dir)
    assert len(ds) == 2


def test_load_paired_dataset_model_name(temp_paired_dirs):
    """Test load_paired_dataset with model name input, mocking DEFAULT_DEST_DIR_BASE."""
    src_dir, syn_dir = temp_paired_dirs

    with patch("sulku.dataset.paired.DEFAULT_DEST_DIR_BASE", syn_dir.parent):
        # The model name is the folder name inside syn_dir's parent
        model_name = syn_dir.name
        ds = load_paired_dataset(path_or_model_name=model_name, source_dir=src_dir)
        assert len(ds) == 2


def test_invalid_directories():
    """Test that FileNotFoundError is raised when directories do not exist."""
    with pytest.raises(FileNotFoundError):
        load_paired_dataset(path_or_model_name="/nonexistent/model", source_dir="/nonexistent/source")


def test_generate_fasttext_sentence_data():
    """Test generating FastText training data from dataset items."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_file = Path(tmp_dir) / "fasttext_data.txt"

        item1_path = Path(tmp_dir) / "item1.md"
        item1_path.write_text("Tämä on ensimmäinen lause. Tämä on toinen lause.", encoding="utf-8")
        item1 = DatasetItem(item1_path)

        item2_path = Path(tmp_dir) / "item2.md"
        item2_path.write_text("Ja tässä on kolmas lause. Neljäs lause on täällä.", encoding="utf-8")
        item2 = DatasetItem(item2_path)

        generate_fasttext_sentence_data(
            items=[item1, item2],
            label="human",
            output_path=out_file,
            min_word_count=4
        )

        assert out_file.exists()
        lines = out_file.read_text(encoding="utf-8").splitlines()

        assert len(lines) == 4
        assert all(line.startswith("__label__human ") for line in lines)
        assert "__label__human Tämä on ensimmäinen lause." in lines
        assert "__label__human Tämä on toinen lause." in lines
        assert "__label__human Ja tässä on kolmas lause." in lines
        assert "__label__human Neljäs lause on täällä." in lines
