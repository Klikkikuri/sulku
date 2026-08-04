"""
Tests for Model Specifications, Store, and Resolution Logic
===========================================================

Tests candidate selection, AmbiguousModelFileError formatting, Hugging Face
downloads via snapshot_download, local file resolution, sidecar metadata
loading/merging, and lazy resolution in ModelStore.
"""

import json
from unittest.mock import patch

import pytest
from sulku.models import (
    AmbiguousModelFileError,
    ModelMetadata,
    ModelPackage,
    ModelSpec,
    ModelStore,
    select_candidate,
)


# ── Candidate Selection & AmbiguousModelFileError Tests ───────────────────────

def test_select_candidate_single_ftz(tmp_path):
    """Test that a single .ftz candidate is deterministically selected."""
    ftz_file = tmp_path / "model.ftz"
    ftz_file.touch()

    selected = select_candidate(tmp_path)
    assert selected == ftz_file


def test_select_candidate_single_bin(tmp_path):
    """Test that a single .bin candidate is selected when no .ftz candidate exists."""
    bin_file = tmp_path / "model.bin"
    bin_file.touch()

    selected = select_candidate(tmp_path)
    assert selected == bin_file


def test_select_candidate_ftz_preferred(tmp_path):
    """Test that .ftz candidate is preferred over .bin candidate when both exist."""
    ftz_file = tmp_path / "model.ftz"
    bin_file = tmp_path / "model.bin"
    ftz_file.touch()
    bin_file.touch()

    selected = select_candidate(tmp_path)
    assert selected == ftz_file


def test_select_candidate_multiple_ftz_raises_ambiguous(tmp_path):
    """Test that multiple .ftz candidates raise AmbiguousModelFileError."""
    ftz1 = tmp_path / "model1.ftz"
    ftz2 = tmp_path / "model2.ftz"
    ftz1.touch()
    ftz2.touch()

    with pytest.raises(AmbiguousModelFileError) as exc_info:
        select_candidate(tmp_path)

    err = exc_info.value
    assert err.location == tmp_path
    assert len(err.candidates) == 2
    assert "model1.ftz" in str(err)
    assert "model2.ftz" in str(err)


def test_select_candidate_multiple_bin_no_ftz_raises_ambiguous(tmp_path):
    """Test that multiple .bin candidates without .ftz raise AmbiguousModelFileError."""
    bin1 = tmp_path / "model1.bin"
    bin2 = tmp_path / "model2.bin"
    bin1.touch()
    bin2.touch()

    with pytest.raises(AmbiguousModelFileError):
        select_candidate(tmp_path)


def test_select_candidate_none_found(tmp_path):
    """Test that select_candidate returns None when no valid candidates exist."""
    assert select_candidate(tmp_path) is None
    non_existent = tmp_path / "non_existent_dir"
    assert select_candidate(non_existent) is None

    # Directory with unrelated files
    (tmp_path / "readme.txt").touch()
    assert select_candidate(tmp_path) is None


def test_model_store_select_candidate_none_found_raises(tmp_path):
    """Test that ModelStore._select_candidate raises AmbiguousModelFileError when no candidates match."""
    with pytest.raises(AmbiguousModelFileError):
        ModelStore._select_candidate(tmp_path, [])


def test_ambiguous_model_file_error_formatting(tmp_path):
    """Test string formatting and attribute retention of AmbiguousModelFileError."""
    c1 = tmp_path / "b_model.ftz"
    c2 = tmp_path / "a_model.ftz"
    err = AmbiguousModelFileError(tmp_path, [c1, c2])

    assert err.location == tmp_path
    assert err.candidates == [c1, c2]
    assert "a_model.ftz, b_model.ftz" in str(err)


# ── Hugging Face Resolution Tests ─────────────────────────────────────────────

def test_resolve_hf_source_invalid_uri(tmp_path):
    """Test that invalid hf:// URIs raise ValueError."""
    store = ModelStore(models=[], cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Invalid hf:// URI"):
        store._resolve_hf_source("hf://invalid_repo_name_without_owner")


@patch("sulku.models.snapshot_download")
def test_resolve_hf_source_repo_root(mock_snapshot, tmp_path):
    """Test resolving repo root URI (hf://owner/repo)."""
    repo_dir = tmp_path / "hf_cache" / "owner_repo"
    repo_dir.mkdir(parents=True)
    model_file = repo_dir / "model.ftz"
    model_file.touch()

    mock_snapshot.return_value = str(repo_dir)

    store = ModelStore(models=[], cache_dir=tmp_path)
    path, pkg_dir = store._resolve_hf_source("hf://owner/repo")

    assert path == model_file
    assert pkg_dir == repo_dir
    mock_snapshot.assert_called_once()
    assert mock_snapshot.call_args.kwargs["repo_id"] == "owner/repo"


@patch("sulku.models.snapshot_download")
def test_resolve_hf_source_subpath_directory(mock_snapshot, tmp_path):
    """Test resolving subfolder URI (hf://owner/repo/subfolder)."""
    repo_dir = tmp_path / "hf_cache" / "owner_repo"
    sub_dir = repo_dir / "subfolder"
    sub_dir.mkdir(parents=True)
    model_file = sub_dir / "classifier.bin"
    model_file.touch()

    mock_snapshot.return_value = str(repo_dir)

    store = ModelStore(models=[], cache_dir=tmp_path)
    path, pkg_dir = store._resolve_hf_source("hf://owner/repo/subfolder")

    assert path == model_file
    assert pkg_dir == sub_dir


@patch("sulku.models.snapshot_download")
def test_resolve_hf_source_explicit_filename(mock_snapshot, tmp_path):
    """Test resolving explicit filename URI (hf://owner/repo/subfolder/custom.ftz)."""
    repo_dir = tmp_path / "hf_cache" / "owner_repo"
    sub_dir = repo_dir / "subfolder"
    sub_dir.mkdir(parents=True)
    model_file = sub_dir / "custom.ftz"
    model_file.touch()

    mock_snapshot.return_value = str(repo_dir)

    store = ModelStore(models=[], cache_dir=tmp_path)
    path, pkg_dir = store._resolve_hf_source("hf://owner/repo/subfolder/custom.ftz")

    assert path == model_file
    assert pkg_dir == sub_dir
    assert "subfolder/custom.ftz" in mock_snapshot.call_args.kwargs["allow_patterns"]


@patch("sulku.models.snapshot_download")
def test_resolve_hf_source_explicit_filename_not_found(mock_snapshot, tmp_path):
    """Test that FileNotFoundError is raised if explicit filename does not exist after download."""
    repo_dir = tmp_path / "hf_cache" / "owner_repo"
    repo_dir.mkdir(parents=True)

    mock_snapshot.return_value = str(repo_dir)

    store = ModelStore(models=[], cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="Explicit model file not found"):
        store._resolve_hf_source("hf://owner/repo/missing.ftz")


@patch("sulku.models.snapshot_download")
def test_resolve_hf_source_no_models_found(mock_snapshot, tmp_path):
    """Test that FileNotFoundError is raised when repo contains no supported model binaries."""
    repo_dir = tmp_path / "hf_cache" / "owner_repo"
    repo_dir.mkdir(parents=True)
    (repo_dir / "README.md").touch()

    mock_snapshot.return_value = str(repo_dir)

    store = ModelStore(models=[], cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="No fastText model file"):
        store._resolve_hf_source("hf://owner/repo")


# ── Local Source Resolution Tests ─────────────────────────────────────────────

def test_resolve_local_source_directory(tmp_path):
    """Test resolving a local directory source containing a model file."""
    model_dir = tmp_path / "local_model"
    model_dir.mkdir()
    model_file = model_dir / "model.ftz"
    model_file.touch()

    store = ModelStore(models=[], cache_dir=tmp_path)
    path, pkg_dir = store._resolve_local_source(str(model_dir))

    assert path == model_file
    assert pkg_dir == model_dir


def test_resolve_local_source_direct_file(tmp_path):
    """Test resolving a local direct file path source."""
    model_file = tmp_path / "direct.ftz"
    model_file.touch()

    store = ModelStore(models=[], cache_dir=tmp_path)
    path, pkg_dir = store._resolve_local_source(f"file://{model_file}")

    assert path == model_file
    assert pkg_dir == tmp_path


def test_resolve_local_source_nonexistent_file(tmp_path):
    """Test that resolving a non-existent local file raises FileNotFoundError."""
    store = ModelStore(models=[], cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="Model file does not exist"):
        store._resolve_local_source(str(tmp_path / "missing.ftz"))


def test_resolve_local_source_directory_no_models(tmp_path):
    """Test that resolving a local directory without model files raises FileNotFoundError."""
    empty_dir = tmp_path / "empty_model_dir"
    empty_dir.mkdir()

    store = ModelStore(models=[], cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="No fastText model file"):
        store._resolve_local_source(str(empty_dir))


# ── Sidecar Metadata Loading & Merging Tests ──────────────────────────────────

def test_metadata_sidecar_package_dir(tmp_path):
    """Test loading sidecar metadata.json located in package directory."""
    model_file = tmp_path / "model.ftz"
    model_file.touch()
    sidecar = tmp_path / "metadata.json"
    sidecar.write_text(json.dumps({"mean": 0.42, "description": "Package dir metadata"}))

    spec = ModelSpec(name="test_m", source=model_file)
    store = ModelStore(models=[spec], cache_dir=tmp_path)

    metadata = store._resolve_metadata(spec, model_file, tmp_path)
    assert metadata.mean == 0.42
    assert metadata.description == "Package dir metadata"


def test_metadata_sidecar_same_stem(tmp_path):
    """Test loading sidecar metadata named <model_stem>.json alongside model binary."""
    model_file = tmp_path / "custom_model.ftz"
    model_file.touch()
    sidecar = tmp_path / "custom_model.json"
    sidecar.write_text(json.dumps({"std": 0.15, "description": "Stem metadata"}))

    spec = ModelSpec(name="test_m", source=model_file)
    store = ModelStore(models=[spec], cache_dir=tmp_path)

    metadata = store._resolve_metadata(spec, model_file, None)
    assert metadata.std == 0.15
    assert metadata.description == "Stem metadata"


def test_metadata_defaults(tmp_path):
    """Test fallback to default ModelMetadata when no sidecar or inline metadata is provided."""
    model_file = tmp_path / "plain.ftz"
    model_file.touch()

    spec = ModelSpec(name="test_m", source=model_file)
    store = ModelStore(models=[spec], cache_dir=tmp_path)

    metadata = store._resolve_metadata(spec, model_file, tmp_path)
    assert metadata.mean == 0.5
    assert metadata.std == 0.10


def test_metadata_corrupt_sidecar_fallback(tmp_path):
    """Test fallback to default ModelMetadata when sidecar metadata JSON is corrupt."""
    model_file = tmp_path / "corrupt.ftz"
    model_file.touch()
    sidecar = tmp_path / "metadata.json"
    sidecar.write_text("{invalid_json:")

    spec = ModelSpec(name="test_m", source=model_file)
    store = ModelStore(models=[spec], cache_dir=tmp_path)

    metadata = store._resolve_metadata(spec, model_file, tmp_path)
    assert metadata.mean == 0.5


def test_metadata_spec_override(tmp_path):
    """Test that inline spec.metadata overrides sidecar metadata fields."""
    model_file = tmp_path / "override.ftz"
    model_file.touch()
    sidecar = tmp_path / "metadata.json"
    sidecar.write_text(json.dumps({"mean": 0.42, "description": "Original description"}))

    spec = ModelSpec(
        name="test_m",
        source=model_file,
        metadata=ModelMetadata(mean=0.88),
    )
    store = ModelStore(models=[spec], cache_dir=tmp_path)

    metadata = store._resolve_metadata(spec, model_file, tmp_path)
    assert metadata.mean == 0.88
    assert metadata.description == "Original description"


# ── ModelStore Registry & Lazy Resolution Tests ──────────────────────────────

def test_model_store_lazy_caching(tmp_path):
    """Test lazy resolution and caching of ModelPackage in ModelStore."""
    model_file = tmp_path / "cached_model.ftz"
    model_file.touch()

    spec = ModelSpec(name="cached_m", source=model_file)
    store = ModelStore(models=[spec], cache_dir=tmp_path)

    with patch.object(store, "_resolve_package", wraps=store._resolve_package) as mock_resolve:
        pkg1 = store.get("cached_m")
        assert isinstance(pkg1, ModelPackage)
        assert pkg1.name == "cached_m"
        assert pkg1.path == model_file
        assert mock_resolve.call_count == 1

        # Second call returns cached package without calling _resolve_package again
        pkg2 = store.get("cached_m")
        assert pkg2 is pkg1
        assert mock_resolve.call_count == 1


def test_model_store_get_spec(tmp_path):
    """Test ModelStore.get_spec method."""
    spec = ModelSpec(name="spec_m", source="hf://owner/repo")
    store = ModelStore(models=[spec], cache_dir=tmp_path)

    assert store.get_spec("spec_m") == spec
    assert store.get_spec("non_existent") is None


def test_model_store_unconfigured_model(tmp_path):
    """Test that requesting an unconfigured model raises ValueError."""
    store = ModelStore(models=[], cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Model 'unknown' not configured"):
        store.get("unknown")


def test_model_store_model_names(tmp_path):
    """Test ModelStore.model_names property."""
    s1 = ModelSpec(name="m1", source="file://a.ftz")
    s2 = ModelSpec(name="m2", source="file://b.ftz")
    store = ModelStore(models=[s1, s2], cache_dir=tmp_path)

    assert store.model_names == ["m1", "m2"]


def test_model_metadata_languages(tmp_path):
    """Test default and custom languages in ModelMetadata."""
    meta_default = ModelMetadata()
    assert meta_default.languages == []

    meta_custom = ModelMetadata(languages=["en", "sv"])
    assert meta_custom.languages == ["en", "sv"]

    model_file = tmp_path / "lang_test.ftz"
    model_file.touch()
    sidecar = tmp_path / "metadata.json"
    sidecar.write_text(json.dumps({"languages": ["en", "fi"]}))

    spec = ModelSpec(name="lang_m", source=model_file)
    store = ModelStore(models=[spec], cache_dir=tmp_path)
    metadata = store._resolve_metadata(spec, model_file, tmp_path)
    assert metadata.languages == ["en", "fi"]

