"""
Model Specifications & Storage
===============================

Provides ModelSpec, ModelMetadata, ModelPackage, and ModelStore for managing
model packages, sidecar metadata, and lazy resolution.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Optional, Union

from huggingface_hub import snapshot_download
from niitti import get_logger
from pydantic import BaseModel, Field

logger = get_logger(__name__)

MODEL_EXTENSIONS = (".ftz", ".bin")


# ── Model Metadata & Specification ───────────────────────────────────────────

class ModelMetadata(BaseModel):
    mean: float = 0.5
    std: float = 0.10
    labels: Dict[str, str] = Field(
        default_factory=lambda: {
            "ai": "__label__synthetic",
            "human": "__label__human",
        }
    )
    description: Optional[str] = None
    created_at: Optional[datetime] = None


class ModelSpec(BaseModel):
    name: str = Field(..., description="Unique model name/identifier (e.g. 'gemini-3.1-flash-lite')")
    source: Union[Path, str] = Field(..., description="Local path or URI (e.g. hf:// or /app/data/models/...)")
    metadata: Optional[ModelMetadata] = Field(default=None, description="Optional inline metadata override")


@dataclass
class ModelPackage:
    """Resolved model package containing local binary file path and metadata."""

    name: str
    path: Path
    metadata: ModelMetadata


class AmbiguousModelFileError(Exception):
    """Raised when multiple candidate model files are found and none can be
    unambiguously preferred."""

    def __init__(self, location: Path, candidates: list[Path]):
        self.location = location
        self.candidates = candidates
        names = ", ".join(sorted(p.name for p in candidates))
        super().__init__(
            f"Multiple model files found in {location} and none could be "
            f"unambiguously selected: {names}. Specify an exact filename in "
            f"the model source (e.g. hf://owner/repo/path/to/model.ftz)."
        )


# ── ModelStore Registry & Lazy Resolver ───────────────────────────────────────

class ModelStore:
    """Registry and lazy package resolver for models configured via Settings."""

    def __init__(
        self,
        models: list[ModelSpec],
        cache_dir: Path,
    ):
        self._specs: dict[str, ModelSpec] = {m.name: m for m in models}
        self.cache_dir = cache_dir
        self._resolved_packages: dict[str, ModelPackage] = {}

    def get_spec(self, name: str) -> ModelSpec | None:
        """Get ModelSpec by name."""
        return self._specs.get(name)

    def get(self, name: str) -> ModelPackage:
        """
        Get and resolve high-level ModelPackage for a model name.

        Lazily resolves local file paths or downloads Hugging Face repositories (hf://)
        on first access, caching the resulting ModelPackage in memory.
        """
        if name not in self._resolved_packages:
            spec = self._specs.get(name)
            if not spec:
                raise ValueError(f"Model '{name}' not configured in ModelStore")
            self._resolved_packages[name] = self._resolve_package(spec)
        return self._resolved_packages[name]

    @property
    def model_names(self) -> list[str]:
        """List all configured model names."""
        return list(self._specs.keys())

    def _resolve_package(self, spec: ModelSpec) -> ModelPackage:
        """
        Internal helper to resolve local path or hf:// model repository,
        locate the model binary file (.ftz / .bin), and load sidecar metadata.json.
        """
        source_str = str(spec.source)

        if source_str.startswith("hf://"):
            model_path, package_dir = self._resolve_hf_source(source_str)
        else:
            model_path, package_dir = self._resolve_local_source(source_str)

        metadata = self._resolve_metadata(spec, model_path, package_dir)

        return ModelPackage(
            name=spec.name,
            path=model_path,
            metadata=metadata,
        )

    # ── HF resolution ────────────────────────────────────────────────────────

    def _resolve_hf_source(self, uri: str) -> tuple[Path, Optional[Path]]:
        """
        Download/cache Hugging Face repository subfolder via snapshot_download.

        URI format is always hf://<owner>/<repo>[/<subpath>], where <subpath>
        may point at a directory or an exact file. Bare (non-namespaced) repo
        IDs are not supported.
        """
        raw_path = uri[5:]  # Strip hf://
        parts = raw_path.split("/")

        if len(parts) < 2:
            raise ValueError(
                f"Invalid hf:// URI: {uri!r}. Expected format is "
                f"hf://<owner>/<repo>[/<subpath>]."
            )

        repo_id = f"{parts[0]}/{parts[1]}"
        subpath_parts = parts[2:]
        subpath = "/".join(subpath_parts) if subpath_parts else ""
        explicit_filename = subpath if subpath.endswith(MODEL_EXTENSIONS) else None

        # Build allow_patterns to avoid downloading unrelated repo files.
        # snapshot_download matches these against the full relative path in
        # the repo, so we must cover any depth under the target directory.
        if explicit_filename:
            allow_patterns = [explicit_filename]
            target_dir = str(Path(explicit_filename).parent)
        elif subpath:
            target_dir = subpath
            allow_patterns = [f"{subpath}/*{ext}" for ext in MODEL_EXTENSIONS] + [
                f"{subpath}/**/*{ext}" for ext in MODEL_EXTENSIONS
            ]
        else:
            target_dir = ""
            allow_patterns = [f"*{ext}" for ext in MODEL_EXTENSIONS] + [
                f"**/*{ext}" for ext in MODEL_EXTENSIONS
            ]

        # Always fetch metadata.json alongside, wherever it lives.
        allow_patterns = allow_patterns + ["*.json", "**/*.json"]

        downloaded_dir = Path(snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns))
        package_dir = downloaded_dir / target_dir if target_dir else downloaded_dir

        if explicit_filename:
            model_path = downloaded_dir / explicit_filename
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Explicit model file not found after download: {model_path} "
                    f"(uri: {uri})"
                )
            return model_path, package_dir

        # Search only within target_dir first (not the whole repo) to avoid
        # picking up an unrelated file elsewhere in the download.
        search_root = package_dir if package_dir.exists() else downloaded_dir
        candidates = [p for ext in MODEL_EXTENSIONS for p in search_root.rglob(f"*{ext}")]
        if not candidates:
            raise FileNotFoundError(
                f"No fastText model file ({'/'.join(MODEL_EXTENSIONS)}) found "
                f"under {search_root} for URI: {uri}"
            )

        model_path = self._select_candidate(search_root, candidates)
        return model_path, package_dir

    # ── Local resolution ─────────────────────────────────────────────────────

    def _resolve_local_source(self, path_str: str) -> tuple[Path, Optional[Path]]:
        """
        Resolve local file path or package directory.
        """
        clean_str = path_str[7:] if path_str.startswith("file://") else path_str
        local_path = Path(clean_str)

        if local_path.is_dir():
            package_dir = local_path
            candidates = [p for ext in MODEL_EXTENSIONS for p in package_dir.glob(f"*{ext}")]
            if not candidates:
                raise FileNotFoundError(
                    f"No fastText model file ({'/'.join(MODEL_EXTENSIONS)}) found "
                    f"in package directory: {package_dir}"
                )
            model_path = self._select_candidate(package_dir, candidates)
        else:
            model_path = local_path
            package_dir = local_path.parent if local_path.parent.exists() else None

        if not model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")

        return model_path, package_dir

    # ── Shared candidate selection ───────────────────────────────────────────

    @staticmethod
    def _select_candidate(location: Path, candidates: list[Path]) -> Path:
        """
        Deterministically select a single model file from candidates found at
        `location`.

        Preference order:
          1. If exactly one .ftz file exists, use it.
          2. Else if no .ftz files but exactly one .bin file exists, use it.
          3. Otherwise (multiple .ftz, or multiple .bin with no .ftz), the
             choice is ambiguous and we raise rather than guess.
        """
        ftz_candidates = sorted(p for p in candidates if p.suffix == ".ftz")
        bin_candidates = sorted(p for p in candidates if p.suffix == ".bin")

        if len(ftz_candidates) == 1:
            return ftz_candidates[0]
        if ftz_candidates:
            raise AmbiguousModelFileError(location, ftz_candidates)

        if len(bin_candidates) == 1:
            return bin_candidates[0]
        if bin_candidates:
            raise AmbiguousModelFileError(location, bin_candidates)

        # Unreachable given callers only invoke this with a non-empty list,
        # but guard anyway.
        raise AmbiguousModelFileError(location, candidates)

    # ── Metadata resolution ──────────────────────────────────────────────────

    def _resolve_metadata(self, spec: ModelSpec, model_path: Path, package_dir: Optional[Path]) -> ModelMetadata:
        """
        Load ModelMetadata from inline spec.metadata or sidecar metadata.json.
        """
        metadata_dict = {}

        sidecar_candidates = []
        if package_dir and (package_dir / "metadata.json").exists():
            sidecar_candidates.append(package_dir / "metadata.json")
        sidecar_same_stem = model_path.with_suffix(".json")
        if sidecar_same_stem.exists():
            sidecar_candidates.append(sidecar_same_stem)

        for candidate in sidecar_candidates:
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    metadata_dict = json.load(f)
                logger.debug(f"Loaded sidecar metadata from {candidate}")
                break
            except Exception as e:
                logger.warning(f"Failed to read sidecar metadata from {candidate}: {e}")

        sidecar_metadata = ModelMetadata(**metadata_dict) if metadata_dict else ModelMetadata()

        if spec.metadata is not None:
            override_data = spec.metadata.model_dump(exclude_unset=True)
            current_data = sidecar_metadata.model_dump()
            current_data.update(override_data)
            return ModelMetadata(**current_data)

        return sidecar_metadata


def spec_name_from_repo(repo_id: str) -> str:
    """Helper to get fallback filename stem from HF repo_id."""
    return repo_id.split("/")[-1]
