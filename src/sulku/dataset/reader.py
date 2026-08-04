"""
Dataset Reader Utility
======================

This module provides utilities to load and traverse datasets consisting of files.
It discovers files under a given dataset path and represents them as lazy-loading
dataset items. It supports custom metadata loaders, filtering, and sampling.
"""

import json
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict, Union

import yaml

from sulku.dataset.base import BaseDataset
from sulku.dataset.mixins import TextDatasetFilterMixin


class DatasetItem:
    """
    Represents a single item in the dataset, corresponding to a file path.

    Provides lazy access to file content and metadata.
    """

    def __init__(
        self,
        path: Path,
        metadata_loader: Optional[Callable[[Path], dict[str, Any]]] = None,
    ):
        """
        Initialize the dataset item.

        :param path: Path to the dataset item file.
        :type path: Path
        :param metadata_loader: Optional callable to load metadata lazily.
        :type metadata_loader: Callable[[Path], dict[str, Any]], optional
        """
        self._path = path
        self._metadata_loader = (
            metadata_loader if metadata_loader is not None else yaml_front_matter_loader
        )
        self._metadata: Optional[dict[str, Any]] = None

    @property
    def path(self) -> Path:
        """
        Get the file path of the dataset item.

        :return: Path to the file.
        :rtype: Path
        """
        return self._path

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Lazy-load and cache the metadata for this item.

        By default, YAML front matter metadata is parsed from the file.

        :return: A dictionary containing metadata.
        :rtype: dict[str, Any]
        """
        if self._metadata is None:
            try:
                self._metadata = self._metadata_loader(self._path)
            except Exception as e:
                self._metadata = {"error": str(e)}
        assert self._metadata is not None
        return self._metadata

    @property
    def content(self) -> str:
        """
        Lazy-load and return the text content of the file.

        If the file content begins with a YAML front matter block (enclosed in
        '---'), the front matter is skipped and only the subsequent body
        content is returned.

        :return: File content as a string.
        :rtype: str
        """
        raw = self._path.read_text(encoding="utf-8")
        if raw.startswith("---"):
            parts = raw.split("---", 2)
            if len(parts) >= 3:
                return parts[2].lstrip("\r\n")
        return raw

    @property
    def bytes_content(self) -> bytes:
        """
        Lazy-load and return the raw bytes content of the file.

        :return: File content as bytes.
        :rtype: bytes
        """
        return self._path.read_bytes()

    def __str__(self) -> str:
        """
        String representation of the DatasetItem.

        :return: String representation of the item.
        :rtype: str
        """
        return self.content

    def __repr__(self) -> str:
        """
        Represent the DatasetItem.

        :return: String representation of the item.
        :rtype: str
        """
        return f"DatasetItem(path={self._path})"


class FrontMatterAuthor(TypedDict, total=False):
    """Common nested author structure found in front matter."""

    name: str
    organization: str


class FrontMatter(TypedDict, total=False):
    """Common front matter fields for markdown and text content."""

    title: str
    description: str
    date: str
    slug: str
    draft: bool
    lang: str
    category: str
    categories: list[str]
    tags: list[str]
    authors: list[str | FrontMatterAuthor]


class FileDataset(BaseDataset[Path, DatasetItem], TextDatasetFilterMixin):
    """
    A collection of file items discovered from a dataset directory path.

    It discovers files matching a glob pattern and supports lazy-loaded metadata,
    filtering, sequence operations (indexing, slicing, length), and sampling.
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        pattern: str = "*",
        recursive: bool = True,
        metadata_loader: Optional[Callable[[Path], dict[str, Any]]] = None,
        item_class: type[DatasetItem] = DatasetItem,
        filter_fn: Optional[Callable[[Path], bool]] = None,
        _files: Optional[list[Path]] = None,
    ):
        """
        Discover files in the dataset path and initialize the dataset.

        :param root_path: Root directory of the dataset.
        :type root_path: Union[str, Path]
        :param pattern: Glob pattern to filter files. Defaults to "*".
        :type pattern: str
        :param recursive: If True, search directory recursively. Defaults to True.
        :type recursive: bool
        :param metadata_loader: Optional callable to load metadata lazily.
        :type metadata_loader: Callable[[Path], dict[str, Any]], optional
        :param item_class: Class to instantiate for dataset items. Defaults to DatasetItem.
        :type item_class: type[DatasetItem]
        :param filter_fn: Optional callable to filter paths.
        :type filter_fn: Callable[[Path], bool], optional
        :param _files: Internal parameter to pass pre-discovered file paths.
        """
        self.root_path = Path(root_path).resolve()
        if not self.root_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.root_path}")

        self.pattern = pattern
        self.recursive = recursive
        self.metadata_loader = (
            metadata_loader if metadata_loader is not None else yaml_front_matter_loader
        )
        self.item_class = item_class
        self.filter_fn = filter_fn

        if _files is None:
            if self.recursive:
                discovered = self.root_path.glob(f"**/{self.pattern}")
            else:
                discovered = self.root_path.glob(self.pattern)

            self._files = sorted(
                [
                    f
                    for f in discovered
                    if f.is_file()
                    and (self.filter_fn(f) if self.filter_fn is not None else True)
                ]
            )
        else:
            self._files = _files

        def loader(p: Path) -> DatasetItem:
            return self.item_class(p, self.metadata_loader)

        super().__init__(self._files, loader)

    def _clone(self, items: list[Path]) -> "FileDataset":
        """
        Return a new instance of FileDataset with the given list of file paths.

        :param items: List of file paths for the cloned dataset.
        :return: A new FileDataset instance.
        """
        return FileDataset(
            root_path=self.root_path,
            pattern=self.pattern,
            recursive=self.recursive,
            metadata_loader=self.metadata_loader,
            item_class=self.item_class,
            filter_fn=self.filter_fn,
            _files=items,
        )

    def with_metadata_loader(
        self, metadata_loader: Callable[[Path], dict[str, Any]]
    ) -> "FileDataset":
        """
        Return a new FileDataset instance with the specified metadata loader.

        :param metadata_loader: A callable that loads metadata from a file path.
        :return: A new FileDataset instance.
        """
        return FileDataset(
            root_path=self.root_path,
            pattern=self.pattern,
            recursive=self.recursive,
            metadata_loader=metadata_loader,
            item_class=self.item_class,
            filter_fn=self.filter_fn,
            _files=self._files,
        )


def json_metadata_loader(path: Path) -> dict[str, Any]:
    """
    Parse metadata from a JSON file.

    :param path: Path to the JSON file.
    :return: A dictionary containing the JSON content.
    """
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def yaml_front_matter_loader(path: Path) -> dict[str, Any]:
    """
    Parse YAML front matter from a Markdown or text file.

    :param path: Path to the markdown file.
    :return: A dictionary containing the parsed metadata.
    """
    metadata: dict[str, Any] = {}
    if not path.exists():
        return metadata

    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return metadata

    if not content.startswith("---"):
        return metadata

    parts = content.split("---", 2)
    if len(parts) < 3:
        return metadata

    yaml_part = parts[1].strip()

    try:
        parsed = yaml.safe_load(yaml_part)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return metadata


def min_words_filter(min_words: int) -> Callable[[DatasetItem], bool]:
    """
    Create a predicate to filter out stubs / short articles.

    :param min_words: Minimum number of words to keep an article.
    :return: Predicate callable.
    """
    from sulku.utils import count_words

    return lambda item: count_words(item.content) >= min_words


def non_empty_filter() -> Callable[[DatasetItem], bool]:
    """
    Create a predicate to filter out empty articles.

    :return: Predicate callable.
    """
    from sulku.utils import count_words

    return lambda item: count_words(item.content) > 0


def language_filter(
    languages: Union[str, list[str], set[str], Any],
) -> Callable[[DatasetItem], bool]:
    """
    Create a predicate to filter articles by language.

    :param languages: A single language string or a collection of allowed languages.
    :return: Predicate callable.
    """
    if isinstance(languages, str):
        allowed = {languages.lower().strip()}
    else:
        allowed = {lang.lower().strip() for lang in languages}

    def predicate(item: DatasetItem) -> bool:
        metadata = item.metadata
        lang_val = metadata.get("language") or metadata.get("lang")
        if not lang_val or not isinstance(lang_val, str):
            return False
        return lang_val.lower().strip() in allowed

    return predicate
