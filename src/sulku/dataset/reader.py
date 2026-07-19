"""
Dataset Reader Utility
======================

This module provides utilities to load and traverse datasets consisting of files.
It discovers files under a given dataset path and represents them as lazy-loading
dataset items. It supports custom metadata loaders, filtering, and sampling.
"""

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypedDict, Union, overload

import yaml


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


class FileDataset(Sequence[DatasetItem]):
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
    ):
        """
        Discover files in the dataset path and initialize the dataset.

        Only stores the list of discovered file paths to be memory-efficient.

        :param root_path: Root directory of the dataset.
        :type root_path: Union[str, Path]
        :param pattern: Glob pattern to filter files. Defaults to "*".
        :type pattern: str
        :param recursive: If True, search directory recursively. Defaults to True.
        :type recursive: bool
        :param metadata_loader: Optional callable to load metadata lazily.
            Defaults to yaml_front_matter_loader.
        :type metadata_loader: Callable[[Path], dict[str, Any]], optional
        :param item_class: Class to instantiate for dataset items. Defaults to DatasetItem.
        :type item_class: type[DatasetItem]
        :param filter_fn: Optional callable to filter paths.
        :type filter_fn: Callable[[Path], bool], optional
        :raises FileNotFoundError: If the root_path does not exist.
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

        # Discover files
        if self.recursive:
            discovered = self.root_path.glob(f"**/{self.pattern}")
        else:
            discovered = self.root_path.glob(self.pattern)

        # Filter out directories and apply filter_fn if provided
        self._files = sorted(
            [
                f
                for f in discovered
                if f.is_file()
                and (self.filter_fn(f) if self.filter_fn is not None else True)
            ]
        )

    def __len__(self) -> int:
        """
        Return the number of files in the dataset.

        :return: Number of items.
        :rtype: int
        """
        return len(self._files)

    @overload
    def __getitem__(self, index: int) -> DatasetItem: ...

    @overload
    def __getitem__(self, index: slice) -> list[DatasetItem]: ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[DatasetItem, list[DatasetItem]]:
        """
        Get dataset items by index or slice.

        :param index: The index or slice.
        :type index: Union[int, slice]
        :return: DatasetItem if index is int, list of DatasetItems if slice.
        :rtype: Union[DatasetItem, list[DatasetItem]]
        """
        if isinstance(index, slice):
            return [
                self.item_class(f, self.metadata_loader) for f in self._files[index]
            ]
        return self.item_class(self._files[index], self.metadata_loader)

    def __iter__(self) -> Iterator[DatasetItem]:
        """
        Iterate over the dataset items.

        :return: Iterator of DatasetItem instances.
        :rtype: Iterator[DatasetItem]
        """
        for f in self._files:
            yield self.item_class(f, self.metadata_loader)

    def sample(self, k: int, seed: Optional[int] = None) -> list[DatasetItem]:
        """
        Get a random sample of k dataset items.

        :param k: Number of items to sample.
        :type k: int
        :param seed: Optional random seed for reproducibility.
        :type seed: int, optional
        :return: List of sampled DatasetItems.
        :rtype: list[DatasetItem]
        :raises ValueError: If sample size k is larger than the dataset.
        """
        if k > len(self):
            raise ValueError(f"Sample size {k} is larger than dataset size {len(self)}")

        if seed is not None:
            rng = random.Random(seed)
            sampled_files = rng.sample(self._files, k)
        else:
            sampled_files = random.sample(self._files, k)

        return [self.item_class(f, self.metadata_loader) for f in sampled_files]

    def with_metadata_loader(
        self, metadata_loader: Callable[[Path], dict[str, Any]]
    ) -> "FileDataset":
        """
        Return a new FileDataset instance with the specified metadata loader.

        This allows extending a basic file list dataset with lazy-loaded metadata
        without re-scanning the disk.

        :param metadata_loader: A callable that loads metadata from a file path.
        :type metadata_loader: Callable[[Path], dict[str, Any]]
        :return: A new FileDataset instance with the loader applied.
        :rtype: FileDataset
        """
        new_ds = FileDataset.__new__(FileDataset)
        new_ds.root_path = self.root_path
        new_ds.pattern = self.pattern
        new_ds.recursive = self.recursive
        new_ds.metadata_loader = metadata_loader
        new_ds.item_class = self.item_class
        new_ds.filter_fn = self.filter_fn
        new_ds._files = self._files
        return new_ds

    def filter(self, predicate: Callable[[DatasetItem], bool]) -> "FileDataset":
        """
        Return a new FileDataset containing only items that match the predicate.

        This eagerly evaluates the predicate on all items currently in the dataset.

        :param predicate: A callable that takes a DatasetItem and returns a boolean.
        :type predicate: Callable[[DatasetItem], bool]
        :return: A new filtered FileDataset instance.
        :rtype: FileDataset
        """
        filtered_files = []
        for f in self._files:
            item = self.item_class(f, self.metadata_loader)
            if predicate(item):
                filtered_files.append(f)

        new_ds = FileDataset.__new__(FileDataset)
        new_ds.root_path = self.root_path
        new_ds.pattern = self.pattern
        new_ds.recursive = self.recursive
        new_ds.metadata_loader = self.metadata_loader
        new_ds.item_class = self.item_class
        new_ds.filter_fn = self.filter_fn
        new_ds._files = filtered_files
        return new_ds


def json_metadata_loader(path: Path) -> dict[str, Any]:
    """
    Parse metadata from a JSON file.

    Loads the entire file as JSON. Useful when the dataset consists of JSON files.

    :param path: Path to the JSON file.
    :type path: Path
    :return: A dictionary containing the JSON content.
    :rtype: dict[str, Any]
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
    :type path: Path
    :return: A dictionary containing the parsed metadata.
    :rtype: dict[str, Any]
    """
    metadata: dict[str, Any] = {}
    if not path.exists():
        return metadata

    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return metadata

    # Check if file has front matter starting boundary
    if not content.startswith("---"):
        return metadata

    # Split on first two '---' boundaries
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

    Keeps only articles with a word count of at least min_words.

    :param min_words: Minimum number of words to keep an article.
    :type min_words: int
    :return: Predicate callable.
    :rtype: Callable[[DatasetItem], bool]
    """
    from sulku.utils import count_words

    return lambda item: count_words(item.content) >= min_words


def non_empty_filter() -> Callable[[DatasetItem], bool]:
    """
    Create a predicate to filter out empty articles.

    Keeps only articles with a word count greater than 0.

    :return: Predicate callable.
    :rtype: Callable[[DatasetItem], bool]
    """
    from sulku.utils import count_words

    return lambda item: count_words(item.content) > 0


def language_filter(
    languages: Union[str, list[str], set[str], Sequence[str]],
) -> Callable[[DatasetItem], bool]:
    """
    Create a predicate to filter articles by language.

    Keeps only articles that match the allowed language(s).

    :param languages: A single language string or a collection of allowed languages.
    :type languages: Union[str, list[str], set[str], Sequence[str]]
    :return: Predicate callable.
    :rtype: Callable[[DatasetItem], bool]
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
