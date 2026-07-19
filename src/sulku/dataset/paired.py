"""
Paired Dataset Utility
======================

This module provides the PairedDataset class and the load_paired_dataset function,
allowing simultaneous access to both source news articles and their generated
synthetic counterparts for model training and evaluation.
"""

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple, Optional, Union, overload

from sulku.constants import DEFAULT_DEST_DIR_BASE, DEFAULT_SOURCE_DIR
from sulku.dataset.reader import DatasetItem, yaml_front_matter_loader


class ItemPair(NamedTuple):
    """
    A pair of original source article and generated synthetic article.
    """

    source: DatasetItem
    synthetic: DatasetItem


class DatasetItemSequence(Sequence[DatasetItem]):
    """
    A read-only sequence of DatasetItem objects.

    Allows lazy instantiation of DatasetItem instances when indexed or sliced.
    """

    def __init__(
        self,
        paths: list[Path],
        metadata_loader: Optional[Callable[[Path], dict[str, Any]]] = None,
    ):
        """
        Initialize the dataset item sequence.

        :param paths: List of file paths.
        :type paths: list[Path]
        :param metadata_loader: Optional callable to load metadata lazily.
        :type metadata_loader: Callable[[Path], dict[str, Any]], optional
        """
        self._paths = paths
        self._metadata_loader = metadata_loader if metadata_loader is not None else yaml_front_matter_loader

    def __len__(self) -> int:
        """
        Return the number of items in the sequence.

        :return: Number of items.
        :rtype: int
        """
        return len(self._paths)

    @overload
    def __getitem__(self, index: int) -> DatasetItem: ...

    @overload
    def __getitem__(self, index: slice) -> list[DatasetItem]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[DatasetItem, list[DatasetItem]]:
        """
        Get dataset items by index or slice.

        :param index: The index or slice.
        :type index: Union[int, slice]
        :return: DatasetItem if index is int, list of DatasetItems if slice.
        :rtype: Union[DatasetItem, list[DatasetItem]]
        """
        if isinstance(index, slice):
            return [DatasetItem(path, self._metadata_loader) for path in self._paths[index]]
        return DatasetItem(self._paths[index], self._metadata_loader)

    def __iter__(self) -> Iterator[DatasetItem]:
        """
        Iterate over the dataset items.

        :return: Iterator of DatasetItem instances.
        :rtype: Iterator[DatasetItem]
        """
        for path in self._paths:
            yield DatasetItem(path, self._metadata_loader)


class PairedDataset(Sequence[ItemPair]):
    """
    A dataset of paired source and synthetic articles.

    Matches generated synthetic articles with their original source articles by
    relative path. Behaves like a PyTorch-compatible map-style dataset.
    """

    def __init__(
        self,
        source_dir: Union[str, Path],
        synthetic_dir: Union[str, Path],
        pattern: str = "*.md",
        recursive: bool = True,
        source_metadata_loader: Optional[Callable[[Path], dict[str, Any]]] = None,
        synth_metadata_loader: Optional[Callable[[Path], dict[str, Any]]] = None,
    ):
        """
        Initialize the paired dataset.

        Discovers matches between the source and synthetic directories using relative paths.

        :param source_dir: Path to the original source news articles.
        :type source_dir: Union[str, Path]
        :param synthetic_dir: Path to the generated synthetic news articles.
        :type synthetic_dir: Union[str, Path]
        :param pattern: Glob pattern to filter files. Defaults to "*.md".
        :type pattern: str
        :param recursive: If True, search directories recursively. Defaults to True.
        :type recursive: bool
        :param source_metadata_loader: Optional loader for source article metadata.
        :type source_metadata_loader: Callable[[Path], dict[str, Any]], optional
        :param synth_metadata_loader: Optional loader for synthetic article metadata.
        :type synth_metadata_loader: Callable[[Path], dict[str, Any]], optional
        """
        self.source_dir = Path(source_dir).resolve()
        self.synthetic_dir = Path(synthetic_dir).resolve()
        self.pattern = pattern
        self.recursive = recursive
        self.source_metadata_loader = (
            source_metadata_loader if source_metadata_loader is not None else yaml_front_matter_loader
        )
        self.synth_metadata_loader = (
            synth_metadata_loader if synth_metadata_loader is not None else yaml_front_matter_loader
        )

        # Discover all files in synthetic directory
        if self.recursive:
            synth_discovered = self.synthetic_dir.glob(f"**/{self.pattern}")
        else:
            synth_discovered = self.synthetic_dir.glob(self.pattern)

        synth_files = sorted([f for f in synth_discovered if f.is_file()])

        # Match synthetic files with source files
        self.paired_paths: list[tuple[Path, Path]] = []
        for synth_file in synth_files:
            rel_path = synth_file.relative_to(self.synthetic_dir)
            source_file = self.source_dir / rel_path
            if source_file.exists() and source_file.is_file():
                self.paired_paths.append((source_file, synth_file))

    def __len__(self) -> int:
        """
        Return the number of paired articles.

        :return: Number of pairs.
        :rtype: int
        """
        return len(self.paired_paths)

    @property
    def source(self) -> DatasetItemSequence:
        """
        Get a sequence of the source articles in the paired dataset.

        :return: A sequence of DatasetItems.
        :rtype: DatasetItemSequence
        """
        source_paths = [src for src, _ in self.paired_paths]
        return DatasetItemSequence(source_paths, self.source_metadata_loader)

    @property
    def synthetic(self) -> DatasetItemSequence:
        """
        Get a sequence of the synthetic articles in the paired dataset.

        :return: A sequence of DatasetItems.
        :rtype: DatasetItemSequence
        """
        synth_paths = [synth for _, synth in self.paired_paths]
        return DatasetItemSequence(synth_paths, self.synth_metadata_loader)

    @overload
    def __getitem__(self, index: int) -> ItemPair: ...

    @overload
    def __getitem__(self, index: slice) -> list[ItemPair]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[ItemPair, list[ItemPair]]:
        """
        Get a pair (or list of pairs) of source and synthetic articles.

        :param index: The index or slice.
        :type index: Union[int, slice]
        :return: A ItemPair containing the corresponding DatasetItems,
            or a list of ItemPairs if sliced.
        :rtype: Union[ItemPair, list[ItemPair]]
        """
        if isinstance(index, slice):
            return [
                ItemPair(
                    source=DatasetItem(src, self.source_metadata_loader),
                    synthetic=DatasetItem(syn, self.synth_metadata_loader),
                )
                for src, syn in self.paired_paths[index]
            ]
        src, syn = self.paired_paths[index]
        return ItemPair(
            source=DatasetItem(src, self.source_metadata_loader),
            synthetic=DatasetItem(syn, self.synth_metadata_loader),
        )

    def __iter__(self) -> Iterator[ItemPair]:
        """
        Iterate over the paired dataset items.

        :return: Iterator of ItemPair objects containing source and synthetic items.
        :rtype: Iterator[ItemPair]
        """
        for src, syn in self.paired_paths:
            yield ItemPair(
                source=DatasetItem(src, self.source_metadata_loader),
                synthetic=DatasetItem(syn, self.synth_metadata_loader),
            )

    def sample(self, k: int, seed: Optional[int] = None) -> list[ItemPair]:
        """
        Get a random sample of k paired dataset items.

        :param k: Number of items to sample.
        :type k: int
        :param seed: Optional random seed for reproducibility.
        :type seed: int, optional
        :return: List of sampled ItemPairs.
        :rtype: list[ItemPair]
        :raises ValueError: If sample size k is larger than the dataset.
        """
        if k > len(self):
            raise ValueError(f"Sample size {k} is larger than dataset size {len(self)}")

        if seed is not None:
            rng = random.Random(seed)
            sampled_paths = rng.sample(self.paired_paths, k)
        else:
            sampled_paths = random.sample(self.paired_paths, k)

        return [
            ItemPair(
                source=DatasetItem(src, self.source_metadata_loader),
                synthetic=DatasetItem(syn, self.synth_metadata_loader),
            )
            for src, syn in sampled_paths
        ]

    def filter(self, predicate: Callable[[ItemPair], bool]) -> "PairedDataset":
        """
        Return a new PairedDataset containing only items that match the predicate.

        This eagerly evaluates the predicate on all items currently in the dataset.

        :param predicate: A callable that takes a ItemPair and returns a boolean.
        :type predicate: Callable[[ItemPair], bool]
        :return: A new filtered PairedDataset instance.
        :rtype: PairedDataset
        """
        filtered_pairs = []
        for src, syn in self.paired_paths:
            item = ItemPair(
                source=DatasetItem(src, self.source_metadata_loader),
                synthetic=DatasetItem(syn, self.synth_metadata_loader),
            )
            if predicate(item):
                filtered_pairs.append((src, syn))

        new_ds = PairedDataset.__new__(PairedDataset)
        new_ds.source_dir = self.source_dir
        new_ds.synthetic_dir = self.synthetic_dir
        new_ds.pattern = self.pattern
        new_ds.recursive = self.recursive
        new_ds.source_metadata_loader = self.source_metadata_loader
        new_ds.synth_metadata_loader = self.synth_metadata_loader
        new_ds.paired_paths = filtered_pairs
        return new_ds


def load_paired_dataset(
    path_or_model_name: Union[str, Path],
    source_dir: Optional[Union[str, Path]] = None,
    pattern: str = "*.md",
    recursive: bool = True,
    source_metadata_loader: Optional[Callable[[Path], dict[str, Any]]] = None,
    synth_metadata_loader: Optional[Callable[[Path], dict[str, Any]]] = None,
) -> PairedDataset:
    """
    Load a paired dataset of original source news and generated synthetic news.

    Resolves the synthetic directory from a given path or model name.

    :param path_or_model_name: An existing directory path or a model name.
    :type path_or_model_name: Union[str, Path]
    :param source_dir: Optional custom source directory path. Defaults to DEFAULT_SOURCE_DIR.
    :type source_dir: Union[str, Path], optional
    :param pattern: Glob pattern to filter files. Defaults to "*.md".
    :type pattern: str
    :param recursive: If True, search recursively. Defaults to True.
    :type recursive: bool
    :param source_metadata_loader: Optional loader for source article metadata.
    :type source_metadata_loader: Callable[[Path], dict[str, Any]], optional
    :param synth_metadata_loader: Optional loader for synthetic article metadata.
    :type synth_metadata_loader: Callable[[Path], dict[str, Any]], optional
    :return: An initialized PairedDataset object.
    :rtype: PairedDataset
    :raises FileNotFoundError: If source or synthetic directory does not exist.
    """
    path_obj = Path(path_or_model_name)
    if path_obj.exists() and path_obj.is_dir():
        synthetic_dir = path_obj.resolve()
    else:
        synthetic_dir = (DEFAULT_DEST_DIR_BASE / path_or_model_name).resolve()

    if not synthetic_dir.exists():
        raise FileNotFoundError(
            f"Synthetic dataset directory not found at: {synthetic_dir}. "
            "Please check if 'path_or_model_name' is a valid model name or path."
        )

    if source_dir is None:
        source_dir = DEFAULT_SOURCE_DIR
    else:
        source_dir = Path(source_dir).resolve()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset directory not found at: {source_dir}")

    return PairedDataset(
        source_dir=source_dir,
        synthetic_dir=synthetic_dir,
        pattern=pattern,
        recursive=recursive,
        source_metadata_loader=source_metadata_loader,
        synth_metadata_loader=synth_metadata_loader,
    )


def generate_fasttext_sentence_data(
    items: Sequence[DatasetItem],
    label: str,
    output_path: Union[str, Path],
    lang: str = "fi",
    min_word_count: int = 4,
    mode: str = "a",
) -> None:
    """
    Generate FastText formatted sentence training data from a sequence of markdown articles.

    Each document's YAML front matter and markdown formatting are stripped. Then, the
    plain text is split into sentences using standard sentencize. Valid sentences with
    word counts meeting or exceeding min_word_count are written to output_path.

    :param items: Sequence of DatasetItem objects containing markdown articles.
    :type items: Sequence[DatasetItem]
    :param label: Class label to prefix each sentence with (e.g. 'human', 'synthetic').
    :type label: str
    :param output_path: Path to the output text file where FastText data will be written.
    :type output_path: Union[str, Path]
    :param lang: Language code for tokenizer/sentencizer. Defaults to 'fi'.
    :type lang: str
    :param min_word_count: Minimum words in a sentence to keep it in training. Defaults to 4.
    :type min_word_count: int
    :param mode: File open mode, 'w' to overwrite or 'a' to append. Defaults to 'a'.
    :type mode: str
    """
    from sulku.utils import count_words, sentencize, strip_markdown

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, mode, encoding="utf-8") as f:
        for item in items:
            raw_content = item.content
            plain_text = strip_markdown(raw_content)
            sentences = sentencize(plain_text, lang=lang)
            for sentence in sentences:
                sentence_cleaned = " ".join(sentence.split())
                if count_words(sentence_cleaned) >= min_word_count:
                    f.write(f"__label__{label} {sentence_cleaned}\n")

