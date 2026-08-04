"""
Paired Dataset Utility
======================

This module provides the PairedDataset class and the load_paired_dataset function,
allowing simultaneous access to both source news articles and their generated
synthetic counterparts for model training and evaluation.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, Optional, Union

from sulku.bootstrap import get_dest_dir_base, get_source_dir
from sulku.constants import LABEL_AI, LABEL_HUMAN
from sulku.dataset.base import BaseDataset
from sulku.dataset.mixins import TextDatasetFilterMixin
from sulku.dataset.reader import DatasetItem, FileDataset, yaml_front_matter_loader
from sulku.utils import count_words, sentencize, strip_markdown

if TYPE_CHECKING:
    from sulku.models import ModelMetadata


class ItemPair(NamedTuple):
    """
    A pair of original source article and generated synthetic article.
    """

    source: DatasetItem
    synthetic: DatasetItem

    @property
    def metadata(self) -> dict[str, Any]:
        return self.source.metadata

    @property
    def content(self) -> str:
        return self.source.content


class PairedDataset(BaseDataset[tuple[Path, Path], ItemPair], TextDatasetFilterMixin):
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
        _paired_paths: Optional[list[tuple[Path, Path]]] = None,
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
        :param _paired_paths: Internal parameter to pass pre-discovered paired file paths.
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

        if _paired_paths is None:
            if self.recursive:
                synth_discovered = self.synthetic_dir.glob(f"**/{self.pattern}")
            else:
                synth_discovered = self.synthetic_dir.glob(self.pattern)

            synth_files = sorted([f for f in synth_discovered if f.is_file()])

            paired_paths: list[tuple[Path, Path]] = []
            for synth_file in synth_files:
                rel_path = synth_file.relative_to(self.synthetic_dir)
                source_file = self.source_dir / rel_path
                if source_file.exists() and source_file.is_file():
                    paired_paths.append((source_file, synth_file))
            self.paired_paths = paired_paths
        else:
            self.paired_paths = _paired_paths

        def loader(paths: tuple[Path, Path]) -> ItemPair:
            return ItemPair(
                source=DatasetItem(paths[0], self.source_metadata_loader),
                synthetic=DatasetItem(paths[1], self.synth_metadata_loader),
            )

        super().__init__(self.paired_paths, loader)

    def _clone(self, items: list[tuple[Path, Path]]) -> "PairedDataset":
        """
        Return a new PairedDataset instance with the given list of paired file paths.

        :param items: List of tuples (source_path, synth_path).
        :return: A new PairedDataset instance.
        """
        return PairedDataset(
            source_dir=self.source_dir,
            synthetic_dir=self.synthetic_dir,
            pattern=self.pattern,
            recursive=self.recursive,
            source_metadata_loader=self.source_metadata_loader,
            synth_metadata_loader=self.synth_metadata_loader,
            _paired_paths=items,
        )

    @property
    def source(self) -> FileDataset:
        """
        Get a FileDataset of the source articles in the paired dataset.

        :return: FileDataset of source items.
        """
        source_paths = [src for src, _ in self.paired_paths]
        return FileDataset(
            root_path=self.source_dir,
            pattern=self.pattern,
            recursive=self.recursive,
            metadata_loader=self.source_metadata_loader,
            _files=source_paths,
        )

    @property
    def synthetic(self) -> FileDataset:
        """
        Get a FileDataset of the synthetic articles in the paired dataset.

        :return: FileDataset of synthetic items.
        """
        synth_paths = [synth for _, synth in self.paired_paths]
        return FileDataset(
            root_path=self.synthetic_dir,
            pattern=self.pattern,
            recursive=self.recursive,
            metadata_loader=self.synth_metadata_loader,
            _files=synth_paths,
        )

    def to_fasttext(
        self,
        output_path: Union[str, Path],
        model_metadata: Optional["ModelMetadata"] = None,
        lang: str = "fi",
        min_word_count: int = 4,
        mode: str = "w",
    ) -> None:
        """
        Export this PairedDataset to a FastText formatted training file.

        :param output_path: Destination path for FastText training file.
        :param model_metadata: Optional ModelMetadata instance supplying target labels.
        :param lang: Language code for sentencizer. Defaults to 'fi'.
        :param min_word_count: Minimum words per sentence. Defaults to 4.
        :param mode: File write mode ('w' to overwrite, 'a' to append).
        """
        export_paired_dataset_to_fasttext(
            self,
            output_path=output_path,
            model_metadata=model_metadata,
            lang=lang,
            min_word_count=min_word_count,
            mode=mode,
        )


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
        synthetic_dir = (get_dest_dir_base() / path_or_model_name).resolve()

    if not synthetic_dir.exists():
        raise FileNotFoundError(
            f"Synthetic dataset directory not found at: {synthetic_dir}. "
            "Please check if 'path_or_model_name' is a valid model name or path."
        )

    if source_dir is None:
        source_dir = get_source_dir()
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
    items: Iterable[DatasetItem],
    label: str,
    output_path: Union[str, Path],
    lang: str = "fi",
    min_word_count: int = 4,
    mode: str = "a",
) -> None:
    """
    Generate FastText formatted sentence training data from a sequence of markdown articles.

    :param items: Iterable of DatasetItem objects containing markdown articles.
    :param label: Class label to prefix each sentence with (e.g. 'human', 'synthetic').
    :param output_path: Path to the output text file where FastText data will be written.
    :param lang: Language code for tokenizer/sentencizer. Defaults to 'fi'.
    :param min_word_count: Minimum words in a sentence to keep it in training. Defaults to 4.
    :param mode: File open mode, 'w' to overwrite or 'a' to append. Defaults to 'a'.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fasttext_label = label if label.startswith("__label__") else f"__label__{label}"

    with open(out_path, mode, encoding="utf-8") as f:
        for item in items:
            raw_content = item.content
            plain_text = strip_markdown(raw_content)
            sentences = sentencize(plain_text, lang=lang)
            for sentence in sentences:
                sentence_cleaned = " ".join(sentence.split())
                if count_words(sentence_cleaned) >= min_word_count:
                    f.write(f"{fasttext_label} {sentence_cleaned}\n")


def export_paired_dataset_to_fasttext(
    dataset: PairedDataset,
    output_path: Union[str, Path],
    model_metadata: Optional["ModelMetadata"] = None,
    lang: str = "fi",
    min_word_count: int = 4,
    mode: str = "w",
) -> None:
    """
    Export a PairedDataset directly to a FastText training file, optionally using labels
    from a ModelMetadata specification.

    :param dataset: PairedDataset instance.
    :param output_path: Destination path for FastText training file.
    :param model_metadata: Optional ModelMetadata instance supplying target class labels.
    :param lang: Language code for sentencizer. Defaults to 'fi'.
    :param min_word_count: Minimum words per sentence. Defaults to 4.
    :param mode: File write mode ('w' to overwrite, 'a' to append).
    """
    if model_metadata and model_metadata.labels:
        human_label = model_metadata.labels.get("human", LABEL_HUMAN)
        ai_label = model_metadata.labels.get("ai", LABEL_AI)
    else:
        human_label = LABEL_HUMAN
        ai_label = LABEL_AI

    generate_fasttext_sentence_data(
        dataset.source,
        label=human_label,
        output_path=output_path,
        lang=lang,
        min_word_count=min_word_count,
        mode=mode,
    )
    generate_fasttext_sentence_data(
        dataset.synthetic,
        label=ai_label,
        output_path=output_path,
        lang=lang,
        min_word_count=min_word_count,
        mode="a",
    )
