"""
Dataset Package
===============

This package contains tools for handling and processing datasets.
"""

from sulku.dataset.base import BaseDataset
from sulku.dataset.generator import SyntheticDatasetGenerator
from sulku.dataset.mixins import TextDatasetFilterMixin
from sulku.dataset.paired import (
    ItemPair,
    PairedDataset,
    export_paired_dataset_to_fasttext,
    generate_fasttext_sentence_data,
    load_paired_dataset,
)
from sulku.dataset.reader import (
    DatasetItem,
    FileDataset,
    json_metadata_loader,
    language_filter,
    min_words_filter,
    non_empty_filter,
    yaml_front_matter_loader,
)

__all__ = [
    "BaseDataset",
    "DatasetItem",
    "FileDataset",
    "ItemPair",
    "PairedDataset",
    "SyntheticDatasetGenerator",
    "TextDatasetFilterMixin",
    "export_paired_dataset_to_fasttext",
    "generate_fasttext_sentence_data",
    "json_metadata_loader",
    "language_filter",
    "load_paired_dataset",
    "min_words_filter",
    "non_empty_filter",
    "yaml_front_matter_loader",
]
