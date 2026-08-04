"""
Dataset Mixins Module
=====================

Provides mixin classes for adding convenience filters to datasets.
"""

from typing import TYPE_CHECKING, Any, Sequence, Union

if TYPE_CHECKING:
    from sulku.dataset.base import BaseDataset


class TextDatasetFilterMixin:
    """
    Mixin class providing text-specific convenience filters for datasets.
    """

    def filter_min_words(self: Union["BaseDataset[Any, Any]", Any], min_words: int):
        """
        Filter dataset items to keep only those with at least min_words words.

        :param min_words: Minimum number of words required.
        :return: Filtered dataset instance.
        """
        from sulku.dataset.reader import min_words_filter

        return self.filter(min_words_filter(min_words))

    def filter_non_empty(self: Union["BaseDataset[Any, Any]", Any]):
        """
        Filter dataset items to keep only non-empty content.

        :return: Filtered dataset instance.
        """
        from sulku.dataset.reader import non_empty_filter

        return self.filter(non_empty_filter())

    def filter_language(
        self: Union["BaseDataset[Any, Any]", Any],
        languages: Union[str, Sequence[str]],
    ):
        """
        Filter dataset items to keep only specified language(s).

        :param languages: Target language code or list of language codes.
        :return: Filtered dataset instance.
        """
        from sulku.dataset.reader import language_filter

        return self.filter(language_filter(languages))
