"""
Dataset Base module
===================

Provides the BaseDataset abstract base class for dataset operations.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
import random
from typing import Callable, Generic, Optional, TypeVar, Union, overload

T_in = TypeVar("T_in")
T_out = TypeVar("T_out")


class BaseDataset(Sequence[T_out], Generic[T_in, T_out], ABC):
    """
    An abstract base class for a dataset that lazily loads items.

    Provides standard sequence operations (indexing, slicing, length),
    along with shuffle, take, filtering, and splitting functionality.
    """

    def __init__(self, items: list[T_in], loader: Callable[[T_in], T_out]):
        self._items = items
        self._loader = loader

    @abstractmethod
    def _clone(self, items: list[T_in]) -> "BaseDataset[T_in, T_out]":
        """
        Subclasses must implement this to return a new instance of their own type.

        :param items: List of raw items for the new instance.
        :return: A new instance of the concrete subclass.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._items)

    @overload
    def __getitem__(self, index: int) -> T_out: ...

    @overload
    def __getitem__(self, index: slice) -> "BaseDataset[T_in, T_out]": ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T_out, "BaseDataset[T_in, T_out]"]:
        if isinstance(index, slice):
            return self._clone(self._items[index])
        return self._loader(self._items[index])

    def shuffle(self, seed: Optional[int] = None) -> "BaseDataset[T_in, T_out]":
        """
        Return a new shuffled dataset.

        :param seed: Optional random seed for deterministic shuffling.
        :return: A new dataset instance with shuffled items.
        """
        rng = random.Random(seed) if seed is not None else random
        shuffled_items = list(self._items)
        rng.shuffle(shuffled_items)
        return self._clone(shuffled_items)

    def take(self, n: int) -> "BaseDataset[T_in, T_out]":
        """
        Return a new dataset with only the first n items.

        :param n: Number of items to take.
        :return: A new dataset instance with at most n items.
        """
        return self._clone(self._items[:n])

    def sample(self, k: int, seed: Optional[int] = None) -> "BaseDataset[T_in, T_out]":
        """
        Get a random sample of k dataset items.

        :param k: Number of items to sample.
        :param seed: Optional random seed for reproducibility.
        :return: A new dataset instance with sampled items.
        """
        if k > len(self._items):
            raise ValueError(f"Sample size {k} is larger than dataset size {len(self._items)}")
        return self.shuffle(seed).take(k)

    def filter(
        self, predicate: Union[Callable[[T_out], bool], list[Callable[[T_out], bool]]]
    ) -> "BaseDataset[T_in, T_out]":
        """
        Return a new dataset containing only items that match the predicate(s).

        If a list of predicates is provided, an item must pass all predicates.

        :param predicate: A single filter predicate or list of filter predicates.
        :return: A new filtered dataset instance.
        """
        if not isinstance(predicate, list):
            predicates = [predicate]
        else:
            predicates = predicate

        filtered = []
        for item in self._items:
            loaded_item = self._loader(item)
            if all(p(loaded_item) for p in predicates):
                filtered.append(item)

        return self._clone(filtered)

    def split(
        self, ratio: float = 0.8, shuffle: bool = True, seed: Optional[int] = None
    ) -> tuple["BaseDataset[T_in, T_out]", "BaseDataset[T_in, T_out]"]:
        """
        Split dataset into two subsets (e.g., train and validation).

        :param ratio: Fraction of dataset for the first subset (default: 0.8).
        :param shuffle: Whether to shuffle items before splitting (default: True).
        :param seed: Optional random seed for reproducible shuffling.
        :return: A tuple of (first_subset, second_subset).
        """
        items_copy = list(self._items)
        if shuffle:
            rng = random.Random(seed) if seed is not None else random
            rng.shuffle(items_copy)

        split_idx = int(len(items_copy) * ratio)
        return self._clone(items_copy[:split_idx]), self._clone(items_copy[split_idx:])

    def __iter__(self):
        for item in self._items:
            yield self._loader(item)
