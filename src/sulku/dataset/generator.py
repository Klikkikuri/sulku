"""
Synthetic Dataset Generator
==========================

This module provides tools to sample a dataset, generate semantic summaries,
cache summaries in user cache directory using platformdirs, and generate
synthetic articles using LLMs.
"""

from datetime import datetime, timezone
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Optional, Union, cast

import yaml
from platformdirs import user_cache_dir

from .models import GenerationDetailsDict, StyleVectorDict, TokenUsageDict
from sulku.dataset.reader import DatasetItem, FileDataset, min_words_filter
from sulku.summarize.llm import create_synthetic_article, summarize_text
from sulku.summarize.models import ArticleSummary
from sulku.constants import (
    CACHE_APP_AUTHOR,
    CACHE_APP_NAME,
    CACHE_SUBDIR,
    DEFAULT_DEST_DIR_BASE,
    DEFAULT_MODEL,
    DEFAULT_SOURCE_DIR,
)

logger = logging.getLogger(__name__)


class SyntheticDatasetGenerator:
    """
    Generates synthetic datasets from a source dataset of articles.

    It handles deterministic sampling, caching of intermediate semantic summaries,
    and generating synthetic articles under a destination directory.
    """

    def __init__(
        self,
        source_dir: Union[str, Path] = DEFAULT_SOURCE_DIR,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the generator.

        :param source_dir: Directory containing the source dataset files.
        :type source_dir: Union[str, Path]
        :param model_name: Name of the LLM model to use.
        :type model_name: str
        :param cache_dir: Optional path to override the summary cache directory.
            If None, uses platformdirs to resolve user cache directory.
        :type cache_dir: Union[str, Path], optional
        """
        self.source_dir = Path(source_dir).resolve()
        if not self.source_dir.exists():
            raise FileNotFoundError(
                f"Source directory does not exist: {self.source_dir}"
            )

        self.model_name = model_name

        if cache_dir is None:
            self.cache_dir = (
                Path(user_cache_dir(CACHE_APP_NAME, CACHE_APP_AUTHOR)) / CACHE_SUBDIR
            )
        else:
            self.cache_dir = Path(cache_dir).resolve()

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_file(self, article: DatasetItem) -> Path:
        """
        Get the path to the cache file for a given dataset item.

        Uses the SHA256 hash of the relative path of the article as the filename.

        :param article: The dataset item.
        :type article: DatasetItem
        :return: Path to the cache JSON file.
        :rtype: Path
        """
        rel_path = str(article.path.relative_to(self.source_dir))
        path_hash = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{path_hash}.json"

    def _get_content_hash(self, content: str) -> str:
        """
        Get SHA256 hash of text content.

        :param content: The text content.
        :type content: str
        :return: SHA256 hex string.
        :rtype: str
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get_or_create_summary(self, article: DatasetItem) -> ArticleSummary:
        """
        Get the summary of the article from cache, or generate and cache it.

        :param article: The dataset item to summarize.
        :type article: DatasetItem
        :return: The generated or cached ArticleSummary.
        :rtype: ArticleSummary
        """
        cache_file = self._get_cache_file(article)
        content_hash = self._get_content_hash(article.content)

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if cached_data.get("content_hash") == content_hash:
                    logger.debug("Cache hit for: %s", article.path)
                    return ArticleSummary.model_validate(cached_data["summary"])
            except Exception as e:
                logger.warning("Failed to load cache file %s: %s", cache_file, e)

        # Generate summary
        logger.info("Generating summary for: %s", article.path)
        summary = summarize_text(article.content, model=self.model_name)

        # Cache the summary
        try:
            rel_path = str(article.path.relative_to(self.source_dir))
            cached_data = {
                "relative_path": rel_path,
                "content_hash": content_hash,
                "summary": summary.model_dump(),
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cached_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to write cache file %s: %s", cache_file, e)

        return summary

    def _serialize_front_matter(self, metadata: dict[str, Any]) -> str:
        """
        Serialize metadata dictionary into a YAML front matter string.

        :param metadata: The metadata dictionary.
        :type metadata: dict[str, Any]
        :return: YAML front matter string.
        :rtype: str
        """
        if not metadata:
            return ""
        # Filter out error keys if any
        filtered_metadata = {k: v for k, v in metadata.items() if k != "error"}
        if not filtered_metadata:
            return ""
        yaml_str = yaml.safe_dump(filtered_metadata, allow_unicode=True)
        return f"---\n{yaml_str}---"

    def generate(
        self,
        n_samples: int,
        seed: Optional[int] = None,
        dest_dir: Optional[Union[str, Path]] = None,
        force: bool = False,
        min_words: Optional[int] = 50,
    ) -> list[Path]:
        """
        Generate a synthetic dataset from sampled articles.

        Samples deterministic N items, generates/caches their summaries, generates
        synthetic articles, and writes them to the destination directory.

        :param n_samples: Number of samples to draw.
        :type n_samples: int
        :param seed: Optional random seed for deterministic sampling.
        :type seed: int, optional
        :param dest_dir: Destination directory. Defaults to `/app/data/genai/{model_name}`.
        :type dest_dir: Union[str, Path], optional
        :param force: Force generation even if synthetic data already exists.
        :type force: bool
        :param min_words: Minimum number of words required to keep an article.
        :type min_words: int, optional
        :return: List of generated synthetic article file paths.
        :rtype: list[Path]
        """
        if dest_dir is None:
            dest_path = DEFAULT_DEST_DIR_BASE / self.model_name
        else:
            dest_path = Path(dest_dir).resolve()

        # Load dataset matching markdown files
        dataset = FileDataset(self.source_dir, pattern="*.md", recursive=True)
        if not dataset:
            raise ValueError(
                f"No markdown files found under source directory: {self.source_dir}"
            )

        if n_samples > len(dataset):
            raise ValueError(
                f"Sample size {n_samples} is larger than dataset size {len(dataset)}"
            )

        # Shuffle the files first to allow lazy filtering without reading the entire dataset
        files = list(dataset._files)
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(files)
        else:
            random.shuffle(files)

        sampled_items = []
        for f in files:
            if len(sampled_items) == n_samples:
                break
            item = dataset.item_class(f, dataset.metadata_loader)
            if min_words is None or min_words_filter(min_words)(item):
                sampled_items.append(item)

        if len(sampled_items) < n_samples:
            raise ValueError(
                f"Sample size {n_samples} is larger than the number of available "
                f"matching files ({len(sampled_items)}) in the dataset."
            )
        generated_paths = []

        for article in sampled_items:
            # Check if synthetic file already exists
            rel_path = article.path.relative_to(self.source_dir)
            out_file = dest_path / rel_path

            if out_file.exists() and not force:
                logger.info("Skipping already generated synthetic article: %s", out_file)
                continue

            # 1. Fetch or create summary (cached)
            summary = self.get_or_create_summary(article)

            # 2. Generate synthetic article content
            logger.info("Generating synthetic article for: %s", article.path)
            generation_metadata = {}
            synthetic_content = create_synthetic_article(
                article,
                summary,
                model=self.model_name,
                metadata_out=generation_metadata,
            )
            if not synthetic_content:
                logger.warning(
                    "Failed to generate synthetic content for: %s", article.path
                )
                continue

            # 3. Format complete article with original front matter metadata
            metadata = dict(article.metadata)
            token_usage = generation_metadata.get("token_usage")
            gen_details: GenerationDetailsDict = {
                "model": self.model_name,
                "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "headline": summary.headline,
                "summary": summary.summary,
                "style": cast(StyleVectorDict, summary.style.model_dump()),
            }
            if token_usage:
                gen_details["token_usage"] = cast(TokenUsageDict, token_usage)
            metadata["generation_details"] = gen_details

            front_matter = self._serialize_front_matter(metadata)
            if front_matter:
                full_text = f"{front_matter}\n\n{synthetic_content}\n"
            else:
                full_text = f"{synthetic_content}\n"

            # 4. Mirror the path structure in destination directory
            out_file.parent.mkdir(parents=True, exist_ok=True)

            # 5. Write the synthetic article to file
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(full_text)

            logger.info("Wrote synthetic article: %s", out_file)
            generated_paths.append(out_file)

        return generated_paths
