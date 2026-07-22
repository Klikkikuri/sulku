"""
CLI Sample Command
==================

This module defines the CLI command to sample files from a dataset.
"""

from pathlib import Path
import sys
import click

from sulku.dataset import FileDataset, language_filter, min_words_filter


@click.command(name="sample")
@click.argument("dataset_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-n",
    "--count",
    type=int,
    required=True,
    help="Number of dataset samples to return.",
)
@click.option("-p", "--pattern", type=str, default="*", help="Glob pattern to filter files.")
@click.option(
    "-r",
    "--recursive/--no-recursive",
    default=True,
    help="Whether to search recursively.",
)
@click.option("-s", "--seed", type=int, default=None, help="Random seed for sampling.")
@click.option(
    "-l",
    "--language",
    type=str,
    default=None,
    help="Language code to filter articles (e.g. 'fi').",
)
@click.option(
    "-mw",
    "--min-words",
    type=int,
    default=None,
    help="Minimum number of words required to keep an article.",
)
def sample_cmd(
    dataset_path: Path,
    count: int,
    pattern: str,
    recursive: bool,
    seed: int | None,
    language: str | None,
    min_words: int | None,
) -> None:
    """
    Sample N items from a dataset and output only their file paths.
    """
    try:
        dataset = FileDataset(dataset_path, pattern=pattern, recursive=recursive)
        if language:
            dataset = dataset.filter(language_filter(language))
        if min_words is not None:
            dataset = dataset.filter(min_words_filter(min_words))

        if not dataset:
            click.echo(
                f"Error: No files found matching pattern '{pattern}' (with applied filters) under {dataset_path}.",
                err=True,
            )
            sys.exit(1)

        if count > len(dataset):
            click.echo(
                f"Error: Requested count {count} is larger than dataset size {len(dataset)}.",
                err=True,
            )
            sys.exit(1)

        sampled_items = dataset.sample(k=count, seed=seed)
        for item in sampled_items:
            click.echo(str(item.path))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
