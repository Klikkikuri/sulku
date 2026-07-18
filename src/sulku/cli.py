"""
Command Line Interface
======================

This module provides command-line interface entry points for the package,
specifically using Click.
"""

from pathlib import Path
import sys
import click
from sulku.dataset.reader import FileDataset


@click.group()
def main() -> None:
    """
    Sulku CLI tool.
    """
    pass


@main.command(name="sample")
@click.argument("dataset_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("-n", "--count", type=int, required=True, help="Number of dataset samples to return.")
@click.option("-p", "--pattern", type=str, default="*", help="Glob pattern to filter files.")
@click.option("-r", "--recursive/--no-recursive", default=True, help="Whether to search recursively.")
@click.option("-s", "--seed", type=int, default=None, help="Random seed for sampling.")
def sample(dataset_path: Path, count: int, pattern: str, recursive: bool, seed: int | None) -> None:
    """
    Sample N items from a dataset and output only their file paths.

    :param dataset_path: The directory path of the dataset.
    :type dataset_path: Path
    :param count: Number of items to sample.
    :type count: int
    :param pattern: Glob pattern to filter files.
    :type pattern: str
    :param recursive: Search recursively if True.
    :type recursive: bool
    :param seed: Random seed for sampling.
    :type seed: int | None
    """
    try:
        dataset = FileDataset(dataset_path, pattern=pattern, recursive=recursive)
        if not dataset:
            click.echo(f"Error: No files found matching pattern '{pattern}' under {dataset_path}.", err=True)
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
