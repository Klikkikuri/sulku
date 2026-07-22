"""
CLI Generate Synthetic Command
==============================

This module defines the CLI command to generate synthetic dataset articles.
"""

from pathlib import Path
import sys
import click

from sulku.constants import DEFAULT_MODEL
from sulku.dataset import SyntheticDatasetGenerator


@click.command(name="generate-synthetic")
@click.argument("dataset_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-n",
    "--count",
    type=int,
    required=True,
    help="Number of dataset samples to generate.",
)
@click.option(
    "-m",
    "--model",
    type=str,
    default=DEFAULT_MODEL,
    help="LLM model name to use.",
)
@click.option(
    "-s",
    "--seed",
    type=int,
    default=None,
    help="Random seed for deterministic sampling.",
)
@click.option(
    "-d",
    "--dest-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Destination directory.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force generation even if synthetic data already exists.",
)
@click.option(
    "-mw",
    "--min-words",
    type=int,
    default=50,
    help="Minimum number of words required to keep an article.",
)
def generate_synthetic_cmd(
    dataset_path: Path,
    count: int,
    model: str,
    seed: int | None,
    dest_dir: Path | None,
    force: bool,
    min_words: int,
) -> None:
    """
    Generate synthetic articles from sampled articles of a dataset.
    """
    try:
        generator = SyntheticDatasetGenerator(
            source_dir=dataset_path,
            model_name=model,
        )

        click.echo(f"Sampling {count} articles and generating synthetic articles using model '{model}'...")
        generated_files = generator.generate(
            n_samples=count, seed=seed, dest_dir=dest_dir, force=force, min_words=min_words
        )

        click.echo(f"Successfully generated {len(generated_files)} synthetic articles:")
        for path in generated_files:
            click.echo(f"  - {path}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
