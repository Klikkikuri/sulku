"""
CLI Generate FastText Command
=============================

This module defines the CLI command to generate FastText formatted training data.
"""

from pathlib import Path
import sys
import click

from sulku.dataset import FileDataset, generate_fasttext_sentence_data


@click.command(name="generate-fasttext")
@click.argument("dataset_path", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "-o",
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to the output file where FastText data will be written.",
)
@click.option(
    "-l",
    "--label",
    type=str,
    required=True,
    help="Class label to prefix each sentence (e.g. 'human', 'machine').",
)
@click.option(
    "-mw",
    "--min-words",
    type=int,
    default=4,
    help="Minimum number of words required to keep a sentence.",
)
@click.option(
    "--lang",
    type=str,
    default="fi",
    help="Language code for the tokenizer/sentencizer (e.g. 'fi', 'en').",
)
@click.option(
    "-p",
    "--pattern",
    type=str,
    default="*",
    help="Glob pattern to filter files.",
)
@click.option(
    "-r",
    "--recursive/--no-recursive",
    default=True,
    help="Whether to search recursively.",
)
@click.option(
    "-a",
    "--append",
    is_flag=True,
    help="Append to the output file instead of overwriting.",
)
def generate_fasttext_cmd(
    dataset_path: Path,
    output_file: Path,
    label: str,
    min_words: int,
    lang: str,
    pattern: str,
    recursive: bool,
    append: bool,
) -> None:
    """
    Generate FastText formatted sentence training data from markdown files.
    """
    try:
        dataset = FileDataset(dataset_path, pattern=pattern, recursive=recursive)
        if not dataset:
            click.echo(
                f"Error: No files found matching pattern '{pattern}' under {dataset_path}.",
                err=True,
            )
            sys.exit(1)

        mode = "a" if append else "w"
        click.echo(f"Generating FastText sentence data from {len(dataset)} items in {dataset_path}...")
        generate_fasttext_sentence_data(
            items=dataset,
            label=label,
            output_path=output_file,
            lang=lang,
            min_word_count=min_words,
            mode=mode,
        )
        click.echo(f"Successfully wrote FastText sentence data to {output_file}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
