"""
CLI Test Subcommands
====================

This module defines the CLI command group 'test' and its subcommands,
such as 'test detect' to evaluate detection service accuracy.
"""

from pathlib import Path
import sys
import click
import httpx

from sulku.dataset import FileDataset
from sulku.cli.detect import detect_text, read_file_content


@click.group(name="test")
def test_group() -> None:
    """
    Subcommands to test and evaluate Sulku components.
    """
    pass


@test_group.command(name="detect")
@click.option(
    "-n",
    "--count",
    type=int,
    default=5,
    help="Number of human and synthetic files to sample.",
)
@click.option(
    "--human-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("/app/data/yle"),
    help="Path to the directory containing human-written articles.",
)
@click.option(
    "--synthetic-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("/app/data/genai"),
    help="Path to the directory containing synthetic articles.",
)
@click.option(
    "--url",
    type=str,
    default="http://127.0.0.1:8000/api/v1/aidetect/",
    help="The URL of the aidetect service endpoint.",
)
@click.option(
    "-s",
    "--seed",
    type=int,
    default=None,
    help="Random seed for deterministic sampling.",
)
def detect_test_cmd(
    count: int,
    human_dir: Path,
    synthetic_dir: Path,
    url: str,
    seed: int | None,
) -> None:
    """
    Test classification accuracy of the detection service.

    Samples N human articles and N synthetic articles, sends them to the
    detection service, and verifies classification accuracy.
    """
    if not human_dir.exists():
        click.echo(f"Error: Human dataset directory '{human_dir}' does not exist.", err=True)
        sys.exit(1)
    if not synthetic_dir.exists():
        click.echo(f"Error: Synthetic dataset directory '{synthetic_dir}' does not exist.", err=True)
        sys.exit(1)

    try:
        # Load and sample human files
        human_dataset = FileDataset(human_dir, pattern="*.md", recursive=True)

        if len(human_dataset) < count:
            click.echo(
                f"Error: Requested count {count} is larger than available human dataset size {len(human_dataset)}.",
                err=True,
            )
            sys.exit(1)

        sampled_human = human_dataset.sample(k=count, seed=seed)
    except Exception as e:
        click.echo(f"Error loading human dataset: {e}", err=True)
        sys.exit(1)

    try:
        # Load and sample synthetic files
        synthetic_dataset = FileDataset(synthetic_dir, pattern="*.md", recursive=True)

        if len(synthetic_dataset) < count:
            click.echo(
                f"Error: Requested count {count} is larger than available synthetic dataset size {len(synthetic_dataset)}.",
                err=True,
            )
            sys.exit(1)

        # Shift seed slightly for synthetic sampling so we don't get identical choices
        seed_syn = seed + 1 if seed is not None else None
        sampled_synthetic = synthetic_dataset.sample(k=count, seed=seed_syn)
    except Exception as e:
        click.echo(f"Error loading synthetic dataset: {e}", err=True)
        sys.exit(1)

    click.echo(f"Testing classification accuracy using endpoint: {url}\n")
    click.echo(f"Sampling {count} human articles from {human_dir}...")
    click.echo(f"Sampling {count} synthetic articles from {synthetic_dir}...\n")

    correct_human = 0
    incorrect_human = 0
    correct_synthetic = 0
    incorrect_synthetic = 0

    click.echo("--- Testing Human Articles (Expected: HUMAN) ---")
    for idx, item in enumerate(sampled_human, start=1):
        try:
            content, content_type = read_file_content(item.path)
            res = detect_text(url, content, content_type)
            is_ai = res.get("is_ai", False)
            if not is_ai:
                correct_human += 1
                status = "Correct (HUMAN)"
            else:
                incorrect_human += 1
                status = "Incorrect (AI)"
            click.echo(f"[{idx}/{count}] {item.path} -> Classified as {status}")
        except httpx.HTTPError as e:
            click.echo(f"Error calling detection service for {item.path.name}: {e}", err=True)
            click.echo("Is the API server running? Start it with 'sulku serve'.", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error processing {item.path.name}: {e}", err=True)
            sys.exit(1)

    click.echo("\n--- Testing Synthetic Articles (Expected: AI) ---")
    for idx, item in enumerate(sampled_synthetic, start=1):
        try:
            content, content_type = read_file_content(item.path)
            res = detect_text(url, content, content_type)
            is_ai = res.get("is_ai", False)
            if is_ai:
                correct_synthetic += 1
                status = "Correct (AI)"
            else:
                incorrect_synthetic += 1
                status = "Incorrect (HUMAN)"
            click.echo(f"[{idx}/{count}] {item.path} -> Classified as {status}")
        except httpx.HTTPError as e:
            click.echo(f"Error calling detection service for {item.path.name}: {e}", err=True)
            click.echo("Is the API server running? Start it with 'sulku serve'.", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error processing {item.path.name}: {e}", err=True)
            sys.exit(1)

    total_human = correct_human + incorrect_human
    total_synthetic = correct_synthetic + incorrect_synthetic
    total_tested = total_human + total_synthetic
    total_correct = correct_human + correct_synthetic
    total_incorrect = incorrect_human + incorrect_synthetic

    accuracy = (total_correct / total_tested * 100) if total_tested > 0 else 0.0
    human_success = (correct_human / total_human * 100) if total_human > 0 else 0.0
    synthetic_success = (correct_synthetic / total_synthetic * 100) if total_synthetic > 0 else 0.0

    click.echo("\n" + "=" * 50)
    click.echo("DETECTION TEST RESULTS")
    click.echo("=" * 50)
    click.echo("Human Articles:")
    click.echo(f"  Total tested: {total_human}")
    click.echo(f"  Correct (Human): {correct_human}")
    click.echo(f"  Incorrect (AI): {incorrect_human}")
    click.echo(f"  Success Rate: {human_success:.2f}%")
    click.echo("")
    click.echo("Synthetic Articles:")
    click.echo(f"  Total tested: {total_synthetic}")
    click.echo(f"  Correct (AI): {correct_synthetic}")
    click.echo(f"  Incorrect (Human): {incorrect_synthetic}")
    click.echo(f"  Success Rate: {synthetic_success:.2f}%")
    click.echo("")
    click.echo("Overall Metrics:")
    click.echo(f"  Total Tested: {total_tested}")
    click.echo(f"  Total Correct: {total_correct}")
    click.echo(f"  Total Incorrect: {total_incorrect}")
    click.echo(f"  Accuracy: {accuracy:.2f}%")
    click.echo("=" * 50)
