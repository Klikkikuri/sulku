"""
Command Line Interface
======================

This module provides command-line interface entry points for the package,
specifically using Click.
"""

from pathlib import Path
import sys
import logging
import click
from dotenv import load_dotenv
from sulku.dataset import (
    FileDataset,
    language_filter,
    min_words_filter,
    SyntheticDatasetGenerator,
)
from sulku.constants import DEFAULT_MODEL


class ExtraFormatter(logging.Formatter):
    """
    Custom logging formatter that outputs 'extra' fields in a readable vertical structure
    when the logger is configured to show extra information (e.g. in debug mode).
    """

    STANDARD_ATTRS = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }

    @staticmethod
    def _format_vertical(val, indent_level=0) -> str:
        indent = "  " * indent_level
        if isinstance(val, dict):
            if not val:
                return "{}"
            lines = []
            for k, v in sorted(val.items(), key=lambda item: str(item[0])):
                formatted_v = ExtraFormatter._format_vertical(v, indent_level + 1)
                if "\n" in formatted_v:
                    lines.append(f"{indent}  {k}:\n{formatted_v}")
                else:
                    lines.append(f"{indent}  {k}: {formatted_v}")
            return "\n".join(lines)
        elif isinstance(val, list):
            if not val:
                return "[]"
            lines = []
            for item in val:
                formatted_item = ExtraFormatter._format_vertical(item, indent_level + 1)
                if "\n" in formatted_item:
                    lines.append(f"{indent}- \n{formatted_item}")
                else:
                    lines.append(f"{indent}- {formatted_item}")
            return "\n".join(lines)
        else:
            return str(val)

    def __init__(self, fmt=None, datefmt=None, style="%", show_extra=False):
        super().__init__(fmt, datefmt, style)
        self.show_extra = show_extra

    def format(self, record: logging.LogRecord) -> str:
        s = super().format(record)
        if self.show_extra:
            extra_keys = set(record.__dict__.keys()) - self.STANDARD_ATTRS
            # Filter out private attributes
            extra_keys = {k for k in extra_keys if not k.startswith("_")}
            if extra_keys:
                extra_lines = []
                for k in sorted(extra_keys):
                    val = getattr(record, k)
                    if isinstance(val, (dict, list)):
                        val_str = self._format_vertical(val, indent_level=1)
                        extra_lines.append(f"  {k}:\n{val_str}")
                    elif isinstance(val, str) and "\n" in val:
                        val_str_indented = "\n".join(f"    {line}" for line in val.splitlines())
                        extra_lines.append(f"  {k}:\n{val_str_indented}")
                    else:
                        extra_lines.append(f"  {k}: {val}")
                s += "\n" + "\n".join(extra_lines)
        return s


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="INFO",
    help="Set the logging level.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Shorthand to set logging level to DEBUG.",
)
def main(log_level: str, debug: bool) -> None:
    """
    Sulku CLI tool.
    """

    load_dotenv()

    level = logging.DEBUG if debug else getattr(logging, log_level.upper())
    handler = logging.StreamHandler(sys.stderr)
    formatter = ExtraFormatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        show_extra=(level == logging.DEBUG),
    )
    handler.setFormatter(formatter)
    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True,
    )


@main.command(name="sample")
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
def sample(
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
    :param language: Language code to filter articles (e.g. 'fi').
    :type language: str | None
    :param min_words: Minimum number of words required to keep an article.
    :type min_words: int | None
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


@main.command(name="generate-synthetic")
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
def generate_synthetic(
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

    :param dataset_path: The directory path of the dataset.
    :type dataset_path: Path
    :param count: Number of items to sample and generate.
    :type count: int
    :param model: LLM model name.
    :type model: str
    :param seed: Random seed for sampling.
    :type seed: int | None
    :param dest_dir: Destination directory.
    :type dest_dir: Path | None
    :param force: Force generation even if synthetic data already exists.
    :type force: bool
    :param min_words: Minimum number of words required to keep an article.
    :type min_words: int
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


@main.command(name="generate-fasttext")
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
def generate_fasttext(
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

    :param dataset_path: Path to the dataset directory.
    :type dataset_path: Path
    :param output_file: Path to the output text file.
    :type output_file: Path
    :param label: Label to prefix sentences with.
    :type label: str
    :param min_words: Minimum words to keep a sentence.
    :type min_words: int
    :param lang: Language for sentence segmenter.
    :type lang: str
    :param pattern: Glob pattern to filter files.
    :type pattern: str
    :param recursive: Search recursively if True.
    :type recursive: bool
    :param append: Append to output file if True.
    :type append: bool
    """
    try:
        from sulku.dataset import FileDataset, generate_fasttext_sentence_data

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


@main.command(name="serve")
@click.option(
    "-h",
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host to bind the server to.",
)
@click.option(
    "-p",
    "--port",
    type=int,
    default=8000,
    help="Port to bind the server to.",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development.",
)
def serve(host: str, port: int, reload: bool) -> None:
    """
    Start the FastAPI HTTP server.

    :param host: Host to bind the server to.
    :type host: str
    :param port: Port to bind the server to.
    :type port: int
    :param reload: Enable auto-reload if True.
    :type reload: bool
    """
    try:
        import uvicorn

        click.echo(f"Starting server on {host}:{port} (reload={reload})...")
        uvicorn.run(
            "sulku.http:create_app",
            host=host,
            port=port,
            reload=reload,
            factory=True,
        )
    except Exception as e:
        click.echo(f"Error starting server: {e}", err=True)
        sys.exit(1)


@main.command(name="detect")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--url",
    type=str,
    default="http://127.0.0.1:8000/api/v1/aidetect/",
    help="The URL of the aidetect service endpoint.",
)
def detect(file_path: Path, url: str) -> None:
    """
    Detect if the text in a file is AI-generated by sending it to the aidetect service.

    :param file_path: Path to the file containing text to analyze.
    :type file_path: Path
    :param url: URL of the aidetect service endpoint.
    :type url: str
    """
    try:
        import httpx

        # Read file contents
        content = file_path.read_text(encoding="utf-8")

        click.echo(f"Sending {file_path} to aidetect service at {url}...")

        # Detect Content-Type using mime magic and/or file extension
        content_type = None
        try:
            import magic

            content_type = magic.from_file(str(file_path), mime=True)
        except (ImportError, Exception):
            pass

        if not content_type:
            import mimetypes

            mimetypes.add_type("text/markdown", ".md")
            mimetypes.add_type("text/markdown", ".markdown")
            content_type, _ = mimetypes.guess_type(str(file_path))

        if not content_type:
            content_type = "text/plain"

        # Send post request to aidetect service with raw body content and Content-Type header
        r = httpx.post(
            url,
            content=content,
            headers={"Content-Type": content_type},
            timeout=15.0,
        )

        if r.status_code == 200:
            result = r.json()
            click.echo(f"Result for {file_path.name}:")
            click.echo(f"  AI-Generated: {result['is_ai']}")
            click.echo(f"  Votes: {result['ai_votes']}/{result['total_models']}")
            if "final_score" in result:
                click.echo(f"  Final Score: {result['final_score']:.4f}")
            if "final_confidence" in result:
                click.echo(f"  Final Confidence: {result['final_confidence']:.4f}")
            click.echo("  Predictions:")
            for name, score in result["predictions"].items():
                model_confidence = result.get("confidences", {}).get(name)
                if model_confidence is None:
                    click.echo(f"    - {name}: {score:.4f}")
                else:
                    click.echo(f"    - {name}: {score:.4f} (confidence: {model_confidence:.4f})")
        else:
            click.echo(f"Error from service (status {r.status_code}): {r.text}", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
