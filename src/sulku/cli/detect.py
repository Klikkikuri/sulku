"""
CLI Detect Command
==================

This module defines the CLI command to run AI text detection on a file or URL.
"""

from pathlib import Path
import sys
from typing import Tuple

import click
import httpx
import trafilatura


def fetch_url_content(url: str) -> str:
    """
    Fetch webpage content and convert it to markdown.

    :param url: The URL of the webpage to fetch.
    :raises RuntimeError: If fetching or extraction fails.
    :return: The extracted markdown content.
    """
    click.echo(f"Fetching content from {url}...")
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise RuntimeError(f"Failed to fetch content from URL {url}")

    content = trafilatura.extract(downloaded, output_format="markdown")
    if not content:
        raise RuntimeError(f"Failed to extract markdown content from URL {url}")

    return content


def read_file_content(file_path: Path) -> Tuple[str, str]:
    """
    Read file content and detect its mime content type.

    :param file_path: Path to the local file.
    :raises FileNotFoundError: If the file does not exist.
    :raises IsADirectoryError: If the path is a directory.
    :return: A tuple of (content, content_type).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File '{file_path}' does not exist.")
    if file_path.is_dir():
        raise IsADirectoryError(f"'{file_path}' is a directory. Please provide a file or URL.")

    content = file_path.read_text(encoding="utf-8")

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

    return content, content_type


def load_input(path_or_url: str) -> Tuple[str, str, str]:
    """
    Load content from either a local file path or a remote URL.

    :param path_or_url: A file path or URL.
    :return: A tuple of (content, content_type, display_name).
    """
    if path_or_url.startswith(("http://", "https://")):
        content = fetch_url_content(path_or_url)
        return content, "text/markdown", path_or_url
    else:
        file_path = Path(path_or_url)
        content, content_type = read_file_content(file_path)
        return content, content_type, file_path.name


def detect_text(api_url: str, content: str, content_type: str) -> dict:
    """
    Send text content to the aidetect service for classification.

    :param api_url: The URL of the aidetect service.
    :param content: The raw text content to analyze.
    :param content_type: The Content-Type header value.
    :raises httpx.HTTPError: If the HTTP request fails.
    :return: The JSON response dictionary from the service.
    """
    r = httpx.post(
        api_url,
        content=content,
        headers={"Content-Type": content_type},
        timeout=15.0,
    )
    r.raise_for_status()
    return r.json()


def print_detection_result(display_name: str, result: dict) -> None:
    """
    Format and print the AI detection results to standard output.

    :param display_name: The name/identifier of the analyzed item.
    :param result: The JSON response dictionary from the service.
    """
    click.echo(f"Result for {display_name}:")
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

    if "paragraphs" in result:
        click.echo("  Paragraphs:")
        for idx, para in enumerate(result["paragraphs"], start=1):
            snippet = para["text"].replace("\n", " ")
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            
            score_str = f"{para['final_score']:.4f}" if para["final_score"] is not None else "Excluded"
            click.echo(f"    - Paragraph {idx} ({len(para['sentences'])} sentences): score: {score_str} | \"{snippet}\"")



@click.command(name="detect")
@click.argument("path_or_url", type=str)
@click.option(
    "--url",
    type=str,
    default="http://127.0.0.1:8000/api/v1/aidetect/",
    help="The URL of the aidetect service endpoint.",
)
def detect_cmd(path_or_url: str, url: str) -> None:
    """
    Detect if the text in a file or web page is AI-generated.

    If given a URL, fetches the page using trafilatura and converts it
    to markdown before sending to the aidetect service.
    """
    try:
        content, content_type, display_name = load_input(path_or_url)
        click.echo(f"Sending {display_name} to aidetect service at {url}...")
        result = detect_text(url, content, content_type)
        print_detection_result(display_name, result)
    except httpx.HTTPStatusError as e:
        click.echo(f"Error from service (status {e.response.status_code}): {e.response.text}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
