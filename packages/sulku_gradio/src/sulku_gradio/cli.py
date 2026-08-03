"""
CLI Entry Point for Sulku Gradio UI
===================================

This module provides the command line interface to start the Gradio UI server.
"""

import sys
import click
from sulku_gradio.app import create_ui


@click.command(name="sulku-gradio")
@click.option(
    "--api-url",
    type=str,
    default="http://127.0.0.1:8000",
    help="Base URL of the running Sulku HTTP API server.",
)
@click.option(
    "-h",
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host address to bind Gradio UI server.",
)
@click.option(
    "-p",
    "--port",
    type=int,
    default=7860,
    help="Port to bind Gradio UI server.",
)
@click.option(
    "--share",
    is_flag=True,
    help="Generate a publicly shareable Gradio link.",
)
def main(api_url: str, host: str, port: int, share: bool) -> None:
    """
    Launch the Sulku Gradio Web UI server.
    """
    try:
        click.echo(f"Starting Sulku Gradio UI on http://{host}:{port}")
        click.echo(f"Targeting Sulku API at {api_url}")
        app = create_ui(default_api_url=api_url)
        app.launch(server_name=host, server_port=port, share=share)
    except Exception as exc:
        click.echo(f"Error starting Gradio UI: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
