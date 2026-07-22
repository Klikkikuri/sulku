"""
CLI Serve Command
=================

This module defines the CLI command to start the FastAPI HTTP server.
"""

import sys
import click
import uvicorn


@click.command(name="serve")
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
def serve_cmd(host: str, port: int, reload: bool) -> None:
    """
    Start the FastAPI HTTP server.
    """
    try:
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
