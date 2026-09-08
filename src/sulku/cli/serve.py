"""
CLI Serve Command
=================

This module defines the CLI command to start the FastAPI HTTP server.
"""

import os
import sys
from pathlib import Path

import click
import uvicorn

import sulku
from sulku.http import create_app


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


@click.command(name="serve")
@click.option(
    "-h",
    "--host",
    type=str,
    default=DEFAULT_HOST,
    help="Host to bind the server to.",
)
@click.option(
    "-p",
    "--port",
    type=int,
    default=DEFAULT_PORT,
    help="Port to bind the server to.",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload for development.",
)
@click.option(
    "--reload-dir",
    "reload_dirs",
    multiple=True,
    type=click.Path(),
    help="Directory to watch in reload mode. Repeatable. Defaults to the sulku package source.",
)
@click.option(
    "--preload",
    is_flag=True,
    help="Load all models eagerly at startup.",
)
@click.option(
    "--keep-alive",
    type=float,
    default=300.0,
    show_default=True,
    help="Idle TTL in seconds before a model is evicted. -1 disables TTL.",
)
@click.option(
    "--max-concurrent",
    type=int,
    default=4,
    show_default=True,
    help="Max concurrent classification tasks (others queue).",
)
@click.option(
    "--max-queue",
    type=int,
    default=32,
    show_default=True,
    help="Max queued requests before 503 is returned.",
)
def serve_cmd(
    host: str,
    port: int,
    reload: bool,
    reload_dirs: tuple[str, ...],
    preload: bool,
    keep_alive: float,
    max_concurrent: int,
    max_queue: int,
) -> None:
    """
    Start the FastAPI HTTP server.
    """
    try:
        click.echo(f"Starting server on {host}:{port} (reload={reload}, preload={preload})...")
        os.environ.setdefault("SULKU_KEEP_ALIVE", str(keep_alive))
        os.environ.setdefault("SULKU_MAX_CONCURRENT", str(max_concurrent))
        os.environ.setdefault("SULKU_MAX_QUEUE", str(max_queue))
        if preload:
            os.environ.setdefault("SULKU_PRELOAD", "true")

        if reload:
            # Uvicorn watches the working directory by default. In the container that is /app,
            # which also holds the .venv volume (~27k files) and the /app/data mount — a watch
            # surface that costs CPU permanently and never contains reloadable source. Watch the
            # package source only.
            watch_dirs = list(reload_dirs) or [str(Path(sulku.__file__).resolve().parent)]
            uvicorn.run(
                "sulku.http:create_app_from_env",
                host=host,
                port=port,
                reload=True,
                reload_dirs=watch_dirs,
                factory=True,
            )
        else:
            app = create_app(preload=preload)
            uvicorn.run(app, host=host, port=port)
    except Exception as e:
        click.echo(f"Error starting server: {e}", err=True)
        sys.exit(1)

