"""
CLI Command Group
=================

This package provides the click CLI command group and registers all CLI commands.
"""

import click
from dotenv import load_dotenv

from sulku.cli.detect import detect_cmd
from sulku.cli.generate_fasttext import generate_fasttext_cmd
from sulku.cli.generate_synthetic import generate_synthetic_cmd
from sulku.cli.sample import sample_cmd
from sulku.cli.serve import serve_cmd
from sulku.cli.test import test_group


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

    level_name = "DEBUG" if debug else log_level.upper()
    from sulku.bootstrap import Settings, setup

    # Activate setup context for CLI execution; with_resource ensures teardown on exit
    ctx = click.get_current_context()
    _, _ = ctx.with_resource(setup(Settings(log_level=level_name, service_name="sulku-cli")))


# Register subcommands
main.add_command(sample_cmd)
main.add_command(generate_synthetic_cmd)
main.add_command(generate_fasttext_cmd)
main.add_command(serve_cmd)
main.add_command(detect_cmd)
main.add_command(test_group)
