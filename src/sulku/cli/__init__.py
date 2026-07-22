"""
CLI Command Group
=================

This package provides the click CLI command group and registers all CLI commands.
"""

import logging
import sys
import click
from dotenv import load_dotenv

from sulku.cli.detect import detect_cmd
from sulku.cli.generate_fasttext import generate_fasttext_cmd
from sulku.cli.generate_synthetic import generate_synthetic_cmd
from sulku.cli.sample import sample_cmd
from sulku.cli.serve import serve_cmd
from sulku.dataset import SyntheticDatasetGenerator


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


# Register subcommands
main.add_command(sample_cmd)
main.add_command(generate_synthetic_cmd)
main.add_command(generate_fasttext_cmd)
main.add_command(serve_cmd)
main.add_command(detect_cmd)
