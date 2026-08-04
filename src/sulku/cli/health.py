"""
Health check CLI command.
"""

import json
import click
import urllib3

from sulku.cli.serve import DEFAULT_HOST, DEFAULT_PORT


@click.command("health")
@click.option(
    "--url",
    default=f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/health",
    show_default=True,
    help="Health check URL.",
)
def health_cmd(url: str) -> None:
    """
    Check the health of the Sulku HTTP service.
    """
    http = urllib3.PoolManager()
    try:
        response = http.request("GET", url, timeout=5.0)
        if response.status != 200:
            click.echo(f"HEALTH CHECK FAILED: HTTP {response.status}", err=True)
            raise click.Abort()

        data = json.loads(response.data.decode("utf-8"))
        if data.get("status") == "ok":
            click.echo("OK")
            return

        click.echo(f"UNHEALTHY: unexpected payload {data}", err=True)
        raise click.Abort()
    except click.Abort:
        raise
    except Exception as exc:
        click.echo(f"HEALTH CHECK FAILED: {exc}", err=True)
        raise click.Abort() from exc
