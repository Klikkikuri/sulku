"""
Sulku Health Probe
==================

Standalone liveness probe for the Sulku HTTP service, installed as the ``sulku-health`` command.

The probe lives outside the ``sulku`` package on purpose: importing ``sulku`` pulls in the dataset,
model and telemetry stack, which costs seconds of start-up on every run. The container healthcheck
runs this every few seconds, so it may only import ``click`` and an HTTP client.
"""

import json

import click
import urllib3

#: Defaults of ``sulku serve``, duplicated here to avoid importing ``sulku.cli.serve``.
DEFAULT_URL = "http://127.0.0.1:8000/health"


@click.command("sulku-health")
@click.option(
    "--url",
    default=DEFAULT_URL,
    show_default=True,
    help="Health check URL.",
)
@click.option(
    "--timeout",
    type=float,
    default=5.0,
    show_default=True,
    help="Request timeout in seconds.",
)
def main(url: str, timeout: float) -> None:
    """
    Check the health of the Sulku HTTP service.
    """
    http = urllib3.PoolManager()
    try:
        response = http.request("GET", url, timeout=timeout)
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


if __name__ == "__main__":
    main()
