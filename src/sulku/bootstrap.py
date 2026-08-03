"""
Bootstrap
=========

Application configuration and setup context management for sulku.

Order of precedence:
    1. Environment variables (with ``SULKU_`` prefix)
    2. Explicit ``Settings`` passed to ``setup()``
    3. Default settings values
"""

from contextlib import contextmanager
from typing import Any, Generator, Literal, cast

from niitti import (
    LoggingSettings,
    SettingsProxy,
    TelemetrySettings,
    get_logger,
    setup_logging,
    setup_tracing,
)
from niitti.sentry import setup_sentry
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = get_logger(__name__)

LogLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_setup_active: bool = False
_active_settings: "Settings | None" = None


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables with ``SULKU_`` prefix or explicit keyword arguments.
    """

    model_config = SettingsConfigDict(
        env_prefix="SULKU_",
        extra="ignore",
    )

    preload: bool = False
    keep_alive: float = 300.0
    max_concurrent: int = 4
    max_queue: int = 32
    log_level: LogLevelType = "INFO"
    service_name: str = "sulku"


def get_settings() -> Settings | None:
    """
    Get the currently active application ``Settings`` instance, or ``None`` if outside app context.
    """
    return _active_settings


def set_active_settings(instance: Settings | None) -> None:
    """
    Set the active application ``Settings`` instance.
    """
    global _active_settings
    _active_settings = instance


def clear_settings() -> None:
    """
    Clear the currently active application ``Settings`` instance.
    """
    global _active_settings
    _active_settings = None


settings: Any = SettingsProxy(get_settings)


@contextmanager
def setup(
    settings_instance: Settings | None = None,
) -> Generator[Settings, None, None]:
    """
    Bootstrap application configuration and settings for sulku.

    Instantiates and activates ``Settings`` on entry, configures niitti logging,
    tracing, and sentry integrations, yielding the settings instance, and cleans
    up active settings state on context exit.

    :param settings_instance: Optional pre-configured ``Settings`` instance. If omitted, default settings are loaded.
    :yield: The active ``Settings`` instance.
    """
    global _setup_active

    if _setup_active:
        logger.debug("setup() is already active in this process.")

    prev_setup = _setup_active
    _setup_active = True

    if settings_instance is None:
        active = get_settings()
        settings_obj = active if active is not None else Settings()
    else:
        settings_obj = settings_instance

    set_active_settings(settings_obj)

    setup_logging(LoggingSettings(LOG_LEVEL=cast(LogLevelType, settings_obj.log_level)))
    setup_tracing(TelemetrySettings(service_name=settings_obj.service_name))
    setup_sentry()

    try:
        yield settings_obj
    finally:
        _setup_active = prev_setup
        if not prev_setup:
            clear_settings()
