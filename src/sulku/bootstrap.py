import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Literal, cast

from platformdirs import user_data_dir

from niitti import (
    LoggingSettings,
    SettingsProxy,
    TelemetrySettings,
    get_logger,
    setup_logging,
    setup_tracing,
)
from niitti.sentry import setup_sentry
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from sulku.models import ModelSpec, ModelStore, select_candidate

logger = get_logger(__name__)

LogLevelType = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_setup_active: bool = False
_active_settings: "Settings | None" = None

# Default data directory resolved via platformdirs (falls back to XDG_DATA_HOME)
def _get_default_data_dir() -> Path:

    _data_dir = os.environ.get("SULKU_DATA_DIR")
    _fallback_data_dir = user_data_dir(appname="sulku", appauthor="klikkikuri")

    if _data_dir and Path(_data_dir).exists():
        DATA_DIR = Path(_data_dir)
    else:
        DATA_DIR = Path(_fallback_data_dir)

    if not DATA_DIR.exists():
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            logger.debug("Created default data directory at '%s'", DATA_DIR)
        except Exception as e:
            logger.error("Failed to create default data directory at '%s': %s", DATA_DIR, e)
    return DATA_DIR

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables with ``SULKU_`` prefix or explicit keyword arguments.
    """

    model_config = SettingsConfigDict(
        env_prefix="SULKU_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    preload: bool = False
    keep_alive: float = 300.0
    max_concurrent: int = 4
    max_queue: int = 32
    log_level: LogLevelType = "INFO"
    service_name: str = "sulku"

    data_dir: Path = Field(default_factory=_get_default_data_dir)
    model_cache_dir: Path = Field(default_factory=lambda: _get_default_data_dir() / "models")

    # TODO: Make clear thease are LLM model names, not inference models. Move out?
    default_model: str = "gemini-3.1-flash-lite"
    summarize_model: str = "gemini-3.1-flash-lite"

    models: list[ModelSpec] | None = None

    @model_validator(mode="after")
    def _discover_models(self) -> "Settings":
        """
        Discover models from ``model_cache_dir`` if no models were explicitly provided.
        """
        if self.models is None or len(self.models) == 0:
            discovered: list[ModelSpec] = []
            if self.model_cache_dir.exists() and self.model_cache_dir.is_dir():
                for entry in sorted(self.model_cache_dir.iterdir()):
                    if entry.is_dir():
                        model_file = select_candidate(entry)
                        if model_file is not None:
                            discovered.append(ModelSpec(name=entry.name, source=model_file))
            self.models = discovered


            logger.debug("Discovered %d models in cache directory '%s'",
                len(discovered),
                self.model_cache_dir,
                models=[m.name for m in discovered],
            )
        return self


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


def get_source_dir() -> Path:
    """Return the configured YLE source data directory from active ``Settings``."""
    s = get_settings()
    return (s.data_dir if s is not None else _get_default_data_dir()) / "yle"


def get_dest_dir_base() -> Path:
    """Return the configured synthetic dataset base directory from active ``Settings``."""
    s = get_settings()
    return (s.data_dir if s is not None else _get_default_data_dir()) / "genai"


settings: SettingsProxy = SettingsProxy(get_settings)


@contextmanager
def setup(
    settings_instance: Settings | None = None,
) -> Generator[tuple[Settings, ModelStore], None, None]:
    """
    Bootstrap application configuration and settings for sulku.

    Instantiates and activates ``Settings`` and ``ModelStore`` on entry, configures niitti logging,
    tracing, and sentry integrations, yielding a tuple of (settings, model_store), and cleans
    up active settings state on context exit.

    :param settings_instance: Optional pre-configured ``Settings`` instance. If omitted, default settings are loaded.
    :yield: Tuple of (active ``Settings`` instance, active ``ModelStore`` instance).
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

    store = ModelStore(
        models=settings_obj.models or [],
        cache_dir=settings_obj.model_cache_dir,
    )

    setup_logging(LoggingSettings(LOG_LEVEL=cast(LogLevelType, settings_obj.log_level)))
    setup_tracing(TelemetrySettings(service_name=settings_obj.service_name))
    setup_sentry()

    try:
        yield settings_obj, store
    finally:
        _setup_active = prev_setup
        if not prev_setup:
            clear_settings()
