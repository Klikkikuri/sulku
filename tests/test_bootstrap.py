"""
Tests for Bootstrap and Settings Management
============================================

Tests for Settings instantiation, setup() context manager, settings proxy,
and integration into FastAPI application creation/lifespan.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sulku.bootstrap import (
    Settings,
    clear_settings,
    get_settings,
    settings,
    setup,
)
from sulku.http import create_app
from sulku.models import ModelSpec


def test_settings_defaults():
    """Test default values of Settings."""
    s = Settings()
    assert s.preload is False
    assert s.keep_alive == 300.0
    assert s.max_concurrent == 4
    assert s.max_queue == 32


def test_settings_env_override(monkeypatch):
    """Test reading settings from environment variables with SULKU_ prefix."""
    monkeypatch.setenv("SULKU_PRELOAD", "true")
    monkeypatch.setenv("SULKU_KEEP_ALIVE", "120.0")
    monkeypatch.setenv("SULKU_MAX_CONCURRENT", "8")
    monkeypatch.setenv("SULKU_MAX_QUEUE", "64")

    s = Settings()
    assert s.preload is True
    assert s.keep_alive == 120.0
    assert s.max_concurrent == 8
    assert s.max_queue == 64


def test_settings_proxy_outside_context():
    """Test that accessing settings proxy outside setup context raises RuntimeError."""
    clear_settings()
    with pytest.raises(RuntimeError, match="Working outside of application context"):
        _ = settings.preload


def test_setup_context_manager():
    """Test setup context manager sets active settings and cleans up on exit."""
    clear_settings()
    assert get_settings() is None

    custom_settings = Settings(
        preload=True,
        keep_alive=60.0,
        models=[
            ModelSpec(
                name="gemini-3.1-flash-lite",
                source=Path("dummy.ftz"),
            )
        ],
    )
    with setup(custom_settings) as (s, store):
        assert s is custom_settings
        assert settings.preload is True
        assert settings.keep_alive == 60.0
        assert get_settings() is custom_settings
        assert store is not None
        assert store.get_spec("gemini-3.1-flash-lite") is not None

    assert get_settings() is None


def test_model_discovery_from_cache(tmp_path):
    """Test that model instances are created by scanning model_cache_dir if models is not provided manually."""
    cache_dir = tmp_path / "models"
    model1_dir = cache_dir / "model_a"
    model1_dir.mkdir(parents=True)
    (model1_dir / "model_a.ftz").touch()

    model2_dir = cache_dir / "model_b"
    model2_dir.mkdir(parents=True)
    # Missing .ftz file in model_b directory
    (model2_dir / "other.txt").touch()

    s = Settings(data_dir=tmp_path, model_cache_dir=cache_dir)
    assert s.models is not None
    assert len(s.models) == 1
    assert s.models[0].name == "model_a"
    assert s.models[0].source == model1_dir / "model_a.ftz"




def test_setup_nested():
    """Test nesting setup() context managers."""
    clear_settings()
    s1 = Settings(keep_alive=100.0)
    s2 = Settings(keep_alive=200.0)

    with setup(s1):
        assert settings.keep_alive == 100.0
        with setup(s2):
            assert settings.keep_alive == 200.0
        # Exiting inner setup maintains outer setup active state
        assert settings.keep_alive == 200.0

    assert get_settings() is None


def test_app_lifespan_integration():
    """Test that FastAPI app creation and lifespan runs setup() and configures prediction service."""
    app = create_app()
    with TestClient(app) as client:
        # Inside active lifespan
        assert get_settings() is not None
        assert app.state.prediction_service.default_keep_alive == 300.0

