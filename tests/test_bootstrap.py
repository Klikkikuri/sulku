"""
Tests for Bootstrap and Settings Management
============================================

Tests for Settings instantiation, setup() context manager, settings proxy,
and integration into FastAPI application creation/lifespan.
"""

import pytest
from fastapi.testclient import TestClient

from sulku.bootstrap import (
    Settings,
    clear_settings,
    get_settings,
    set_active_settings,
    settings,
    setup,
)
from sulku.http import create_app
from sulku.prediction import prediction_service


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

    custom_settings = Settings(preload=True, keep_alive=60.0)
    with setup(custom_settings) as s:
        assert s is custom_settings
        assert settings.preload is True
        assert settings.keep_alive == 60.0
        assert get_settings() is custom_settings

    assert get_settings() is None


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
    with TestClient(app):
        # Inside active lifespan
        assert get_settings() is not None
        assert prediction_service.default_keep_alive == 300.0
