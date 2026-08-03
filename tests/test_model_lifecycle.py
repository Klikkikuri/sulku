"""
Tests for PredictionService lifecycle methods, model loading/eviction, and ref-counting.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from sulku.prediction import PredictionService


@pytest.mark.anyio
async def test_ensure_loaded_and_idempotent(monkeypatch):
    """Verify ensure_loaded loads model and concurrent calls are idempotent."""
    service = PredictionService()
    dummy_model = MagicMock()

    # Mock fasttext.load_model and MODEL_PATHS
    monkeypatch.setattr("sulku.prediction.MODEL_PATHS", {"mock_model": MagicMock()})
    monkeypatch.setattr("fasttext.load_model", lambda path: dummy_model)

    # First load
    await service.ensure_loaded("mock_model")
    assert "mock_model" in service.models
    assert service.models["mock_model"] == dummy_model
    assert service._in_flight.get("mock_model") == 0
    assert service._loaded_at.get("mock_model") is not None
    assert service._last_used.get("mock_model") is not None

    # Concurrent ensure_loaded calls
    await asyncio.gather(
        service.ensure_loaded("mock_model"),
        service.ensure_loaded("mock_model"),
    )
    assert len(service.models) == 1


@pytest.mark.anyio
async def test_ensure_loaded_unknown_model():
    """Verify ensure_loaded raises KeyError for invalid model names."""
    service = PredictionService()
    with pytest.raises(KeyError, match="Unknown model"):
        await service.ensure_loaded("non_existent_model")


def test_evict_model_and_unload_model():
    """Verify evict_model cleans up internal tracking state and unload_model enforces existence."""
    service = PredictionService()
    service.models["test_model"] = MagicMock()
    service._last_used["test_model"] = 100.0
    service._loaded_at["test_model"] = 50.0
    service._in_flight["test_model"] = 0
    service._keep_alive["test_model"] = 120.0

    # evict_model
    service.evict_model("test_model", reason="ttl")
    assert "test_model" not in service.models
    assert "test_model" not in service._last_used
    assert "test_model" not in service._loaded_at
    assert "test_model" not in service._in_flight
    assert "test_model" not in service._keep_alive

    # evicting non-loaded model is a no-op
    service.evict_model("test_model")

    # unload_model raises KeyError when not loaded
    with pytest.raises(KeyError):
        service.unload_model("test_model")

    # manual unload
    service.models["test_model_2"] = MagicMock()
    service.unload_model("test_model_2")
    assert "test_model_2" not in service.models


def test_ref_counting_track_in_flight():
    """Verify track_in_flight context manager updates in_flight counters correctly."""
    service = PredictionService()
    name = "m1"

    assert service._in_flight.get(name, 0) == 0

    with service.track_in_flight(name):
        assert service._in_flight[name] == 1
        with service.track_in_flight(name):
            assert service._in_flight[name] == 2
        assert service._in_flight[name] == 1

    assert service._in_flight[name] == 0


def test_get_keep_alive_fallback():
    """Verify get_keep_alive respects per-model TTL overrides and defaults."""
    service = PredictionService()
    assert service.get_keep_alive("model_a") == 300.0

    service._keep_alive["model_a"] = 60.0
    assert service.get_keep_alive("model_a") == 60.0
    assert service.get_keep_alive("model_b") == 300.0
