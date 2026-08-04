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
    mock_store = MagicMock()
    mock_package = MagicMock()
    mock_package.path = MagicMock()
    mock_package.path.exists.return_value = True
    mock_store.get_spec.return_value = MagicMock()
    mock_store.get.return_value = mock_package

    service = PredictionService(store=mock_store)
    dummy_model = MagicMock()

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
    mock_store = MagicMock()
    mock_store.get_spec.return_value = None
    service = PredictionService(store=mock_store)
    with pytest.raises(KeyError, match="Unknown model"):
        await service.ensure_loaded("non_existent_model")


def test_evict_model_and_unload_model(tmp_path):
    """Verify evict_model cleans up internal tracking state and unload_model enforces existence."""
    from sulku.models import ModelStore
    store = ModelStore(models=[], cache_dir=tmp_path)
    service = PredictionService(store=store)
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


def test_ref_counting_track_in_flight(tmp_path):
    """Verify track_in_flight context manager updates in_flight counters correctly."""
    from sulku.models import ModelStore
    store = ModelStore(models=[], cache_dir=tmp_path)
    service = PredictionService(store=store)
    name = "m1"

    assert service._in_flight.get(name, 0) == 0

    with service.track_in_flight(name):
        assert service._in_flight[name] == 1
        with service.track_in_flight(name):
            assert service._in_flight[name] == 2
        assert service._in_flight[name] == 1

    assert service._in_flight[name] == 0


def test_get_keep_alive_fallback(tmp_path):
    """Verify get_keep_alive respects per-model TTL overrides and defaults."""
    from sulku.models import ModelStore
    store = ModelStore(models=[], cache_dir=tmp_path)
    service = PredictionService(store=store)
    assert service.get_keep_alive("model_a") == 300.0

    service._keep_alive["model_a"] = 60.0
    assert service.get_keep_alive("model_a") == 60.0
    assert service.get_keep_alive("model_b") == 300.0


@pytest.mark.anyio
async def test_api_load_unload(monkeypatch, tmp_path):
    """Round-trip test for model load/unload API endpoints."""
    from fastapi.testclient import TestClient
    from sulku.http import create_app

    mock_model = MagicMock()
    monkeypatch.setattr("fasttext.load_model", lambda path: mock_model)

    dummy_model_file = tmp_path / "gemini-3.1-flash-lite.ftz"
    dummy_model_file.touch()
    monkeypatch.setenv("SULKU_MODELS", f'[{{"name": "gemini-3.1-flash-lite", "source": "{dummy_model_file}"}}]')

    with TestClient(create_app()) as client:
        # Initial status: not loaded
        resp = client.get("/api/v1/aidetect/models")
        assert resp.status_code == 200
        target = resp.json()["models"][0]["name"]
        assert resp.json()["models"][0]["loaded"] is False

        # Load model via endpoint
        load_resp = client.post(f"/api/v1/aidetect/models/{target}/load")
        assert load_resp.status_code == 200
        assert load_resp.json() == {"model": target, "loaded": True}

        # Verify load status
        resp = client.get("/api/v1/aidetect/models")
        assert any(m["name"] == target and m["loaded"] is True for m in resp.json()["models"])

        # Unload model via endpoint
        unload_resp = client.post(f"/api/v1/aidetect/models/{target}/unload")
        assert unload_resp.status_code == 200
        assert unload_resp.json() == {"model": target, "loaded": False}

        # Unload non-loaded model returns 404
        err_resp = client.post(f"/api/v1/aidetect/models/{target}/unload")
        assert err_resp.status_code == 404


def test_metrics_endpoint():
    """Verify GET /metrics returns 200 and Prometheus metrics format."""
    from fastapi.testclient import TestClient
    from sulku.http import create_app

    with TestClient(create_app()) as client:
        resp = client.get("/metrics/")
        assert resp.status_code == 200
        assert "sulku_model_loads_total" in resp.text


@pytest.mark.anyio
async def test_semaphore_overload():
    """Verify ClassifySemaphore raises OverloadError when queue is full."""
    from sulku.concurrency import ClassifySemaphore

    sem = ClassifySemaphore(max_concurrent=1, max_queue=1)

    # Acquire the single concurrent slot
    await sem.__aenter__()

    # Create worker to enter waiting queue (queue depth = 1)
    queue_task = asyncio.create_task(sem.__aenter__())
    await asyncio.sleep(0.01)
    assert sem.queue_depth == 1

    # Next call exceeds max_queue=1 and raises OverloadError immediately
    with pytest.raises(ClassifySemaphore.OverloadError, match="queue full"):
        await sem.__aenter__()

    # Clean up tasks
    await sem.__aexit__()
    await queue_task
    await sem.__aexit__()

