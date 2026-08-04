"""
Tests for Phase 2 adaptive intelligence: EWMARate, speculative pre-load, and _update_adaptive_ttl.
"""

import time
import pytest
from unittest.mock import MagicMock

from sulku.concurrency import EWMARate
from sulku.prediction import PredictionService


def test_ewma_rate_idle_gap():
    """Verify EWMARate triggers burst onset after long idle gap."""
    rate = EWMARate(half_life=10.0)

    # First tick with _last_tick=0.0 -> elapsed > 2 * half_life -> burst onset True
    assert rate.tick() is True


def test_ewma_rate_threshold_crossing(monkeypatch):
    """Verify EWMARate triggers burst onset when rate crosses 0.01 threshold."""
    rate = EWMARate(half_life=30.0)

    start_time = 1000.0
    now = start_time
    monkeypatch.setattr(time, "monotonic", lambda: now)

    # Initial tick at t=1000s (was_idle is True)
    assert rate.tick() is True

    # Reset rate to sub-threshold manually to test threshold crossing
    rate._rate = 0.005

    # Advance time slightly (e.g. 1s) for next request
    now += 1.0
    # instant = 1.0 / 1.0 = 1.0. New rate = decay * 0.005 + (1-decay) * 1.0 > 0.01
    assert rate.tick() is True
    assert rate.rate >= 0.01


def test_adaptive_ttl_update(tmp_path):
    """Verify _update_adaptive_ttl updates keep_alive based on inter-request interval."""
    from sulku.models import ModelStore
    store = ModelStore(models=[], cache_dir=tmp_path)
    service = PredictionService(store=store)
    model_name = "test_model"
    service.default_keep_alive = 300.0

    # Sample 1: elapsed = 10.0s -> s=10.0, n=1, k=3 -> adaptive = 30s, capped at min_ttl (60s)
    service._update_adaptive_ttl(model_name, 10.0)
    assert service.get_keep_alive(model_name) == 60.0

    # Sample 2: elapsed = 100.0s -> s=110.0, n=2 -> avg=55s, k=3 -> adaptive = 165s
    service._update_adaptive_ttl(model_name, 100.0)
    assert service.get_keep_alive(model_name) == pytest.approx(165.0)

    # Max bound test: elapsed = 1000.0s -> large mean -> capped at max_ttl (1800s)
    service._update_adaptive_ttl(model_name, 2000.0)
    assert service.get_keep_alive(model_name) == 1800.0


@pytest.mark.anyio
async def test_ensure_loaded_calls_adaptive_ttl(monkeypatch):
    """Verify ensure_loaded invokes _update_adaptive_ttl on subsequent requests."""
    model_name = "test_model"
    mock_store = MagicMock()
    mock_package = MagicMock()
    mock_package.path = MagicMock()
    mock_store.get_spec.return_value = MagicMock()
    mock_store.get.return_value = mock_package

    service = PredictionService(store=mock_store)
    monkeypatch.setattr("fasttext.load_model", lambda path: MagicMock())

    now = 100.0
    monkeypatch.setattr(time, "monotonic", lambda: now)

    await service.ensure_loaded(model_name)
    assert service._last_used[model_name] == 100.0

    # Second call 50 seconds later
    now = 150.0
    await service.ensure_loaded(model_name)

    assert service._inter_request_n[model_name] == 1
    assert service._inter_request_sum[model_name] == 50.0
    # Mean interval = 50s * 3 = 150s
    assert service.get_keep_alive(model_name) == 150.0
