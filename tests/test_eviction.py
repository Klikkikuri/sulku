"""
Tests for sulku.eviction module.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from sulku.eviction import (
    PSI_CRITICAL_FULL,
    PSI_HARD_THRESHOLD,
    PSI_SOFT_THRESHOLD,
    PsiSnapshot,
    eviction_loop,
    read_memory_psi,
)


def test_read_memory_psi():
    """Verify read_memory_psi returns PsiSnapshot or None."""
    res = read_memory_psi()
    if res is not None:
        assert isinstance(res, PsiSnapshot)
        assert res.some_avg10 >= 0.0
        assert res.full_avg10 >= 0.0
        assert isinstance(res.source, str)


class MockPredictionService:
    def __init__(self):
        self.default_keep_alive = 300.0
        self.min_loaded_time = 0.0  # 0 for testing cooldown skip
        self.models = {"model_a": MagicMock(), "model_b": MagicMock()}
        self._last_used = {"model_a": 100.0, "model_b": 200.0}
        self._loaded_at = {"model_a": 50.0, "model_b": 50.0}
        self._in_flight = {"model_a": 0, "model_b": 0}
        self._keep_alive = {}
        self.evicted = []

    def get_keep_alive(self, name: str) -> float:
        return self._keep_alive.get(name, self.default_keep_alive)

    def evict_model(self, name: str, reason: str = "eviction") -> None:
        if name in self.models:
            del self.models[name]
            self._last_used.pop(name, None)
            self._loaded_at.pop(name, None)
            self.evicted.append((name, reason))


@pytest.mark.slow
@pytest.mark.anyio
async def test_eviction_loop_ttl(monkeypatch):
    """Test TTL expiration in eviction_loop."""
    service = MockPredictionService()
    service._last_used["model_a"] = 0.0  # long ago
    service._last_used["model_b"] = 1000.0  # recent

    monkeypatch.setattr("time.monotonic", lambda: 500.0)
    monkeypatch.setattr("sulku.eviction.read_memory_psi", lambda: None)

    task = asyncio.create_task(eviction_loop(service, poll_interval=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert ("model_a", "ttl") in service.evicted
    assert "model_b" in service.models


@pytest.mark.slow
@pytest.mark.anyio
async def test_eviction_loop_psi_critical(monkeypatch):
    """Test PSI critical full stall emergency eviction."""
    service = MockPredictionService()
    service._in_flight["model_b"] = 1  # model_b in flight

    monkeypatch.setattr("time.monotonic", lambda: 500.0)
    monkeypatch.setattr(
        "sulku.eviction.read_memory_psi",
        lambda: PsiSnapshot(some_avg10=50.0, full_avg10=10.0, source="test"),
    )

    task = asyncio.create_task(eviction_loop(service, poll_interval=0.01))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert ("model_a", "psi_critical") in service.evicted
    assert "model_b" in service.models  # preserved due to in_flight > 0
