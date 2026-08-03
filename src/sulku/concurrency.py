"""
Concurrency
===========

Provides two primitives for burst-safe classification:

* ``ClassifySemaphore`` — caps concurrent active classifications and tracks
  queue depth for metrics and load-shedding decisions.
* ``EWMARate`` — exponentially-weighted moving average request rate used for
  speculative pre-load on burst onset.
"""

import asyncio
import math
import time
from dataclasses import dataclass, field


@dataclass
class EWMARate:
    """Thread-safe EWMA request-rate estimator (requests / second).

    Uses a half-life decay so the weight of old observations halves every
    *half_life* seconds, giving a responsive but smooth rate signal.
    """

    half_life: float = 30.0  # seconds; tune to expected idle gap length
    _rate: float = field(default=0.0, init=False, repr=False)
    # Initialise to 0.0 so that `elapsed` on the first tick equals the full
    # time since service start — a large idle gap — which correctly triggers
    # burst-onset detection via the `was_idle` fallback below.
    _last_tick: float = field(default=0.0, init=False, repr=False)

    @property
    def rate(self) -> float:
        """Current smoothed rate (requests / second)."""
        return self._rate

    def tick(self) -> bool:
        """Record one request arrival. Returns True on burst onset.

        Burst onset is detected in two complementary ways:

        1. **Idle-gap fallback** — if no request arrived for > ``2 × half_life``
           seconds the EWMA rate has decayed to effectively zero, but the first
           incoming request would produce a tiny ``instant`` value
           (``1 / elapsed ≈ 0``) that never crosses the ``0.01`` threshold.
           Detecting the gap directly means the very first request of a burst is
           never missed, regardless of the EWMA value.

        2. **EWMA threshold crossing** — for sustained bursts, rate rises above
           ``0.01`` req/s, covering the second and subsequent requests.
        """
        now = time.monotonic()
        elapsed = now - self._last_tick
        self._last_tick = now
        # Long idle gap: EWMA is near zero AND instant ≈ 0, so threshold
        # crossing won't fire. Detect onset directly from the gap size.
        was_idle = elapsed > 2 * self.half_life
        decay = math.exp(-math.log(2) * elapsed / self.half_life)
        prev = self._rate
        instant = 1.0 / max(elapsed, 1e-6)
        self._rate = decay * self._rate + (1 - decay) * instant
        return was_idle or (prev < 0.01 and self._rate >= 0.01)


class ClassifySemaphore:
    """Bounded concurrency gate with queue-depth tracking.

    Wraps an asyncio.Semaphore to:
    * cap simultaneous classify tasks to *max_concurrent*
    * expose ``queue_depth`` for metrics and load-shedding
    * raise ``OverloadError`` before queuing if *max_queue* is exceeded
    """

    class OverloadError(Exception):
        """Raised when the request queue is full and load-shedding is active."""

    def __init__(self, max_concurrent: int = 4, max_queue: int = 32) -> None:
        self._sem = asyncio.Semaphore(max_concurrent)
        self.max_queue = max_queue
        self._waiting: int = 0

    @property
    def queue_depth(self) -> int:
        return self._waiting

    async def __aenter__(self) -> "ClassifySemaphore":
        if self._waiting >= self.max_queue:
            raise self.OverloadError(
                f"Classify queue full ({self._waiting}/{self.max_queue}); retry later."
            )
        self._waiting += 1
        try:
            await self._sem.acquire()
        finally:
            self._waiting -= 1
        return self

    async def __aexit__(self, *_) -> None:
        self._sem.release()
