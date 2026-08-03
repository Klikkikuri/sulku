"""
Eviction
========

PSI-aware model eviction loop for PredictionService.

Eviction is a single operation
------------------------------
fastText 0.9.3 (Facebook C++ via pybind11) loads models with ``std::ifstream``
into ``AlignedVector<float>`` heap memory. No mmap is involved. Calling
``del model`` drops the Python reference; once the pybind11 ``shared_ptr``
refcount hits zero the C++ destructor frees the heap immediately (~92 MB for
the current model, measured via ``/proc/self/smaps``). There is no warm
page-cache layer to preserve — ``madvise(MADV_DONTNEED)`` has no applicable
target and is not used.

The PSI tiers below differ in **which** models to evict (LRU first /
all-but-MRU / all not in-flight), not in eviction depth.

Three PSI tiers
---------------
* **Soft** (some_avg10 ≥ 10 %): LRU-evict idle models ahead of their TTL.
* **Hard** (some_avg10 ≥ 40 %): evict all models except the most-recently-used.
* **Critical** (full_avg10 ≥ 5 %): emergency evict everything not in-flight.

PSI source
----------
Cgroup v2 ``/sys/fs/cgroup/memory.pressure`` is preferred over system-wide
``/proc/pressure/memory`` — it reflects pressure within the container only.
Both are confirmed available on this host (kernel 7.0.13, cgroup v2).
"""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Protocol

from niitti import get_logger

logger = get_logger(__name__)

_PSI_CANDIDATES = [
    Path("/sys/fs/cgroup/memory.pressure"),  # cgroup v2 — container-scoped
    Path("/proc/pressure/memory"),           # system-wide fallback
]

PSI_SOFT_THRESHOLD: float = 10.0   # % some.avg10
PSI_HARD_THRESHOLD: float = 40.0   # % some.avg10
PSI_CRITICAL_FULL:  float = 5.0    # % full.avg10


@dataclass
class PsiSnapshot:
    """Parsed PSI memory metric at one point in time."""
    some_avg10: float
    full_avg10: float
    source: str  # path that was read, for logging


def read_memory_psi() -> PsiSnapshot | None:
    """Read memory PSI from the best available source. Returns None on non-Linux."""
    for path in _PSI_CANDIDATES:
        try:
            text = path.read_text()
            some = full = 0.0
            for line in text.splitlines():
                parts = dict(p.split("=") for p in line.split() if "=" in p)
                if line.startswith("some"):
                    some = float(parts.get("avg10", 0))
                elif line.startswith("full"):
                    full = float(parts.get("avg10", 0))
            return PsiSnapshot(some_avg10=some, full_avg10=full, source=str(path))
        except OSError:
            continue
    return None


class EvictionTarget(Protocol):
    """Structural interface that PredictionService must satisfy."""
    default_keep_alive: float
    min_loaded_time: float
    models: Dict[str, Any]
    _last_used: Dict[str, float]
    _loaded_at: Dict[str, float]
    _in_flight: Dict[str, int]
    def get_keep_alive(self, name: str) -> float: ...
    def evict_model(self, name: str, reason: str = "eviction") -> None: ...


async def eviction_loop(service: EvictionTarget, poll_interval: float = 10.0) -> None:
    """Background coroutine: evict models based on TTL and PSI pressure.

    Checks every *poll_interval* seconds (default 10 s — cheap enough to
    be PSI-responsive without meaningful overhead).

    Skip-conditions applied before any eviction:
    * Model has active in-flight requests (ref count > 0).
    * Model was loaded less than *min_loaded_time* seconds ago (cooldown).
    """
    while True:
        await asyncio.sleep(poll_interval)
        now = time.monotonic()
        psi = read_memory_psi()
        loaded = list(service.models.keys())
        if not loaded:
            continue

        def _evictable(name: str) -> bool:
            if service._in_flight.get(name, 0) > 0:
                return False  # active request — never evict
            age = now - service._loaded_at.get(name, now)
            if age < service.min_loaded_time:
                return False  # within cooldown window
            return True

        # ── Critical: full stall — evict everything evictable ───────────────
        if psi and psi.full_avg10 >= PSI_CRITICAL_FULL:
            logger.warning(
                "emergency evict all idle models",
                full_avg10=psi.full_avg10,
                source=psi.source,
            )
            for name in [n for n in loaded if _evictable(n)]:
                service.evict_model(name, reason="psi_critical")
            continue

        # ── Hard pressure: evict all but MRU ─────────────────────────────────
        if psi and psi.some_avg10 >= PSI_HARD_THRESHOLD:
            candidates = [n for n in loaded if _evictable(n)]
            if candidates:
                mru = max(loaded, key=lambda n: service._last_used.get(n, 0))
                for name in candidates:
                    if name != mru:
                        service.evict_model(name, reason="psi_hard")
            continue

        # ── Soft pressure or TTL: LRU candidates ─────────────────────────────
        ttl_expired = [
            n for n in loaded
            if _evictable(n)
            and service.get_keep_alive(n) >= 0
            and (now - service._last_used.get(n, 0)) >= service.get_keep_alive(n)
        ]

        if psi and psi.some_avg10 >= PSI_SOFT_THRESHOLD:
            # Under soft pressure, accelerate: include models not yet TTL-expired,
            # sorted coldest-first so the most-idle model goes first.
            idle_sorted = sorted(
                [n for n in loaded if _evictable(n)],
                key=lambda n: service._last_used.get(n, 0),
            )
            # dict.fromkeys preserves insertion order and deduplicates
            for name in list(dict.fromkeys(ttl_expired + idle_sorted)):
                service.evict_model(name, reason="psi_soft")
        else:
            for name in ttl_expired:
                service.evict_model(name, reason="ttl")
