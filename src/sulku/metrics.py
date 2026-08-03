"""
Metrics
=======

OpenTelemetry instrument registry for sulku.

All instruments are created once at import time from a shared ``Meter``.
Other modules import specific instruments from here rather than creating
their own, preventing duplicate-registration errors.

Prometheus metrics are exposed at ``GET /metrics`` via the FastAPI app.
"""

import time as _time
from opentelemetry import metrics as otel_metrics
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader

_reader = PrometheusMetricReader()
_provider = MeterProvider(metric_readers=[_reader])
otel_metrics.set_meter_provider(_provider)
meter = otel_metrics.get_meter("sulku", version="0.1.0")


# ── Observable gauge callbacks ────────────────────────────────────────────────
# Each callback is a top-level generator function using *lazy imports* inside
# its body to avoid circular imports.


def _model_loaded_cb(opts: CallbackOptions):
    from sulku.prediction import prediction_service
    from sulku.constants import MODEL_PATHS
    for name in MODEL_PATHS:
        yield Observation(1 if name in prediction_service.models else 0, {"model": name})


def _model_last_used_cb(opts: CallbackOptions):
    from sulku.prediction import prediction_service
    now_mono = _time.monotonic()
    now_wall = _time.time()
    for name, mono in prediction_service._last_used.items():
        yield Observation(now_wall - (now_mono - mono), {"model": name})


def _queue_depth_cb(opts: CallbackOptions):
    from sulku.http import _semaphore
    yield Observation(_semaphore.queue_depth, {})


def _psi_some_cb(opts: CallbackOptions):
    from sulku.eviction import read_memory_psi
    snap = read_memory_psi()
    if snap:
        yield Observation(snap.some_avg10, {})


def _psi_full_cb(opts: CallbackOptions):
    from sulku.eviction import read_memory_psi
    snap = read_memory_psi()
    if snap:
        yield Observation(snap.full_avg10, {})


def _request_rate_cb(opts: CallbackOptions):
    from sulku.http import _rate
    yield Observation(_rate.rate, {})


# ── Model lifecycle ───────────────────────────────────────────────────────────
model_loads = meter.create_counter(
    "sulku_model_loads_total",
    description="Number of times a model has been loaded into memory.",
)
model_unloads = meter.create_counter(
    "sulku_model_unloads_total",
    description="Number of times a model has been unloaded. 'reason' label: "
                "ttl | psi_soft | psi_hard | psi_critical | manual",
)
model_loaded_gauge = meter.create_observable_gauge(
    "sulku_model_loaded",
    callbacks=[_model_loaded_cb],
    description="1 if the model is currently in memory, 0 otherwise.",
)
model_last_used = meter.create_observable_gauge(
    "sulku_model_last_used_timestamp",
    callbacks=[_model_last_used_cb],
    description="Unix timestamp of the most recent classification using this model.",
)

# ── Per-request ───────────────────────────────────────────────────────────────
classification_duration = meter.create_histogram(
    "sulku_classification_duration_seconds",
    description="End-to-end classification time per model.",
    unit="s",
)
classifications_total = meter.create_counter(
    "sulku_classifications_total",
    description="Classifications completed. 'status' label: success | error",
)
model_in_flight = meter.create_up_down_counter(
    "sulku_model_in_flight_requests",
    description="Number of active classification requests per model.",
)

# ── Queue / concurrency ───────────────────────────────────────────────────────
queue_depth = meter.create_observable_gauge(
    "sulku_classify_queue_depth",
    callbacks=[_queue_depth_cb],
    description="Requests waiting for the concurrency semaphore.",
)
requests_shed = meter.create_counter(
    "sulku_requests_shed_total",
    description="Requests rejected with 503 due to overload.",
)

# ── Pressure signals ──────────────────────────────────────────────────────────
psi_some_avg10 = meter.create_observable_gauge(
    "sulku_memory_pressure_some_avg10",
    callbacks=[_psi_some_cb],
    description="PSI memory some.avg10 (% time at least one task stalled).",
    unit="%",
)
psi_full_avg10 = meter.create_observable_gauge(
    "sulku_memory_pressure_full_avg10",
    callbacks=[_psi_full_cb],
    description="PSI memory full.avg10 (% time all tasks stalled).",
    unit="%",
)

# ── Traffic shape ─────────────────────────────────────────────────────────────
request_rate_ewma = meter.create_observable_gauge(
    "sulku_request_rate_ewma",
    callbacks=[_request_rate_cb],
    description="Exponentially-weighted moving average of classify requests per second.",
)


