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


# ── Model lifecycle ───────────────────────────────────────────────────────────
model_loaded_gauge = meter.create_observable_gauge(
    "sulku_model_loaded",
    callbacks=[_model_loaded_cb],
    description="1 if the model is currently in memory, 0 otherwise.",
)


# TODO: Implement additional metrics instruments when their corresponding modules are created:
# - model_loads (counter: sulku_model_loads_total)
# - model_unloads (counter: sulku_model_unloads_total)
# - model_last_used (observable gauge: sulku_model_last_used_timestamp)
# - classification_duration (histogram: sulku_classification_duration_seconds)
# - classifications_total (counter: sulku_classifications_total)
# - model_in_flight (up_down_counter: sulku_model_in_flight_requests)
# - queue_depth (observable gauge: sulku_classify_queue_depth)
# - requests_shed (counter: sulku_requests_shed_total)
# - psi_some_avg10 (observable gauge: sulku_memory_pressure_some_avg10)
# - psi_full_avg10 (observable gauge: sulku_memory_pressure_full_avg10)
# - request_rate_ewma (observable gauge: sulku_request_rate_ewma)
