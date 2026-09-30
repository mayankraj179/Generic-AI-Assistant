"""OpenTelemetry setup shared with google-adk.

google-adk creates its spans through the global OpenTelemetry API
(google/adk/telemetry/tracing.py: ``trace.get_tracer(...)``) and only installs
a TracerProvider itself inside its own CLI. So this module installs the one SDK
TracerProvider for the process, and ADK's spans (agent run, call_llm,
execute_tool) nest under this application's spans in the same trace.

Environment:
  OTEL_TRACES_EXPORTER   "none" (default), "otlp" (e.g. local Jaeger), "console"
  OTEL_EXPORTER_OTLP_ENDPOINT  default http://localhost:4318 (OTLP over HTTP)
  OTEL_SERVICE_NAME      default "assistant-framework"
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import Span, Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("assistant_framework")

_configured = False


def configure_tracing() -> None:
    """Idempotent. Never raises: a missing exporter package or an unreachable
    collector must not stop the application from serving requests."""
    global _configured
    if _configured or isinstance(trace.get_tracer_provider(), TracerProvider):
        _configured = True
        return

    service = os.getenv("OTEL_SERVICE_NAME", "assistant-framework")
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    exporter_name = os.getenv("OTEL_TRACES_EXPORTER", "none").lower()
    if exporter_name == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            logger.warning(
                "OTEL_TRACES_EXPORTER=otlp but opentelemetry-exporter-otlp-proto-http is "
                "not installed; traces will not be exported (pip install '.[tracing]')"
            )
        else:
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces"))
            )
            logger.info("exporting traces over OTLP to %s", endpoint)
    elif exporter_name == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True


def _attr_value(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(v, (bool, int, float, str)) for v in value
    ):
        return list(value)
    return str(value)


def set_attributes(span: Span, attributes: Mapping[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, _attr_value(value))


@contextmanager
def start_span(name: str, **attributes: Any) -> Iterator[Span]:
    """A child span of the current context; records an escaping exception
    as the span's error status."""
    with tracer.start_as_current_span(
        name, record_exception=True, set_status_on_exception=True
    ) as span:
        set_attributes(span, attributes)
        yield span


def mark_error(span: Span, kind: str, message: str) -> None:
    span.set_attribute("error.kind", kind)
    span.set_status(Status(StatusCode.ERROR, message))


def current_trace_id() -> str | None:
    context = trace.get_current_span().get_span_context()
    return f"{context.trace_id:032x}" if context.is_valid else None
