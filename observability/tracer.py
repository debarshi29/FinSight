from __future__ import annotations

import functools
import time
from typing import Any, Callable

import structlog

from core.config import settings

log = structlog.get_logger()

_configured = False


def setup_tracing() -> None:
    global _configured
    if _configured:
        return

    import logging

    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if not settings.otel_enabled
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    if settings.otel_enabled:
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider = TracerProvider()
            exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            log.info("tracing.otel_enabled", endpoint=settings.otel_endpoint)
        except ImportError:
            log.warning("tracing.otel_not_installed")

    _configured = True


def traced(name: str | None = None) -> Callable:
    """Decorator that logs entry/exit and latency for any async function."""

    def decorator(fn: Callable) -> Callable:
        span_name = name or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()
            log.debug(f"{span_name}.start")
            try:
                result = await fn(*args, **kwargs)
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                log.debug(f"{span_name}.complete", latency_ms=elapsed_ms)
                return result
            except Exception as e:
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                log.error(f"{span_name}.error", error=str(e), latency_ms=elapsed_ms)
                raise

        return wrapper

    return decorator
