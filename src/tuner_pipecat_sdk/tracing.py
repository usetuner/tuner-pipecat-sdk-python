"""OpenTelemetry trace forwarding to Tuner (ENG-1233).

Pipecat already instruments each conversation with OTel spans when tracing is enabled.
This wires those spans to Tuner's OTLP endpoint and tags them with the call id, so the
Trace tab on the call details page can show the tree.

Without this the customer wires it by hand — build a TracerProvider, pick the HTTP
exporter, set the endpoint and auth header, stamp the call id — and the failure mode is
silent: traces simply never appear. Everything needed is already on TunerConfig, so the
SDK can do it.

Optional: the OTel packages are an extra (`pip install tuner-pipecat-sdk[traces]`). If they
are absent this degrades to a no-op with a debug log rather than failing the call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from .config import TunerConfig

# The attribute Tuner correlates on.
#
# Stamped on every span rather than using Pipecat's own `additional_span_attributes`, which
# only reaches the root `conversation` span. Tuner can work from that — it correlates at
# trace level — but the root span covers the whole call, so it is not exported until the
# call ends. Tagging every span means correlation happens on the first batch instead of the
# last, and there is nothing to reconcile afterwards.
CALL_ID_ATTRIBUTE = "tuner.call_id"

# Tuner accepts OTLP over HTTP with protobuf encoding. gRPC is not supported.
_TRACES_PATH = "/api/v1/traces"


def _import_otel() -> tuple[Any, Any, Any, Any] | None:
    """Import the optional OTel pieces, or return None when the extra is not installed."""
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.trace import get_tracer_provider

        return (OTLPSpanExporter, TracerProvider, BatchSpanProcessor, get_tracer_provider)
    except ImportError:
        return None


def _build_call_id_processor(call_id: str) -> Any:
    """A SpanProcessor that stamps the call id onto every span as it starts."""
    from opentelemetry.sdk.trace import SpanProcessor

    class _CallIdSpanProcessor(SpanProcessor):
        def on_start(self, span: Any, parent_context: Any = None) -> None:
            # on_start, not on_end: a span's attributes are frozen once it ends.
            span.set_attribute(CALL_ID_ATTRIBUTE, call_id)

        def on_end(self, span: Any) -> None:  # pragma: no cover - nothing to do
            return None

        def shutdown(self) -> None:  # pragma: no cover - nothing to own
            return None

        def force_flush(self, timeout_millis: int = 30_000) -> bool:  # pragma: no cover
            return True

    return _CallIdSpanProcessor()


def setup_call_tracing(*, config: TunerConfig) -> bool:
    """Forward this call's OTel spans to Tuner, tagged with the call id.

    Deliberately additive. Pipecat's `setup_tracing` calls `trace.set_tracer_provider(...)`,
    which replaces the global provider — so a customer who already exports traces to their
    own backend would silently lose them if we did the same. Instead: if a real provider
    already exists, attach to it; only create and register one when nothing is configured.

    Never raises. Traces are a debugging aid and must not be able to fail a call.

    Args:
        config: The observer's configuration, which already carries the call id, API key
            and base URL

    Returns:
        True if span forwarding was set up, False if it was skipped
    """
    otel = _import_otel()
    if otel is None:
        logger.debug(
            "[tuner] skipping trace forwarding: OpenTelemetry packages are not installed. "
            "Install with: pip install 'tuner-pipecat-sdk[traces]'"
        )
        return False

    otlp_span_exporter, tracer_provider_cls, batch_span_processor, get_tracer_provider = otel

    try:
        exporter = otlp_span_exporter(
            endpoint=f"{config.base_url.rstrip('/')}{_TRACES_PATH}",
            headers={"authorization": f"Bearer {config.api_key}"},
        )

        existing = get_tracer_provider()
        if isinstance(existing, tracer_provider_cls):
            # The customer already configured tracing — most likely via Pipecat's own
            # setup_tracing(). Add ours alongside whatever they already export to.
            provider = existing
            owns_provider = False
        else:
            # Nothing configured: the default global is a no-op proxy provider.
            provider = tracer_provider_cls()
            owns_provider = True

        provider.add_span_processor(batch_span_processor(exporter))
        provider.add_span_processor(_build_call_id_processor(config.call_id))

        if owns_provider:
            from opentelemetry.trace import set_tracer_provider

            set_tracer_provider(provider)

        logger.debug(
            "[tuner] forwarding OTel spans to Tuner for call {} ({} provider)",
            config.call_id,
            "new" if owns_provider else "existing",
        )
        return True

    except Exception as exc:
        logger.warning("[tuner] could not set up trace forwarding to Tuner: {}", exc)
        return False
