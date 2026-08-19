"""Tests for tuner_pipecat_sdk.tracing (ENG-1233).

Two silent failure modes are what these pin down: clobbering a customer's existing tracing,
and the optional OTel extra being absent.
"""

from unittest.mock import MagicMock, patch

from tuner_pipecat_sdk.config import TunerConfig
from tuner_pipecat_sdk.tracing import CALL_ID_ATTRIBUTE, setup_call_tracing

_STUB_EXPORTER = patch(
    "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
    MagicMock(),
)


def test_returns_false_when_otel_is_not_installed(tuner_config: TunerConfig):
    """The extra is optional, so its absence must be a no-op rather than an error."""
    with patch("tuner_pipecat_sdk.tracing._import_otel", return_value=None):
        assert setup_call_tracing(config=tuner_config) is False


def test_attaches_to_an_existing_provider_without_replacing_it(tuner_config: TunerConfig):
    """A customer who already called Pipecat's setup_tracing() must keep their exporter.

    Pipecat's setup_tracing calls trace.set_tracer_provider(), which replaces the global
    provider. Doing the same here would silently drop whatever they export to.
    """
    from opentelemetry.sdk.trace import TracerProvider

    existing = TracerProvider()
    before = len(existing._active_span_processor._span_processors)

    with (
        patch("opentelemetry.trace.get_tracer_provider", return_value=existing),
        patch("opentelemetry.trace.set_tracer_provider") as set_provider,
        _STUB_EXPORTER,
    ):
        assert setup_call_tracing(config=tuner_config) is True

    assert len(existing._active_span_processor._span_processors) == before + 2
    set_provider.assert_not_called()


def test_creates_and_registers_a_provider_when_none_exists(tuner_config: TunerConfig):
    """With nothing configured the global default is a no-op proxy, so we must supply one."""
    with (
        patch("opentelemetry.trace.get_tracer_provider", return_value=MagicMock()),
        patch("opentelemetry.trace.set_tracer_provider") as set_provider,
        _STUB_EXPORTER,
    ):
        assert setup_call_tracing(config=tuner_config) is True

    set_provider.assert_called_once()


def test_the_call_id_is_stamped_on_every_span(tuner_config: TunerConfig):
    """Not just the root.

    Pipecat's own `additional_span_attributes` only reaches the root `conversation` span,
    and that span covers the whole call so it is not exported until the call ends. Stamping
    every span means Tuner can correlate from the first batch instead of the last.
    """
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    with (
        patch("opentelemetry.trace.get_tracer_provider", return_value=provider),
        patch("opentelemetry.trace.set_tracer_provider"),
        _STUB_EXPORTER,
    ):
        setup_call_tracing(config=tuner_config)

    tracer = provider.get_tracer("test")
    for span_name in ("conversation", "turn", "llm", "tts"):
        with tracer.start_as_current_span(span_name) as span:
            assert span.attributes[CALL_ID_ATTRIBUTE] == "call-123"


def test_the_endpoint_and_auth_header_come_from_config(tuner_config: TunerConfig):
    with patch("tuner_pipecat_sdk.tracing._import_otel") as import_otel:
        exporter_cls = MagicMock()
        import_otel.return_value = (
            exporter_cls,
            MagicMock,
            MagicMock(),
            MagicMock(return_value=MagicMock()),
        )
        setup_call_tracing(config=tuner_config)

    kwargs = exporter_cls.call_args.kwargs
    assert kwargs["endpoint"] == "https://tuner.example.com/api/v1/traces"
    assert kwargs["headers"] == {"authorization": "Bearer test-api-key"}


def test_a_trailing_slash_on_base_url_does_not_double_up(tuner_config: TunerConfig):
    config = tuner_config.model_copy(update={"base_url": "https://tuner.example.com/"})

    with patch("tuner_pipecat_sdk.tracing._import_otel") as import_otel:
        exporter_cls = MagicMock()
        import_otel.return_value = (
            exporter_cls,
            MagicMock,
            MagicMock(),
            MagicMock(return_value=MagicMock()),
        )
        setup_call_tracing(config=config)

    assert exporter_cls.call_args.kwargs["endpoint"] == "https://tuner.example.com/api/v1/traces"


def test_a_failure_is_swallowed_rather_than_breaking_the_call(tuner_config: TunerConfig):
    """Traces are a debugging aid; they must never take a call down."""
    with patch("tuner_pipecat_sdk.tracing._import_otel") as import_otel:
        import_otel.return_value = (
            MagicMock(side_effect=RuntimeError("exporter blew up")),
            MagicMock,
            MagicMock(),
            MagicMock(),
        )
        assert setup_call_tracing(config=tuner_config) is False


def test_config_defaults_traces_on(tuner_config: TunerConfig):
    assert tuner_config.traces_enabled is True


def test_traces_can_be_turned_off(tuner_config: TunerConfig):
    assert tuner_config.model_copy(update={"traces_enabled": False}).traces_enabled is False


# --- observer wiring -----------------------------------------------------------------


def _observer_kwargs(**overrides) -> dict:
    kwargs = {
        "api_key": "test-api-key",
        "workspace_id": 42,
        "agent_id": "test-agent",
        "call_id": "call-123",
    }
    kwargs.update(overrides)
    return kwargs


def test_the_observer_sets_up_tracing_with_the_call_id():
    """Pins the timing too: it happens in the constructor, before any span is emitted.

    Setting it up later would miss the spans from the start of the call.
    """
    from tuner_pipecat_sdk.observer import Observer

    with patch("tuner_pipecat_sdk._base.setup_call_tracing") as setup:
        Observer(**_observer_kwargs())

    setup.assert_called_once()
    assert setup.call_args.kwargs["config"].call_id == "call-123"


def test_the_observer_skips_tracing_when_it_is_turned_off():
    from tuner_pipecat_sdk.observer import Observer

    with patch("tuner_pipecat_sdk._base.setup_call_tracing") as setup:
        Observer(**_observer_kwargs(traces_enabled=False))

    setup.assert_not_called()
