"""Tests for CallAccumulator.on_metrics_frame."""

from types import SimpleNamespace

from tuner_pipecat_sdk.accumulator import CallAccumulator


def _metric(cls_name, **kwargs):
    """Create a fake metric object whose type().__name__ == cls_name."""
    cls = type(cls_name, (), kwargs)
    return cls()


def _llm_usage(total_tokens):
    value = SimpleNamespace(total_tokens=total_tokens)
    return _metric("LLMUsageMetricsData", value=value)


def _tts_usage(chars):
    return _metric("TTSUsageMetricsData", value=chars)


def _processing(processor, value_s):
    return _metric("ProcessingMetricsData", processor=processor, value=value_s)


def _frame(*data_items):
    return SimpleNamespace(data=list(data_items))


def test_llm_usage_accumulates():
    acc = CallAccumulator()
    acc.on_metrics_frame(_frame(_llm_usage(100)))
    acc.on_metrics_frame(_frame(_llm_usage(50)))
    assert acc.get_total_llm_tokens() == 150


def test_tts_usage_accumulates():
    acc = CallAccumulator()
    acc.on_metrics_frame(_frame(_tts_usage(200)))
    acc.on_metrics_frame(_frame(_tts_usage(75)))
    assert acc.get_total_tts_characters() == 275


def test_processing_llm_processor():
    acc = CallAccumulator()
    acc.on_metrics_frame(_frame(_processing("OpenAILLMService", 1.2)))
    assert acc._pending_pipecat_llm_processing_s == 1.2
    assert acc._pending_pipecat_tts_processing_s == 0.0


def test_processing_tts_processor():
    acc = CallAccumulator()
    acc.on_metrics_frame(_frame(_processing("ElevenLabsTTSService", 0.8)))
    assert acc._pending_pipecat_tts_processing_s == 0.8
    assert acc._pending_pipecat_llm_processing_s == 0.0


def test_empty_frame_no_crash():
    acc = CallAccumulator()
    acc.on_metrics_frame(_frame())
    assert acc.get_total_llm_tokens() == 0
    assert acc.get_total_tts_characters() == 0


def test_frame_without_data_attribute_no_crash():
    acc = CallAccumulator()
    acc.on_metrics_frame(SimpleNamespace())  # no 'data' attribute
    assert acc.get_total_llm_tokens() == 0


def test_multiple_metric_types_in_one_frame():
    acc = CallAccumulator()
    acc.on_metrics_frame(
        _frame(
            _llm_usage(300),
            _tts_usage(120),
            _processing("ElevenLabsTTSService", 0.9),
        )
    )
    assert acc.get_total_llm_tokens() == 300
    assert acc.get_total_tts_characters() == 120
    assert acc._pending_pipecat_tts_processing_s == 0.9


def test_unknown_metric_type_no_state_change():
    acc = CallAccumulator()
    unknown = _metric("UnknownMetricsData")
    acc.on_metrics_frame(_frame(unknown))
    assert acc.get_total_llm_tokens() == 0
    assert acc.get_total_tts_characters() == 0
