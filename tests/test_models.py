"""Tests for data models (PendingTransition, LatencyTurn, CallPayload, etc.)."""

from pipecat_flows_tuner.models import (
    AiModels,
    CallPayload,
    GeneralMetaData,
    LatencyTurn,
    ToolInfo,
    TranscriptMetadata,
    TranscriptSegment,
    TranscriptWord,
    UsageToken,
)


def test_latency_turn():
    t = LatencyTurn(
        turn_index=0,
        node="greeting",
        ttfb_ms=150,
        llm_ms=80,
        tts_ms=50,
        bot_started_ms=300,
        user_stopped_ms=100,
        user_started_ms=50,
        bot_stopped_ms=500,
    )
    assert t.turn_index == 0
    assert t.ttfb_ms == 150
    assert t.bot_stopped_ms == 500


# ---------------------------------------------------------------------------
# TranscriptWord
# ---------------------------------------------------------------------------


def test_transcript_word_required_fields():
    w = TranscriptWord(word="Hello", start_ms=1000, end_ms=1500)
    assert w.word == "Hello"
    assert w.start_ms == 1000
    assert w.end_ms == 1500
    assert w.confidence is None


def test_transcript_word_with_confidence():
    w = TranscriptWord(word="Hello", start_ms=1000, end_ms=1500, confidence=0.98)
    assert w.confidence == 0.98


# ---------------------------------------------------------------------------
# TranscriptMetadata
# ---------------------------------------------------------------------------


def test_transcript_metadata_defaults_to_none():
    m = TranscriptMetadata()
    assert m.e2e_latency is None
    assert m.interrupted is None
    assert m.llm_node_ttft is None
    assert m.tts_node_ttfb is None


def test_transcript_metadata_known_fields():
    m = TranscriptMetadata(
        e2e_latency=1200.5,
        interrupted=False,
        llm_node_ttft=845.0,
        tts_node_ttfb=300.0,
        transcript_confidence=0.95,
    )
    assert m.e2e_latency == 1200.5
    assert m.interrupted is False
    assert m.llm_node_ttft == 845.0


def test_transcript_metadata_allows_extra_fields():
    m = TranscriptMetadata(e2e_latency=500.0, response_id=3, turn_index=1)  # type: ignore[call-arg]
    assert m.e2e_latency == 500.0
    assert m.model_extra["response_id"] == 3  # type: ignore[index]
    assert m.model_extra["turn_index"] == 1  # type: ignore[index]


# ---------------------------------------------------------------------------
# ToolInfo
# ---------------------------------------------------------------------------


def test_tool_info_minimal():
    ti = ToolInfo(name="transfer", request_id="tc-1", params={"to": "sales"})
    assert ti.name == "transfer"
    assert ti.request_id == "tc-1"
    assert ti.params == {"to": "sales"}
    assert ti.result is None
    assert ti.is_error is None
    assert ti.error is None
    assert ti.start_ms is None


def test_tool_info_with_result():
    ti = ToolInfo(
        name="book_viewing",
        request_id="tc-2",
        params={"property_id": "A101"},
        result={"booking_id": "BK-001", "status": "confirmed"},
        is_error=False,
        start_ms=17500,
    )
    assert ti.result == {"booking_id": "BK-001", "status": "confirmed"}
    assert ti.is_error is False
    assert ti.start_ms == 17500


def test_tool_info_error():
    ti = ToolInfo(
        name="check_availability",
        request_id="tc-3",
        result={"error": "timeout"},
        is_error=True,
        error="Service timed out",
    )
    assert ti.is_error is True
    assert ti.error == "Service timed out"


# ---------------------------------------------------------------------------
# TranscriptSegment
# ---------------------------------------------------------------------------


def test_transcript_segment_minimal():
    seg = TranscriptSegment(
        role="user",
        text="Hello",
        start_ms=0,
        end_ms=100,
        metadata={"id": "x"},
    )
    assert seg.role == "user"
    assert seg.text == "Hello"
    assert seg.tool is None
    assert seg.words is None
    assert seg.duration_ms is None


def test_transcript_segment_with_words():
    seg = TranscriptSegment(
        role="user",
        text="Hello there",
        start_ms=1000,
        end_ms=2000,
        metadata={},
        words=[
            TranscriptWord(word="Hello", start_ms=1000, end_ms=1500),
            TranscriptWord(word="there", start_ms=1500, end_ms=2000),
        ],
    )
    assert len(seg.words) == 2  # type: ignore[arg-type]
    assert seg.words[0].word == "Hello"
    assert seg.words[1].start_ms == 1500


def test_transcript_segment_with_tool():
    seg = TranscriptSegment(
        role="agent_function",
        text="book_viewing(property_id=A101)",
        start_ms=17500,
        end_ms=17500,
        metadata={"node": "collect_details"},
        tool=ToolInfo(
            name="book_viewing",
            request_id="tc-001",
            params={"property_id": "A101"},
            start_ms=17500,
        ),
    )
    assert seg.tool is not None
    assert seg.tool.name == "book_viewing"
    assert seg.tool.start_ms == 17500


def test_transcript_segment_serialization_excludes_none():
    seg = TranscriptSegment(
        role="agent",
        text="Hello!",
        start_ms=4897,
        end_ms=6800,
        metadata={"response_id": 1},
    )
    d = seg.model_dump(by_alias=True, exclude_none=True)
    assert "tool" not in d
    assert "words" not in d
    assert "duration_ms" not in d
    assert d["role"] == "agent"
    assert d["text"] == "Hello!"


# ---------------------------------------------------------------------------
# CallPayload
# ---------------------------------------------------------------------------


def test_call_payload_to_dict():
    payload = CallPayload(
        call_id="c1",
        call_type="web_call",
        start_timestamp=1000,
        end_timestamp=2000,
        recording_url="https://example.com/rec",
        transcript_with_tool_calls=[],
        call_status="call_ended",
        duration_ms=1000,
        general_meta_data_raw=GeneralMetaData(
            ai_models=AiModels(asr_model="dg", llm_model="gpt", tts_model="eleven"),
            usage_token=UsageToken(asr_duration=1000, llm_token=50, tts_character_count=200),
        ),
    )
    d = payload.to_dict()
    assert d["call_id"] == "c1"
    assert d["duration_ms"] == 1000
    assert "general_meta_data_raw" in d
    assert d["general_meta_data_raw"]["ai_models"]["asr_model"] == "dg"
    assert d["general_meta_data_raw"]["usage_token"]["llm_token"] == 50
    assert "transcript_with_tool_calls" in d


