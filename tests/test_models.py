"""Tests for data models (PendingTransition, LatencyTurn, CallPayload, etc.)."""

from pipecat_flows_tuner.models import (
    AiModels,
    CallPayload,
    GeneralMetaData,
    LatencyTurn,
    NodeInfo,
    NodeTransitionRecord,
    PendingTransition,
    ToolInfo,
    TranscriptSegment,
    UsageToken,
)


def test_pending_transition():
    pt = PendingTransition(function_name="transfer", arguments={"to": "sales"}, timestamp_ms=100)
    assert pt.function_name == "transfer"
    assert pt.arguments == {"to": "sales"}
    assert pt.timestamp_ms == 100


def test_node_transition_record():
    r = NodeTransitionRecord(
        from_node="greeting",
        to_node="transfer",
        trigger_function="transfer",
        trigger_args={"to": "sales"},
        state_snapshot={"key": "value"},
        task_messages=[],
        functions_available=["transfer", "hangup"],
        timestamp_ms=200,
    )
    assert r.from_node == "greeting"
    assert r.to_node == "transfer"
    assert r.trigger_function == "transfer"
    assert r.timestamp_ms == 200


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
        user_confidence=0.95,
        bot_stopped_ms=500,
    )
    assert t.turn_index == 0
    assert t.ttfb_ms == 150
    assert t.bot_stopped_ms == 500


def test_tool_info():
    ti = ToolInfo(name="transfer", request_id="tc-1", params={"to": "sales"})
    assert ti.name == "transfer"
    assert ti.request_id == "tc-1"
    assert ti.params == {"to": "sales"}


def test_node_info_serialization_alias():
    ni = NodeInfo(from_node="A", to="B", reason="transfer")
    d = ni.model_dump(by_alias=True)
    assert "from" in d
    assert d["from"] == "A"
    assert d["to"] == "B"


def test_transcript_segment():
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
    assert seg.node is None


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
    # by_alias: NodeInfo "from_node" -> "from"
    assert "transcript_with_tool_calls" in d
