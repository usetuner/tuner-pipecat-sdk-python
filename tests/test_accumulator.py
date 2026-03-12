"""Tests for FlowsAccumulator: timing, turns, node transitions, payload building."""

import time
from unittest.mock import MagicMock

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMFullResponseStartFrame,
    TTSStartedFrame,
    TTSTextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)

from pipecat_flows_tuner.accumulator import FlowsAccumulator
from pipecat_flows_tuner.models import LatencyTurn, NodeTransitionRecord, PendingTransition


# ─── Helpers (_rel_ms, _ns_to_ms) ─────────────────────────────────────────────

def test_rel_ms_zero_when_no_start():
    acc = FlowsAccumulator()
    assert acc._rel_ms(1_000_000_000) == 0
    assert acc._rel_ms(0) == 0


def test_rel_ms_relative_to_call_start():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1_000_000_000  # 1s
    assert acc._rel_ms(1_500_000_000) == 500
    assert acc._rel_ms(1_000_000_000) == 0


def test_ns_to_ms_returns_none_when_zero():
    acc = FlowsAccumulator()
    assert acc._ns_to_ms(0, 1_000_000) is None
    assert acc._ns_to_ms(1_000_000, 0) is None


def test_ns_to_ms_computes_milliseconds():
    acc = FlowsAccumulator()
    assert acc._ns_to_ms(0, 1_000_000) is None  # 0 first arg
    # 1e9 ns = 1s -> 1000 ms
    assert acc._ns_to_ms(1_000_000_000, 2_000_000_000) == 1_000
    assert acc._ns_to_ms(1_000_000_000, 1_500_000_000) == 500


# ─── on_start ─────────────────────────────────────────────────────────────────

def test_on_start_sets_call_start():
    acc = FlowsAccumulator()
    ns = time.time_ns()
    acc.on_start(ns)
    assert acc.call_start_abs_ns == ns


# ─── User / VAD ──────────────────────────────────────────────────────────────

def test_on_user_started_sets_once():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.on_user_started(100)
    acc.on_user_started(200)
    assert acc._user_started_ns == 100


def test_on_user_stopped_sets_anchor_and_clears_llm_tts_bot():
    acc = FlowsAccumulator()
    acc._current_node = "greeting"
    acc.on_user_stopped(MagicMock(stop_secs=0), 200)
    assert acc._user_stopped_ns == 200
    assert acc._llm_started_ns == 0
    assert acc._tts_started_ns == 0
    assert acc._latency_node == "greeting"


def test_on_user_stopped_applies_stop_correction():
    acc = FlowsAccumulator()
    frame = MagicMock(stop_secs=0.5)
    acc.on_user_stopped(frame, 2_000_000_000)  # 2s in ns
    # correction: 0.5s = 500_000_000 ns
    assert acc._user_stopped_ns == 2_000_000_000 - 500_000_000


# ─── Transcription (confidence) ───────────────────────────────────────────────

def test_on_transcription_accumulates_confidence():
    acc = FlowsAccumulator()
    for c in [0.9, 0.95]:
        frame = MagicMock()
        frame.result.channel.alternatives = [MagicMock(confidence=c)]
        acc.on_transcription(frame)
    assert acc._current_turn_confidences == [0.9, 0.95]


def test_on_transcription_handles_missing_confidence():
    acc = FlowsAccumulator()
    frame = MagicMock()
    frame.result.channel.alternatives = [MagicMock(confidence=None)]
    acc.on_transcription(frame)
    assert acc._current_turn_confidences == []


def test_on_transcription_handles_exception():
    acc = FlowsAccumulator()
    frame = MagicMock()
    del frame.result
    acc.on_transcription(frame)  # no raise
    assert acc._current_turn_confidences == []


# ─── LLM / TTS / Bot ─────────────────────────────────────────────────────────

def test_on_llm_started_sets_once_after_user_stopped():
    acc = FlowsAccumulator()
    acc._user_stopped_ns = 100
    acc.on_llm_started(150)
    acc.on_llm_started(160)
    assert acc._llm_started_ns == 150


def test_on_llm_started_ignored_before_user_stopped():
    acc = FlowsAccumulator()
    acc.on_llm_started(100)
    assert acc._llm_started_ns == 0


def test_on_tts_started_sets_once_after_user_stopped():
    acc = FlowsAccumulator()
    acc._user_stopped_ns = 100
    acc.on_tts_started(200)
    acc.on_tts_started(210)
    assert acc._tts_started_ns == 200


def test_on_tts_text_chars_accumulates():
    acc = FlowsAccumulator()
    acc.on_tts_text_chars(MagicMock(text="Hello"))
    acc.on_tts_text_chars(MagicMock(text=" world"))
    assert acc._tts_chars == 11


def test_on_tts_text_chars_handles_empty():
    acc = FlowsAccumulator()
    acc.on_tts_text_chars(MagicMock(text=""))
    assert acc._tts_chars == 0


def test_on_bot_started_speaking_flushes_latency_turn():
    base_ns = 1_000_000_000
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = base_ns
    acc._user_started_ns = base_ns + 50_000_000
    acc._user_stopped_ns = base_ns + 100_000_000
    acc._llm_started_ns = base_ns + 150_000_000
    acc._tts_started_ns = base_ns + 200_000_000
    acc._latency_node = "greeting"
    acc._current_turn_confidences = [0.9]

    acc.on_bot_started_speaking(base_ns + 250_000_000)

    assert len(acc.latency_turns) == 1
    turn = acc.latency_turns[0]
    assert turn.turn_index == 0
    assert turn.node == "greeting"
    assert turn.user_confidence == 0.9
    assert turn.bot_started_ms == 250
    assert turn.user_stopped_ms == 100
    assert turn.user_started_ms == 50
    assert turn.llm_ms == 50
    assert turn.tts_ms == 50
    assert turn.ttfb_ms == 100  # user_stopped -> tts_started
    assert acc._user_stopped_ns == 0
    assert acc._current_turn_confidences == []


def test_on_bot_stopped_updates_last_turn():
    acc = FlowsAccumulator()
    # _rel_ms(ts) = (ts - call_start_abs_ns) // 1_000_000; use start=1 so 500ms => 1 + 500*1e6
    acc.call_start_abs_ns = 1
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0, node=None, ttfb_ms=100, llm_ms=50, tts_ms=50,
            bot_started_ms=200, user_stopped_ms=100, user_started_ms=50,
            user_confidence=None, bot_stopped_ms=None,
        )
    ]
    acc.on_bot_stopped(1 + 500 * 1_000_000)  # 500 ms relative to start=1
    assert acc.latency_turns[-1].bot_stopped_ms == 500


def test_on_bot_stopped_no_op_when_done():
    acc = FlowsAccumulator()
    acc.done = True
    acc.latency_turns = [LatencyTurn(
        turn_index=0, node=None, ttfb_ms=0, llm_ms=0, tts_ms=0,
        bot_started_ms=0, user_stopped_ms=0, user_started_ms=0,
        user_confidence=None, bot_stopped_ms=None,
    )]
    acc.on_bot_stopped(999_000_000)
    assert acc.latency_turns[-1].bot_stopped_ms is None


# ─── Function call / pending transition ───────────────────────────────────────

def test_on_function_call_in_progress_sets_pending():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1
    frame = MagicMock(function_name="transfer", arguments={"to": "sales"})
    acc.on_function_call_in_progress(frame, 1 + 300 * 1_000_000)  # 300 ms
    pt = acc.get_pending_transition()
    assert pt is not None
    assert pt.function_name == "transfer"
    assert pt.arguments == {"to": "sales"}
    assert pt.timestamp_ms == 300


# ─── Node entered ─────────────────────────────────────────────────────────────

def test_on_node_entered_appends_transition_and_clears_pending():
    acc = FlowsAccumulator()
    acc._pending_transition = PendingTransition(
        function_name="transfer", arguments={"to": "sales"}, timestamp_ms=100
    )
    acc.call_start_abs_ns = 1
    acc.on_node_entered(
        from_node="greeting",
        to_node="transfer",
        node_config={"functions": [{"name": "hangup"}], "task_messages": []},
        trigger=acc._pending_transition,
        state_snapshot={"x": 1},
        timestamp_ns=1 + 200 * 1_000_000,  # 200 ms
    )
    assert len(acc.node_transitions) == 1
    rec = acc.node_transitions[0]
    assert rec.from_node == "greeting"
    assert rec.to_node == "transfer"
    assert rec.trigger_function == "transfer"
    assert rec.trigger_args == {"to": "sales"}
    assert rec.state_snapshot == {"x": 1}
    assert rec.functions_available == ["hangup"]
    assert rec.timestamp_ms == 200
    assert acc._current_node == "transfer"
    assert acc._pending_transition is None


def test_on_node_entered_extracts_function_names_from_objects():
    acc = FlowsAccumulator()
    class Fn:
        def __init__(self, name):
            self.name = name
    acc.on_node_entered(
        from_node=None,
        to_node="start",
        node_config={"functions": [Fn("a"), {"name": "b"}], "task_messages": []},
        trigger=None,
        state_snapshot={},
        timestamp_ns=0,
    )
    assert acc.node_transitions[0].functions_available == ["a", "b"]


# ─── Call end ───────────────────────────────────────────────────────────────

def test_on_call_end_sets_done_and_end_time():
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 100
    acc.on_call_end(200)
    assert acc.done is True
    assert acc.call_end_abs_ns == 200


def test_on_call_end_idempotent_when_done():
    acc = FlowsAccumulator()
    acc.done = True
    acc.call_end_abs_ns = 100
    acc.on_call_end(999)
    assert acc.call_end_abs_ns == 100


# ─── build_payload ───────────────────────────────────────────────────────────

def test_build_payload_basic(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 1_000_000_000
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    acc._tts_chars = 50
    transcript = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    assert payload.call_id == tuner_config.call_id
    assert payload.call_type == tuner_config.call_type
    assert payload.start_timestamp == 1
    assert payload.end_timestamp == 2
    assert payload.duration_ms == 1_000
    assert payload.call_status == "call_ended"
    assert payload.general_meta_data_raw.usage_token.tts_character_count == 50
    assert payload.general_meta_data_raw.ai_models.asr_model == tuner_config.asr_model
    assert len(payload.transcript_with_tool_calls) >= 2  # user + agent segments


# ─── _enrich_transcript (via build_payload) ──────────────────────────────────

def test_enrich_transcript_user_and_assistant(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0, node="n", ttfb_ms=100, llm_ms=50, tts_ms=50,
            bot_started_ms=200, user_stopped_ms=100, user_started_ms=50,
            user_confidence=0.9, bot_stopped_ms=300,
        ),
    ]
    transcript = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    roles = [s.role for s in payload.transcript_with_tool_calls]
    assert "user" in roles
    assert "agent" in roles


def test_enrich_transcript_skips_system(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    transcript = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hi!"},
    ]
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0, node=None, ttfb_ms=0, llm_ms=0, tts_ms=0,
            bot_started_ms=0, user_stopped_ms=0, user_started_ms=0,
            user_confidence=None, bot_stopped_ms=100,
        ),
    ]
    payload = acc.build_payload(tuner_config, transcript)
    roles = [s.role for s in payload.transcript_with_tool_calls]
    assert "system" not in roles


def test_enrich_transcript_tool_call_and_result_and_node_transition(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    acc.node_transitions = [
        NodeTransitionRecord(
            from_node="greeting",
            to_node="transfer",
            trigger_function="transfer",
            trigger_args={"to": "sales"},
            state_snapshot={},
            task_messages=[],
            functions_available=[],
            timestamp_ms=100,
        ),
    ]
    transcript = [
        {"role": "user", "content": "Transfer me"},
        {"role": "assistant", "tool_calls": [{
            "id": "tc-1",
            "function": {"name": "transfer", "arguments": '{"to": "sales"}'},
        }]},
        {"role": "tool", "tool_call_id": "tc-1", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Done."},
    ]
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0, node="greeting", ttfb_ms=50, llm_ms=30, tts_ms=20,
            bot_started_ms=100, user_stopped_ms=0, user_started_ms=0,
            user_confidence=None, bot_stopped_ms=150,
        ),
        LatencyTurn(
            turn_index=1, node="transfer", ttfb_ms=50, llm_ms=30, tts_ms=20,
            bot_started_ms=200, user_stopped_ms=150, user_started_ms=100,
            user_confidence=None, bot_stopped_ms=250,
        ),
    ]
    payload = acc.build_payload(tuner_config, transcript)
    roles = [s.role for s in payload.transcript_with_tool_calls]
    assert "agent_function" in roles
    assert "agent_result" in roles
    assert "node_transition" in roles


def test_enrich_transcript_initial_node_transition(tuner_config):
    acc = FlowsAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    acc.node_transitions = [
        NodeTransitionRecord(
            from_node=None,
            to_node="start",
            trigger_function=None,
            trigger_args=None,
            state_snapshot={"x": 1},
            task_messages=[],
            functions_available=["next"],
            timestamp_ms=0,
        ),
    ]
    transcript = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hi!"},
    ]
    acc.latency_turns = [
        LatencyTurn(
            turn_index=0, node="start", ttfb_ms=0, llm_ms=0, tts_ms=0,
            bot_started_ms=0, user_stopped_ms=0, user_started_ms=0,
            user_confidence=None, bot_stopped_ms=100,
        ),
    ]
    payload = acc.build_payload(tuner_config, transcript)
    first = payload.transcript_with_tool_calls[0]
    assert first.role == "node_transition"
    assert first.node is not None
    assert first.node.to == "start"
