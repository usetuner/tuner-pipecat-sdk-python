"""End-to-end: native pipecat transcript merged with LangChain-sourced tool/node data."""

from functools import partial

import pytest

pytest.importorskip("tuner_langchain")

from tuner_langchain import TunerAccumulator

from tuner_pipecat_sdk.langchain_bridge import merge_langchain_segments


def _attach_lg_accumulator(acc, lg_acc: TunerAccumulator) -> None:
    """Mirrors what LangchainIntegration._ensure_accumulator() does at wrap_graph()/
    wrap_chain() time -- registers the merge as a segment enricher rather than
    passing it directly to build_payload(), which no longer accepts it."""
    acc.register_segment_enricher(
        partial(
            merge_langchain_segments,
            lg_accumulator=lg_acc,
            call_start_abs_ns=acc.call_start_abs_ns,
        )
    )


def test_native_and_langchain_segments_interleave_in_true_timestamp_order(replay, tuner_config):
    r = replay()
    base_ns = r.base

    # Native call: user speaks, bot replies (LLM text + voiced TTS), all via frame events.
    r.turn_start(1, 0).user_start(0).transcription("book me a table", 20).user_stop(200)
    r.bot_says("Sure, let me check.", 300, 700)

    # LangChain-sourced node + tool call happening *during* that same response, driven
    # directly through TunerAccumulator's event methods (no real LangChain run needed).
    lg_acc = TunerAccumulator()
    lg_acc.on_graph_start("root-1", base_ns + 350_000_000)  # +350ms, inside the bot's turn
    lg_acc.on_node_start("booking_node", "node-1", "root-1", base_ns + 380_000_000, {})
    lg_acc.on_tool_start(
        "check_availability", "tool-1", "node-1", base_ns + 400_000_000, '{"date": "2024-06-15"}'
    )
    lg_acc.on_tool_end("tool-1", base_ns + 450_000_000, '{"available": true}')
    lg_acc.on_node_end("node-1", base_ns + 500_000_000, {})
    lg_acc.on_graph_end("root-1", base_ns + 550_000_000)

    # Usage reported the way _AccumulatorUsageSink relays it from tuner-langchain's on_llm_end.
    r.acc.record_external_llm_usage(42, 17)

    _attach_lg_accumulator(r.acc, lg_acc)
    r.end(1_000)
    payload = r.acc.build_payload(tuner_config)
    rows = payload.transcript_with_tool_calls

    roles_in_order = [row.role for row in rows]
    assert roles_in_order == [
        "user",
        "agent",
        "node_transition",
        "agent_function",
        "agent_result",
    ]

    # The langchain-sourced rows fall strictly between the user row's end and the
    # agent row's end, in correct chronological order relative to each other.
    starts_by_role = {row.role: row.start_ms for row in rows}
    assert starts_by_role["user"] <= starts_by_role["node_transition"]
    assert starts_by_role["node_transition"] < starts_by_role["agent_function"]
    assert starts_by_role["agent_function"] < starts_by_role["agent_result"]

    node_row = next(row for row in rows if row.role == "node_transition")
    assert node_row.node.to == "booking_node"

    call_row = next(row for row in rows if row.role == "agent_function")
    assert call_row.tool.name == "check_availability"

    result_row = next(row for row in rows if row.role == "agent_result")
    assert result_row.tool.result == {"value": {"available": True}}

    # Usage reported via the LangChain path reaches the payload's usage summary.
    assert payload.general_meta_data_raw.usage_token.llm_token == 42 + 17


def test_langchain_llm_duration_surfaces_as_llm_node_ttft(replay, tuner_config):
    """LangChain-sourced LLM call duration (reported the way _AccumulatorUsageSink
    relays it from tuner-langchain's on_llm_end) must reach the agent row's
    llm_node_ttft metadata -- the whole point of wiring up the previously-unused
    _last_llm_duration_ms hook, since pipecat's LangchainProcessor never emits the
    MetricsFrame data that field would otherwise come from."""
    r = replay()
    r.turn_start(1, 0).user_start(0).transcription("book me a table", 20).user_stop(200)
    r.bot_says("Sure, let me check.", 300, 700)

    r.acc.record_external_llm_duration_ms(88)
    # ttfb_secs=0.1 + user.stop_ms=200 recomputes bot.start_ms to 300, matching
    # bot_says(300, 700) above exactly -- chosen so this breakdown call doesn't
    # perturb the already-scripted bot timing.
    r.latency(0.1, "LLM", user_start_ms=0)

    r.end(1_000)
    payload = r.acc.build_payload(tuner_config)
    agent_row = next(row for row in payload.transcript_with_tool_calls if row.role == "agent")

    assert agent_row.metadata["llm_node_ttft"] == 88
    assert agent_row.start_ms == 300  # confirms the breakdown didn't shift bot timing


def test_native_tool_calls_omitted_when_langchain_accumulator_present(replay, tuner_config):
    """Mirrors the observer-level dedup guard: build_payload's own merge doesn't
    re-add native tool call rows when a LangChain accumulator is supplied -- those
    rows simply never exist here because the observer already skipped recording
    them (see test_tool_call_frames_skipped_when_lg_accumulator_attached)."""
    r = replay()
    r.turn_start(1, 0).user_start(0).transcription("hi", 20).user_stop(200)
    r.bot_says("Hello!", 300, 700)
    r.end(1_000)

    # Snapshot "without" first -- registering a segment enricher is a one-way,
    # cumulative operation now (matching how wrap_graph()/wrap_chain() actually
    # register once, early, and stay registered for the rest of the call), so
    # there's no "lg_accumulator=None" override to ask for after the fact.
    payload_without_lg = r.acc.build_payload(tuner_config)

    lg_acc = TunerAccumulator()
    _attach_lg_accumulator(r.acc, lg_acc)
    payload_with_lg = r.acc.build_payload(tuner_config)

    # No LangChain invocations recorded -> merge is a no-op either way; both payloads match.
    assert [s.role for s in payload_with_lg.transcript_with_tool_calls] == [
        s.role for s in payload_without_lg.transcript_with_tool_calls
    ]
