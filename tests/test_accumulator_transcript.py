"""Accumulator transcript enrichment with tools, transitions, and gap-based merging."""

import pytest

from tuner_pipecat_sdk.accumulator import CallAccumulator
from tuner_pipecat_sdk.models import LatencyMeasurement, SpeechSegment


def _seg(seg_id, speaker, start, stop=None, **kw):
    return SpeechSegment(id=seg_id, speaker=speaker, start_ms=start, stop_ms=stop, **kw)


def _meas(user_id, bot_id, **kw):
    return LatencyMeasurement(user_segment_id=user_id, bot_segment_id=bot_id, **kw)


def test_enrich_transcript_tool_call_and_result(tuner_config):
    acc = CallAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.call_end_abs_ns = base_ns + 2_000_000_000
    acc.done = True
    acc.registry.record_invocation_ns("tc-1", base_ns + 60_000_000)
    acc.registry.record_completion_ns("tc-1", base_ns + 90_000_000)
    acc.speech_segments = [
        _seg(0, "user", 100, 150),
        _seg(1, "bot", 200, 250),
    ]
    acc.latency_measurements = [_meas(0, 1, ttfb_ms=50, llm_ms=30, tts_ms=20, e2e_ms=50)]
    transcript = [
        {"role": "user", "content": "Transfer me"},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "tc-1", "function": {"name": "transfer", "arguments": '{"to": "sales"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "tc-1", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Done."},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    roles = [segment.role for segment in payload.transcript_with_tool_calls]
    assert "agent_function" in roles
    assert "agent_result" in roles
    assert "node_transition" not in roles
    user_segments = [s for s in payload.transcript_with_tool_calls if s.role == "user"]
    agent_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent"]
    assert "asr_node_ttft" not in user_segments[0].metadata
    assert agent_segments[0].metadata["tts_node_ttfb"] == 50

    func_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent_function"]
    result_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent_result"]
    assert func_segments[0].start_ms == 60
    assert result_segments[0].start_ms == 90
    assert func_segments[0].start_ms != result_segments[0].start_ms
    assert result_segments[0].text is None
    assert result_segments[0].tool.result == {"ok": True}


def test_consecutive_assistant_messages_merged_into_one_segment(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    acc.speech_segments = [_seg(0, "user", 10, 50), _seg(1, "bot", 100, 300)]
    acc.latency_measurements = [_meas(0, 1, ttfb_ms=10, llm_ms=20, tts_ms=30)]
    transcript = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there,"},
        {"role": "assistant", "content": "how can I help?"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    agent_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent"]
    assert len(agent_segments) == 1
    assert agent_segments[0].text == "Hi there, how can I help?"
    assert agent_segments[0].start_ms == 100
    assert agent_segments[0].end_ms == 300


def test_enrich_transcript_uses_assistant_turn_events_to_skip_ghost_messages(tuner_config):
    # Ghost messages appear in the same user-turn window as the spoken text, before a tool call
    # triggers a node transition. The last plain assistant text in the window is the spoken one.
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    acc.speech_segments = [_seg(0, "user", 100, 200), _seg(1, "bot", 300, 500)]
    acc.latency_measurements = [_meas(0, 1, ttfb_ms=20, llm_ms=10, tts_ms=10)]
    transcript = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Draft answer"},  # ghost — not the last in window
        {
            "role": "assistant",
            "tool_calls": [{"id": "c1", "function": {"name": "choose", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        {"role": "assistant", "content": "Spoken answer"},  # spoken — last plain text in window
    ]
    payload = acc.build_payload(tuner_config, transcript)
    agent_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent"]
    assert len(agent_segments) == 2
    assert agent_segments[0].text == "Draft answer"
    assert agent_segments[0].start_ms == 0
    assert agent_segments[1].text == "Spoken answer"
    assert agent_segments[1].start_ms == 300


def test_last_plain_assistant_in_window_gets_segment_by_order_not_text(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    acc.speech_segments = [_seg(0, "user", 100, 200), _seg(1, "bot", 300, 600)]
    acc.latency_measurements = [_meas(0, 1, ttfb_ms=30, llm_ms=10, tts_ms=10)]
    transcript = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Draft answer"},
        {"role": "assistant", "content": "Final spoken answer"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    agent_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent"]
    assert len(agent_segments) == 1
    assert agent_segments[0].text == "Draft answer Final spoken answer"
    assert agent_segments[0].start_ms == 300
    assert agent_segments[0].end_ms == 600


def test_agent_metadata_node_comes_from_bot_segment(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 1_000_000_000
    acc.done = True
    acc.speech_segments = [_seg(0, "user", 100, 200), _seg(1, "bot", 300, 600, node="size")]
    acc.latency_measurements = [_meas(0, 1, ttfb_ms=30, llm_ms=10, tts_ms=10)]
    transcript = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "What size pizza would you like?"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    agent_seg = next(s for s in payload.transcript_with_tool_calls if s.role == "agent")
    assert agent_seg.metadata["node"] == "size"


def test_all_trailing_assistant_messages_after_last_user_are_spoken(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    acc.speech_segments = [_seg(0, "user", 600, 700), _seg(1, "bot", 800, 1200)]
    acc.latency_measurements = [_meas(0, 1, ttfb_ms=20, llm_ms=10, tts_ms=10)]
    transcript = [
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "Thank you for your order!"},
        {"role": "assistant", "content": "Enjoy your meal!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    agent_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent"]
    assert len(agent_segments) == 1
    assert agent_segments[0].text == "Thank you for your order! Enjoy your meal!"
    assert agent_segments[0].start_ms == 800


def test_agent_result_uses_registry_completion_when_available(tuner_config):
    """agent_result.start_ms uses the registry completion time, not invocation_ms."""
    acc = CallAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.call_end_abs_ns = base_ns + 2_000_000_000
    acc.done = True
    acc.registry.record_completion_ns("call_xyz", base_ns + 350_000_000)
    transcript = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_xyz", "function": {"name": "greet", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_xyz", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Done!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    result_segs = [s for s in payload.transcript_with_tool_calls if s.role == "agent_result"]
    assert len(result_segs) == 1
    assert result_segs[0].start_ms == 350
    assert result_segs[0].end_ms is None


def test_parallel_same_name_tools_use_distinct_invocation_ms_by_id(tuner_config):
    """Two add_topping calls with different tool_call_ids get distinct invocation times."""
    acc = CallAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.call_end_abs_ns = base_ns + 2_000_000_000
    acc.done = True
    acc.registry.record_invocation_ns("tc-a", base_ns + 100_000_000)
    acc.registry.record_invocation_ns("tc-b", base_ns + 200_000_000)
    transcript = [
        {"role": "user", "content": "add mushrooms and olives"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tc-a",
                    "function": {"name": "add_topping", "arguments": '{"topping": "mushrooms"}'},
                },
                {
                    "id": "tc-b",
                    "function": {"name": "add_topping", "arguments": '{"topping": "olives"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "tc-a", "content": '{"ok": true}'},
        {"role": "tool", "tool_call_id": "tc-b", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Added both!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    func_segs = [s for s in payload.transcript_with_tool_calls if s.role == "agent_function"]
    assert len(func_segs) == 2
    assert func_segs[0].start_ms == 100
    assert func_segs[1].start_ms == 200


def test_agent_result_without_registry_completion_is_zero(tuner_config):
    """Without registry completion, agent_result timing stays unset (0)."""
    acc = CallAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.call_end_abs_ns = base_ns + 2_000_000_000
    acc.done = True
    acc.registry.record_invocation_ns("tc-1", base_ns + 75_000_000)
    transcript = [
        {"role": "user", "content": "transfer"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "tc-1", "function": {"name": "transfer", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc-1", "content": '{"ok": true}'},
        {"role": "assistant", "content": "done"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    result_seg = next(s for s in payload.transcript_with_tool_calls if s.role == "agent_result")
    assert result_seg.start_ms == 0


def test_payload_monotonic_guard_corrects_agent_end_before_start(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 10_000_000_000
    acc.done = True
    acc.speech_segments = [_seg(0, "user", 500, 1000), _seg(1, "bot", 5000, 2000)]
    acc.latency_measurements = [_meas(0, 1, ttfb_ms=10, llm_ms=20, tts_ms=30)]
    transcript = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    agent_seg = next(s for s in payload.transcript_with_tool_calls if s.role == "agent")
    assert agent_seg.start_ms == 5000
    assert agent_seg.end_ms == 5000


def test_agent_result_with_no_matching_tool_call_has_null_function_name(tuner_config):
    """Tool result with no matching tool call in context — function_name should be None."""
    acc = CallAccumulator()
    base_ns = 1_000_000_000
    acc.call_start_abs_ns = base_ns
    acc.call_end_abs_ns = base_ns + 2_000_000_000
    acc.done = True
    acc.registry.record_completion_ns("orphan-id", base_ns + 200_000_000)
    transcript = [
        {"role": "user", "content": "hello"},
        {"role": "tool", "tool_call_id": "orphan-id", "content": '{"ok": true}'},
        {"role": "assistant", "content": "Done!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    result_segs = [s for s in payload.transcript_with_tool_calls if s.role == "agent_result"]
    assert len(result_segs) == 1
    assert result_segs[0].tool is not None
    assert result_segs[0].tool.name is None
    assert result_segs[0].start_ms == 200


def test_agent_result_non_json_uses_text_not_tool_result(tuner_config):
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    transcript = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "tc-1", "function": {"name": "do_thing", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc-1", "content": "plain text result"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    result_seg = next(s for s in payload.transcript_with_tool_calls if s.role == "agent_result")
    assert result_seg.text == "plain text result"
    assert result_seg.tool.result is None


@pytest.mark.parametrize("injected_role", ["assistant", "system"])
def test_pre_seeded_preamble_instruction_excluded_from_transcript(tuner_config, injected_role):
    """Developer-injected instructions before the first LLM response — whether
    role=assistant or role=system — must not appear in the output transcript."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 2_000_000_000
    acc.done = True
    # Proactive greeting: bot speaks first, no real user segment.
    acc.speech_segments = [_seg(0, "bot", 500, 2000, is_proactive=True)]
    acc.latency_measurements = [
        _meas(-1, 0, is_proactive=True, ttfb_ms=100, llm_ms=2120, tts_ms=1940)
    ]
    transcript = [
        {
            "role": injected_role,
            "content": "Greet the customer warmly and ask which pizza they would like.",
        },
        {"role": "assistant", "content": "Hi there, thanks for calling Pipecat Pizza!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    agent_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent"]
    assert len(agent_segments) == 1
    assert agent_segments[0].text == "Hi there, thanks for calling Pipecat Pizza!"


def test_injected_user_message_without_speech_excluded_from_transcript(tuner_config):
    """Mid-conversation injected user messages (silence handlers) produce no VAD speech
    window, so they must not appear as Customer rows; the bot's reply still does."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 60_000_000_000
    acc.done = True
    acc.speech_segments = [
        _seg(0, "user", 500, 1000, windows=[[500, 1000]]),  # one real utterance, one window
        _seg(1, "bot", 2000, 5000),
        _seg(2, "bot", 46000, 47000),  # triggered by injected user message — no user window
    ]
    acc.latency_measurements = [
        _meas(0, 1, ttfb_ms=100, llm_ms=1130, tts_ms=1010),
        _meas(0, 2),  # answers user seg 0 again — no new user turn fired
    ]
    transcript = [
        {"role": "user", "content": "Yes, sir. My name is Sallym."},
        {"role": "assistant", "content": "Hi Sallym, it's great to connect with you."},
        {
            "role": "user",
            "content": "The user has been quiet. Politely and briefly ask if they're still there.",
        },
        {"role": "assistant", "content": "Are you still there?"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    user_segments = [s for s in payload.transcript_with_tool_calls if s.role == "user"]
    agent_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent"]
    assert len(user_segments) == 1
    assert user_segments[0].text == "Yes, sir. My name is Sallym."
    assert len(agent_segments) == 2
    assert agent_segments[1].text == "Are you still there?"
    assert agent_segments[1].start_ms == 46000


def test_coalesced_two_utterance_turn_merges_without_stealing_next_segment(tuner_config):
    """Two consecutive user messages = one coalesced turn (one segment, two windows). They
    merge into one row and must NOT consume the next turn's segment (the regression)."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 30_000_000_000
    acc.done = True
    acc.speech_segments = [
        _seg(0, "user", 100, 1000, windows=[[100, 500], [600, 1000]]),  # turn 1: 2 utterances
        _seg(1, "bot", 1500, 3000),
        _seg(2, "user", 5000, 6000, windows=[[5000, 6000]]),  # turn 2
        _seg(3, "bot", 7000, 9000),
    ]
    acc.latency_measurements = [_meas(0, 1, ttfb_ms=50), _meas(2, 3, ttfb_ms=50)]
    transcript = [
        {"role": "user", "content": "I would like to order"},
        {"role": "user", "content": "margarita."},
        {"role": "assistant", "content": "Great choice! What size?"},
        {"role": "user", "content": "Small."},
        {"role": "assistant", "content": "Got it."},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    user_rows = [s for s in payload.transcript_with_tool_calls if s.role == "user"]
    assert len(user_rows) == 2
    assert user_rows[0].text == "I would like to order margarita."
    assert user_rows[0].start_ms == 100
    assert user_rows[0].metadata["fragments"] == 2
    assert user_rows[1].text == "Small."
    assert user_rows[1].start_ms == 5000  # kept its own segment — not stolen


def test_silence_gap_within_one_turn_splits_via_windows(tuner_config):
    """User speaks, waits (bot never replies), speaks again — one coalesced turn with two
    windows far apart. The gap splits them into two rows so the silence stays visible."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 60_000_000_000
    acc.done = True
    acc.speech_segments = [
        _seg(0, "user", 200, 9500, windows=[[200, 1000], [9000, 9500]]),  # 8s gap between
        _seg(1, "bot", 10000, 12000),
    ]
    acc.latency_measurements = [_meas(0, 1, ttfb_ms=100)]
    transcript = [
        {"role": "user", "content": "Book me a flight"},
        {"role": "user", "content": "Are you still there?"},
        {"role": "assistant", "content": "Yes I am here!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    rows = payload.transcript_with_tool_calls
    user_rows = [s for s in rows if s.role == "user"]
    assert [r.text for r in user_rows] == ["Book me a flight", "Are you still there?"]
    assert user_rows[0].start_ms == 200
    assert user_rows[1].start_ms == 9000  # gap preserved as a separate row
    # Two rows from one coalesced turn must still get DISTINCT, sequential turn_index.
    turn_indices = [r.metadata["turn_index"] for r in rows if "turn_index" in r.metadata]
    assert turn_indices == sorted(set(turn_indices))  # unique and ascending


def test_turn_index_is_unique_per_row_across_a_coalesced_turn(tuner_config):
    """Regression for the duplicate-turn_index bug: a coalesced turn that splits into two
    rows (two utterances around a tool call, like 'I can take medium.' then 'do you have
    Coke?') must give each rendered row a distinct, sequential turn_index — not the shared
    segment id."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 60_000_000_000
    acc.done = True
    acc.speech_segments = [
        _seg(0, "bot", 500, 2000, is_proactive=True),  # greeting
        # one user turn, two utterances split by a tool call (gap > merge threshold)
        _seg(1, "user", 5000, 9000, windows=[[5000, 6000], [8000, 9000]]),
        _seg(2, "bot", 10000, 12000),
    ]
    acc.latency_measurements = [
        _meas(-1, 0, is_proactive=True, ttfb_ms=100),
        _meas(1, 2, ttfb_ms=50),
    ]
    transcript = [
        {"role": "assistant", "content": "Hi! What would you like?"},
        {"role": "user", "content": "I can take medium."},
        {
            "role": "assistant",
            "tool_calls": [{"id": "cs", "function": {"name": "choose_size", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "cs", "content": '{"size": "medium"}'},
        {"role": "user", "content": "By the way, do you have Coke?"},
        {"role": "assistant", "content": "Sorry, only pizza."},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    rows = payload.transcript_with_tool_calls
    user_rows = [s for s in rows if s.role == "user"]
    # The single coalesced turn produced two distinct rows...
    assert [r.text for r in user_rows] == ["I can take medium.", "By the way, do you have Coke?"]
    assert user_rows[0].metadata["turn_index"] != user_rows[1].metadata["turn_index"]
    # ...and every row carrying a turn_index has a unique, ascending value.
    indices = [r.metadata["turn_index"] for r in rows if "turn_index" in r.metadata]
    assert len(indices) == len(set(indices))
    assert indices == sorted(indices)


def test_user_speaks_during_tool_call_is_kept(tuner_config):
    """A real utterance spoken while a tool runs fires no turn-start, so its VAD window is
    appended to the active (previous) user segment. It must still be rendered and timed
    from that window — not dropped."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 60_000_000_000
    acc.done = True
    acc.registry.record_invocation_ns("tc1", 28_000_000)
    acc.speech_segments = [
        # One user turn, two utterances: "margarita" then "Coke" spoken during the tool call.
        _seg(0, "user", 27101, 39877, stt_ms=179, windows=[[27101, 27483], [37053, 39877]]),
        _seg(1, "bot", 40000, 46000),  # "I'm sorry..." reply
    ]
    acc.latency_measurements = [_meas(0, 1, ttfb_ms=377)]
    transcript = [
        {"role": "user", "content": "I would like to order margarita."},
        {
            "role": "assistant",
            "tool_calls": [{"id": "tc1", "function": {"name": "choose_pizza", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "tc1", "content": '{"pizza": "margherita"}'},
        {"role": "user", "content": "Can I also order, a Coke?"},
        {"role": "assistant", "content": "I'm sorry, we only offer pizzas here."},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    segs = payload.transcript_with_tool_calls
    user_rows = [s for s in segs if s.role == "user"]
    assert [r.text for r in user_rows] == [
        "I would like to order margarita.",
        "Can I also order, a Coke?",
    ]
    coke = next(s for s in segs if s.text == "Can I also order, a Coke?")
    assert coke.start_ms == 37053  # timed from its own VAD window


def test_outbound_call_user_speaks_first_is_not_filtered(tuner_config):
    """Outbound calls where the callee speaks before the bot must keep the real user row."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 10_000_000_000
    acc.done = True
    acc.speech_segments = [_seg(0, "user", 150, 2000), _seg(1, "bot", 2500, 5000)]
    acc.latency_measurements = [_meas(0, 1, ttfb_ms=50, llm_ms=800, tts_ms=500, e2e_ms=500)]
    transcript = [
        {"role": "user", "content": "Hello?"},
        {"role": "assistant", "content": "Hi there! This is Sam calling from Acme."},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    user_segments = [s for s in payload.transcript_with_tool_calls if s.role == "user"]
    agent_segments = [s for s in payload.transcript_with_tool_calls if s.role == "agent"]
    assert len(user_segments) == 1
    assert user_segments[0].text == "Hello?"
    assert user_segments[0].start_ms == 150
    assert len(agent_segments) == 1
    assert agent_segments[0].start_ms == 2500


def test_silence_gap_splits_into_two_user_rows(tuner_config):
    """The triggering bug: user speaks, waits through a long silence, speaks again — the
    two utterances must stay as separate rows so the gap is visible."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 60_000_000_000
    acc.done = True
    acc.speech_segments = [
        _seg(0, "user", 200, 1000),  # "Book me a flight"
        _seg(1, "user", 9000, 9500),  # 8s later: "Are you still there?"
        _seg(2, "bot", 10000, 12000),
    ]
    acc.latency_measurements = [_meas(1, 2, ttfb_ms=100, llm_ms=500, tts_ms=300, e2e_ms=500)]
    transcript = [
        {"role": "user", "content": "Book me a flight"},
        {"role": "user", "content": "Are you still there?"},
        {"role": "assistant", "content": "Yes I am here!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    segs = payload.transcript_with_tool_calls
    assert [s.role for s in segs] == ["user", "user", "agent"]
    assert segs[0].text == "Book me a flight"
    assert segs[0].start_ms == 200
    assert segs[1].text == "Are you still there?"
    assert segs[1].start_ms == 9000  # gap preserved — not merged


def test_mid_sentence_pause_merges_into_one_user_row(tuner_config):
    """A short VAD pause between fragments of one thought merges into a single row."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 10_000_000_000
    acc.done = True
    acc.speech_segments = [
        _seg(0, "user", 200, 1000),  # "I want to book a"
        _seg(1, "user", 1300, 2000),  # 300ms pause: "flight to Cairo"
        _seg(2, "bot", 2500, 4000),
    ]
    acc.latency_measurements = [_meas(1, 2, ttfb_ms=50, llm_ms=200, tts_ms=100)]
    transcript = [
        {"role": "user", "content": "I want to book a"},
        {"role": "user", "content": "flight to Cairo"},
        {"role": "assistant", "content": "Sure!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    segs = payload.transcript_with_tool_calls
    assert [s.role for s in segs] == ["user", "agent"]
    assert segs[0].text == "I want to book a flight to Cairo"
    assert segs[0].metadata["fragments"] == 2


def test_user_speaks_then_session_closes_is_orphan(tuner_config):
    """User speaks, then the session closes before any bot reply: the user row is kept
    with timing and stt, no LatencyMeasurement, no agent row, no crash."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 5_000_000_000
    acc.done = True
    acc.speech_segments = [_seg(0, "user", 500, 2000, stt_ms=150)]
    acc.latency_measurements = []
    transcript = [{"role": "user", "content": "Hello, is anyone there?"}]
    payload = acc.build_payload(tuner_config, transcript)
    segs = payload.transcript_with_tool_calls
    assert [s.role for s in segs] == ["user"]
    assert segs[0].text == "Hello, is anyone there?"
    assert segs[0].start_ms == 500
    assert segs[0].end_ms == 2000
    assert segs[0].metadata["stt_node_ttfb"] == 150


def test_proactive_greeting_then_real_turn(tuner_config):
    """Bot greets first (proactive, no e2e), then a real user→bot exchange follows."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 10_000_000_000
    acc.done = True
    acc.speech_segments = [
        _seg(0, "bot", 800, 2000, is_proactive=True),
        _seg(1, "user", 5000, 6000),
        _seg(2, "bot", 7000, 9000),
    ]
    acc.latency_measurements = [
        _meas(-1, 0, is_proactive=True, ttfb_ms=769),
        _meas(1, 2, ttfb_ms=50, llm_ms=200, tts_ms=100, e2e_ms=1000),
    ]
    transcript = [
        {"role": "assistant", "content": "Hello! Welcome to Pipecat Pizza."},
        {"role": "user", "content": "Hi there"},
        {"role": "assistant", "content": "How can I help?"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    segs = payload.transcript_with_tool_calls
    assert [s.role for s in segs] == ["agent", "user", "agent"]
    assert segs[0].text == "Hello! Welcome to Pipecat Pizza."
    assert segs[0].metadata.get("e2e_latency") is None  # proactive
    assert segs[2].metadata["e2e_latency"] == 1000


def test_speak_tool_speak_both_voiced_are_timed_not_ghosted(tuner_config):
    """When the bot voices a line, runs tools, then voices another line, the captured TTS
    text confirms both were spoken — so the first is timed, not wrongly ghosted."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 30_000_000_000
    acc.done = True
    acc.speech_segments = [
        _seg(0, "user", 1000, 2000),
        _seg(
            1,
            "bot",
            3000,
            6000,
            spoken_text="Just to confirm, you'd like a large and a medium. Let me check.",
        ),
        _seg(2, "bot", 8000, 11000, spoken_text="Your total is $27.98. Should I confirm?"),
    ]
    acc.latency_measurements = [
        _meas(0, 1, ttfb_ms=400, e2e_ms=1000),
        _meas(0, 2, ttfb_ms=300),
    ]
    acc.user_transcriptions = [("and another one medium", 1000)]
    transcript = [
        {"role": "user", "content": "And another one medium."},
        {
            "role": "assistant",
            "content": "Just to confirm, you'd like a large and a medium. Let me check.",
        },
        {
            "role": "assistant",
            "tool_calls": [{"id": "p", "function": {"name": "choose_pizza", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "p", "content": "{}"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "s", "function": {"name": "choose_size", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "s", "content": "{}"},
        {"role": "assistant", "content": "Your total is $27.98. Should I confirm?"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    agents = [s for s in payload.transcript_with_tool_calls if s.role == "agent"]
    assert len(agents) == 2
    assert agents[0].text.startswith("Just to confirm")
    assert agents[0].start_ms == 3000  # timed, not a 0/0 ghost
    assert agents[1].start_ms == 8000


def test_unvoiced_draft_before_tool_call_stays_ghost(tuner_config):
    """A draft the LLM produced but never voiced (superseded by a tool call) matches no TTS
    text and is correctly left as a ghost."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 30_000_000_000
    acc.done = True
    acc.speech_segments = [
        _seg(0, "user", 1000, 2000),
        _seg(1, "bot", 8000, 11000, spoken_text="What size would you like?"),
    ]
    acc.latency_measurements = [_meas(0, 1, ttfb_ms=300)]
    acc.user_transcriptions = [("i want pizza", 1000)]
    transcript = [
        {"role": "user", "content": "I want pizza."},
        {"role": "assistant", "content": "Let me look that up for you."},  # never voiced
        {
            "role": "assistant",
            "tool_calls": [{"id": "x", "function": {"name": "lookup", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "x", "content": "{}"},
        {"role": "assistant", "content": "What size would you like?"},  # voiced
    ]
    payload = acc.build_payload(tuner_config, transcript)
    agents = [s for s in payload.transcript_with_tool_calls if s.role == "agent"]
    draft = next(s for s in agents if "look that up" in s.text)
    voiced = next(s for s in agents if "size" in s.text)
    assert draft.start_ms == 0  # ghost
    assert voiced.start_ms == 8000


def test_user_interruption_flags_propagate(tuner_config):
    """When the user interrupts the agent, the agent row is flagged interrupted and the
    following user row is flagged as the interrupting turn."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 10_000_000_000
    acc.done = True
    acc.speech_segments = [
        _seg(0, "user", 200, 500),
        _seg(1, "bot", 800, 1500, interrupted=True, interrupted_at_ms=1200),
        _seg(2, "user", 1200, 1800),  # the interrupting utterance
        _seg(3, "bot", 2200, 3000),
    ]
    acc.latency_measurements = [
        _meas(0, 1, ttfb_ms=50, was_interrupted=True, interrupted_at_ms=1200),
        _meas(2, 3, ttfb_ms=50, e2e_ms=400),
    ]
    transcript = [
        {"role": "user", "content": "Tell me about your menu"},
        {"role": "assistant", "content": "We have margherita, pepperoni, veggie, and"},
        {"role": "user", "content": "Just pepperoni please"},
        {"role": "assistant", "content": "Great choice!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    segs = payload.transcript_with_tool_calls
    assert [s.role for s in segs] == ["user", "agent", "user", "agent"]
    assert segs[1].metadata["interrupted"] is True  # agent was interrupted
    assert segs[2].metadata["interrupted"] is True  # this user did the interrupting


def test_clean_end_call_hangup_not_marked_interrupted(tuner_config):
    """TurnTrackingObserver reports was_interrupted=True when the pipeline ends on a clean
    end_call hangup. With no interrupted_at_ms (no user cut in), the final agent row must
    NOT be flagged interrupted."""
    acc = CallAccumulator()
    acc.call_start_abs_ns = 0
    acc.call_end_abs_ns = 10_000_000_000
    acc.done = True
    acc.speech_segments = [
        _seg(0, "user", 200, 500, windows=[[200, 500]]),
        # final bot turn ended by end_call: was_interrupted set by the observer, but the
        # user never cut in → no interrupted_at_ms.
        _seg(1, "bot", 800, 3000, interrupted=True),
    ]
    acc.latency_measurements = [_meas(0, 1, ttfb_ms=50, was_interrupted=True)]
    transcript = [
        {"role": "user", "content": "Yes, confirm."},
        {"role": "assistant", "content": "Thank you! Have a great day!"},
    ]
    payload = acc.build_payload(tuner_config, transcript)
    agent = next(s for s in payload.transcript_with_tool_calls if s.role == "agent")
    assert agent.metadata["interrupted"] is False  # clean hangup, not a real interruption
