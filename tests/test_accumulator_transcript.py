"""Event-sourced transcript construction.

The transcript is built LIVE from the frame stream (mirroring pipecat's own conversation
construction), not reconstructed from the final LLM context. These tests script a call through
the accumulator's on_* event methods via the Replay helper and assert the rendered rows: order
comes from arrival order, timing from the linked speech segments, and a turn exists iff its
frames occurred (so developer-injected messages, which emit no STT/LLM frames, simply never
appear — no dropping logic needed).
"""


def _roles(rows):
    return [r.role for r in rows]


def _texts(rows, role):
    return [r.text for r in rows if r.role == role]


def test_basic_user_then_agent_exchange(replay, tuner_config):
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("Hello there.", 1200)
        .user_stop(1500)
        .generate("Hi! How can I help?", 1800, 2000)
        .tts("Hi! How can I help?", 1900)
        .bot_start(2100)
        .bot_stop(3000)
        .latency(0.05, "TTSService", user_start_ms=1000)
        .turn_end(1)
        .rows(tuner_config)
    )
    assert _roles(rows) == ["user", "agent"]
    user, agent = rows
    assert user.text == "Hello there." and user.start_ms == 1000 and user.end_ms == 1500
    assert agent.text == "Hi! How can I help?" and agent.start_ms == 2100 and agent.end_ms == 3000
    assert agent.metadata["tts_node_ttfb"] == 50


def test_order_is_arrival_order(replay, tuner_config):
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("first", 1100)
        .user_stop(1200)
        .bot_says("reply one", 1500, 1800)
        .turn_start(2, 3000)
        .user_start(3000)
        .transcription("second", 3100)
        .user_stop(3200)
        .bot_says("reply two", 3500, 3800)
        .rows(tuner_config)
    )
    assert _roles(rows) == ["user", "agent", "user", "agent"]
    assert [r.text for r in rows] == ["first", "reply one", "second", "reply two"]


def test_proactive_greeting_has_no_e2e_latency(replay, tuner_config):
    rows = (
        replay()
        .generate("Welcome to Pizza!", 100, 300)
        .tts("Welcome to Pizza!", 150)
        .bot_start(400)
        .bot_stop(2000)
        .latency(0.2, "TTSService")  # proactive: no user_turn_start_time
        .rows(tuner_config)
    )
    assert _roles(rows) == ["agent"]
    assert rows[0].text == "Welcome to Pizza!"
    assert rows[0].metadata.get("e2e_latency") is None
    assert rows[0].metadata["tts_node_ttfb"] == 200


def test_user_turn_without_transcription_produces_no_row(replay, tuner_config):
    """A user row requires a TranscriptionFrame. A turn boundary that produces none (a spurious
    VAD trigger, or pipecat's proactive ghost turn) yields no row.

    This is precisely *why* a developer-injected ``{"role":"user"}`` context message can never
    appear: the transcript is built only from frames (acc.live_turns), and an injected message —
    added straight to the LLM context — emits no TranscriptionFrame, so there is no channel
    through which it could enter. (In the old reconstruction model the injected message lived in
    the context and had to be detected and dropped; here it is absent by construction.)
    """
    rows = (
        replay()
        .turn_start(1, 1000)  # boundary fires but no transcription arrives → no user row
        .user_start(1000)
        .user_stop(1500)
        .turn_start(2, 5000)
        .user_start(5000)
        .transcription("I would like a pizza.", 5200)
        .user_stop(5500)
        .generate("Sure!", 5800, 6000)
        .bot_start(6100)
        .bot_stop(6500)
        .rows(tuner_config)
    )
    assert _texts(rows, "user") == ["I would like a pizza."]


def test_assistant_response_with_only_tool_call_has_no_text_row(replay, tuner_config):
    """An agent text row requires generated text (or voiced TTS). An LLM response that emits no
    text — e.g. one that only makes a tool call — produces no agent row, only the tool rows.

    This is the assistant-side analog: a developer-injected ``{"role":"assistant"}`` context
    message emits no LLMTextFrame and is never voiced, so it cannot enter the transcript.
    """
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("order a pizza", 1100)
        .user_stop(1200)
        .llm_start(1300)  # LLM response with no text — only a tool call
        .llm_end(1350)
        .tool_call("choose_pizza", "tc-1", {"pizza": "margherita"}, 1400)
        .tool_result("choose_pizza", "tc-1", {"ok": True}, 1500)
        .generate("Great choice!", 1700, 1800)
        .bot_start(1900)
        .bot_stop(2300)
        .rows(tuner_config)
    )
    assert _texts(rows, "agent") == ["Great choice!"]
    assert "agent_function" in _roles(rows)


def test_tool_call_and_result_rows(replay, tuner_config):
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("order a pizza", 1100)
        .user_stop(1200)
        .tool_call("choose_pizza", "tc-1", {"pizza": "margherita"}, 1500)
        .tool_result("choose_pizza", "tc-1", {"pizza": "margherita", "price": 10.99}, 1700)
        .generate("Great choice!", 1900, 2000)
        .bot_start(2100)
        .bot_stop(2500)
        .rows(tuner_config)
    )
    assert _roles(rows) == ["user", "agent_function", "agent_result", "agent"]
    func = rows[1]
    assert func.text == "choose_pizza(pizza=margherita)"
    assert func.start_ms == 1500
    assert func.tool.name == "choose_pizza" and func.tool.params == {"pizza": "margherita"}
    result = rows[2]
    assert result.text is None  # structured result
    assert result.start_ms == 1700
    assert result.tool.result == {"pizza": "margherita", "price": 10.99}


def test_parallel_tool_calls_keep_distinct_timing(replay, tuner_config):
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("do both", 1100)
        .user_stop(1200)
        .tool_call("lookup", "a", {}, 1500)
        .tool_call("lookup", "b", {}, 1700)
        .rows(tuner_config)
    )
    funcs = [r for r in rows if r.role == "agent_function"]
    assert [f.start_ms for f in funcs] == [1500, 1700]


def test_non_json_tool_result_is_text(replay, tuner_config):
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("x", 1100)
        .user_stop(1200)
        .tool_call("note", "n1", {}, 1500)
        .tool_result("note", "n1", "done", 1600)
        .rows(tuner_config)
    )
    result = next(r for r in rows if r.role == "agent_result")
    assert result.text == "done"
    assert result.tool.result is None


def test_unvoiced_draft_never_committed_is_dropped(replay, tuner_config):
    # A draft the LLM generated but that was superseded (a tool call ran, then a new response)
    # never receives a commit (LLMContextAssistantTimestampFrame), so — exactly like pipecat —
    # it never appears. Only the committed response shows.
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("i want pizza", 1100)
        .user_stop(1200)
        .generate("Let me look that up.", 1500, 1700)  # generated, never voiced, never committed
        .tool_call("lookup", "t1", {}, 1800)
        .tool_result("lookup", "t1", {"ok": True}, 1900)
        .bot_says("What size?", 2300, 2800)  # voiced + committed
        .rows(tuner_config)
    )
    agents = [r for r in rows if r.role == "agent"]
    assert [a.text for a in agents] == ["What size?"]
    assert agents[0].start_ms == 2300


def test_speak_tool_speak_both_committed_are_timed(replay, tuner_config):
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("confirm", 1100)
        .user_stop(1200)
        .generate("Just to confirm.", 1500, 1700)
        .tts("Just to confirm.", 1600)
        .bot_start(1800)
        .bot_stop(2200)
        .assistant_commit(2200)
        .tool_call("check", "t1", {}, 2300)
        .tool_result("check", "t1", {"ok": True}, 2400)
        .generate("Your total is $10.", 2600, 2700)
        .tts("Your total is $10.", 2650)
        .bot_start(2800)
        .bot_stop(3200)
        .assistant_commit(3200)
        .rows(tuner_config)
    )
    agents = [r for r in rows if r.role == "agent"]
    assert [a.text for a in agents] == ["Just to confirm.", "Your total is $10."]
    assert [a.start_ms for a in agents] == [1800, 2800]  # both timed, neither a ghost


def test_orphan_user_turn_no_bot_reply(replay, tuner_config):
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("anyone there?", 1200)
        .stt_ttfb(0.3)
        .user_stop(1500)
        .end(2000)
        .rows(tuner_config)
    )
    assert _roles(rows) == ["user"]
    assert rows[0].text == "anyone there?"
    assert rows[0].metadata["stt_node_ttfb"] == 300


def test_coalesced_utterances_in_one_turn_make_one_row(replay, tuner_config):
    # Several transcriptions within one user turn (intermediate pauses are INCOMPLETE, so only
    # one UserStoppedSpeaking fires) → one row, exactly as pipecat commits one user message.
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("I would like", 1100)
        .transcription("a margherita", 1600)
        .user_stop(1800)
        .rows(tuner_config)
    )
    users = [r for r in rows if r.role == "user"]
    assert len(users) == 1
    assert users[0].text == "I would like a margherita"
    assert users[0].metadata["fragments"] == 2


def test_separate_user_turn_stops_make_separate_rows(replay, tuner_config):
    # Two utterances each ending in their own UserStoppedSpeaking → two rows (one per pipecat
    # user-message boundary), regardless of the gap length.
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("hello", 1100)
        .user_stop(1300)
        .user_start(5000)
        .transcription("are you there", 5100)
        .user_stop(5300)
        .rows(tuner_config)
    )
    users = [r for r in rows if r.role == "user"]
    assert [u.text for u in users] == ["hello", "are you there"]
    assert [u.start_ms for u in users] == [1000, 5000]
    # turn_index is unique and ascending across the split
    assert users[0].metadata["turn_index"] != users[1].metadata["turn_index"]


def test_interruption_flag_propagates(replay, tuner_config):
    # User cuts in while the bot is speaking → agent row interrupted, next user row flagged.
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("question", 1100)
        .user_stop(1200)
        .generate("Here is a long answer", 1400, 1600)
        .tts("Here is a long answer", 1500)
        .bot_start(1700)
        .turn_start(2, 2000)  # interrupting turn
        .user_start(2000)  # interrupts the bot
        .bot_stop(2050)
        .assistant_commit(2050)  # the spoken portion is committed
        .transcription("stop", 2100)
        .user_stop(2300)
        .turn_end(1, interrupted=True)
        .rows(tuner_config)
    )
    agent = next(r for r in rows if r.role == "agent")
    assert agent.metadata["interrupted"] is True
    assert agent.metadata.get("interrupted_at_ms") == 2000
    # the interrupting user row carries the flag
    interrupting_user = [r for r in rows if r.role == "user" and r.text == "stop"][0]
    assert interrupting_user.metadata["interrupted"] is True


def test_interrupted_uncommitted_greeting_is_dropped(replay, tuner_config):
    # The greeting is interrupted and regenerated. The interrupted attempt never receives a
    # commit (LLMContextAssistantTimestampFrame), so — like pipecat — it does not appear. The
    # committed re-greeting and the next turn show with their real timing.
    full = "Welcome to Pizza! Today we have margherita and pepperoni. What can I get you?"
    rows = (
        replay()
        # turn 1: greeting interrupted; never committed
        .turn_start(1, 100)
        .generate("Welcome to Pizza!", 200, 300)
        .bot_start(400)
        .user_start(450)  # barge-in
        .bot_stop(460)
        .transcription("okay", 500)
        .user_stop(700)
        .turn_end(1, interrupted=True)
        # turn 2: full re-greeting, voiced + committed
        .turn_start(2, 800)
        .bot_says(full, 1300, 5000)
        # turn 3: user orders, bot replies (committed)
        .turn_start(3, 7000)
        .user_start(7000)
        .transcription("a margherita", 7200)
        .user_stop(7500)
        .bot_says("Coming up!", 8100, 9000)
        .turn_end(3)
        .rows(tuner_config)
    )
    agents = [r for r in rows if r.role == "agent"]
    assert [a.text for a in agents] == [full, "Coming up!"]  # interrupted attempt absent
    coming = next(a for a in agents if a.text == "Coming up!")
    assert coming.start_ms == 8100


def test_interrupted_assistant_responses_without_commit_are_dropped(replay, tuner_config):
    """Real nova-clinic trial: the caller interrupts twice while the bot is responding. Pipecat
    discards those interrupted assistant responses (they never receive an
    LLMContextAssistantTimestampFrame), committing only the greeting, the later answer, and the
    final (call-ended) response. Tuner must match that committed set exactly.

    Pipecat committed: greeting, U, U, U, "Of course. Which doctor…", U, final answer.
    """
    full_final = (
        "Thank you for asking. Currently, Dr. Ahmad Waheed is not part of our clinic. "
        "Our available doctors are Dr. Sarah Patel and Dr. James Lee."
    )
    greeting = "Hello, thank you for calling Nova Clinic. How can I help you today?"
    r = replay()
    # greeting — committed
    r.turn_start(1, 0).bot_says(greeting, 4075, 9152)
    r.user_start(10442).transcription("Hello, I would like to book an appointment.", 13201)
    r.user_stop(13224)
    # response 2 — generated, never voiced, interrupted → no commit → dropped
    r.generate("Of course, I can help you with that. May I have your full name?", 13226, 14907)
    r.turn_start(2, 15499).user_start(15499)
    # response 3 — voiced briefly then interrupted → no commit → dropped
    r.generate("Of course, I can help you with both booking and your doctor inquiry.", 21868, 23541)
    r.transcription("You know what?", 17381)
    r.transcription("I would also want to inquire about a doctor.", 21831).user_stop(21866)
    r.bot_start(26085).bot_stop(27145)  # spoke a bit, no commit
    r.turn_start(3, 27138).user_start(27138)  # interrupts response 3
    # response 4 — committed
    r.transcription("And I want to start first with the doctor inquiry.", 32130).user_stop(32164)
    r.bot_says("Of course. Which doctor would you like to know more about?", 35470, 41421)
    # user 4
    r.turn_start(4, 42340).user_start(42340)
    r.transcription("Does Dr. Ahmad work here?", 49641).user_stop(49670)
    # response 5 — spoken; call ends before its own commit frame → committed at call end
    r.generate(full_final, 49672, 56524).bot_start(53024)
    rows = r.end(60500).rows(tuner_config)

    assert _roles(rows) == ["agent", "user", "user", "user", "agent", "user", "agent"]
    agents = _texts(rows, "agent")
    assert agents[0].startswith("Hello, thank you for calling")
    assert agents[1] == "Of course. Which doctor would you like to know more about?"
    assert agents[2] == full_final
    # the two interrupted responses never appear
    assert not any("help you with that" in t for t in agents)
    assert not any("both booking" in t for t in agents)


def test_latency_metadata_per_node(replay, tuner_config):
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("hi", 1100)
        .stt_ttfb(0.1)
        .user_stop(1500)
        .generate("hello", 1800, 2000)
        .tts("hello", 1900)
        .bot_start(2100)
        .bot_stop(2600)
        .latency(0.05, "TTSService", user_start_ms=1000)
        .turn_end(1)
        .rows(tuner_config)
    )
    user = next(r for r in rows if r.role == "user")
    agent = next(r for r in rows if r.role == "agent")
    assert user.metadata["stt_node_ttfb"] == 100
    assert agent.metadata["tts_node_ttfb"] == 50
    assert agent.metadata["e2e_latency"] == 2100 - 1500


def test_agent_text_is_generation_even_when_tts_voiced_less(replay, tuner_config):
    # Decided behavior: agent text mirrors the LLM generation (pipecat UI parity), even if TTS
    # voiced only part of it before an interruption.
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("hi", 1100)
        .user_stop(1200)
        .generate("Full generated answer with lots of detail.", 1400, 1600)
        .tts("Full generated", 1500)  # only part voiced
        .bot_start(1700)
        .bot_stop(2000)
        .rows(tuner_config)
    )
    agent = next(r for r in rows if r.role == "agent")
    assert agent.text == "Full generated answer with lots of detail."
