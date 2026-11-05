"""Caller speech timed by pipecat's SpeakingObserver (pipecat >= 1.9).

Stretches of speech are matched to user turns by time, so user rows and latency come from when
the caller really spoke. A stretch with no turn, or a turn with no stretch, changes nothing else.
"""

import asyncio

import pytest

from tuner_pipecat_sdk.accumulator import CallAccumulator


def _call(replay, *, speech_timing=True):
    r = replay()
    r.acc.speech_timing = speech_timing
    return r


def _speech(r, start_ms, end_ms):
    """One stretch of speech, as SpeakingObserver reports it (unix seconds)."""
    base_s = r.base / 1_000_000_000
    r.acc.on_user_speech(base_s + start_ms / 1000, base_s + end_ms / 1000)
    return r


def _users(rows):
    return [(row.start_ms, row.end_ms) for row in rows if row.role == "user"]


def test_speech_bounds_joins_the_stretches_overlapping_a_turn():
    acc = CallAccumulator()
    acc.user_speech = [(400, 900), (1200, 1700), (5000, 5300)]
    assert acc.speech_bounds(700, 2200) == (400, 1700)
    assert acc.speech_bounds(2500, 4000) is None


def test_user_row_spans_the_callers_real_speech(replay, tuner_config):
    r = _call(replay)
    r.turn_start(1, 1200).user_start(1200)
    _speech(r, 1000, 1400)  # speech began at 1000; UserStartedSpeakingFrame arrived at 1200
    r.transcription("hello there", 1450).user_stop(1900)
    r.bot_says("hi, how can I help?", 2500, 3500)
    assert _users(r.rows(tuner_config)) == [(1000, 1400)]


def test_a_stretch_shorter_than_the_vad_start_delay_is_kept(replay, tuner_config):
    """'Eh' (0.18 s) ends before UserStartedSpeakingFrame arrives, 0.2 s after it began."""
    r = _call(replay)
    r.turn_start(1, 192643).user_start(192643)
    _speech(r, 192442, 192621)
    _speech(r, 193274, 193701)
    r.transcription("Eh", 193816).user_stop(194634).transcription("That's all of them.", 194728)
    r.bot_says("Any as-needed medications?", 196803, 199000)
    assert _users(r.rows(tuner_config)) == [(192442, 193701)]


def test_a_stretch_across_two_turns_leaves_both_on_their_frame_times(replay, tuner_config):
    """The STT split one stretch of speech into two turns (server-side endpointing). The
    stretch can't tell where one turn ended, so neither row takes it."""
    r = _call(replay)
    r.turn_start(1, 1000).user_start(1000).transcription("part one", 1400).user_stop(1500)
    r.user_start(1800).transcription("part two", 2200).user_stop(2300)
    _speech(r, 900, 3000)
    assert _users(r.rows(tuner_config)) == [(1000, 1500), (1800, 2300)]


def test_a_stretch_is_used_by_one_row_only(replay, tuner_config):
    """The first turn's stretch ends inside the next turn's lead window."""
    r = _call(replay)
    r.turn_start(1, 1200).user_start(1200)
    _speech(r, 1000, 5000)
    r.transcription("a long first answer", 5100).user_stop(5300)
    r.user_start(5400)
    _speech(r, 5200, 6000)
    r.transcription("second", 6100).user_stop(6300)
    assert _users(r.rows(tuner_config)) == [(1000, 5000), (5200, 6000)]


def test_speech_that_starts_before_a_turn_closes_is_the_next_turns(replay, tuner_config):
    """From a live call: the caller started '1958.' 182 ms before 'March fourteen' closed."""
    r = _call(replay)
    r.turn_start(1, 35643).user_start(35643)
    _speech(r, 35443, 36822)
    r.transcription("March fourteen", 38254).user_stop(38261)
    r.user_start(38280)
    _speech(r, 38079, 39410)
    r.transcription("1958.", 40411).user_stop(40412)
    assert _users(r.rows(tuner_config)) == [(35443, 36822), (38079, 39410)]


def test_speech_that_ended_long_before_a_turn_started_is_not_its(replay, tuner_config):
    """The caller talked over the bot without starting a turn, and stopped just before the
    turn that followed."""
    r = _call(replay)
    r.bot_says("let me check that for you", 500, 2500)
    _speech(r, 1000, 2900)
    r.turn_start(1, 3200).user_start(3200)
    _speech(r, 3000, 3600)
    r.transcription("ok", 3650).user_stop(4100)
    assert _users(r.rows(tuner_config)) == [(3000, 3600)]


def _reply_after_speech(replay, tuner_config, *, speech_timing):
    r = _call(replay, speech_timing=speech_timing)
    r.turn_start(1, 1200).user_start(1200)
    _speech(r, 1000, 1400)
    r.transcription("hello", 1450).user_stop(1900)
    r.generate("hi", 2000, 2200).tts("hi", 2300).bot_start(2400)
    r.acc.on_latency_measured(1.0)  # pipecat's own latency for this reply
    r.latency(0.05, "TTSService", user_start_ms=1000).bot_stop(3000).turn_end(1)
    return next(row for row in r.rows(tuner_config) if row.role == "agent")


def test_latency_runs_from_the_end_of_speech_and_the_bot_start_stays(replay, tuner_config):
    agent = _reply_after_speech(replay, tuner_config, speech_timing=True)
    assert agent.start_ms == 2400
    assert agent.metadata["e2e_latency"] == 2400 - 1400


def test_without_speech_timing_the_queued_latency_moves_the_bot_start(replay, tuner_config):
    agent = _reply_after_speech(replay, tuner_config, speech_timing=False)
    assert agent.start_ms == 1900 + 1000
    assert agent.metadata["e2e_latency"] == 1000


def test_latency_counts_speech_after_the_turn_closed(replay, tuner_config):
    """The caller kept talking after the turn closed, with nothing transcribed: they were still
    speaking, so the wait starts when they stopped, as pipecat measures it."""
    r = _call(replay)
    r.turn_start(1, 1200).user_start(1200)
    _speech(r, 1000, 1400)
    r.transcription("hello", 1450).user_stop(1500)
    _speech(r, 1600, 1800)
    r.generate("hi", 2000, 2200).tts("hi", 2300).bot_start(2400)
    r.latency(0.05, "TTSService", user_start_ms=1000).bot_stop(3000).turn_end(1)
    rows = r.rows(tuner_config)
    agent = next(row for row in rows if row.role == "agent")
    assert _users(rows) == [(1000, 1400)]
    assert agent.metadata["e2e_latency"] == 2400 - 1800


def test_latency_uses_the_speech_end_pipecat_sends_as_the_turn_start(replay, tuner_config):
    """Pipecat 1.12's breakdown carries user_turn_start_time set to the user's stop time, and
    the accumulator writes it onto the user segment's start. Latency must still find the stop."""
    r = _call(replay)
    r.turn_start(1, 1200).user_start(1200)
    _speech(r, 1000, 1400)
    r.transcription("hello", 1450).user_stop(1900)
    r.generate("hi", 2000, 2200).tts("hi", 2300).bot_start(2400)
    r.latency(0.05, "TTSService", user_start_ms=1400).bot_stop(3000).turn_end(1)
    agent = next(row for row in r.rows(tuner_config) if row.role == "agent")
    assert agent.metadata["e2e_latency"] == 2400 - 1400


def test_latency_ignores_speech_from_an_earlier_turn(replay, tuner_config):
    r = _call(replay)
    r.turn_start(1, 1200).user_start(1200)
    _speech(r, 1000, 1400)
    r.transcription("hello", 1450).user_stop(1900)
    r.bot_says("hi", 2400, 3000)
    r.turn_start(2, 4000).user_start(4000)  # no stretch reported for this turn
    r.transcription("bye", 4100).user_stop(4600)
    r.generate("goodbye", 4700, 4900).tts("goodbye", 5000).bot_start(5100)
    r.latency(0.05, "TTSService", user_start_ms=4000).bot_stop(5600).turn_end(2)
    agents = [row for row in r.rows(tuner_config) if row.role == "agent"]
    assert agents[-1].metadata["e2e_latency"] == 5100 - 4600  # frame times, not turn 1's speech


def test_a_turn_without_a_stretch_keeps_its_times_and_moves_nothing_else(replay, tuner_config):
    r = _call(replay)
    r.turn_start(1, 1200).user_start(1200).transcription("yes", 1300).user_stop(1700)
    r.bot_says("great", 2000, 2600)
    r.turn_start(2, 3200).user_start(3200)
    _speech(r, 3000, 3500)
    r.transcription("march third", 3550).user_stop(4000)
    r.bot_says("thanks", 4400, 5000)
    assert _users(r.rows(tuner_config)) == [(1200, 1700), (3000, 3500)]


def test_speech_outside_any_turn_is_ignored(replay, tuner_config):
    r = _call(replay)
    r.bot_says("let me check that for you", 500, 2500)
    _speech(r, 1500, 1700)  # a cough while the bot spoke, never a turn
    r.turn_start(1, 3200).user_start(3200)
    _speech(r, 3000, 3600)
    r.transcription("ok", 3650).user_stop(4100)
    assert _users(r.rows(tuner_config)) == [(3000, 3600)]


async def test_observer_records_speech_from_vad_frames():
    pytest.importorskip("pipecat.observers.speaking_observer")
    from unittest.mock import Mock

    from pipecat.frames.frames import VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame
    from pipecat.observers.base_observer import FramePushed
    from pipecat.processors.frame_processor import FrameDirection

    from tuner_pipecat_sdk.observer import Observer

    observer = Observer(
        api_key="test-key",
        workspace_id=1,
        agent_id="agent-1",
        call_id="call-1",
        base_url="https://tuner.test",
        forward_traces=False,
    )
    observer._acc.call_start_abs_ns = 1000 * 1_000_000_000
    for frame in (
        VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=1001.2),
        VADUserStoppedSpeakingFrame(stop_secs=0.8, timestamp=1003.0),
    ):
        await observer.on_push_frame(
            FramePushed(
                source=Mock(),
                destination=Mock(),
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )
    await asyncio.sleep(0)  # SpeakingObserver runs its event handler as a task

    assert observer._acc.speech_timing is True
    assert observer._acc.user_speech == [(1000, 2200)]
