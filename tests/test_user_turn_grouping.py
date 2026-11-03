"""User rows are grouped around the bot's replies, as pipecat's client shows them.

A final transcript that lands just after its turn closed still belongs to it, and a turn that is
open when the bot starts its reply ends there.
"""


def _rows(rows):
    return [(row.role, row.text) for row in rows if row.role in ("user", "agent")]


def test_a_late_transcript_joins_the_turn_it_was_said_in(replay, tuner_config):
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("March 14 2019", 1500)
        .user_stop(1700)
        .transcription("Nineteen.", 1730)  # STT finalized after the turn closed
        .bot_says("That date doesn't match. Could you repeat it?", 2500, 3500)
        .rows(tuner_config)
    )
    assert _rows(rows) == [
        ("user", "March 14 2019 Nineteen."),
        ("agent", "That date doesn't match. Could you repeat it?"),
    ]


def test_a_turn_that_closed_before_its_transcript_keeps_its_times(replay, tuner_config):
    """The whole transcript arrived after the turn closed (a slow STT response)."""
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .user_stop(1800)
        .transcription("Sixty tablets every ninety days.", 14800)
        .bot_says("Thanks. Is that right?", 16000, 17000)
        .rows(tuner_config)
    )
    user = next(row for row in rows if row.role == "user")
    assert (user.text, user.start_ms, user.end_ms) == (
        "Sixty tablets every ninety days.",
        1000,
        1800,
    )
    assert [row.role for row in rows if row.role in ("user", "agent")] == ["user", "agent"]


def test_a_transcript_after_the_bot_started_starts_a_new_turn(replay, tuner_config):
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("Yes.", 1300)
        .user_stop(1500)
        .bot_says("Great. What's your date of birth?", 2000, 3000)
        .transcription("Okay.", 3200)
        .user_stop(3400)
        .rows(tuner_config)
    )
    assert _rows(rows) == [
        ("user", "Yes."),
        ("agent", "Great. What's your date of birth?"),
        ("user", "Okay."),
    ]


def test_a_turn_open_when_the_bot_starts_ends_there(replay, tuner_config):
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("Carvedilol", 1400)
        .user_stop(1600)
        .turn_start(2, 1800)
        .user_start(1800)
        .transcription("25 mg tablets, one tablet twice a day", 2600)
        # the turn is still open when the bot answers "Carvedilol"
        .bot_says("What's the strength?", 3000, 4000)
        .transcription("Sixty tablets every ninety days.", 5200)
        .user_stop(5400)
        .rows(tuner_config)
    )
    assert _rows(rows) == [
        ("user", "Carvedilol"),
        ("user", "25 mg tablets, one tablet twice a day"),
        ("agent", "What's the strength?"),
        ("user", "Sixty tablets every ninety days."),
    ]


def test_turns_without_a_reply_between_them_stay_separate(replay, tuner_config):
    rows = (
        replay()
        .turn_start(1, 1000)
        .user_start(1000)
        .transcription("Twice a day.", 1300)
        .user_stop(1500)
        .user_start(1800)
        .transcription("Sixty tablets.", 2100)
        .user_stop(2300)
        .bot_says("Thanks.", 3000, 3500)
        .rows(tuner_config)
    )
    assert _rows(rows) == [
        ("user", "Twice a day."),
        ("user", "Sixty tablets."),
        ("agent", "Thanks."),
    ]
