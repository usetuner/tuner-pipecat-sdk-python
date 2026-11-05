"""The caller's speech as pipecat's SpeakingObserver times it (pipecat >= 1.9).

Each stretch is (start_ms, end_ms) from the start of the call: when the caller really began and
stopped, with the VAD's confirmation delay taken off. User rows and latency use these instead of
the turn frames, which arrive later. Every rule for matching stretches to turns lives here.
"""

from dataclasses import dataclass, field

# UserStartedSpeakingFrame trails the speech it reports by the VAD's start_secs (0.2 s by
# default). Speech that started up to this long before a turn's start frame is still that turn's.
_SPEECH_LEAD_MS = 500


@dataclass
class CallerSpeech:
    """Stretches of the caller's speech. Matched to user turns by time, never by order, so a
    missed stretch moves nothing else."""

    stretches: list[tuple[int, int]] = field(default_factory=list)

    def add(self, start_ms: int, end_ms: int) -> None:
        """Record one stretch, as SpeakingObserver reports it when the stretch ends."""
        if end_ms > start_ms:
            self.stretches.append((start_ms, end_ms))

    def bounds(self, start_ms: int, end_ms: int, after_ms: int = 0) -> tuple[int, int] | None:
        """The caller's real speech within a user turn seen from start_ms to end_ms: the first
        start and last end of the stretches that belong to it. None when none does.

        A stretch belongs to the turn when it ended by end_ms, since UserStoppedSpeakingFrame
        trails the speech it closes; one still going then is the next turn's. It must start at
        or after after_ms, the previous user row's end, so no two rows share a stretch.

        start_ms is when UserStartedSpeakingFrame arrived, which trails the speech by the VAD's
        start_secs. A stretch shorter than that ("Eh") ends before it, so one that started up to
        _SPEECH_LEAD_MS earlier still counts. One that ended before start_ms but started earlier
        than that was other speech."""
        lead_ms = start_ms - _SPEECH_LEAD_MS
        inside = [
            (s, e)
            for s, e in self.stretches
            if after_ms <= s and e <= end_ms and (e > start_ms or s >= lead_ms)
        ]
        if not inside:
            return None
        return min(s for s, _ in inside), max(e for _, e in inside)

    def last_end_between(self, since_ms: int, until_ms: int) -> int | None:
        """When the caller last stopped speaking after since_ms and by until_ms."""
        ends = [e for _, e in self.stretches if since_ms < e <= until_ms]
        return max(ends) if ends else None


class UserRowTimes:
    """Times user rows in call order. Each row spans the caller's real speech in its turn, and
    the next row looks only at speech after this row's end, so a stretch goes to one row."""

    def __init__(self, speech: CallerSpeech | None) -> None:
        self._speech = speech
        self._last_end_ms = 0

    def for_turn(self, start_ms: int, end_ms: int) -> tuple[int, int]:
        """Times for the next user row, whose turn ran from start_ms to end_ms by its frames.
        Without speech timing, or with no stretch for the turn, the frame times stay."""
        bounds = (
            self._speech.bounds(start_ms, end_ms, after_ms=self._last_end_ms)
            if self._speech is not None
            else None
        )
        if bounds is not None:
            start_ms, end_ms = bounds
        self._last_end_ms = end_ms
        return start_ms, end_ms
