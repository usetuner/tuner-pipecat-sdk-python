# Integration Guide

## Integration Contract

`Observer` reads runtime events from your pipeline and emits one payload at call end.
Turn timing is captured through `attach_turn_tracking_observer(...)`.

## Recommended Flow

1. Create your `LLMContext`.
2. Create `Observer` with call metadata and model names.
3. Call `observer.attach_context(context)` once before starting the task.
4. Run your pipeline with observer after TTS.

## Captured Signals

- Call start/end timestamps and total duration
- Per-turn latency: user stop to LLM, TTS, bot start, and bot stop
- ASR confidence per turn
- Transcript segments:
  - user turns
  - agent turns
  - tool call segments
  - tool result segments

## Notes on Interruptions and TTS

Interruption behavior in transcript context depends on the TTS provider behavior in `pipecat`:

- Providers with word timestamps typically commit only spoken words.
- Providers without word timestamps often commit full synthesized text before playback.
