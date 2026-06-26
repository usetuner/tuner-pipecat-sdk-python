# Integration Guide

## Integration Contract

`Observer` reads runtime events from your pipeline and emits one payload at call end.
Turn timing is captured through `attach_turn_tracking_observer(...)`.

## Recommended Flow

1. Create `Observer` with call metadata and model names.
2. Register the observer on the task: `PipelineTask(pipeline, observers=[observer, observer.latency_observer, turn_tracker])` (it is a pipeline-level observer, not a processor in `Pipeline([...])`).

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
