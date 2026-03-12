# Integration Guide

## Integration Contract

`FlowsObserver` currently depends on `FlowManager` for transcript retrieval and transition capture.
The SDK does not expose `attach_context_aggregator` at this time.

## Recommended Flow

1. Create your `FlowManager`.
2. Create `FlowsObserver` with call metadata and model names.
3. Call `observer.attach_flow_manager(flow_manager)` once before starting the task.
4. Run your pipeline with observer after TTS.

## Captured Signals

- Call start/end timestamps and total duration
- Per-turn latency: user stop to LLM, TTS, bot start, and bot stop
- ASR confidence per turn
- Flow node transitions:
  - from/to node
  - triggering function and arguments
  - state snapshot at transition time
  - functions available in target node
- Transcript segments:
  - user turns
  - agent turns
  - tool call segments
  - tool result segments
  - node transition segments

## Notes on Interruptions and TTS

Interruption behavior in transcript context depends on the TTS provider behavior in `pipecat`:

- Providers with word timestamps typically commit only spoken words.
- Providers without word timestamps often commit full synthesized text before playback.

For strongly structured flows, `pipecat-flows` context reset strategies can help keep context clean between nodes.
