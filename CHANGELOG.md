# Changelog

All notable changes to `tuner-pipecat-sdk` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- **`pizza_order_google` example** — full Google stack variant of the pizzeria bot: `GoogleSTTService` (Cloud Speech-to-Text V2), `GoogleLLMService` (Gemini 2.5 Flash), and `GoogleTTSService` (Cloud TTS, Chirp 3 HD).

### Changed
- **`Observer` is now a pipeline-level observer (BREAKING).** Pass it to `PipelineTask(observers=[...])` instead of inserting it into the `Pipeline([...])` processor list. It now sees every frame at every processor boundary, so it stays out of the audio path and captures frames an intermediate processor consumes — notably `TranscriptionFrame`, which the user aggregator swallows before the old end-of-pipeline position could see it.
  - Migration: remove `observer` from `Pipeline([...])` and add it to `observers=[observer, observer.latency_observer, turn_tracker]`. `attach_turn_tracking_observer()` is unchanged.

### Deprecated
- **`attach_context()`** is now a no-op and will be removed in a future release. The transcript is built live from the frame stream and no longer reads the LLM context object. Existing calls are safe to remove.

### Fixed
- **Interrupted turns no longer emit a stale assistant message** — the observer now correctly discards partial LLM output when a turn is interrupted before the bot finishes speaking.
- **Developer-injected `{"role": "user"}` messages (e.g. a `"Greet the customer…"` kickoff) no longer appear as user turns or shift the transcript out of sync.** Real user turns are now matched against captured STT transcriptions; when no transcriptions are available the observer falls back to dropping user messages that precede a proactive greeting.
- **LLM latency (`llm_node_ttft`) is now populated for providers that emit a TTFB metric but no processing-time metric (e.g. Google/Gemini).** It falls back to the LLM service's TTFB when processing time is absent.
- **Per-node TTFB attribution corrected.** `tts_node_ttfb` now reports the TTS service's own TTFB instead of the first TTFB in the turn (which was the STT's on real turns).

### Changed
- **`stt_node_ttfb` now reports the STT service's TTFB** (pure model latency) instead of the VAD-stop→turn-finalized gap. Values are lower and no longer include smart-turn analysis time.

---

## [0.2.2] 2026-06-11

### Added
- **Call cost reporting** — pass a `cost_calculator` callable to `Observer` to compute the cost of a call from its usage data (`CallUsage`: LLM prompt/completion tokens, TTS characters, STT audio seconds). The result (in USD cents) is included in the payload sent to the Tuner API.
- `tuner_pipecat_sdk.CallUsage` — usage data type passed to `cost_calculator`.
- `pizza_order` example updated to demonstrate `cost_calculator` with Deepgram + OpenAI pricing.

### Removed
- **Dropped `pipecat-ai-flows` support** — removed the `FlowsObserver` class, the `flows` optional-dependency extra, and the `pipecat-ai-flows` dependency. Use `Observer` with `attach_context()` for all pipelines.
- The `pizza_order` example was migrated from pipecat-flows to the plain `Observer` + LLM tool-calling API.

---

## [0.2.0] 2026-05-15

### Changed
- Migrate all examples to pipecat-ai v1 (`>=1.0.0`) and pipecat-ai-flows v1 (`>=1.0.0`).
- Dropped support of Python 3.10

---

## [0.1.1] – 2026-05-14

### Added
- **SIP provider support** — `attach_sip_from_telephony()` accepts a built-in provider name (`"twilio"`, `"telnyx"`, `"jambonz"`, `"plivo"`, `"exotel"`) or a custom extractor callable to populate SIP Call-ID and headers on the Tuner record.
- `attach_sip_from_context()` for typed provider contexts (e.g. `JambonzCallContext`).
- `attach_sip_from_dialin()` for Pipecat's `DialinSettings` (Daily PSTN).
- Non-2xx Tuner API responses now log the full request body and response body for easier debugging without enabling `debug=True`.

### Changed
- `nova_clinic_pipecat` example extended with Twilio, Telnyx, and Jambonz telephony server scripts.

---

## [0.1.0] – 2026-04-13

### Added
- **`Observer`** — pipeline `FrameProcessor` for plain pipecat pipelines. Attach via `attach_context()`.
- **`FlowsObserver`** — pipeline `FrameProcessor` for pipecat-flows pipelines. Attach via `attach_flow_manager()`.
- `attach_turn_tracking_observer()` wires pipecat's `TurnTrackingObserver` into the Tuner accumulator so turn start/end timestamps and interruption flags are captured.
- `UserBotLatencyObserver` latency events (TTFB, TTFT, per-turn breakdown) forwarded to the Tuner payload.
- Captures: transcript, per-turn latency breakdown, function-call timings, ASR/LLM/TTS usage metrics, disconnection reason.
- `agent_version` parameter — resolved from the explicit argument, `APP_VERSION` env var, or common CI env vars (`GITHUB_RUN_NUMBER`, etc.).
- `nova_clinic_pipecat` example demonstrating a full WebRTC voice assistant with tool use.
- `pizza_order`, `customer_support`, `appointment_booking` examples demonstrating pipecat-flows integration.

### Fixed
- Turn-order calculation corrected for the first turn (#3).
- Latency breakdown no longer corrupted when turns overlap or arrive out of order.

---

## [0.0.x] – 2026-03-19 and earlier

Initial development: accumulator, HTTP client, frame handling, turn tracking, and first round of examples.

[Unreleased]: https://github.com/usetuner/tuner-pipecat-sdk-python/compare/v0.2.0...HEAD
[0.1.1]: https://github.com/usetuner/tuner-pipecat-sdk-python/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/usetuner/tuner-pipecat-sdk-python/compare/v0.0.x...v0.1.0
