# Changelog

All notable changes to `tuner-pipecat-sdk` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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

[Unreleased]: https://github.com/usetuner/tuner-pipecat-sdk-python/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/usetuner/tuner-pipecat-sdk-python/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/usetuner/tuner-pipecat-sdk-python/compare/v0.0.x...v0.1.0
