# Changelog

All notable changes to `tuner-pipecat-sdk` will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0]

### Changed
- **Breaking:** minimum `pipecat-ai` version bumped to `>=1.0.0`
- Updated `Observer.attach_context` to use `LLMContext` (replaces `OpenAILLMContext` removed in pipecat 1.0.0)

---

## [0.1.1]

### Added
- Support for both plain `pipecat` pipelines (`Observer`) and `pipecat-flows` pipelines (`FlowsObserver`)
- `TurnTrackingObserver` integration for accurate turn-level metrics
- `attach_turn_tracking_observer` on both observer types

### Fixed
- Turn calculation for plain pipecat pipelines

---

## [0.1.0]

### Added
- Initial release
- `FlowsObserver` for `pipecat-flows` pipelines — captures flow transitions, latency, transcript, and usage metadata
- Structured `CallPayload` sent to the Tuner API on call end
- Pydantic models for all data structures
- Lazy import of `FlowsObserver` to avoid hard dependency on `pipecat-flows` when unused
