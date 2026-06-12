# Examples

Each example shows a different voice bot use case built with [Pipecat](https://github.com/pipecat-ai/pipecat) and observed with `tuner-pipecat-sdk`.

Every example is self-contained with its own `pyproject.toml`. Run `uv sync` inside the example directory and follow its README.

---

## Examples

| Example | Use case | Key concepts |
|---------|----------|-------------|
| [`pizza_order/`](pizza_order/) | Pipecat Pizza ordering | LLM tool-calling, running price total, confirmation branch, cost reporting |
| [`nova_clinic_pipecat/`](nova_clinic_pipecat/) | Medical clinic receptionist | Tool-calling, SIP/telephony transports, cost reporting |

---

## Prerequisites

All examples share the same requirements:

- Python 3.10+, [`uv`](https://docs.astral.sh/uv/)
- For **installing the SDK with pip**, Python version issues, or local path deps, see the **repository root README** (not repeated here).
- API keys: `DEEPGRAM_API_KEY`, `OPENAI_API_KEY` (see each example's `.env.example`)
- Optional: Tuner API credentials (`TUNER_API_KEY`, `TUNER_WORKSPACE_ID`, `TUNER_AGENT_ID`, `TUNER_BASE_URL`)

---

## Quick start

```bash
cd examples/<example_name>
uv sync
cp .env.example .env   # fill in your API keys
uv run <example_name>.py
```

Then open http://localhost:7860 and click **Connect**.

---

## How the SDK fits in

```
transport.input()
    └─► STT
        └─► context_aggregator.user()
            └─► LLM
                └─► TTS
                    └─► Observer   ← tuner-pipecat-sdk
                        └─► transport.output()
                            └─► context_aggregator.assistant()
```

`Observer` sits after TTS in the pipeline. It intercepts metrics frames, reads the transcript via `attach_context()`, and posts a structured `CallPayload` to the Tuner API when the call ends.
