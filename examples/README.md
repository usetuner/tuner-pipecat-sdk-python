# Examples

Each example shows a different voice bot use case built with [Pipecat Flows](https://github.com/pipecat-ai/pipecat-flows) and observed with `pipecat-flows-tuner`.

Every example is self-contained with its own `pyproject.toml`. Run `uv sync` inside the example directory and follow its README.

---

## Examples

| Example | Use case | Key concepts |
|---------|----------|-------------|
| [`customer_support/`](customer_support/) | Acme Corp support agent | Multi-branch flow, state accumulation, resolve vs. escalate |
| [`appointment_booking/`](appointment_booking/) | Medical clinic receptionist | Linear 7-node flow, enum inputs, confirmation + reschedule |
| [`pizza_order/`](pizza_order/) | Pipecat Pizza ordering | Toppings selection, delivery/pickup branch, running price total |

---

## Prerequisites

All examples share the same requirements:

- Python 3.10+, [`uv`](https://docs.astral.sh/uv/)
- For **installing the SDK with pip**, Python version issues, or local path deps, see the **repository root README** (not repeated here).
- API keys: `CARTESIA_API_KEY`, `DEEPGRAM_API_KEY`, `OPENAI_API_KEY`
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
                    └─► FlowsObserver   ← pipecat-flows-tuner
                        └─► transport.output()
                            └─► context_aggregator.assistant()
```

`FlowsObserver` sits after TTS in the pipeline. It intercepts metrics frames, tracks node transitions via `attach_flow_manager()`, and posts a structured `CallPayload` to the Tuner API when the call ends.
