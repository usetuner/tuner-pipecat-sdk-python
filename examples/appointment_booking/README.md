# Appointment Booking Bot

A medical clinic appointment booking bot built with Pipecat Flows and the `pipecat-flows-tuner` SDK.

**What it demonstrates:**
- Linear 7-node data-collection flow
- Enum-constrained inputs (service type, day, time slot)
- State accumulation across nodes (`patient_name`, `service`, `day`, `time_slot`, `phone`)
- Confirmation node with reschedule branch
- Full call observability via `FlowsObserver`

## Flow diagram

```
greeting
   └─► service_type
           └─► preferred_day
                   └─► preferred_time
                               └─► contact_info
                                       └─► confirm
                                               ├─► farewell (confirmed)
                                               └─► farewell (reschedule)
```

## Prerequisites

- Python 3.10+
- `uv` package manager

## Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Create a `.env` file:

   ```env
   CARTESIA_API_KEY=your_cartesia_key
   DEEPGRAM_API_KEY=your_deepgram_key
   OPENAI_API_KEY=your_openai_key

   # Optional — Tuner observability (defaults to local dev server)
   TUNER_API_KEY=dev
   TUNER_WORKSPACE_ID=1
   TUNER_AGENT_ID=appointment-booking-bot
   TUNER_BASE_URL=http://localhost:8000
   ```

## Run

```bash
uv run appointment_booking.py
```

Open http://localhost:7860 in your browser and click **Connect**.

## Services used

| Role | Service |
|------|---------|
| STT  | Deepgram Nova-3 |
| LLM  | OpenAI GPT-4o-mini |
| TTS  | Cartesia Sonic |
| Transport | SmallWebRTC (default) |
