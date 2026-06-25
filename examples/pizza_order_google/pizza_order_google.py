#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""Pizzeria ordering bot built with Pipecat and the Tuner Observer.

This version uses an all-Google stack:
- STT:  GoogleSTTService   (Google Cloud Speech-to-Text V2)
- LLM:  GoogleLLMService   (Gemini 2.5 Flash)
- TTS:  GoogleTTSService   (Google Cloud TTS, Chirp 3 HD voices)

Requirements:
- GOOGLE_APPLICATION_CREDENTIALS (path to a GCP service-account JSON file
  with Speech-to-Text and Text-to-Speech API access), OR set
  GOOGLE_CREDENTIALS to the raw JSON string.
- GOOGLE_API_KEY (for Gemini / GoogleLLMService — this is a separate
  credential from the GCP service account above; get it from Google AI
  Studio or a Vertex-enabled API key)

Run the example:
uv run pizza_order_google.py
"""

import os
import uuid

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndTaskFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams

from tuner_pipecat_sdk import CallUsage, Observer

load_dotenv(override=True)

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# ---------------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------------

MENU = {
    "margherita": 10.99,
    "pepperoni": 12.99,
    "veggie": 11.99,
    "bbq chicken": 13.99,
}

SIZE_SURCHARGE = {"small": 0.0, "medium": 2.0, "large": 4.0}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

AGENT_INSTRUCTIONS = f"""
You are a friendly cashier at 'Pipecat Pizza'.
Keep responses short and conversational. This is a voice call — never use
emojis, formatting, bullet points, or special characters in your responses.

## Today's menu
{chr(10).join(f"- {name} (${price:.2f})" for name, price in MENU.items())}

## Sizes
- small (no extra charge)
- medium (+$2.00)
- large (+$4.00)

## How to take an order
1. Greet the customer warmly and present today's menu, then ask which pizza
   they would like.
2. Once they pick a pizza, call the choose_pizza function with their choice.
3. Ask what size they want, then call the choose_size function.
4. Read back the full order — pizza, size, and total price — and ask the
   customer to confirm. Call the confirm_order function with their answer.
5. If confirmed, thank them enthusiastically and tell them their order is being
   prepared. If not confirmed, apologise politely and tell them they can call
   back anytime.
6. After your final closing line, call the end_call function to end the call.

## Important rules
- Only accept pizzas from the menu above.
- Never confirm an order without calling confirm_order first.
- Always read back the pizza, size, and total before asking to confirm.
"""


# ---------------------------------------------------------------------------
# Debug logging processor
# ---------------------------------------------------------------------------


class DebugLogProcessor(FrameProcessor):
    """Logs transcriptions and LLM response text as they flow through the pipeline."""

    def __init__(self):
        super().__init__()
        self._bot_response_buf = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            logger.info(f"[TRANSCRIPT] User said: '{frame.text}'")

        elif isinstance(frame, LLMFullResponseStartFrame):
            self._bot_response_buf = []

        elif isinstance(frame, TextFrame) and self._bot_response_buf is not None:
            self._bot_response_buf.append(frame.text)

        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._bot_response_buf:
                full = "".join(self._bot_response_buf)
                logger.info(f"[BOT RESPONSE] {full}")
            self._bot_response_buf = []

        await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Order state + tools
# ---------------------------------------------------------------------------

# Single-call demo: keep the in-progress order in module state.
_order: dict = {}


async def choose_pizza(params):
    """Record which pizza the customer wants to order."""
    pizza = params.arguments.get("pizza", "").lower()
    price = MENU.get(pizza, 0.0)
    logger.info(f"[ORDER] Pizza chosen: {pizza} (${price:.2f})")
    _order["pizza"] = pizza
    _order["price"] = price
    await params.result_callback({"pizza": pizza, "price": price})


async def choose_size(params):
    """Record the pizza size the customer wants."""
    size = params.arguments.get("size", "").lower()
    surcharge = SIZE_SURCHARGE.get(size, 0.0)
    total = _order.get("price", 0.0) + surcharge
    logger.info(f"[ORDER] Size chosen: {size} | total=${total:.2f}")
    _order["size"] = size
    _order["total"] = total
    await params.result_callback({"size": size, "total": total})


async def confirm_order(params):
    """Confirm or cancel the order."""
    confirmed = bool(params.arguments.get("confirmed"))
    logger.info(f"[ORDER] Confirmed: {confirmed} | order={_order}")
    await params.result_callback({"confirmed": confirmed})


async def end_call(params):
    logger.info("[END CALL] Reason: agent_hangup")
    await params.result_callback({"status": "ending", "reason": "agent_hangup"})
    await params.llm.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)


def build_tools() -> ToolsSchema:
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="choose_pizza",
                description="Record which pizza the customer wants to order.",
                properties={"pizza": {"type": "string", "enum": list(MENU.keys())}},
                required=["pizza"],
            ),
            FunctionSchema(
                name="choose_size",
                description="Record the pizza size the customer wants.",
                properties={"size": {"type": "string", "enum": list(SIZE_SURCHARGE.keys())}},
                required=["size"],
            ),
            FunctionSchema(
                name="confirm_order",
                description="Confirm or cancel the order.",
                properties={"confirmed": {"type": "boolean"}},
                required=["confirmed"],
            ),
            FunctionSchema(
                name="end_call",
                description=(
                    "End the call once you have delivered your final spoken response "
                    "to the customer."
                ),
                properties={},
                required=[],
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Bot entrypoint
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    # --- STT: Google Cloud Speech-to-Text V2 -----------------------------
    # Needs GCP service-account credentials (NOT a plain API key).
    # Either pass a path to the JSON key file, or the raw JSON string.
    stt = GoogleSTTService(
        credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        settings=GoogleSTTService.Settings(
            languages=[Language.EN_US],
            model="latest_long",
            enable_automatic_punctuation=True,
        ),
    )

    # --- TTS: Google Cloud TTS streaming (Chirp 3 HD / Journey voices) ---
    tts = GoogleTTSService(
        credentials_path=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        settings=GoogleTTSService.Settings(
            voice=os.getenv("GOOGLE_TTS_VOICE", "en-US-Chirp3-HD-Charon"),
            language=Language.EN_US,
        ),
    )

    # --- LLM: Gemini 2.5 Flash (text in/out, via google-genai) -----------
    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model=os.getenv("GOOGLE_LLM_MODEL", "gemini-2.5-flash"),
        system_instruction=AGENT_INSTRUCTIONS,
    )

    llm.register_function("choose_pizza", choose_pizza)
    llm.register_function("choose_size", choose_size)
    llm.register_function("confirm_order", confirm_order)
    llm.register_function("end_call", end_call)

    context = LLMContext(tools=build_tools())
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    debug_logger = DebugLogProcessor()
    turn_tracker = TurnTrackingObserver()

    def calculate_cost(usage: CallUsage) -> float:
        # Gemini 2.5 Flash pricing (per token) — update if your tier differs
        llm_cost  = (usage.llm_prompt_tokens     or 0) * 0.000_000_075
        llm_cost += (usage.llm_completion_tokens or 0) * 0.000_000_300
        # Google Cloud TTS (Chirp 3 HD): ~$0.030 per 1K characters
        tts_cost  = (usage.tts_characters        or 0) * 0.000_030
        # Google Cloud STT V2 (standard streaming): ~$0.0240 per audio minute
        stt_cost  = usage.stt_audio_seconds            * (0.0240 / 60)
        return (llm_cost + tts_cost + stt_cost) * 100

    observer = Observer(
        api_key=os.getenv("TUNER_API_KEY", "dev"),
        workspace_id=int(os.getenv("TUNER_WORKSPACE_ID")),
        agent_id=os.getenv("TUNER_AGENT_ID", "pizzeria-bot"),
        call_id=str(uuid.uuid4()),
        base_url=os.getenv("TUNER_BASE_URL", "https://voice-api.staging.ginni.ai"),
        asr_model=os.getenv("TUNER_ASR_MODEL", "google/stt-v2"),
        llm_model=os.getenv("TUNER_LLM_MODEL", "gemini-2.5-flash"),
        tts_model=os.getenv("TUNER_TTS_MODEL", "google/chirp3-hd"),
        cost_calculator=calculate_cost,
        debug=True,
    )
    observer.attach_turn_tracking_observer(turn_tracker)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            debug_logger,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[observer, observer.latency_observer, turn_tracker],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("[BOT] Client connected — starting pizzeria flow")
        _order.clear()
        context.add_message(
            {
                "role": "user",
                "content": (
                    "Greet the customer warmly, present the menu, "
                    "and ask which pizza they would like."
                ),
            }
        )
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("[BOT] Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()