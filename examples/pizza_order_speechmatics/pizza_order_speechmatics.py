#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""Pizzeria ordering bot built with Pipecat and the Tuner Observer.

This version uses Speechmatics for both STT and TTS:
- STT: SpeechmaticsSTTService (Agent STT preview / linden-1, or standard RT —
       switch with SPEECHMATICS_STT_MODE)
- TTS: SpeechmaticsTTSService (preview HTTP API, 16 kHz)
- LLM: OpenAILLMService       (gpt-4o-mini)

Requirements:
- SPEECHMATICS_API_KEY
- OPENAI_API_KEY

Run the example:
uv run pizza_order_speechmatics.py
"""

import os
import uuid

import aiohttp
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
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.speechmatics.stt import SpeechmaticsSTTService
from pipecat.services.speechmatics.tts import SpeechmaticsTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams

from tuner_pipecat_sdk import CallUsage, Observer

load_dotenv(override=True)


def env(name: str, default: str | None = None) -> str | None:
    """Read an env var, treating a blank value (bare ``KEY=`` in .env) as unset."""
    return os.getenv(name) or default


def env_int(name: str, default: int | None = None) -> int | None:
    """Read an env var as an int, treating a blank value as unset."""
    raw = env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from e


transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


# ---------------------------------------------------------------------------
# STT modes
# ---------------------------------------------------------------------------

# Selected with SPEECHMATICS_STT_MODE. `model` and `operating_point` are merged
# into the StartRecognition transcription config; None means "do not send".
STT_MODES = {
    "agent": {
        "url": "wss://preview.rt.speechmatics.com/v2/agent",
        "model": "linden-1",
        "operating_point": None,
    },
    "rt": {
        "url": "wss://eu2.rt.speechmatics.com/v2",
        "model": None,
        "operating_point": "enhanced",
    },
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
    # --- STT: Speechmatics ------------------------------------------------
    # SPEECHMATICS_STT_MODE picks the engine:
    #
    #   "agent" (default) — Agent STT preview. Needs BOTH the `/v2/agent`
    #       endpoint path AND `model: "linden-1"` in the StartRecognition
    #       transcription config; miss either one and you silently get the
    #       standard RT API instead.
    #   "rt"              — the standard real-time API: regional endpoint,
    #       no model name, `enhanced` operating point.
    #
    # Pipecat's Settings has no `model` field (it reports the operating point
    # there), so the model is injected through `extra_params` ->
    # `advanced_engine_control`, which the Speechmatics SDK merges straight
    # into the transcription config it sends in StartRecognition. For Agent
    # STT `operating_point` is cleared, since linden-1 selects the engine
    # itself.
    #
    # SPEECHMATICS_RT_URL / SPEECHMATICS_STT_MODEL override the mode's
    # defaults if you need a different region or model build.
    stt_mode = (env("SPEECHMATICS_STT_MODE", "agent") or "").lower()
    if stt_mode not in STT_MODES:
        raise ValueError(
            f"SPEECHMATICS_STT_MODE must be one of {sorted(STT_MODES)}, got {stt_mode!r}"
        )
    stt_defaults = STT_MODES[stt_mode]

    stt_url = env("SPEECHMATICS_RT_URL", stt_defaults["url"])
    stt_model = env("SPEECHMATICS_STT_MODEL", stt_defaults["model"])
    logger.info(f"[STT] Speechmatics mode={stt_mode} model={stt_model or '-'} url={stt_url}")

    # TurnDetectionMode.EXTERNAL leaves endpointing to Pipecat's Silero VAD,
    # which is what the rest of this pipeline expects.
    stt = SpeechmaticsSTTService(
        api_key=os.getenv("SPEECHMATICS_API_KEY"),
        base_url=stt_url,
        settings=SpeechmaticsSTTService.Settings(
            language=Language.EN,
            turn_detection_mode=SpeechmaticsSTTService.TurnDetectionMode.EXTERNAL,
            max_delay=1.0,
            enable_diarization=False,
            extra_params={
                "advanced_engine_control": {
                    "model": stt_model,
                    "operating_point": stt_defaults["operating_point"],
                },
            },
        ),
    )

    # --- TTS: Speechmatics (preview) -------------------------------------
    # HTTP-based (one request per utterance, so TTFB is higher than a
    # streaming socket) and 16 kHz only — the output transport resamples.
    # Needs a shared aiohttp session, closed at the end of run_bot.
    session = aiohttp.ClientSession()
    tts = SpeechmaticsTTSService(
        api_key=os.getenv("SPEECHMATICS_API_KEY"),
        base_url=env("SPEECHMATICS_TTS_URL", "https://preview.tts.speechmatics.com"),
        aiohttp_session=session,
        settings=SpeechmaticsTTSService.Settings(
            voice=env("SPEECHMATICS_VOICE", "sarah"),
        ),
    )
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        settings=OpenAILLMService.Settings(system_instruction=AGENT_INSTRUCTIONS),
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
        # OpenAI gpt-4o-mini pricing (per token)
        llm_cost = (usage.llm_prompt_tokens or 0) * 0.000_000_150
        llm_cost += (usage.llm_completion_tokens or 0) * 0.000_000_600
        # Speechmatics TTS is in preview with no published per-character
        # rate — placeholder at $0.030 per 1K characters. Update once priced.
        tts_cost = (usage.tts_characters or 0) * 0.000_030
        # Agent STT (linden-1) is in preview with no published rate —
        # placeholder at the RT enhanced price of ~$0.0173 per audio minute.
        stt_cost = usage.stt_audio_seconds * (0.0173 / 60)
        return (llm_cost + tts_cost + stt_cost) * 100

    observer = Observer(
        api_key=env("TUNER_API_KEY", "dev"),
        workspace_id=int(os.getenv("TUNER_WORKSPACE_ID")),
        agent_id=env("TUNER_AGENT_ID", "pizza-order-bot-speechmatics"),
        call_id=str(uuid.uuid4()),
        base_url=env("TUNER_BASE_URL", "https://api.usetuner.ai"),
        asr_model=env("TUNER_ASR_MODEL", f"speechmatics/{stt_model or stt_mode}"),
        llm_model=env("TUNER_LLM_MODEL", "gpt-4o-mini"),
        tts_model=env("TUNER_TTS_MODEL", "speechmatics/tts-preview"),
        # Deployment version number (int). Falls back to the SDK's own
        # APP_VERSION / CI build-number lookup when left unset.
        agent_version=env_int("TUNER_AGENT_VERSION"),
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
                "role": "assistant",
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
    try:
        await runner.run(task)
    finally:
        await session.close()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
