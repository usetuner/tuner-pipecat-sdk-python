#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License

"""Pizzeria ordering bot built with Pipecat Flows.

Requirements:
- DEEPGRAM_API_KEY
- OPENAI_API_KEY

Run the example:
uv run pizza_order.py
"""

import json
import os
import uuid

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
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
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowsFunctionSchema,
    NodeConfig,
)

from tuner_pipecat_sdk import FlowsObserver

load_dotenv(override=True)

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


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
# Helpers
# ---------------------------------------------------------------------------

MENU = {
    "margherita": 10.99,
    "pepperoni": 12.99,
    "veggie": 11.99,
    "bbq chicken": 13.99,
}


def log_node(node: dict, label: str) -> None:
    logger.debug(f"[NODE CREATED] {label}:\n{json.dumps(node, indent=2, default=str)}")


# ---------------------------------------------------------------------------
# Flow nodes
# ---------------------------------------------------------------------------


def create_greeting_node() -> NodeConfig:
    choose_pizza_func = FlowsFunctionSchema(
        name="choose_pizza",
        description="Record which pizza the customer wants to order.",
        required=["pizza"],
        properties={"pizza": {"type": "string", "enum": list(MENU.keys())}},
        handler=handle_choose_pizza,
    )

    node = NodeConfig(
        name="greeting",
        role_messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly pizzeria cashier at 'Pipecat Pizza'. "
                    "Keep responses short and conversational. "
                    "Your responses will be converted to audio — no emojis or special characters."
                ),
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Greet the customer warmly and present today's menu: "
                    f"{', '.join(f'{k} (${v:.2f})' for k, v in MENU.items())}. "
                    "Ask them which pizza they'd like."
                ),
            }
        ],
        functions=[choose_pizza_func],
    )
    log_node(dict(node), "greeting")
    return node


async def handle_choose_pizza(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    pizza = args["pizza"].lower()
    price = MENU.get(pizza, 0.0)
    logger.info(f"[ORDER] Pizza chosen: {pizza} (${price:.2f})")
    flow_manager.state["pizza"] = pizza
    flow_manager.state["price"] = price
    return {"pizza": pizza, "price": price}, create_size_node()


def create_size_node() -> NodeConfig:
    choose_size_func = FlowsFunctionSchema(
        name="choose_size",
        description="Record the pizza size the customer wants.",
        required=["size"],
        properties={"size": {"type": "string", "enum": ["small", "medium", "large"]}},
        handler=handle_choose_size,
    )

    node = NodeConfig(
        name="size",
        task_messages=[
            {
                "role": "system",
                "content": "Ask the customer what size they want: small ($0 extra), medium (+$2), or large (+$4).",
            }
        ],
        functions=[choose_size_func],
    )
    log_node(dict(node), "size")
    return node


async def handle_choose_size(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    size = args["size"].lower()
    surcharge = {"small": 0.0, "medium": 2.0, "large": 4.0}.get(size, 0.0)
    total = flow_manager.state["price"] + surcharge
    logger.info(f"[ORDER] Size chosen: {size} | total=${total:.2f}")
    flow_manager.state["size"] = size
    flow_manager.state["total"] = total
    return {"size": size, "total": total}, create_confirm_node()


def create_confirm_node() -> NodeConfig:
    confirm_func = FlowsFunctionSchema(
        name="confirm_order",
        description="Confirm or cancel the order.",
        required=["confirmed"],
        properties={"confirmed": {"type": "boolean"}},
        handler=handle_confirm,
    )

    node = NodeConfig(
        name="confirm",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Read back the order summary to the customer and ask them to confirm. "
                    "Include the pizza name, size, and total price from what was just ordered."
                ),
            }
        ],
        functions=[confirm_func],
    )
    log_node(dict(node), "confirm")
    return node


async def handle_confirm(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    confirmed = args["confirmed"]
    logger.info(f"[ORDER] Confirmed: {confirmed} | state={flow_manager.state}")
    return {"confirmed": confirmed}, create_farewell_node(confirmed)


def create_farewell_node(confirmed: bool = True) -> NodeConfig:
    if confirmed:
        content = (
            "Thank the customer enthusiastically, tell them their order is being prepared, "
            "and wish them a great meal. Then end the conversation."
        )
    else:
        content = (
            "Apologise politely, tell them they can call back anytime, and end the conversation."
        )

    node = NodeConfig(
        name="farewell",
        task_messages=[{"role": "system", "content": content}],
        post_actions=[{"type": "end_conversation"}],
    )
    log_node(dict(node), "farewell")
    return node


# ---------------------------------------------------------------------------
# Bot entrypoint
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice=os.getenv("DEEPGRAM_VOICE", "aura-2-thalia-en"),
    )
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    debug_logger = DebugLogProcessor()
    turn_tracker = TurnTrackingObserver()

    observer = FlowsObserver(
        api_key=os.getenv("TUNER_API_KEY", "dev"),
        workspace_id=int(os.getenv("TUNER_WORKSPACE_ID")),
        agent_id=os.getenv("TUNER_AGENT_ID", "pizzeria-bot"),
        call_id=str(uuid.uuid4()),
        base_url=os.getenv("TUNER_BASE_URL", "http://localhost:8000"),
        asr_model=os.getenv("TUNER_ASR_MODEL", "deepgram/nova-3"),
        llm_model=os.getenv("TUNER_LLM_MODEL", "gpt-4o-mini"),
        tts_model=os.getenv("TUNER_TTS_MODEL", "deepgram/aura-2-thalia-en"),
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
            observer,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            observers=[observer.latency_observer, turn_tracker],
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    flow_manager = FlowManager(
        task=task,
        llm=llm,
        context_aggregator=context_aggregator,
        transport=transport,
    )
    observer.attach_flow_manager(flow_manager)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("[BOT] Client connected — starting pizzeria flow")
        await flow_manager.initialize(create_greeting_node())

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
