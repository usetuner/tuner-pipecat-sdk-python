#
# Copyright (c) 2024-2026, Tuner Team
#
# SPDX-License-Identifier: BSD 2-Clause License

"""Appointment booking bot built with Pipecat Flows + pipecat-flows-tuner.

This example demonstrates:
- Linear multi-step data-collection flow
- Numeric enum validation (morning/afternoon slots)
- State accumulation across nodes (service, date, time, name, phone)
- Confirmation node with reschedule branch
- Full call observability via FlowsObserver

Requirements:
- CARTESIA_API_KEY
- DEEPGRAM_API_KEY
- OPENAI_API_KEY

Run the example:
    uv run appointment_booking.py
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
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams

from pipecat_flows import (
    FlowArgs,
    FlowManager,
    FlowsFunctionSchema,
    NodeConfig,
)
from pipecat_flows_tuner import FlowsObserver

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

SERVICES = ["general_checkup", "specialist_consultation", "urgent_care", "follow_up"]
AVAILABLE_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
TIME_SLOTS = ["morning", "afternoon"]


def log_node(node: dict, label: str) -> None:
    logger.debug(f"[NODE CREATED] {label}:\n{json.dumps(node, indent=2, default=str)}")


# ---------------------------------------------------------------------------
# Flow nodes
# ---------------------------------------------------------------------------


def create_greeting_node() -> NodeConfig:
    provide_name_func = FlowsFunctionSchema(
        name="provide_name",
        description="Record the patient's name to begin the booking.",
        required=["name"],
        properties={"name": {"type": "string"}},
        handler=handle_provide_name,
    )

    node = NodeConfig(
        name="greeting",
        role_messages=[
            {
                "role": "system",
                "content": (
                    "You are a friendly receptionist at Greenfield Medical Clinic. "
                    "Keep responses brief and welcoming. "
                    "Your responses will be converted to audio — no emojis or special characters."
                ),
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Welcome the patient to Greenfield Medical Clinic and ask for their full name."
                ),
            }
        ],
        functions=[provide_name_func],
    )
    log_node(dict(node), "greeting")
    return node


async def handle_provide_name(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    name = args["name"]
    flow_manager.state["patient_name"] = name
    logger.info(f"[BOOKING] Patient name: {name}")
    return {"name": name}, create_service_type_node()


def create_service_type_node() -> NodeConfig:
    select_service_func = FlowsFunctionSchema(
        name="select_service",
        description="Record the type of medical service the patient needs.",
        required=["service"],
        properties={
            "service": {
                "type": "string",
                "enum": SERVICES,
            },
        },
        handler=handle_select_service,
    )

    node = NodeConfig(
        name="service_type",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Ask the patient what type of appointment they need. "
                    "The options are: general checkup, specialist consultation, urgent care, or follow-up visit."
                ),
            }
        ],
        functions=[select_service_func],
    )
    log_node(dict(node), "service_type")
    return node


async def handle_select_service(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    service = args["service"]
    flow_manager.state["service"] = service
    logger.info(f"[BOOKING] Service: {service}")
    return {"service": service}, create_preferred_day_node()


def create_preferred_day_node() -> NodeConfig:
    choose_day_func = FlowsFunctionSchema(
        name="choose_day",
        description="Record the patient's preferred day of the week for their appointment.",
        required=["day"],
        properties={
            "day": {
                "type": "string",
                "enum": AVAILABLE_DAYS,
            },
        },
        handler=handle_choose_day,
    )

    node = NodeConfig(
        name="preferred_day",
        task_messages=[
            {
                "role": "system",
                "content": (
                    f"Ask the patient which day of the week works best for them. "
                    f"We have availability {', '.join(AVAILABLE_DAYS)}."
                ),
            }
        ],
        functions=[choose_day_func],
    )
    log_node(dict(node), "preferred_day")
    return node


async def handle_choose_day(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    day = args["day"]
    flow_manager.state["day"] = day
    logger.info(f"[BOOKING] Day: {day}")
    return {"day": day}, create_preferred_time_node()


def create_preferred_time_node() -> NodeConfig:
    choose_time_func = FlowsFunctionSchema(
        name="choose_time",
        description="Record the patient's preferred time slot — morning or afternoon.",
        required=["time_slot"],
        properties={
            "time_slot": {
                "type": "string",
                "enum": TIME_SLOTS,
            },
        },
        handler=handle_choose_time,
    )

    node = NodeConfig(
        name="preferred_time",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Ask the patient whether they prefer a morning slot (9am-12pm) "
                    "or an afternoon slot (1pm-5pm)."
                ),
            }
        ],
        functions=[choose_time_func],
    )
    log_node(dict(node), "preferred_time")
    return node


async def handle_choose_time(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    time_slot = args["time_slot"]
    flow_manager.state["time_slot"] = time_slot
    logger.info(f"[BOOKING] Time slot: {time_slot}")
    return {"time_slot": time_slot}, create_contact_info_node()


def create_contact_info_node() -> NodeConfig:
    provide_phone_func = FlowsFunctionSchema(
        name="provide_phone",
        description="Record the patient's contact phone number for the booking confirmation.",
        required=["phone"],
        properties={"phone": {"type": "string"}},
        handler=handle_provide_phone,
    )

    node = NodeConfig(
        name="contact_info",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Ask the patient for the best phone number to send their appointment confirmation to."
                ),
            }
        ],
        functions=[provide_phone_func],
    )
    log_node(dict(node), "contact_info")
    return node


async def handle_provide_phone(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    phone = args["phone"]
    flow_manager.state["phone"] = phone
    logger.info(f"[BOOKING] Phone: {phone}")
    return {"phone": phone}, create_confirm_node()


def create_confirm_node() -> NodeConfig:
    confirm_booking_func = FlowsFunctionSchema(
        name="confirm_booking",
        description="Confirm or reschedule the appointment.",
        required=["confirmed"],
        properties={"confirmed": {"type": "boolean"}},
        handler=handle_confirm_booking,
    )

    node = NodeConfig(
        name="confirm",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Read back the full appointment details to the patient: "
                    "name, service type, day, time slot, and phone number. "
                    "Ask them to confirm or if they would like to make any changes."
                ),
            }
        ],
        functions=[confirm_booking_func],
    )
    log_node(dict(node), "confirm")
    return node


async def handle_confirm_booking(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    confirmed = args["confirmed"]
    flow_manager.state["confirmed"] = confirmed
    logger.info(f"[BOOKING] Confirmed: {confirmed} | state={flow_manager.state}")
    return {"confirmed": confirmed}, create_farewell_node(confirmed)


def create_farewell_node(confirmed: bool) -> NodeConfig:
    if confirmed:
        content = (
            "Thank the patient by name for booking with Greenfield Medical Clinic. "
            "Let them know a confirmation will be sent to their phone number. "
            "Remind them to arrive 10 minutes early and bring their insurance card. "
            "Wish them well and end the conversation."
        )
    else:
        content = (
            "Apologise for any inconvenience. "
            "Let the patient know they can call back at any time to reschedule. "
            "Thank them for their time and end the conversation."
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

    observer = FlowsObserver(
        api_key=os.getenv("TUNER_API_KEY", "dev"),
        workspace_id=int(os.getenv("TUNER_WORKSPACE_ID")),
        agent_id=os.getenv("TUNER_AGENT_ID", "appointment-booking-bot"),
        call_id=str(uuid.uuid4()),
        base_url=os.getenv("TUNER_BASE_URL", "http://localhost:8000"),
        asr_model=os.getenv("TUNER_ASR_MODEL", "deepgram/nova-3"),
        llm_model=os.getenv("TUNER_LLM_MODEL", "gpt-4o-mini"),
        tts_model=os.getenv("TUNER_TTS_MODEL", "cartesia/sonic"),
        debug=True,
    )

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
        logger.info("[BOT] Client connected — starting appointment booking flow")
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
