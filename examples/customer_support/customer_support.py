#
# Copyright (c) 2024-2026, Tuner Team
#
# SPDX-License-Identifier: BSD 2-Clause License

"""Customer support bot built with Pipecat Flows + pipecat-flows-tuner.

This example demonstrates:
- Multi-node support flow with category branching
- State accumulation across nodes (name, category, description)
- Conditional farewell (resolved vs. escalated to human agent)
- Full call observability via FlowsObserver

Requirements:
- CARTESIA_API_KEY
- DEEPGRAM_API_KEY
- OPENAI_API_KEY

Run the example:
    uv run customer_support.py
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
from pipecat.services.cartesia.tts import CartesiaTTSService
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

ISSUE_CATEGORIES = ["billing", "technical", "account"]

RESOLUTION_SCRIPTS = {
    "billing": (
        "For billing issues, explain that all charges are reviewed within 3-5 business days. "
        "Offer to waive any late fees if applicable. Ask if this resolves their concern."
    ),
    "technical": (
        "For technical issues, walk the customer through these steps: "
        "1) Clear cache and restart the app, 2) Check internet connection, 3) Reinstall if needed. "
        "Ask if any of these steps resolved the problem."
    ),
    "account": (
        "For account issues, guide the customer to Settings > Account > Manage Profile. "
        "Remind them that password resets are sent to their registered email. "
        "Ask if this resolves their issue."
    ),
}


def log_node(node: dict, label: str) -> None:
    logger.debug(f"[NODE CREATED] {label}:\n{json.dumps(node, indent=2, default=str)}")


# ---------------------------------------------------------------------------
# Flow nodes
# ---------------------------------------------------------------------------


def create_greeting_node() -> NodeConfig:
    provide_name_func = FlowsFunctionSchema(
        name="provide_name",
        description="Record the customer's name to personalise the interaction.",
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
                    "You are a professional customer support agent at Acme Corp. "
                    "Keep responses concise and empathetic. "
                    "Your responses will be converted to audio — no emojis or special characters."
                ),
            }
        ],
        task_messages=[
            {
                "role": "system",
                "content": "Greet the customer warmly and ask for their name.",
            }
        ],
        functions=[provide_name_func],
    )
    log_node(dict(node), "greeting")
    return node


async def handle_provide_name(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    name = args["name"]
    flow_manager.state["name"] = name
    logger.info(f"[SUPPORT] Customer name: {name}")
    return {"name": name}, create_issue_category_node()


def create_issue_category_node() -> NodeConfig:
    select_category_func = FlowsFunctionSchema(
        name="select_category",
        description="Record the issue category the customer needs help with.",
        required=["category"],
        properties={
            "category": {"type": "string", "enum": ISSUE_CATEGORIES},
        },
        handler=handle_select_category,
    )

    node = NodeConfig(
        name="issue_category",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Ask the customer what type of issue they're experiencing. "
                    "The options are: billing, technical support, or account management."
                ),
            }
        ],
        functions=[select_category_func],
    )
    log_node(dict(node), "issue_category")
    return node


async def handle_select_category(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    category = args["category"].lower()
    flow_manager.state["category"] = category
    logger.info(f"[SUPPORT] Category: {category}")
    return {"category": category}, create_collect_context_node()


def create_collect_context_node() -> NodeConfig:
    describe_issue_func = FlowsFunctionSchema(
        name="describe_issue",
        description="Record a brief description of the customer's issue.",
        required=["description"],
        properties={"description": {"type": "string"}},
        handler=handle_describe_issue,
    )

    node = NodeConfig(
        name="collect_context",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Ask the customer to briefly describe the specific problem they are experiencing. "
                    "Keep it open-ended so they can explain in their own words."
                ),
            }
        ],
        functions=[describe_issue_func],
    )
    log_node(dict(node), "collect_context")
    return node


async def handle_describe_issue(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    description = args["description"]
    flow_manager.state["description"] = description
    logger.info(f"[SUPPORT] Issue description: {description}")
    return {"description": description}, create_resolution_node()


def create_resolution_node() -> NodeConfig:
    mark_resolved_func = FlowsFunctionSchema(
        name="mark_resolved",
        description="Indicate whether the customer's issue was resolved or needs escalation.",
        required=["resolved"],
        properties={"resolved": {"type": "boolean"}},
        handler=handle_mark_resolved,
    )

    node = NodeConfig(
        name="resolution",
        task_messages=[
            {
                "role": "system",
                "content": (
                    "Attempt to resolve the customer's issue using the appropriate script for their category. "
                    f"Billing: {RESOLUTION_SCRIPTS['billing']} "
                    f"Technical: {RESOLUTION_SCRIPTS['technical']} "
                    f"Account: {RESOLUTION_SCRIPTS['account']} "
                    "Use the category and description captured earlier to tailor your response. "
                    "After providing guidance, ask if this resolved their issue."
                ),
            }
        ],
        functions=[mark_resolved_func],
    )
    log_node(dict(node), "resolution")
    return node


async def handle_mark_resolved(args: FlowArgs, flow_manager: FlowManager) -> tuple:
    resolved = args["resolved"]
    flow_manager.state["resolved"] = resolved
    logger.info(f"[SUPPORT] Resolved: {resolved} | state={flow_manager.state}")
    return {"resolved": resolved}, create_farewell_node(resolved)


def create_farewell_node(resolved: bool) -> NodeConfig:
    if resolved:
        content = (
            "Thank the customer by name for contacting Acme Corp support. "
            "Let them know their issue has been resolved and wish them a great day. "
            "Then end the conversation."
        )
    else:
        content = (
            "Apologise that the issue could not be resolved immediately. "
            "Inform the customer that a senior support specialist will contact them within 24 hours. "
            "Thank them for their patience and end the conversation."
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
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="32b3f3c5-7171-46aa-abe7-b598964aa793",
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
        agent_id=os.getenv("TUNER_AGENT_ID", "customer-support-bot"),
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
        logger.info("[BOT] Client connected — starting customer support flow")
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
