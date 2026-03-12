"""Integration concern: FlowManager patching used by the observer runtime."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from .accumulator import FlowsAccumulator
from .config import TunerConfig


def attach_flow_manager_patch(
    flow_manager: Any,
    accumulator: FlowsAccumulator,
    config: TunerConfig,
) -> None:
    original_set_node = flow_manager._set_node

    async def patched_set_node(node_id: str, node_config: dict[str, object]) -> None:
        from_node = flow_manager._current_node
        pending = accumulator.get_pending_transition()
        await original_set_node(node_id, node_config)
        state = dict(flow_manager.state)
        accumulator.on_node_entered(
            from_node=from_node,
            to_node=node_id,
            node_config=node_config,
            trigger=pending,
            state_snapshot=state,
            timestamp_ns=time.time_ns(),
        )
        if config.debug:
            logger.debug(
                "[flows-tuner] node entered: {} → {}  trigger={}",
                from_node,
                node_id,
                pending.function_name if pending else None,
            )

    flow_manager._set_node = patched_set_node
    logger.info("[flows-tuner] attached to FlowManager (patched _set_node)")
