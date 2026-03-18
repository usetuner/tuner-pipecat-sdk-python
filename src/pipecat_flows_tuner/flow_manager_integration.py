"""Integration concern: FlowManager patching used by the observer runtime."""

from __future__ import annotations

import time
from typing import Any

from loguru import logger

from .accumulator import FlowsAccumulator
from .config import TunerConfig


# TODO this is a temp sol (limitaiton), till i find another way to have access to the node transitions :)
def attach_flow_manager_patch(
    flow_manager: Any,
    accumulator: FlowsAccumulator,
    config: TunerConfig,
) -> None:
    # NOTE: This bridge depends on private pipecat-flows internals will be removed in the future 
    # once we find another way to have access to the node transitions.
    original_set_node = getattr(flow_manager, "_set_node", None)
    if not callable(original_set_node):
        logger.warning("[flows-tuner] FlowManager has no callable _set_node; skipping patch")
        return
    if getattr(flow_manager, "_flows_tuner_patch_applied", False):
        return

    async def patched_set_node(node_id: str, node_config: dict[str, object]) -> None:
        from_node = getattr(flow_manager, "_current_node", None)
        pending = accumulator.get_pending_transition()
        await original_set_node(node_id, node_config)
        accumulator.on_node_entered(
            from_node=from_node,
            to_node=node_id,
            trigger=pending,
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
    flow_manager._flows_tuner_patch_applied = True
    logger.info("[flows-tuner] attached to FlowManager (patched _set_node)")
