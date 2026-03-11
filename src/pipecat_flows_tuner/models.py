"""Data models for pipecat_flows_tuner.

These are plain dicts in the payload, but typed aliases are provided here for documentation.
The actual runtime representation is always dict so it is JSON-serializable without extra work.
"""

from typing import Any


# ── type aliases (documentation only) ────────────────────────────────────────

# NodeTransition: one entry in payload["flow_transitions"]
# {
#   "from_node": str | None,
#   "to_node": str,
#   "trigger_function": str | None,
#   "trigger_args": dict | None,
#   "state_snapshot": dict,
#   "task_messages": list,
#   "functions_available": list[str],
#   "timestamp_ms": int,
# }
NodeTransition = dict[str, Any]

# FlowSegment: one entry in payload["transcript_with_tool_calls"]
# {
#   "role": str,            # "user" | "agent" | "function_call" | "function_result"
#   "text": str,
#   "start_ms": int,
#   "end_ms": int,
#   "node": str | None,
#   ... (role-specific keys)
# }
FlowSegment = dict[str, Any]

# LatencyTurn: one entry in payload["latency_turns"]
# {
#   "turn_index": int,
#   "node": str | None,
#   "ttfb_ms": int | None,
#   "llm_ms": int | None,
#   "tts_ms": int | None,
# }
LatencyTurn = dict[str, Any]

# CallData: the top-level payload sent to the API
CallData = dict[str, Any]
