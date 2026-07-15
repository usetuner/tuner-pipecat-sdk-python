"""Tool call timing registry keyed by tool_call_id."""

from dataclasses import dataclass, field


@dataclass
class ToolTimingRegistry:
    """Per-tool-call-id timing store for function invocation and completion."""

    _invocations_ns: dict[str, int] = field(default_factory=dict)
    _completions_ns: dict[str, int] = field(default_factory=dict)

    def record_invocation_ns(self, tool_call_id: str, abs_ns: int) -> None:
        """Record when a tool call was invoked."""
        self._invocations_ns[tool_call_id] = abs_ns

    def record_completion_ns(self, tool_call_id: str, abs_ns: int) -> None:
        """Record when a tool call completed."""
        self._completions_ns[tool_call_id] = abs_ns

    def get_invocation_ns(self, tool_call_id: str) -> int | None:
        """Return the invocation timestamp for `tool_call_id`, or `None` if it was never recorded."""
        return self._invocations_ns.get(tool_call_id)

    def get_completion_ns(self, tool_call_id: str) -> int | None:
        """Return the completion timestamp for `tool_call_id`, or `None` if it was never recorded."""
        return self._completions_ns.get(tool_call_id)
