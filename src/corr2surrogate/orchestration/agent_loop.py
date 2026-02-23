"""Generic agent loop with strict JSON actions and tool dispatch."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal, Protocol

from .tool_registry import (
    ToolExecutionError,
    ToolRegistry,
    ToolValidationError,
    UnknownToolError,
)


class AgentLoopError(RuntimeError):
    """Base agent loop error."""


class InvalidAgentActionError(AgentLoopError):
    """Raised when agent output does not follow action contract."""


class MaxTurnsExceededError(AgentLoopError):
    """Raised when loop turn limit is exceeded."""


@dataclass(frozen=True)
class AgentLoopLimits:
    """Operational limits for one loop run."""

    max_turns_per_phase: int = 20
    max_invalid_actions: int = 3


@dataclass(frozen=True)
class AgentAction:
    """Normalized action returned by the agent."""

    action: Literal["tool_call", "respond"]
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    message: str = ""


@dataclass(frozen=True)
class AgentTurnEvent:
    """Structured record for each orchestration turn."""

    turn: int
    status: Literal[
        "tool_result",
        "respond",
        "needs_user_confirmation",
        "blocked",
        "invalid_action",
        "tool_error",
    ]
    action: AgentAction
    message: str = ""
    tool_output: Any = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentResponder(Protocol):
    """Callable contract for an LLM wrapper."""

    def __call__(
        self,
        *,
        history: list[AgentTurnEvent],
        context: dict[str, Any],
    ) -> dict[str, Any] | str:
        """Return one strict JSON action."""


class AgentLoop:
    """A minimal, deterministic loop that dispatches validated tool calls."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        limits: AgentLoopLimits | None = None,
    ) -> None:
        self.registry = registry
        self.limits = limits or AgentLoopLimits()
        self.history: list[AgentTurnEvent] = []
        self._invalid_actions = 0

    def step(self, raw_action: dict[str, Any] | str) -> AgentTurnEvent:
        """Process one action from the active agent."""
        turn = len(self.history) + 1
        if turn > self.limits.max_turns_per_phase:
            raise MaxTurnsExceededError(
                f"Exceeded max turns ({self.limits.max_turns_per_phase})."
            )

        try:
            action = parse_agent_action(raw_action)
        except InvalidAgentActionError as exc:
            self._invalid_actions += 1
            event = AgentTurnEvent(
                turn=turn,
                status="invalid_action",
                action=AgentAction(action="respond", message=""),
                error=str(exc),
            )
            self.history.append(event)
            self._enforce_invalid_limit()
            return event

        if action.action == "respond":
            event = AgentTurnEvent(
                turn=turn,
                status="respond",
                action=action,
                message=action.message,
            )
            self.history.append(event)
            return event

        try:
            result = self.registry.execute(action.tool_name, action.arguments)
        except (UnknownToolError, ToolValidationError, ToolExecutionError) as exc:
            event = AgentTurnEvent(
                turn=turn,
                status="tool_error",
                action=action,
                error=str(exc),
            )
            self.history.append(event)
            return event

        status_map = {
            "ok": "tool_result",
            "needs_confirmation": "needs_user_confirmation",
            "blocked": "blocked",
        }
        event = AgentTurnEvent(
            turn=turn,
            status=status_map[result.status],
            action=action,
            message=result.message,
            tool_output=result.output,
        )
        self.history.append(event)
        return event

    def run(
        self,
        *,
        responder: AgentResponder,
        context: dict[str, Any] | None = None,
    ) -> AgentTurnEvent:
        """Run loop until a terminal condition is reached."""
        state = dict(context or {})
        while True:
            raw_action = responder(history=list(self.history), context=state)
            event = self.step(raw_action)
            state["last_event"] = event.to_dict()
            if event.status in {"respond", "needs_user_confirmation", "blocked"}:
                return event

    def _enforce_invalid_limit(self) -> None:
        if self._invalid_actions > self.limits.max_invalid_actions:
            raise InvalidAgentActionError(
                "Too many invalid actions from agent output."
            )


def parse_agent_action(raw_action: dict[str, Any] | str) -> AgentAction:
    """Parse and validate strict JSON action format."""
    payload = _coerce_payload(raw_action)
    if not isinstance(payload, dict):
        raise InvalidAgentActionError("Agent output must be a JSON object.")

    action_type = payload.get("action")
    if action_type == "tool_call":
        tool_name = payload.get("tool_name")
        arguments = payload.get("arguments")
        if not isinstance(tool_name, str) or tool_name.strip() == "":
            raise InvalidAgentActionError("tool_call requires non-empty 'tool_name'.")
        if not isinstance(arguments, dict):
            raise InvalidAgentActionError("tool_call requires object 'arguments'.")
        reason = payload.get("reason", "")
        if reason is not None and not isinstance(reason, str):
            raise InvalidAgentActionError("'reason' must be a string if provided.")
        return AgentAction(
            action="tool_call",
            tool_name=tool_name.strip(),
            arguments=arguments,
            reason=str(reason or ""),
        )

    if action_type == "respond":
        message = payload.get("message")
        if not isinstance(message, str) or message.strip() == "":
            raise InvalidAgentActionError("respond requires non-empty 'message'.")
        return AgentAction(action="respond", message=message.strip())

    raise InvalidAgentActionError("Action must be either 'tool_call' or 'respond'.")


def _coerce_payload(raw_action: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(raw_action, dict):
        return raw_action
    if not isinstance(raw_action, str):
        raise InvalidAgentActionError("Agent output must be dict or JSON string.")
    try:
        loaded = json.loads(raw_action)
    except json.JSONDecodeError as exc:
        raise InvalidAgentActionError(f"Agent output is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise InvalidAgentActionError("Agent JSON must decode to an object.")
    return loaded
