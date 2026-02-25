from corr2surrogate.orchestration.agent_loop import AgentLoop, AgentLoopLimits, parse_agent_action
from corr2surrogate.orchestration.tool_registry import ToolRegistry


def test_parse_agent_action_accepts_json_string() -> None:
    action = parse_agent_action(
        '{"action":"tool_call","tool_name":"echo","arguments":{"value":"x"}}'
    )
    assert action.action == "tool_call"
    assert action.tool_name == "echo"
    assert action.arguments["value"] == "x"


def test_agent_loop_executes_tool_call() -> None:
    registry = ToolRegistry()

    def echo(value: str) -> str:
        return value

    registry.register_function(
        name="echo",
        description="Echo tool.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        handler=echo,
    )
    loop = AgentLoop(registry=registry)
    event = loop.step(
        {"action": "tool_call", "tool_name": "echo", "arguments": {"value": "ok"}}
    )
    assert event.status == "tool_result"
    assert event.tool_output == "ok"


def test_agent_loop_run_returns_on_response() -> None:
    registry = ToolRegistry()
    loop = AgentLoop(registry=registry)

    def responder(*, history, context):
        return {"action": "respond", "message": "done"}

    event = loop.run(responder=responder, context={})
    assert event.status == "respond"
    assert event.message == "done"


def test_agent_loop_run_falls_back_after_repeated_tool_errors() -> None:
    registry = ToolRegistry()
    loop = AgentLoop(
        registry=registry,
        limits=AgentLoopLimits(max_consecutive_tool_errors=2),
    )

    def responder(*, history, context):
        return {"action": "tool_call", "tool_name": "missing_tool", "arguments": {}}

    event = loop.run(responder=responder, context={})
    assert event.status == "respond"
    assert "repeated tool argument errors" in event.message


def test_agent_loop_run_falls_back_after_repeated_same_tool_result_error() -> None:
    registry = ToolRegistry()

    def failing_tool() -> dict:
        return {"status": "error", "message": "bad input"}

    registry.register_function(
        name="failing_tool",
        description="Returns deterministic error payload.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler=failing_tool,
    )
    loop = AgentLoop(registry=registry)

    def responder(*, history, context):
        return {"action": "tool_call", "tool_name": "failing_tool", "arguments": {}}

    event = loop.run(responder=responder, context={})
    assert event.status == "respond"
    assert "repeating the same failing tool call result" in event.message
