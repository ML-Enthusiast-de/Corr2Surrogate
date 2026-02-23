from corr2surrogate.orchestration.agent_loop import AgentLoop, parse_agent_action
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
