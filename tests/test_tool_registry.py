from corr2surrogate.orchestration.tool_registry import (
    ToolRegistry,
    ToolValidationError,
)


def test_tool_registry_executes_valid_call() -> None:
    registry = ToolRegistry()

    def add_one(value: int) -> int:
        return value + 1

    registry.register_function(
        name="add_one",
        description="Increment by one.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        handler=add_one,
        risk_level="low",
    )
    result = registry.execute("add_one", {"value": 2})
    assert result.status == "ok"
    assert result.output == 3


def test_tool_registry_rejects_unknown_fields() -> None:
    registry = ToolRegistry()

    def noop() -> str:
        return "ok"

    registry.register_function(
        name="noop",
        description="No-op tool.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler=noop,
    )

    try:
        registry.validate_arguments("noop", {"extra": 1})
    except ToolValidationError:
        return
    raise AssertionError("Expected ToolValidationError for unknown field.")


def test_tool_registry_respects_confirm_risk_level() -> None:
    registry = ToolRegistry()

    def noop() -> str:
        return "ok"

    registry.register_function(
        name="needs_confirm",
        description="Needs user confirmation.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        handler=noop,
        risk_level="confirm",
    )
    result = registry.execute("needs_confirm", {})
    assert result.status == "needs_confirmation"
