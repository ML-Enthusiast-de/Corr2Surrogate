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


def test_tool_registry_allows_optional_null_field_values() -> None:
    registry = ToolRegistry()

    def echo(value: int, optional_value: int | None = None) -> dict[str, int | None]:
        return {"value": value, "optional_value": optional_value}

    registry.register_function(
        name="echo",
        description="Echo values.",
        input_schema={
            "type": "object",
            "properties": {
                "value": {"type": "integer"},
                "optional_value": {"type": "integer"},
            },
            "required": ["value"],
            "additionalProperties": False,
        },
        handler=echo,
    )
    registry.validate_arguments("echo", {"value": 1, "optional_value": None})


def test_tool_registry_rejects_required_null_field_values() -> None:
    registry = ToolRegistry()

    def echo(value: int) -> int:
        return value

    registry.register_function(
        name="echo_required",
        description="Echo required value.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        handler=echo,
    )

    try:
        registry.validate_arguments("echo_required", {"value": None})
    except ToolValidationError:
        return
    raise AssertionError("Expected ToolValidationError for required null value.")
