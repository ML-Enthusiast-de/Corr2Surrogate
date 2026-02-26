"""Tool registry and argument validation for agent tool use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal


RiskLevel = Literal["low", "confirm", "blocked"]


class ToolRegistryError(RuntimeError):
    """Base error for registry failures."""


class UnknownToolError(ToolRegistryError):
    """Raised when a tool name is not registered."""


class ToolValidationError(ToolRegistryError):
    """Raised when tool arguments do not match schema."""


class ToolExecutionError(ToolRegistryError):
    """Raised when a tool handler fails at runtime."""


@dataclass(frozen=True)
class ToolSpec:
    """Description of one callable tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    risk_level: RiskLevel = "low"


@dataclass(frozen=True)
class ToolExecutionResult:
    """Structured result from one tool call."""

    tool_name: str
    status: Literal["ok", "needs_confirmation", "blocked"]
    output: Any = None
    message: str = ""


class ToolRegistry:
    """In-memory tool registry with schema validation."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register one tool."""
        if spec.name in self._tools:
            raise ToolRegistryError(f"Tool '{spec.name}' is already registered.")
        self._tools[spec.name] = spec

    def register_function(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        handler: Callable[..., Any],
        risk_level: RiskLevel = "low",
    ) -> None:
        """Convenience API for function-based tool registration."""
        self.register(
            ToolSpec(
                name=name,
                description=description,
                input_schema=input_schema,
                handler=handler,
                risk_level=risk_level,
            )
        )

    def get(self, name: str) -> ToolSpec:
        """Return tool spec by name."""
        if name not in self._tools:
            raise UnknownToolError(f"Unknown tool '{name}'.")
        return self._tools[name]

    def list_tools(self) -> list[dict[str, str]]:
        """List basic metadata for all registered tools."""
        return [
            {"name": spec.name, "description": spec.description, "risk_level": spec.risk_level}
            for spec in self._tools.values()
        ]

    def validate_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Validate arguments against a JSON-schema-like object schema."""
        spec = self.get(tool_name)
        _validate_object(arguments, schema=spec.input_schema, path=tool_name)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        """Validate and execute a tool call."""
        spec = self.get(tool_name)
        self.validate_arguments(tool_name, arguments)

        if spec.risk_level == "blocked":
            return ToolExecutionResult(
                tool_name=tool_name,
                status="blocked",
                message=f"Tool '{tool_name}' is blocked by policy.",
            )
        if spec.risk_level == "confirm":
            return ToolExecutionResult(
                tool_name=tool_name,
                status="needs_confirmation",
                message=f"Tool '{tool_name}' requires explicit user confirmation.",
            )

        try:
            output = spec.handler(**arguments)
        except Exception as exc:  # pragma: no cover - handler-specific failures
            raise ToolExecutionError(
                f"Tool '{tool_name}' failed during execution: {exc}"
            ) from exc
        return ToolExecutionResult(tool_name=tool_name, status="ok", output=output)


def _validate_object(payload: Any, *, schema: dict[str, Any], path: str) -> None:
    if schema.get("type") != "object":
        raise ToolValidationError(f"{path}: root schema type must be 'object'.")
    if not isinstance(payload, dict):
        raise ToolValidationError(f"{path}: arguments must be an object.")

    properties = schema.get("properties", {})
    required = schema.get("required", [])
    required_set = set(required)
    additional_properties = schema.get("additionalProperties", True)

    for key in required:
        if key not in payload:
            raise ToolValidationError(f"{path}: missing required field '{key}'.")

    if not additional_properties:
        unknown = [key for key in payload if key not in properties]
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ToolValidationError(f"{path}: unknown fields not allowed: {joined}.")

    for key, value in payload.items():
        key_path = f"{path}.{key}"
        key_schema = properties.get(key)
        if key_schema is None:
            if additional_properties:
                continue
            raise ToolValidationError(f"{key_path}: field is not defined in schema.")
        if value is None:
            if key in required_set:
                raise ToolValidationError(f"{key_path}: required field cannot be null.")
            continue
        _validate_value(value=value, schema=key_schema, path=key_path)


def _validate_value(*, value: Any, schema: dict[str, Any], path: str) -> None:
    expected = schema.get("type")
    if expected is None:
        return

    if expected == "string":
        if not isinstance(value, str):
            raise ToolValidationError(f"{path}: expected string.")
        return

    if expected == "boolean":
        if not isinstance(value, bool):
            raise ToolValidationError(f"{path}: expected boolean.")
        return

    if expected == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ToolValidationError(f"{path}: expected number.")
        return

    if expected == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ToolValidationError(f"{path}: expected integer.")
        return

    if expected == "array":
        if not isinstance(value, list):
            raise ToolValidationError(f"{path}: expected array.")
            return
        item_schema = schema.get("items")
        if item_schema is not None:
            for idx, item in enumerate(value):
                _validate_value(value=item, schema=item_schema, path=f"{path}[{idx}]")
        return

    if expected == "object":
        _validate_object(value, schema=schema, path=path)
        return

    raise ToolValidationError(f"{path}: unsupported schema type '{expected}'.")
