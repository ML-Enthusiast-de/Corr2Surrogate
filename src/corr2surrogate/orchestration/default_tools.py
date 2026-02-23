"""Default tool registration for the Corr2Surrogate harness."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .tool_registry import ToolRegistry
from .workflow import (
    build_modeling_directives,
    evaluate_training_iteration,
    prepare_ingestion_step,
)


def build_default_registry() -> ToolRegistry:
    """Create the default tool registry for analyst/modeler loops."""
    registry = ToolRegistry()

    registry.register_function(
        name="prepare_ingestion_step",
        description="Load CSV/XLSX and return ingestion readiness status.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "sheet_name": {"type": "string"},
                "header_row": {"type": "integer"},
                "data_start_row": {"type": "integer"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=_tool_prepare_ingestion_step,
        risk_level="low",
    )

    registry.register_function(
        name="evaluate_training_iteration",
        description="Evaluate one model attempt against acceptance criteria.",
        input_schema={
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "object",
                    "additionalProperties": True,
                },
                "acceptance_criteria": {
                    "type": "object",
                    "additionalProperties": True,
                },
                "attempt": {"type": "integer"},
                "max_attempts": {"type": "integer"},
                "min_relative_improvement": {"type": "number"},
                "previous_best_score": {"type": "number"},
            },
            "required": ["metrics", "acceptance_criteria", "attempt", "max_attempts"],
            "additionalProperties": False,
        },
        handler=_tool_evaluate_training_iteration,
        risk_level="low",
    )

    registry.register_function(
        name="build_modeling_directives",
        description=(
            "Merge ranked candidates and forced requests into actionable modeling directives."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "ranked_signals": {"type": "array"},
                "forced_requests": {"type": "array"},
            },
            "required": ["ranked_signals"],
            "additionalProperties": False,
        },
        handler=_tool_build_modeling_directives,
        risk_level="low",
    )

    return registry


def _tool_prepare_ingestion_step(
    *,
    path: str,
    sheet_name: str | None = None,
    header_row: int | None = None,
    data_start_row: int | None = None,
) -> dict[str, Any]:
    result = prepare_ingestion_step(
        path=path,
        sheet_name=sheet_name,
        header_row=header_row,
        data_start_row=data_start_row,
    )
    payload = {
        "status": result.status,
        "message": result.message,
        "options": result.options or [],
    }
    return payload


def _tool_evaluate_training_iteration(
    *,
    metrics: dict[str, float],
    acceptance_criteria: dict[str, float],
    attempt: int,
    max_attempts: int,
    min_relative_improvement: float = 0.02,
    previous_best_score: float | None = None,
) -> dict[str, Any]:
    result = evaluate_training_iteration(
        metrics=metrics,
        acceptance_criteria=acceptance_criteria,
        attempt=attempt,
        max_attempts=max_attempts,
        min_relative_improvement=min_relative_improvement,
        previous_best_score=previous_best_score,
    )
    return asdict(result)


def _tool_build_modeling_directives(
    *,
    ranked_signals: list[dict[str, Any]],
    forced_requests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from corr2surrogate.analytics.ranking import ForcedModelingDirective, RankedSignal

    ranked = [RankedSignal(**item) for item in ranked_signals]
    forced = [ForcedModelingDirective(**item) for item in (forced_requests or [])]
    directives = build_modeling_directives(ranked_signals=ranked, forced_requests=forced)
    return {"directives": [asdict(item) for item in directives]}
