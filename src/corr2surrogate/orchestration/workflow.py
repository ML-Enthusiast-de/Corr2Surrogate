"""Minimal orchestration helpers for interactive, user-informed execution."""

from __future__ import annotations

from dataclasses import dataclass

from corr2surrogate.agents.agent1_analyst import build_ingestion_status_message
from corr2surrogate.ingestion import (
    IngestionError,
    IngestionResult,
    SheetSelectionRequiredError,
    load_tabular_data,
)


@dataclass(frozen=True)
class WorkflowStepResult:
    """Generic workflow response shape for CLI/API integrations."""

    status: str
    message: str
    ingestion_result: IngestionResult | None = None
    options: list[str] | None = None


def prepare_ingestion_step(
    *,
    path: str,
    sheet_name: str | int | None = None,
    header_row: int | None = None,
    data_start_row: int | None = None,
) -> WorkflowStepResult:
    """Load tabular input and return an interaction-safe status."""
    try:
        result = load_tabular_data(
            path,
            sheet_name=sheet_name,
            header_row=header_row,
            data_start_row=data_start_row,
        )
        return WorkflowStepResult(
            status="ok",
            message=build_ingestion_status_message(result),
            ingestion_result=result,
        )
    except SheetSelectionRequiredError as exc:
        return WorkflowStepResult(
            status="needs_user_input",
            message="Excel file has multiple sheets. Please choose one.",
            options=exc.sheets,
        )
    except IngestionError as exc:
        return WorkflowStepResult(status="error", message=str(exc))
