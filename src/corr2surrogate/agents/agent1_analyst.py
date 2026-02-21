"""Agent 1 helpers for user-facing ingestion/science communication."""

from __future__ import annotations

from corr2surrogate.ingestion import IngestionResult


def build_ingestion_status_message(result: IngestionResult) -> str:
    """Create a concise user update about detected file structure."""
    inferred = result.inferred_header
    lines = [
        f"Loaded `{result.source_path.name}` as {result.file_type.upper()}",
        f"Inferred header row: {inferred.header_row}, data starts at row: {inferred.data_start_row}",
        f"Header confidence: {inferred.confidence:.2f}",
    ]
    if result.selected_sheet:
        lines.append(f"Selected sheet: {result.selected_sheet}")
    if inferred.needs_user_confirmation:
        lines.append(
            "I am not fully confident about the header/data start. Please confirm or provide row indices."
        )
    return "\n".join(lines)
