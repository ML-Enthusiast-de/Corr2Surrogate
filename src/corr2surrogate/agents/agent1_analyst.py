"""Agent 1 helpers for user-facing ingestion and analysis communication."""

from __future__ import annotations

from corr2surrogate.analytics.ranking import ForcedModelingDirective, RankedSignal
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


def build_forced_request_message(directive: ForcedModelingDirective) -> str:
    """Explain that a user-defined signal pair/modeling request will be honored."""
    predictors = ", ".join(directive.predictor_signals)
    reason = f" Reason: {directive.user_reason}" if directive.user_reason else ""
    return (
        f"Forced modeling request acknowledged: target `{directive.target_signal}` "
        f"will be modeled using `{predictors}` regardless of correlation strength.{reason}"
    )


def build_dependency_risk_message(ranked_signal: RankedSignal) -> str:
    """Explain dependency risks in candidate ranking."""
    if ranked_signal.feasible:
        return (
            f"`{ranked_signal.target_signal}` is feasible with adjusted score "
            f"{ranked_signal.adjusted_score:.3f}. {ranked_signal.rationale}"
        )
    return (
        f"`{ranked_signal.target_signal}` is currently infeasible: {ranked_signal.rationale} "
        "Consider keeping some required inputs physical."
    )
