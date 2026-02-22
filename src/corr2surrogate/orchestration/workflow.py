"""Minimal orchestration helpers for interactive, user-informed execution."""

from __future__ import annotations

from dataclasses import dataclass, field

from corr2surrogate.agents.agent1_analyst import build_ingestion_status_message
from corr2surrogate.analytics.ranking import ForcedModelingDirective, RankedSignal
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


@dataclass(frozen=True)
class ModelingDirective:
    """Actionable modeling request for Agent 2."""

    target_signal: str
    predictor_signals: list[str]
    force_run_regardless_of_correlation: bool
    source: str
    reason: str = ""


@dataclass(frozen=True)
class LoopEvaluation:
    """Decision after one model-training iteration."""

    should_continue: bool
    attempt: int
    max_attempts: int
    unmet_criteria: list[str]
    recommendations: list[str] = field(default_factory=list)
    summary: str = ""


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


def build_modeling_directives(
    *,
    ranked_signals: list[RankedSignal],
    forced_requests: list[ForcedModelingDirective] | None = None,
) -> list[ModelingDirective]:
    """Merge dependency-aware ranking with user-forced requests."""
    directives: list[ModelingDirective] = []
    seen_targets: set[str] = set()

    for forced in forced_requests or []:
        directives.append(
            ModelingDirective(
                target_signal=forced.target_signal,
                predictor_signals=list(forced.predictor_signals),
                force_run_regardless_of_correlation=True,
                source="user_forced",
                reason=forced.user_reason,
            )
        )
        seen_targets.add(forced.target_signal)

    for ranked in ranked_signals:
        if not ranked.feasible:
            continue
        if ranked.target_signal in seen_targets:
            continue
        directives.append(
            ModelingDirective(
                target_signal=ranked.target_signal,
                predictor_signals=list(ranked.required_signals),
                force_run_regardless_of_correlation=False,
                source="ranked",
                reason=ranked.rationale,
            )
        )
    return directives


def evaluate_training_iteration(
    *,
    metrics: dict[str, float],
    acceptance_criteria: dict[str, float],
    attempt: int,
    max_attempts: int,
    min_relative_improvement: float = 0.02,
    previous_best_score: float | None = None,
) -> LoopEvaluation:
    """Assess model quality and decide whether to continue agentic loop."""
    unmet = _find_unmet_criteria(metrics=metrics, acceptance_criteria=acceptance_criteria)
    if not unmet:
        return LoopEvaluation(
            should_continue=False,
            attempt=attempt,
            max_attempts=max_attempts,
            unmet_criteria=[],
            recommendations=["Quality criteria met. Proceed to artifact export."],
            summary="Model meets all acceptance criteria.",
        )

    recommendations = _build_recommendations(
        metrics=metrics,
        unmet_criteria=unmet,
        min_relative_improvement=min_relative_improvement,
        previous_best_score=previous_best_score,
    )
    can_retry = attempt < max_attempts
    summary = (
        "Acceptance criteria not met. Continuing optimization loop."
        if can_retry
        else "Acceptance criteria not met and max attempts reached."
    )
    return LoopEvaluation(
        should_continue=can_retry,
        attempt=attempt,
        max_attempts=max_attempts,
        unmet_criteria=unmet,
        recommendations=recommendations,
        summary=summary,
    )


def _find_unmet_criteria(
    *, metrics: dict[str, float], acceptance_criteria: dict[str, float]
) -> list[str]:
    unmet: list[str] = []
    for metric_name, threshold in acceptance_criteria.items():
        actual = metrics.get(metric_name)
        if actual is None:
            unmet.append(metric_name)
            continue
        higher_better = _higher_is_better(metric_name)
        if higher_better and actual < threshold:
            unmet.append(metric_name)
        if not higher_better and actual > threshold:
            unmet.append(metric_name)
    return unmet


def _higher_is_better(metric_name: str) -> bool:
    key = metric_name.lower()
    positive_markers = ("r2", "auc", "accuracy", "f1", "precision", "recall")
    return any(marker in key for marker in positive_markers)


def _build_recommendations(
    *,
    metrics: dict[str, float],
    unmet_criteria: list[str],
    min_relative_improvement: float,
    previous_best_score: float | None,
) -> list[str]:
    recommendations: list[str] = []
    train_mae = metrics.get("train_mae")
    val_mae = metrics.get("val_mae")
    sample_count = metrics.get("n_samples")

    if sample_count is not None and sample_count < 500:
        recommendations.append(
            "Dataset appears small for robust surrogate quality. Collect more representative data."
        )
    if train_mae is not None and val_mae is not None and val_mae > train_mae * 1.25:
        recommendations.append(
            "Validation gap suggests overfitting. Increase regularization or simplify architecture."
        )
    if previous_best_score is not None and previous_best_score > 0:
        current_score = metrics.get("val_mae")
        if current_score is not None:
            relative_gain = (previous_best_score - current_score) / previous_best_score
            if relative_gain < min_relative_improvement:
                recommendations.append(
                    "Recent gains are marginal. Try alternate architectures or feature engineering."
                )

    if not recommendations:
        recommendations.append(
            "Try alternate architecture and tune window/lag features for unmet metrics: "
            + ", ".join(unmet_criteria)
        )
    return recommendations
