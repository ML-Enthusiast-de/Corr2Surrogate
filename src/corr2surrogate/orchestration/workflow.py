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
    trajectory_recommendations: list[str] = field(default_factory=list)
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
    task_type_hint: str | None = None,
    data_mode: str | None = None,
    feature_columns: list[str] | None = None,
    target_column: str | None = None,
    lag_horizon_samples: int | None = None,
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
    trajectory_recommendations = _build_trajectory_recommendations(
        metrics=metrics,
        unmet_criteria=unmet,
        task_type_hint=task_type_hint,
        data_mode=data_mode,
        feature_columns=feature_columns or [],
        target_column=target_column,
        lag_horizon_samples=lag_horizon_samples,
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
        trajectory_recommendations=trajectory_recommendations,
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
    sample_count = metrics.get("n_samples")

    if sample_count is not None and sample_count < 500:
        recommendations.append(
            "Dataset appears small for robust surrogate quality. Collect more representative data."
        )
    classification_like = any(
        token in name.lower()
        for name in metrics.keys()
        for token in ("accuracy", "precision", "recall", "f1", "auc", "log_loss")
    )
    if classification_like:
        train_f1 = metrics.get("train_f1")
        val_f1 = metrics.get("val_f1")
        train_log_loss = metrics.get("train_log_loss")
        val_log_loss = metrics.get("val_log_loss")
        val_recall = metrics.get("val_recall")
        val_pr_auc = metrics.get("val_pr_auc")

        if train_f1 is not None and val_f1 is not None and train_f1 > max(val_f1 + 0.10, val_f1 * 1.10):
            recommendations.append(
                "Validation F1 trails training F1 materially. Reduce model complexity or add regularization."
            )
        if train_log_loss is not None and val_log_loss is not None and val_log_loss > train_log_loss * 1.20:
            recommendations.append(
                "Validation log-loss is materially worse than training, which suggests overfitting or unstable probabilities."
            )
        if any(name in unmet_criteria for name in ("recall", "pr_auc")):
            if val_recall is not None and val_recall < 0.70:
                recommendations.append(
                    "Recall is still weak on the minority class. Collect more positive-class examples or try a stronger tree classifier."
                )
            elif val_pr_auc is not None and val_pr_auc < 0.40:
                recommendations.append(
                    "Precision-recall separation is weak. Add discriminative features or gather harder negative examples."
                )
        if previous_best_score is not None and previous_best_score > 0:
            current_score = metrics.get("val_f1")
            if current_score is not None:
                relative_gain = (current_score - previous_best_score) / previous_best_score
                if relative_gain < min_relative_improvement:
                    recommendations.append(
                        "Recent classification gains are marginal. Try alternate architectures, threshold tuning, or better class coverage."
                    )
    else:
        train_mae = metrics.get("train_mae")
        val_mae = metrics.get("val_mae")
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


def _build_trajectory_recommendations(
    *,
    metrics: dict[str, float],
    unmet_criteria: list[str],
    task_type_hint: str | None,
    data_mode: str | None,
    feature_columns: list[str],
    target_column: str | None,
    lag_horizon_samples: int | None,
) -> list[str]:
    if not unmet_criteria:
        return []

    task_type = str(task_type_hint or "").strip().lower()
    mode = str(data_mode or "").strip().lower()
    predictors = [str(item).strip() for item in feature_columns if str(item).strip()]
    primary_predictors = predictors[: min(3, len(predictors))]
    predictor_text = ", ".join(primary_predictors) if primary_predictors else "the current predictor set"
    target_text = str(target_column or "the selected target").strip() or "the selected target"
    lag_horizon = max(int(lag_horizon_samples or 0), 0)

    suggestions: list[str] = []
    if task_type in {"fraud_detection", "anomaly_detection"}:
        suggestions.append(
            f"Collect additional positive-event runs around the operating region covered by {predictor_text} so `{target_text}` sees more rare-event examples."
        )
        suggestions.append(
            "Add hard negative and near-boundary examples that look similar to positives to improve precision-recall separation."
        )
        if mode == "time_series":
            suggestions.append(
                f"Record longer sequential windows spanning the lead-in to each event so temporal precursors are visible across at least {max(lag_horizon, 3)} samples."
            )
        return suggestions

    if task_type in {"binary_classification", "multiclass_classification"}:
        suggestions.append(
            f"Collect more balanced label coverage in the predictor space defined by {predictor_text} so `{target_text}` sees all decision regions in train/validation/test."
        )
        suggestions.append(
            "Add boundary-crossing examples near class transitions rather than only easy, well-separated cases."
        )
        if mode == "time_series":
            suggestions.append(
                f"Capture contiguous sequences through class transitions so lagged features can observe the state change over at least {max(lag_horizon, 3)} samples."
            )
        return suggestions

    suggestions.append(
        f"Run dense sweeps across the strongest current inputs ({predictor_text}) and include repeated low/mid/high operating points for `{target_text}`."
    )
    if mode == "time_series":
        suggestions.append(
            f"Add ramp-up and ramp-down trajectories that hold excitation long enough to cover at least {max(lag_horizon + 1, 4)} sequential samples."
        )
    else:
        suggestions.append(
            "Collect mixed-condition steady-state points that combine the current feature extremes instead of varying one factor at a time."
        )
    sample_count = metrics.get("n_samples")
    if sample_count is not None and sample_count < 500:
        suggestions.append(
            "Increase dataset size with another acquisition batch before escalating to a more complex model family."
        )
    return suggestions
