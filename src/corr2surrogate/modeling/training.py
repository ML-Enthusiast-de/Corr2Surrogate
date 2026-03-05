"""Split-safe training and model comparison helpers for Agent 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from corr2surrogate.analytics.task_detection import assess_task_profile, is_classification_task
from corr2surrogate.core.json_utils import write_json
from corr2surrogate.persistence.artifact_store import ArtifactStore
from .baselines import IncrementalLinearSurrogate
from .checkpoints import ModelCheckpointStore
from .classifiers import (
    BaggedTreeClassifierSurrogate,
    BoostedTreeClassifierSurrogate,
    LogisticClassificationSurrogate,
)
from .evaluation import classification_metrics, regression_metrics
from .normalization import MinMaxNormalizer
from .performance_feedback import analyze_model_performance
from .splitters import DatasetSplit, build_train_validation_test_split


@dataclass(frozen=True)
class CandidateMetrics:
    """Train/validation/test metrics for one candidate model."""

    model_family: str
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_MODEL_TIEBREAK = {
    "linear_ridge": 0,
    "lagged_linear": 1,
    "bagged_tree_ensemble": 2,
    "boosted_tree_ensemble": 3,
    "lagged_tree_ensemble": 4,
    "logistic_regression": 0,
    "lagged_logistic_regression": 1,
    "bagged_tree_classifier": 2,
    "boosted_tree_classifier": 3,
    "lagged_tree_classifier": 4,
}


def normalize_candidate_model_family(requested_model_family: str) -> str | None:
    """Normalize user-facing model names to implemented candidate identifiers."""
    normalized = requested_model_family.strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "auto": "auto",
        "recommended": "auto",
        "linear_ridge": "linear_ridge",
        "ridge": "linear_ridge",
        "linear": "linear_ridge",
        "incremental_linear_surrogate": "linear_ridge",
        "logistic_regression": "logistic_regression",
        "logistic": "logistic_regression",
        "logit": "logistic_regression",
        "linear_classifier": "logistic_regression",
        "classifier": "logistic_regression",
        "lagged_logistic_regression": "lagged_logistic_regression",
        "lagged_logistic": "lagged_logistic_regression",
        "lagged_logit": "lagged_logistic_regression",
        "temporal_logistic": "lagged_logistic_regression",
        "temporal_classifier": "lagged_logistic_regression",
        "lagged_classifier": "lagged_logistic_regression",
        "lagged_linear": "lagged_linear",
        "lagged": "lagged_linear",
        "temporal_linear": "lagged_linear",
        "arx": "lagged_linear",
        "bagged_tree_classifier": "bagged_tree_classifier",
        "tree_classifier": "bagged_tree_classifier",
        "classifier_tree": "bagged_tree_classifier",
        "fraud_tree": "bagged_tree_classifier",
        "lagged_tree_classifier": "lagged_tree_classifier",
        "temporal_tree_classifier": "lagged_tree_classifier",
        "lag_window_tree_classifier": "lagged_tree_classifier",
        "temporal_fraud_tree": "lagged_tree_classifier",
        "lagged_tree_ensemble": "lagged_tree_ensemble",
        "lagged_tree": "lagged_tree_ensemble",
        "lag_window_tree": "lagged_tree_ensemble",
        "temporal_tree": "lagged_tree_ensemble",
        "temporal_tree_ensemble": "lagged_tree_ensemble",
        "bagged_tree_ensemble": "bagged_tree_ensemble",
        "tree_ensemble": "bagged_tree_ensemble",
        "tree_ensemble_candidate": "bagged_tree_ensemble",
        "tree": "bagged_tree_ensemble",
        "extra_trees": "bagged_tree_ensemble",
        "boosted_tree_ensemble": "boosted_tree_ensemble",
        "boosted_tree": "boosted_tree_ensemble",
        "gradient_boosting": "boosted_tree_ensemble",
        "hist_gradient_boosting": "boosted_tree_ensemble",
        "gbdt": "boosted_tree_ensemble",
        "boosted_tree_classifier": "boosted_tree_classifier",
        "gradient_boosting_classifier": "boosted_tree_classifier",
        "hist_gradient_boosting_classifier": "boosted_tree_classifier",
        "gbdt_classifier": "boosted_tree_classifier",
    }
    return aliases.get(normalized)


def train_surrogate_candidates(
    *,
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    requested_model_family: str,
    timestamp_column: str | None = None,
    normalize: bool = True,
    missing_data_strategy: str = "fill_median",
    fill_constant_value: float | None = None,
    compare_against_baseline: bool = True,
    lag_horizon_samples: int | None = None,
    threshold_policy: str | None = None,
    decision_threshold: float | None = None,
    task_type: str | None = None,
    run_id: str | None = None,
    checkpoint_tag: str | None = None,
    data_references: list[str] | None = None,
) -> dict[str, Any]:
    """Train a split-safe linear baseline and optional nonlinear comparator."""
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty.")
    _require_columns(frame, list(feature_columns) + [target_column])

    data_mode = _infer_data_mode(frame=frame, timestamp_column=timestamp_column)
    task_profile = assess_task_profile(
        frame=frame,
        target_column=target_column,
        data_mode=data_mode,
        task_type_hint=task_type,
    )
    split = build_train_validation_test_split(
        n_rows=len(frame),
        data_mode=data_mode,
        task_type=task_profile.task_type,
        stratify_labels=frame[target_column] if is_classification_task(task_profile.task_type) else None,
    )
    requested = normalize_candidate_model_family(requested_model_family)
    if requested is None:
        raise ValueError(
            "Requested model is not implemented. "
            "Supported: auto, linear_ridge/ridge/linear, logistic_regression/logistic/linear_classifier, "
            "lagged_logistic_regression/lagged_logistic/temporal_classifier, "
            "lagged_linear/lagged/temporal_linear/arx, "
            "lagged_tree_classifier/temporal_tree_classifier, "
            "lagged_tree_ensemble/lagged_tree/lag_window_tree/temporal_tree, "
            "bagged_tree_ensemble/tree/tree_ensemble/extra_trees, "
            "boosted_tree_ensemble/gradient_boosting/hist_gradient_boosting, "
            "bagged_tree_classifier/tree_classifier/classifier_tree, "
            "boosted_tree_classifier/gradient_boosting_classifier."
        )
    requested = _resolve_requested_for_task(
        requested=requested,
        task_type=task_profile.task_type,
    )
    if is_classification_task(task_profile.task_type):
        return _train_classification_candidates(
            frame=frame,
            target_column=target_column,
            feature_columns=feature_columns,
            requested=requested,
            normalize=normalize,
            missing_data_strategy=missing_data_strategy,
            fill_constant_value=fill_constant_value,
            compare_against_baseline=compare_against_baseline,
            lag_horizon_samples=lag_horizon_samples,
            split=split,
            task_profile=task_profile,
            threshold_policy=threshold_policy,
            decision_threshold=decision_threshold,
            run_id=run_id,
            checkpoint_tag=checkpoint_tag,
            data_references=data_references,
        )
    split_frames = {
        "train": frame.iloc[split.train_indices].reset_index(drop=True),
        "validation": frame.iloc[split.validation_indices].reset_index(drop=True),
        "test": frame.iloc[split.test_indices].reset_index(drop=True),
    }
    prepared = _prepare_split_safe_frames(
        split_frames=split_frames,
        feature_columns=feature_columns,
        target_column=target_column,
        missing_data_strategy=missing_data_strategy,
        fill_constant_value=fill_constant_value,
    )
    prepared_frames = prepared["frames"]
    preprocessing = prepared["preprocessing"]

    normalizer: MinMaxNormalizer | None = None
    if normalize:
        normalizer = MinMaxNormalizer()
        normalizer.fit(prepared_frames["train"], feature_columns=feature_columns)
        prepared_frames = {
            name: normalizer.transform_features(part)
            for name, part in prepared_frames.items()
        }

    linear_model = IncrementalLinearSurrogate(
        feature_columns=feature_columns,
        target_column=target_column,
    )
    linear_rows_used = linear_model.fit_dataframe(prepared_frames["train"])
    rows_used_by_model: dict[str, int] = {"linear_ridge": int(linear_rows_used)}
    linear_candidate = _candidate_metrics_from_model(
        model_family="linear_ridge",
        model=linear_model,
        frames=prepared_frames,
        notes="Split-safe ridge baseline with train-only preprocessing.",
    )

    candidates: list[CandidateMetrics] = [linear_candidate]
    lagged_model: LaggedLinearSurrogate | None = None
    lagged_horizon = _resolve_lag_horizon(
        data_mode=data_mode,
        requested=requested,
        lag_horizon_samples=lag_horizon_samples,
        split=split,
    )
    if requested in {"lagged_linear", "lagged_tree_ensemble"} and lagged_horizon is None:
        raise ValueError(
            "Lagged model families require time-series structure and a usable timestamp column."
        )
    if lagged_horizon is not None and (compare_against_baseline or requested == "lagged_linear"):
        lagged_model = LaggedLinearSurrogate(
            feature_columns=feature_columns,
            target_column=target_column,
            lag_horizon=lagged_horizon,
        )
        lagged_rows_used = lagged_model.fit_dataframe(prepared_frames["train"])
        rows_used_by_model["lagged_linear"] = int(lagged_rows_used)
        candidates.append(
            _candidate_metrics_with_context(
                model_family="lagged_linear",
                model=lagged_model,
                frames=prepared_frames,
                notes=(
                    "Lagged tabular ridge baseline with current and historical predictor windows. "
                    f"Lag horizon={lagged_horizon} samples. Train rows used={lagged_rows_used}."
                ),
            )
        )

    lagged_tree_model: LaggedTreeEnsembleSurrogate | None = None
    if lagged_horizon is not None and (compare_against_baseline or requested == "lagged_tree_ensemble"):
        lagged_tree_model = LaggedTreeEnsembleSurrogate(
            feature_columns=feature_columns,
            target_column=target_column,
            lag_horizon=lagged_horizon,
        )
        lagged_tree_rows_used = lagged_tree_model.fit_dataframe(prepared_frames["train"])
        rows_used_by_model["lagged_tree_ensemble"] = int(lagged_tree_rows_used)
        candidates.append(
            _candidate_metrics_with_context(
                model_family="lagged_tree_ensemble",
                model=lagged_tree_model,
                frames=prepared_frames,
                notes=(
                    "Lag-window bagged depth-limited regression trees over current and historical "
                    f"predictor windows. Lag horizon={lagged_horizon} samples. "
                    f"Train rows used={lagged_tree_rows_used}."
                ),
            )
        )

    tree_model: BaggedTreeEnsembleSurrogate | None = None
    if compare_against_baseline or requested == "bagged_tree_ensemble":
        tree_model = BaggedTreeEnsembleSurrogate(
            feature_columns=feature_columns,
            target_column=target_column,
        )
        tree_rows_used = tree_model.fit_dataframe(prepared_frames["train"])
        rows_used_by_model["bagged_tree_ensemble"] = int(tree_rows_used)
        tree_notes = (
            "Bagged depth-limited regression trees as the first nonlinear local baseline. "
            f"Train rows used={tree_rows_used}."
        )
        candidates.append(
            _candidate_metrics_from_model(
                model_family="bagged_tree_ensemble",
                model=tree_model,
                frames=prepared_frames,
                notes=tree_notes,
            )
        )

    boosted_tree_model: BoostedTreeEnsembleSurrogate | None = None
    if compare_against_baseline or requested == "boosted_tree_ensemble":
        boosted_tree_model = BoostedTreeEnsembleSurrogate(
            feature_columns=feature_columns,
            target_column=target_column,
        )
        boosted_rows_used = boosted_tree_model.fit_dataframe(prepared_frames["train"])
        rows_used_by_model["boosted_tree_ensemble"] = int(boosted_rows_used)
        boosted_notes = (
            "Stage-wise boosted depth-limited regression trees for stronger nonlinear fitting. "
            f"Train rows used={boosted_rows_used}."
        )
        candidates.append(
            _candidate_metrics_from_model(
                model_family="boosted_tree_ensemble",
                model=boosted_tree_model,
                frames=prepared_frames,
                notes=boosted_notes,
            )
        )

    best_by_validation = _select_best_candidate(candidates, task_type=task_profile.task_type)
    selected_candidate = _resolve_selected_candidate(
        requested=requested,
        candidates=candidates,
        task_type=task_profile.task_type,
    )
    selected_model_name = selected_candidate.model_family
    selected_model_obj: Any
    if selected_model_name == "linear_ridge":
        selected_model_obj = linear_model
    elif selected_model_name == "lagged_linear" and lagged_model is not None:
        selected_model_obj = lagged_model
    elif selected_model_name == "lagged_tree_ensemble" and lagged_tree_model is not None:
        selected_model_obj = lagged_tree_model
    elif selected_model_name == "bagged_tree_ensemble" and tree_model is not None:
        selected_model_obj = tree_model
    elif selected_model_name == "boosted_tree_ensemble" and boosted_tree_model is not None:
        selected_model_obj = boosted_tree_model
    else:
        raise RuntimeError("Selected model object is unavailable.")

    artifact_store = ArtifactStore()
    run_dir = artifact_store.create_run_dir(run_id=run_id)
    normalizer_path: str | None = None
    if normalizer is not None:
        normalizer_path = str(artifact_store.save_normalizer(run_dir=run_dir, normalizer=normalizer))

    model_state_path = _save_selected_model_state(
        model=selected_model_obj,
        run_dir=run_dir,
        model_name=selected_model_name,
    )
    selected_hyperparameters = _best_params_payload(
        selected_model_name=selected_model_name,
        linear_rows_used=linear_rows_used,
        lagged_model=lagged_model,
        lagged_tree_model=lagged_tree_model,
        tree_model=tree_model,
        boosted_tree_model=boosted_tree_model,
        requested=requested,
    )
    professional_analysis = _build_regression_professional_analysis(
        selected_model_name=selected_model_name,
        selected_model_obj=selected_model_obj,
        selected_candidate=selected_candidate,
        prepared_frames=prepared_frames,
        feature_columns=feature_columns,
        target_column=target_column,
        lag_horizon_samples=lagged_horizon,
    )
    params_path = artifact_store.save_model_params(
        run_dir=run_dir,
        model_name=selected_model_name,
        best_params=selected_hyperparameters,
        metrics=selected_candidate.test_metrics,
        feature_columns=feature_columns,
        target_column=target_column,
        split_strategy=split.strategy,
        normalizer_path=normalizer_path,
        extra={
            "requested_model_family": requested,
            "selected_model_family": selected_candidate.model_family,
            "best_validation_model_family": best_by_validation.model_family,
            "data_mode": data_mode,
            "lag_horizon_samples": int(lagged_horizon) if lagged_horizon is not None else 0,
            "split": split.to_dict(),
            "preprocessing": preprocessing,
            "comparison": [item.to_dict() for item in candidates],
            "professional_analysis": professional_analysis,
        },
    )
    checkpoint_store = ModelCheckpointStore()
    checkpoint = checkpoint_store.create_checkpoint(
        model_name=selected_model_name,
        run_dir=run_dir,
        model_state_path=model_state_path,
        target_column=target_column,
        feature_columns=feature_columns,
        metrics=selected_candidate.test_metrics,
        data_references=list(data_references or []),
        notes=checkpoint_tag or "",
        tags=[checkpoint_tag] if checkpoint_tag else [],
    )

    return {
        "status": "ok",
        "data_mode": data_mode,
        "task_profile": task_profile.to_dict(),
        "requested_model_family": requested,
        "selected_model_family": selected_candidate.model_family,
        "best_validation_model_family": best_by_validation.model_family,
        "comparison": [item.to_dict() for item in candidates],
        "split": split.to_dict(),
        "preprocessing": preprocessing,
        "normalization": {
            "enabled": bool(normalizer is not None),
            "method": "minmax" if normalizer is not None else "none",
            "normalizer_path": normalizer_path,
        },
        "checkpoint_id": checkpoint.checkpoint_id,
        "run_dir": str(run_dir),
        "model_state_path": str(model_state_path),
        "model_params_path": str(params_path),
        "selected_hyperparameters": selected_hyperparameters,
        "feature_columns": list(feature_columns),
        "lag_horizon_samples": int(lagged_horizon) if lagged_horizon is not None else 0,
        "rows_used": int(rows_used_by_model.get(selected_model_name, prepared_frames["train"].shape[0])),
        "rows_used_by_model": rows_used_by_model,
        "professional_analysis": professional_analysis,
        "selected_metrics": {
            "train": selected_candidate.train_metrics,
            "validation": selected_candidate.validation_metrics,
            "test": selected_candidate.test_metrics,
        },
    }


def _resolve_requested_for_task(*, requested: str, task_type: str) -> str:
    if not is_classification_task(task_type):
        return requested
    if requested in {
        "auto",
        "logistic_regression",
        "lagged_logistic_regression",
        "bagged_tree_classifier",
        "boosted_tree_classifier",
        "lagged_tree_classifier",
    }:
        return requested
    if requested == "linear_ridge":
        return "logistic_regression"
    if requested == "lagged_linear":
        return "lagged_logistic_regression"
    if requested == "bagged_tree_ensemble":
        return "bagged_tree_classifier"
    if requested == "boosted_tree_ensemble":
        return "boosted_tree_classifier"
    if requested == "lagged_tree_ensemble":
        return "lagged_tree_classifier"
    return requested


def _train_classification_candidates(
    *,
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    requested: str,
    normalize: bool,
    missing_data_strategy: str,
    fill_constant_value: float | None,
    compare_against_baseline: bool,
    lag_horizon_samples: int | None,
    split: DatasetSplit,
    task_profile: Any,
    threshold_policy: str | None,
    decision_threshold: float | None,
    run_id: str | None,
    checkpoint_tag: str | None,
    data_references: list[str] | None,
) -> dict[str, Any]:
    split_frames = {
        "train": frame.iloc[split.train_indices].reset_index(drop=True),
        "validation": frame.iloc[split.validation_indices].reset_index(drop=True),
        "test": frame.iloc[split.test_indices].reset_index(drop=True),
    }
    prepared = _prepare_split_safe_frames_classification(
        split_frames=split_frames,
        feature_columns=feature_columns,
        target_column=target_column,
        missing_data_strategy=missing_data_strategy,
        fill_constant_value=fill_constant_value,
    )
    prepared_frames = prepared["frames"]
    preprocessing = prepared["preprocessing"]

    normalizer: MinMaxNormalizer | None = None
    if normalize:
        normalizer = MinMaxNormalizer()
        normalizer.fit(prepared_frames["train"], feature_columns=feature_columns)
        prepared_frames = {
            name: normalizer.transform_features(part)
            for name, part in prepared_frames.items()
        }

    lagged_horizon = _resolve_lag_horizon(
        data_mode="time_series" if split.data_mode == "time_series" else "steady_state",
        requested=requested,
        lag_horizon_samples=lag_horizon_samples,
        split=split,
    )
    if requested in {"lagged_logistic_regression", "lagged_tree_classifier"} and lagged_horizon is None:
        raise ValueError(
            "Lagged classifier families require time-series structure and a usable timestamp column."
        )

    logistic_model = LogisticClassificationSurrogate(
        feature_columns=feature_columns,
        target_column=target_column,
    )
    logistic_rows_used = logistic_model.fit_dataframe(prepared_frames["train"])
    rows_used_by_model: dict[str, int] = {"logistic_regression": int(logistic_rows_used)}
    classifier_thresholds: dict[str, float] = {}
    logistic_candidate = _classification_candidate_metrics_from_model(
        model_family="logistic_regression",
        model=logistic_model,
        frames=prepared_frames,
        task_type=str(task_profile.task_type),
        threshold_policy=threshold_policy,
        decision_threshold=decision_threshold,
        notes=(
            "One-vs-rest logistic classifier as the first split-safe classification baseline. "
            f"Train rows used={logistic_rows_used}."
        ),
    )
    classifier_thresholds["logistic_regression"] = float(
        logistic_candidate.train_metrics.get("decision_threshold", 0.5)
    )

    candidates: list[CandidateMetrics] = [logistic_candidate]
    lagged_logistic_model: LaggedLogisticClassificationSurrogate | None = None
    if lagged_horizon is not None and (compare_against_baseline or requested == "lagged_logistic_regression"):
        lagged_logistic_model = LaggedLogisticClassificationSurrogate(
            feature_columns=feature_columns,
            target_column=target_column,
            lag_horizon=lagged_horizon,
        )
        lagged_logistic_rows = lagged_logistic_model.fit_dataframe(prepared_frames["train"])
        rows_used_by_model["lagged_logistic_regression"] = int(lagged_logistic_rows)
        lagged_logistic_candidate = _classification_candidate_metrics_with_context(
            model_family="lagged_logistic_regression",
            model=lagged_logistic_model,
            frames=prepared_frames,
            task_type=str(task_profile.task_type),
            threshold_policy=threshold_policy,
            decision_threshold=decision_threshold,
            notes=(
                "Lagged logistic classifier over current and historical predictor windows. "
                f"Lag horizon={lagged_horizon} samples. Train rows used={lagged_logistic_rows}."
            ),
        )
        candidates.append(lagged_logistic_candidate)
        classifier_thresholds["lagged_logistic_regression"] = float(
            lagged_logistic_candidate.train_metrics.get("decision_threshold", 0.5)
        )

    tree_classifier: BaggedTreeClassifierSurrogate | None = None
    if compare_against_baseline or requested == "bagged_tree_classifier":
        tree_classifier = BaggedTreeClassifierSurrogate(
            feature_columns=feature_columns,
            target_column=target_column,
        )
        tree_rows_used = tree_classifier.fit_dataframe(prepared_frames["train"])
        rows_used_by_model["bagged_tree_classifier"] = int(tree_rows_used)
        candidates.append(
            _classification_candidate_metrics_from_model(
                model_family="bagged_tree_classifier",
                model=tree_classifier,
                frames=prepared_frames,
                task_type=str(task_profile.task_type),
                threshold_policy=threshold_policy,
                decision_threshold=decision_threshold,
                notes=(
                    "Bagged depth-limited tree classifier as the first nonlinear classification baseline. "
                    f"Train rows used={tree_rows_used}."
                ),
            )
        )
        classifier_thresholds["bagged_tree_classifier"] = float(
            candidates[-1].train_metrics.get("decision_threshold", 0.5)
        )

    boosted_tree_classifier: BoostedTreeClassifierSurrogate | None = None
    if compare_against_baseline or requested == "boosted_tree_classifier":
        boosted_tree_classifier = BoostedTreeClassifierSurrogate(
            feature_columns=feature_columns,
            target_column=target_column,
        )
        boosted_rows_used = boosted_tree_classifier.fit_dataframe(prepared_frames["train"])
        rows_used_by_model["boosted_tree_classifier"] = int(boosted_rows_used)
        boosted_candidate = _classification_candidate_metrics_from_model(
            model_family="boosted_tree_classifier",
            model=boosted_tree_classifier,
            frames=prepared_frames,
            task_type=str(task_profile.task_type),
            threshold_policy=threshold_policy,
            decision_threshold=decision_threshold,
            notes=(
                "Stage-wise boosted depth-limited tree classifier for stronger nonlinear decision boundaries. "
                f"Train rows used={boosted_rows_used}."
            ),
        )
        candidates.append(boosted_candidate)
        classifier_thresholds["boosted_tree_classifier"] = float(
            boosted_candidate.train_metrics.get("decision_threshold", 0.5)
        )

    lagged_tree_classifier: LaggedTreeClassifierSurrogate | None = None
    if lagged_horizon is not None and (compare_against_baseline or requested == "lagged_tree_classifier"):
        lagged_tree_classifier = LaggedTreeClassifierSurrogate(
            feature_columns=feature_columns,
            target_column=target_column,
            lag_horizon=lagged_horizon,
        )
        lagged_tree_rows = lagged_tree_classifier.fit_dataframe(prepared_frames["train"])
        rows_used_by_model["lagged_tree_classifier"] = int(lagged_tree_rows)
        lagged_tree_candidate = _classification_candidate_metrics_with_context(
            model_family="lagged_tree_classifier",
            model=lagged_tree_classifier,
            frames=prepared_frames,
            task_type=str(task_profile.task_type),
            threshold_policy=threshold_policy,
            decision_threshold=decision_threshold,
            notes=(
                "Lag-window bagged tree classifier over current and historical predictor windows. "
                f"Lag horizon={lagged_horizon} samples. Train rows used={lagged_tree_rows}."
            ),
        )
        candidates.append(lagged_tree_candidate)
        classifier_thresholds["lagged_tree_classifier"] = float(
            lagged_tree_candidate.train_metrics.get("decision_threshold", 0.5)
        )

    best_by_validation = _select_best_candidate(candidates, task_type=str(task_profile.task_type))
    selected_candidate = _resolve_selected_candidate(
        requested=requested,
        candidates=candidates,
        task_type=str(task_profile.task_type),
    )
    selected_model_name = selected_candidate.model_family
    if selected_model_name == "logistic_regression":
        selected_model_obj: Any = logistic_model
    elif selected_model_name == "lagged_logistic_regression" and lagged_logistic_model is not None:
        selected_model_obj = lagged_logistic_model
    elif selected_model_name == "bagged_tree_classifier" and tree_classifier is not None:
        selected_model_obj = tree_classifier
    elif selected_model_name == "boosted_tree_classifier" and boosted_tree_classifier is not None:
        selected_model_obj = boosted_tree_classifier
    elif selected_model_name == "lagged_tree_classifier" and lagged_tree_classifier is not None:
        selected_model_obj = lagged_tree_classifier
    else:
        raise RuntimeError("Selected classifier model object is unavailable.")

    artifact_store = ArtifactStore()
    run_dir = artifact_store.create_run_dir(run_id=run_id)
    normalizer_path: str | None = None
    if normalizer is not None:
        normalizer_path = str(artifact_store.save_normalizer(run_dir=run_dir, normalizer=normalizer))

    model_state_path = _save_selected_model_state(
        model=selected_model_obj,
        run_dir=run_dir,
        model_name=selected_model_name,
    )
    selected_hyperparameters = {
        "requested_model_family": requested,
        "task_type": str(task_profile.task_type),
        "threshold_policy": str(threshold_policy or "").strip() or "auto",
    }
    if selected_model_name == "logistic_regression":
        selected_hyperparameters.update(
            {
                "learning_rate": float(logistic_model.learning_rate),
                "epochs": int(logistic_model.epochs),
                "l2": float(logistic_model.l2),
                "training_rows_used": int(logistic_rows_used),
                "class_count": int(len(logistic_model.class_labels)),
                "decision_threshold": float(classifier_thresholds.get("logistic_regression", 0.5)),
            }
        )
    elif selected_model_name == "lagged_logistic_regression" and lagged_logistic_model is not None:
        selected_hyperparameters.update(
            {
                "learning_rate": float(lagged_logistic_model.learning_rate),
                "epochs": int(lagged_logistic_model.epochs),
                "l2": float(lagged_logistic_model.l2),
                "lag_horizon_samples": int(lagged_logistic_model.lag_horizon),
                "training_rows_used": int(rows_used_by_model.get("lagged_logistic_regression", 0)),
                "class_count": int(len(lagged_logistic_model.class_labels)),
                "training_feature_count": int(len(lagged_logistic_model._lagged_feature_columns)),
                "decision_threshold": float(
                    classifier_thresholds.get("lagged_logistic_regression", 0.5)
                ),
            }
        )
    elif selected_model_name == "lagged_tree_classifier" and lagged_tree_classifier is not None:
        selected_hyperparameters.update(
            {
                "n_estimators": int(lagged_tree_classifier.n_estimators),
                "max_depth": int(lagged_tree_classifier.max_depth),
                "min_leaf": int(lagged_tree_classifier.min_leaf),
                "lag_horizon_samples": int(lagged_tree_classifier.lag_horizon),
                "training_rows_used": int(rows_used_by_model.get("lagged_tree_classifier", 0)),
                "class_count": int(len(lagged_tree_classifier.class_labels)),
                "training_feature_count": int(len(lagged_tree_classifier._lagged_feature_columns)),
                "decision_threshold": float(
                    classifier_thresholds.get("lagged_tree_classifier", 0.5)
                ),
            }
        )
    elif selected_model_name == "boosted_tree_classifier" and boosted_tree_classifier is not None:
        selected_hyperparameters.update(
            {
                "n_estimators": int(boosted_tree_classifier.n_estimators),
                "learning_rate": float(boosted_tree_classifier.learning_rate),
                "max_depth": int(boosted_tree_classifier.max_depth),
                "min_leaf": int(boosted_tree_classifier.min_leaf),
                "training_rows_used": int(rows_used_by_model.get("boosted_tree_classifier", 0)),
                "class_count": int(len(boosted_tree_classifier.class_labels)),
                "decision_threshold": float(
                    classifier_thresholds.get("boosted_tree_classifier", 0.5)
                ),
            }
        )
    elif tree_classifier is not None:
        selected_hyperparameters.update(
            {
                "n_estimators": int(tree_classifier.n_estimators),
                "max_depth": int(tree_classifier.max_depth),
                "min_leaf": int(tree_classifier.min_leaf),
                "training_rows_used": int(rows_used_by_model.get("bagged_tree_classifier", 0)),
                "class_count": int(len(tree_classifier.class_labels)),
                "decision_threshold": float(
                    classifier_thresholds.get("bagged_tree_classifier", 0.5)
                ),
            }
        )

    professional_analysis = _build_classification_professional_analysis(
        selected_model_name=selected_model_name,
        selected_model_obj=selected_model_obj,
        selected_candidate=selected_candidate,
        prepared_frames=prepared_frames,
        feature_columns=feature_columns,
        target_column=target_column,
        task_type=str(task_profile.task_type),
    )

    params_path = artifact_store.save_model_params(
        run_dir=run_dir,
        model_name=selected_model_name,
        best_params=selected_hyperparameters,
        metrics=selected_candidate.test_metrics,
        feature_columns=feature_columns,
        target_column=target_column,
        split_strategy=split.strategy,
        normalizer_path=normalizer_path,
        extra={
            "requested_model_family": requested,
            "selected_model_family": selected_candidate.model_family,
            "best_validation_model_family": best_by_validation.model_family,
            "data_mode": "time_series" if split.data_mode == "time_series" else "steady_state",
            "split": split.to_dict(),
            "preprocessing": preprocessing,
            "comparison": [item.to_dict() for item in candidates],
            "task_profile": task_profile.to_dict(),
            "decision_thresholds": classifier_thresholds,
            "threshold_policy": str(threshold_policy or "").strip() or "auto",
            "lag_horizon_samples": int(lagged_horizon) if lagged_horizon is not None else 0,
            "professional_analysis": professional_analysis,
        },
    )
    checkpoint_store = ModelCheckpointStore()
    checkpoint = checkpoint_store.create_checkpoint(
        model_name=selected_model_name,
        run_dir=run_dir,
        model_state_path=model_state_path,
        target_column=target_column,
        feature_columns=feature_columns,
        metrics=selected_candidate.test_metrics,
        data_references=list(data_references or []),
        notes=checkpoint_tag or "",
        tags=[checkpoint_tag] if checkpoint_tag else [],
    )
    return {
        "status": "ok",
        "data_mode": "time_series" if split.data_mode == "time_series" else "steady_state",
        "task_profile": task_profile.to_dict(),
        "requested_model_family": requested,
        "selected_model_family": selected_candidate.model_family,
        "best_validation_model_family": best_by_validation.model_family,
        "comparison": [item.to_dict() for item in candidates],
        "split": split.to_dict(),
        "preprocessing": preprocessing,
        "normalization": {
            "enabled": bool(normalizer is not None),
            "method": "minmax" if normalizer is not None else "none",
            "normalizer_path": normalizer_path,
        },
        "checkpoint_id": checkpoint.checkpoint_id,
        "run_dir": str(run_dir),
        "model_state_path": str(model_state_path),
        "model_params_path": str(params_path),
        "selected_hyperparameters": selected_hyperparameters,
        "feature_columns": list(feature_columns),
        "lag_horizon_samples": int(lagged_horizon) if lagged_horizon is not None else 0,
        "rows_used": int(rows_used_by_model.get(selected_model_name, prepared_frames["train"].shape[0])),
        "rows_used_by_model": rows_used_by_model,
        "professional_analysis": professional_analysis,
        "selected_metrics": {
            "train": selected_candidate.train_metrics,
            "validation": selected_candidate.validation_metrics,
            "test": selected_candidate.test_metrics,
        },
    }


class BaggedTreeEnsembleSurrogate:
    """Small local regression tree ensemble without external dependencies."""

    def __init__(
        self,
        *,
        feature_columns: list[str],
        target_column: str,
        n_estimators: int = 12,
        max_depth: int = 4,
        min_leaf: int = 6,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_leaf = int(min_leaf)
        self._estimators: list[dict[str, Any]] = []

    def fit_dataframe(self, frame: pd.DataFrame) -> int:
        x, y = _prepare_xy(frame=frame, feature_columns=self.feature_columns, target_column=self.target_column)
        if x.shape[0] == 0:
            raise ValueError("No valid numeric rows available for tree training.")
        self._estimators = []
        feature_count = x.shape[1]
        feature_subsample = max(1, int(feature_count))
        for seed in range(self.n_estimators):
            rng = np.random.default_rng(100 + seed)
            row_idx = rng.integers(0, x.shape[0], size=x.shape[0])
            feat_idx = np.sort(
                rng.choice(feature_count, size=feature_subsample, replace=False)
            ).astype(int)
            tree = _fit_regression_tree(
                x_train=x[row_idx][:, feat_idx],
                y_train=y[row_idx],
                depth=0,
                max_depth=self.max_depth,
                min_leaf=min(self.min_leaf, max(2, x.shape[0] // 8)),
            )
            self._estimators.append(
                {
                    "feature_indices": feat_idx.tolist(),
                    "tree": tree,
                }
            )
        return int(x.shape[0])

    def predict_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._estimators:
            raise RuntimeError("Tree ensemble has not been fitted yet.")
        x = _prepare_feature_matrix(frame=frame, feature_columns=self.feature_columns)
        preds = []
        for estimator in self._estimators:
            feat_idx = np.asarray(estimator["feature_indices"], dtype=int)
            pred = _predict_regression_tree_batch(
                tree=estimator["tree"],
                x=x[:, feat_idx],
            )
            preds.append(pred)
        return np.mean(np.vstack(preds), axis=0).astype(float)

    def evaluate_dataframe(self, frame: pd.DataFrame) -> dict[str, float]:
        y_true = _prepare_target_vector(frame=frame, target_column=self.target_column)
        y_pred = self.predict_dataframe(frame)
        return regression_metrics(y_true=y_true, y_pred=y_pred).to_dict()

    def state_dict(self) -> dict[str, Any]:
        if not self._estimators:
            raise RuntimeError("Tree ensemble has not been fitted yet.")
        return {
            "model_name": "bagged_tree_ensemble",
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_leaf": self.min_leaf,
            "estimators": self._estimators,
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return write_json(output, self.state_dict(), indent=2)


class BoostedTreeEnsembleSurrogate:
    """Simple gradient-boosted regression trees with local weak learners."""

    def __init__(
        self,
        *,
        feature_columns: list[str],
        target_column: str,
        n_estimators: int = 28,
        learning_rate: float = 0.25,
        max_depth: int = 3,
        min_leaf: int = 5,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_leaf = int(min_leaf)
        self._bias: float = 0.0
        self._estimators: list[dict[str, Any]] = []

    def fit_dataframe(self, frame: pd.DataFrame) -> int:
        x, y = _prepare_xy(frame=frame, feature_columns=self.feature_columns, target_column=self.target_column)
        if x.shape[0] == 0:
            raise ValueError("No valid numeric rows available for boosted tree training.")
        self._bias = float(np.mean(y))
        residual = y - self._bias
        self._estimators = []
        feature_count = x.shape[1]
        feature_subsample = max(1, int(math.sqrt(feature_count)))
        for step in range(self.n_estimators):
            rng = np.random.default_rng(300 + step)
            feat_idx = np.sort(
                rng.choice(feature_count, size=feature_subsample, replace=False)
            ).astype(int)
            tree = _fit_regression_tree(
                x_train=x[:, feat_idx],
                y_train=residual,
                depth=0,
                max_depth=self.max_depth,
                min_leaf=min(self.min_leaf, max(2, x.shape[0] // 10)),
            )
            update = _predict_regression_tree_batch(tree=tree, x=x[:, feat_idx])
            residual = residual - (self.learning_rate * update)
            self._estimators.append(
                {
                    "feature_indices": feat_idx.tolist(),
                    "tree": tree,
                }
            )
            if float(np.std(residual)) <= 1e-7:
                break
        if not self._estimators:
            raise RuntimeError("Boosted tree ensemble failed to produce any weak learner.")
        return int(x.shape[0])

    def predict_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._estimators:
            raise RuntimeError("Boosted tree ensemble has not been fitted yet.")
        x = _prepare_feature_matrix(frame=frame, feature_columns=self.feature_columns)
        pred = np.full(x.shape[0], self._bias, dtype=float)
        for estimator in self._estimators:
            feat_idx = np.asarray(estimator["feature_indices"], dtype=int)
            pred += self.learning_rate * _predict_regression_tree_batch(
                tree=estimator["tree"],
                x=x[:, feat_idx],
            )
        return pred.astype(float)

    def evaluate_dataframe(self, frame: pd.DataFrame) -> dict[str, float]:
        y_true = _prepare_target_vector(frame=frame, target_column=self.target_column)
        y_pred = self.predict_dataframe(frame)
        return regression_metrics(y_true=y_true, y_pred=y_pred).to_dict()

    def state_dict(self) -> dict[str, Any]:
        if not self._estimators:
            raise RuntimeError("Boosted tree ensemble has not been fitted yet.")
        return {
            "model_name": "boosted_tree_ensemble",
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "n_estimators": int(self.n_estimators),
            "learning_rate": float(self.learning_rate),
            "max_depth": int(self.max_depth),
            "min_leaf": int(self.min_leaf),
            "bias": float(self._bias),
            "estimators": self._estimators,
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return write_json(output, self.state_dict(), indent=2)


class LaggedLinearSurrogate:
    """Linear surrogate over current and lagged predictor windows."""

    def __init__(
        self,
        *,
        feature_columns: list[str],
        target_column: str,
        lag_horizon: int = 3,
        ridge: float = 1e-8,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        if int(lag_horizon) <= 0:
            raise ValueError("lag_horizon must be > 0.")
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.lag_horizon = int(lag_horizon)
        self.ridge = float(ridge)
        self._lagged_feature_columns = _lagged_feature_names(
            feature_columns=self.feature_columns,
            lag_horizon=self.lag_horizon,
        )
        self._delegate: IncrementalLinearSurrogate | None = None

    def fit_dataframe(self, frame: pd.DataFrame) -> int:
        lagged = _build_lagged_design_frame(
            frame=frame,
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            lag_horizon=self.lag_horizon,
        )
        if lagged.shape[0] == 0:
            raise ValueError("No valid rows available after lagged feature construction.")
        self._delegate = IncrementalLinearSurrogate(
            feature_columns=list(self._lagged_feature_columns),
            target_column=self.target_column,
            ridge=self.ridge,
        )
        return self._delegate.fit_dataframe(lagged)

    def predict_dataframe(
        self,
        frame: pd.DataFrame,
        *,
        context_frame: pd.DataFrame | None = None,
    ) -> np.ndarray:
        lagged = self._lagged_frame(frame=frame, context_frame=context_frame)
        if lagged.shape[0] == 0:
            raise ValueError("Lagged prediction frame is empty for the requested split.")
        return self._require_delegate().predict_dataframe(lagged)

    def evaluate_dataframe(
        self,
        frame: pd.DataFrame,
        *,
        context_frame: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        lagged = self._lagged_frame(frame=frame, context_frame=context_frame)
        if lagged.shape[0] == 0:
            raise ValueError("Lagged evaluation frame is empty for the requested split.")
        return self._require_delegate().evaluate_dataframe(lagged)

    def state_dict(self) -> dict[str, Any]:
        delegate = self._require_delegate()
        return {
            "model_name": "lagged_linear",
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "lag_horizon": int(self.lag_horizon),
            "ridge": float(self.ridge),
            "lagged_feature_columns": list(self._lagged_feature_columns),
            "linear_state": delegate.state_dict(),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return write_json(output, self.state_dict(), indent=2)

    def _lagged_frame(
        self,
        *,
        frame: pd.DataFrame,
        context_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return _build_lagged_design_frame(
            frame=frame,
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            lag_horizon=self.lag_horizon,
            context_frame=context_frame,
        )

    def _require_delegate(self) -> IncrementalLinearSurrogate:
        if self._delegate is None:
            raise RuntimeError("Lagged linear surrogate has not been fitted yet.")
        return self._delegate


class LaggedTreeEnsembleSurrogate:
    """Tree ensemble over current and lagged predictor windows."""

    def __init__(
        self,
        *,
        feature_columns: list[str],
        target_column: str,
        lag_horizon: int = 3,
        n_estimators: int = 12,
        max_depth: int = 4,
        min_leaf: int = 6,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        if int(lag_horizon) <= 0:
            raise ValueError("lag_horizon must be > 0.")
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.lag_horizon = int(lag_horizon)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_leaf = int(min_leaf)
        self._lagged_feature_columns = _lagged_feature_names(
            feature_columns=self.feature_columns,
            lag_horizon=self.lag_horizon,
        )
        self._delegate: BaggedTreeEnsembleSurrogate | None = None

    def fit_dataframe(self, frame: pd.DataFrame) -> int:
        lagged = _build_lagged_design_frame(
            frame=frame,
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            lag_horizon=self.lag_horizon,
        )
        if lagged.shape[0] == 0:
            raise ValueError("No valid rows available after lagged feature construction.")
        self._delegate = BaggedTreeEnsembleSurrogate(
            feature_columns=list(self._lagged_feature_columns),
            target_column=self.target_column,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_leaf=self.min_leaf,
        )
        return self._delegate.fit_dataframe(lagged)

    def predict_dataframe(
        self,
        frame: pd.DataFrame,
        *,
        context_frame: pd.DataFrame | None = None,
    ) -> np.ndarray:
        lagged = self._lagged_frame(frame=frame, context_frame=context_frame)
        if lagged.shape[0] == 0:
            raise ValueError("Lagged prediction frame is empty for the requested split.")
        return self._require_delegate().predict_dataframe(lagged)

    def evaluate_dataframe(
        self,
        frame: pd.DataFrame,
        *,
        context_frame: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        lagged = self._lagged_frame(frame=frame, context_frame=context_frame)
        if lagged.shape[0] == 0:
            raise ValueError("Lagged evaluation frame is empty for the requested split.")
        return self._require_delegate().evaluate_dataframe(lagged)

    def state_dict(self) -> dict[str, Any]:
        delegate = self._require_delegate()
        return {
            "model_name": "lagged_tree_ensemble",
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "lag_horizon": int(self.lag_horizon),
            "n_estimators": int(self.n_estimators),
            "max_depth": int(self.max_depth),
            "min_leaf": int(self.min_leaf),
            "lagged_feature_columns": list(self._lagged_feature_columns),
            "tree_state": delegate.state_dict(),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return write_json(output, self.state_dict(), indent=2)

    def _lagged_frame(
        self,
        *,
        frame: pd.DataFrame,
        context_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return _build_lagged_design_frame(
            frame=frame,
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            lag_horizon=self.lag_horizon,
            context_frame=context_frame,
        )

    def _require_delegate(self) -> BaggedTreeEnsembleSurrogate:
        if self._delegate is None:
            raise RuntimeError("Lagged tree ensemble has not been fitted yet.")
        return self._delegate


class LaggedLogisticClassificationSurrogate:
    """Logistic classifier over current and lagged predictor windows."""

    def __init__(
        self,
        *,
        feature_columns: list[str],
        target_column: str,
        lag_horizon: int = 3,
        learning_rate: float = 0.25,
        epochs: int = 350,
        l2: float = 1e-4,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        if int(lag_horizon) <= 0:
            raise ValueError("lag_horizon must be > 0.")
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.lag_horizon = int(lag_horizon)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self._lagged_feature_columns = _lagged_feature_names(
            feature_columns=self.feature_columns,
            lag_horizon=self.lag_horizon,
        )
        self._delegate: LogisticClassificationSurrogate | None = None

    @property
    def class_labels(self) -> list[str]:
        return list(self._require_delegate().class_labels)

    def fit_dataframe(self, frame: pd.DataFrame) -> int:
        lagged = self._lagged_frame(frame=frame)
        if lagged.shape[0] == 0:
            raise ValueError("No valid rows available after lagged feature construction.")
        self._delegate = LogisticClassificationSurrogate(
            feature_columns=list(self._lagged_feature_columns),
            target_column=self.target_column,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            l2=self.l2,
        )
        return self._delegate.fit_dataframe(lagged)

    def predict_proba_dataframe(
        self,
        frame: pd.DataFrame,
        *,
        context_frame: pd.DataFrame | None = None,
    ) -> np.ndarray:
        lagged = self._lagged_frame(frame=frame, context_frame=context_frame)
        if lagged.shape[0] == 0:
            raise ValueError("Lagged prediction frame is empty for the requested split.")
        return self._require_delegate().predict_proba_dataframe(lagged)

    def predict_dataframe(
        self,
        frame: pd.DataFrame,
        *,
        context_frame: pd.DataFrame | None = None,
    ) -> np.ndarray:
        lagged = self._lagged_frame(frame=frame, context_frame=context_frame)
        if lagged.shape[0] == 0:
            raise ValueError("Lagged prediction frame is empty for the requested split.")
        return self._require_delegate().predict_dataframe(lagged)

    def state_dict(self) -> dict[str, Any]:
        delegate = self._require_delegate()
        return {
            "model_name": "lagged_logistic_regression",
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "lag_horizon": int(self.lag_horizon),
            "learning_rate": float(self.learning_rate),
            "epochs": int(self.epochs),
            "l2": float(self.l2),
            "lagged_feature_columns": list(self._lagged_feature_columns),
            "logistic_state": delegate.state_dict(),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return write_json(output, self.state_dict(), indent=2)

    def _lagged_frame(
        self,
        *,
        frame: pd.DataFrame,
        context_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return _build_lagged_design_frame(
            frame=frame,
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            lag_horizon=self.lag_horizon,
            context_frame=context_frame,
        )

    def _require_delegate(self) -> LogisticClassificationSurrogate:
        if self._delegate is None:
            raise RuntimeError("Lagged logistic classifier has not been fitted yet.")
        return self._delegate


class LaggedTreeClassifierSurrogate:
    """Bagged tree classifier over current and lagged predictor windows."""

    def __init__(
        self,
        *,
        feature_columns: list[str],
        target_column: str,
        lag_horizon: int = 3,
        n_estimators: int = 12,
        max_depth: int = 4,
        min_leaf: int = 6,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        if int(lag_horizon) <= 0:
            raise ValueError("lag_horizon must be > 0.")
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.lag_horizon = int(lag_horizon)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_leaf = int(min_leaf)
        self._lagged_feature_columns = _lagged_feature_names(
            feature_columns=self.feature_columns,
            lag_horizon=self.lag_horizon,
        )
        self._delegate: BaggedTreeClassifierSurrogate | None = None

    @property
    def class_labels(self) -> list[str]:
        return list(self._require_delegate().class_labels)

    def fit_dataframe(self, frame: pd.DataFrame) -> int:
        lagged = self._lagged_frame(frame=frame)
        if lagged.shape[0] == 0:
            raise ValueError("No valid rows available after lagged feature construction.")
        self._delegate = BaggedTreeClassifierSurrogate(
            feature_columns=list(self._lagged_feature_columns),
            target_column=self.target_column,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_leaf=self.min_leaf,
        )
        return self._delegate.fit_dataframe(lagged)

    def predict_proba_dataframe(
        self,
        frame: pd.DataFrame,
        *,
        context_frame: pd.DataFrame | None = None,
    ) -> np.ndarray:
        lagged = self._lagged_frame(frame=frame, context_frame=context_frame)
        if lagged.shape[0] == 0:
            raise ValueError("Lagged prediction frame is empty for the requested split.")
        return self._require_delegate().predict_proba_dataframe(lagged)

    def predict_dataframe(
        self,
        frame: pd.DataFrame,
        *,
        context_frame: pd.DataFrame | None = None,
    ) -> np.ndarray:
        lagged = self._lagged_frame(frame=frame, context_frame=context_frame)
        if lagged.shape[0] == 0:
            raise ValueError("Lagged prediction frame is empty for the requested split.")
        return self._require_delegate().predict_dataframe(lagged)

    def state_dict(self) -> dict[str, Any]:
        delegate = self._require_delegate()
        return {
            "model_name": "lagged_tree_classifier",
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "lag_horizon": int(self.lag_horizon),
            "n_estimators": int(self.n_estimators),
            "max_depth": int(self.max_depth),
            "min_leaf": int(self.min_leaf),
            "lagged_feature_columns": list(self._lagged_feature_columns),
            "tree_state": delegate.state_dict(),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return write_json(output, self.state_dict(), indent=2)

    def _lagged_frame(
        self,
        *,
        frame: pd.DataFrame,
        context_frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        return _build_lagged_design_frame(
            frame=frame,
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            lag_horizon=self.lag_horizon,
            context_frame=context_frame,
        )

    def _require_delegate(self) -> BaggedTreeClassifierSurrogate:
        if self._delegate is None:
            raise RuntimeError("Lagged tree classifier has not been fitted yet.")
        return self._delegate


def _candidate_metrics_from_model(
    *,
    model_family: str,
    model: Any,
    frames: dict[str, pd.DataFrame],
    notes: str,
) -> CandidateMetrics:
    return CandidateMetrics(
        model_family=model_family,
        train_metrics={k: float(v) for k, v in model.evaluate_dataframe(frames["train"]).items()},
        validation_metrics={
            k: float(v) for k, v in model.evaluate_dataframe(frames["validation"]).items()
        },
        test_metrics={k: float(v) for k, v in model.evaluate_dataframe(frames["test"]).items()},
        notes=notes,
    )


def _classification_candidate_metrics_from_model(
    *,
    model_family: str,
    model: Any,
    frames: dict[str, pd.DataFrame],
    task_type: str,
    threshold_policy: str | None,
    decision_threshold: float | None,
    notes: str,
) -> CandidateMetrics:
    class_labels = [str(item) for item in getattr(model, "class_labels", [])]
    target_column = str(getattr(model, "target_column", "")).strip()
    if not class_labels or not target_column:
        raise RuntimeError("Classification candidate is missing class labels or target column.")
    positive_label = _minority_label_token(
        frame=frames["train"],
        target_column=target_column,
    )
    decision_threshold = _select_binary_decision_threshold(
        model=model,
        validation_frame=frames["validation"],
        class_labels=class_labels,
        positive_label=positive_label,
        task_type=task_type,
        threshold_policy=threshold_policy,
        explicit_threshold=decision_threshold,
    )
    train_metrics = _classification_metrics_for_frame(
        model=model,
        frame=frames["train"],
        class_labels=class_labels,
        positive_label=positive_label,
        decision_threshold=decision_threshold,
    )
    validation_metrics = _classification_metrics_for_frame(
        model=model,
        frame=frames["validation"],
        class_labels=class_labels,
        positive_label=positive_label,
        decision_threshold=decision_threshold,
    )
    test_metrics = _classification_metrics_for_frame(
        model=model,
        frame=frames["test"],
        class_labels=class_labels,
        positive_label=positive_label,
        decision_threshold=decision_threshold,
    )
    for bundle in (train_metrics, validation_metrics, test_metrics):
        bundle["decision_threshold"] = float(decision_threshold)
    return CandidateMetrics(
        model_family=model_family,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        notes=notes,
    )


def _classification_candidate_metrics_with_context(
    *,
    model_family: str,
    model: Any,
    frames: dict[str, pd.DataFrame],
    task_type: str,
    threshold_policy: str | None,
    decision_threshold: float | None,
    notes: str,
) -> CandidateMetrics:
    train_design = model._lagged_frame(frame=frames["train"])
    validation_design = model._lagged_frame(
        frame=frames["validation"],
        context_frame=frames["train"],
    )
    test_design = model._lagged_frame(
        frame=frames["test"],
        context_frame=pd.concat([frames["train"], frames["validation"]], ignore_index=True),
    )
    if train_design.shape[0] == 0 or validation_design.shape[0] == 0 or test_design.shape[0] == 0:
        raise ValueError("Lagged classification design frame is empty for one or more splits.")

    delegate_model = model._require_delegate()
    class_labels = [str(item) for item in getattr(model, "class_labels", [])]
    target_column = str(getattr(model, "target_column", "")).strip()
    if not class_labels or not target_column:
        raise RuntimeError("Lagged classification candidate is missing class labels or target column.")
    positive_label = _minority_label_token(
        frame=train_design,
        target_column=target_column,
    )
    resolved_threshold = _select_binary_decision_threshold(
        model=delegate_model,
        validation_frame=validation_design,
        class_labels=class_labels,
        positive_label=positive_label,
        task_type=task_type,
        threshold_policy=threshold_policy,
        explicit_threshold=decision_threshold,
    )
    train_metrics = _classification_metrics_for_frame(
        model=delegate_model,
        frame=train_design,
        class_labels=class_labels,
        positive_label=positive_label,
        decision_threshold=resolved_threshold,
    )
    validation_metrics = _classification_metrics_for_frame(
        model=delegate_model,
        frame=validation_design,
        class_labels=class_labels,
        positive_label=positive_label,
        decision_threshold=resolved_threshold,
    )
    test_metrics = _classification_metrics_for_frame(
        model=delegate_model,
        frame=test_design,
        class_labels=class_labels,
        positive_label=positive_label,
        decision_threshold=resolved_threshold,
    )
    for bundle in (train_metrics, validation_metrics, test_metrics):
        bundle["decision_threshold"] = float(resolved_threshold)
    return CandidateMetrics(
        model_family=model_family,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        notes=notes,
    )


def _candidate_metrics_with_context(
    *,
    model_family: str,
    model: Any,
    frames: dict[str, pd.DataFrame],
    notes: str,
) -> CandidateMetrics:
    train_metrics = model.evaluate_dataframe(frames["train"])
    validation_metrics = model.evaluate_dataframe(
        frames["validation"],
        context_frame=frames["train"],
    )
    test_metrics = model.evaluate_dataframe(
        frames["test"],
        context_frame=pd.concat([frames["train"], frames["validation"]], ignore_index=True),
    )
    return CandidateMetrics(
        model_family=model_family,
        train_metrics={k: float(v) for k, v in train_metrics.items()},
        validation_metrics={k: float(v) for k, v in validation_metrics.items()},
        test_metrics={k: float(v) for k, v in test_metrics.items()},
        notes=notes,
    )


def _classification_metrics_for_frame(
    *,
    model: Any,
    frame: pd.DataFrame,
    class_labels: list[str],
    positive_label: str | None,
    decision_threshold: float,
) -> dict[str, float]:
    y_true = [str(item) for item in frame[str(getattr(model, "target_column", ""))].tolist()]
    probabilities = model.predict_proba_dataframe(frame)
    metrics = classification_metrics(
        y_true=y_true,
        probabilities=probabilities,
        class_labels=class_labels,
        positive_label=positive_label,
        decision_threshold=decision_threshold if len(class_labels) == 2 else None,
    ).to_dict()
    return {str(key): float(value) for key, value in metrics.items()}


def _minority_label_token(*, frame: pd.DataFrame, target_column: str) -> str | None:
    if target_column not in frame.columns:
        return None
    counts = frame[target_column].astype(str).value_counts()
    if counts.empty:
        return None
    return str(counts.sort_values(kind="stable").index[0])


def _select_binary_decision_threshold(
    *,
    model: Any,
    validation_frame: pd.DataFrame,
    class_labels: list[str],
    positive_label: str | None,
    task_type: str,
    threshold_policy: str | None,
    explicit_threshold: float | None,
) -> float:
    if len(class_labels) != 2 or not positive_label:
        return 0.5
    if explicit_threshold is not None:
        return float(max(0.01, min(0.99, explicit_threshold)))
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    best_threshold = 0.5
    best_rank: tuple[float, ...] | None = None
    normalized_policy = str(threshold_policy or "").strip().lower() or "auto"
    y_true = [str(item) for item in validation_frame[str(getattr(model, "target_column", ""))].tolist()]
    probabilities = model.predict_proba_dataframe(validation_frame)
    for threshold in thresholds:
        metrics = classification_metrics(
            y_true=y_true,
            probabilities=probabilities,
            class_labels=class_labels,
            positive_label=positive_label,
            decision_threshold=threshold,
        ).to_dict()
        if normalized_policy == "favor_recall":
            rank = (
                -float(metrics.get("recall", 0.0)),
                -float(metrics.get("f1", 0.0)),
                -float(metrics.get("pr_auc", 0.0)),
                -float(metrics.get("precision", 0.0)),
                float(metrics.get("log_loss", float("inf"))),
            )
        elif normalized_policy == "favor_precision":
            rank = (
                -float(metrics.get("precision", 0.0)),
                -float(metrics.get("f1", 0.0)),
                -float(metrics.get("recall", 0.0)),
                -float(metrics.get("pr_auc", 0.0)),
                float(metrics.get("log_loss", float("inf"))),
            )
        elif normalized_policy == "favor_pr_auc":
            rank = (
                -float(metrics.get("pr_auc", 0.0)),
                -float(metrics.get("recall", 0.0)),
                -float(metrics.get("f1", 0.0)),
                -float(metrics.get("precision", 0.0)),
                float(metrics.get("log_loss", float("inf"))),
            )
        elif normalized_policy == "favor_f1":
            rank = (
                -float(metrics.get("f1", 0.0)),
                -float(metrics.get("recall", 0.0)),
                -float(metrics.get("precision", 0.0)),
                -float(metrics.get("pr_auc", 0.0)),
                float(metrics.get("log_loss", float("inf"))),
            )
        elif task_type in {"fraud_detection", "anomaly_detection"}:
            rank = (
                -float(metrics.get("recall", 0.0)),
                -float(metrics.get("pr_auc", 0.0)),
                -float(metrics.get("f1", 0.0)),
                -float(metrics.get("precision", 0.0)),
                float(metrics.get("log_loss", float("inf"))),
            )
        else:
            rank = (
                -float(metrics.get("f1", 0.0)),
                -float(metrics.get("accuracy", 0.0)),
                -float(metrics.get("precision", 0.0)),
                -float(metrics.get("recall", 0.0)),
                float(metrics.get("log_loss", float("inf"))),
            )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_threshold = float(threshold)
    return best_threshold


def _select_best_candidate(
    candidates: list[CandidateMetrics],
    *,
    task_type: str,
) -> CandidateMetrics:
    ranked = sorted(
        candidates,
        key=lambda item: (
            _candidate_validation_rank(item=item, task_type=task_type),
            _MODEL_TIEBREAK.get(item.model_family, 9),
        ),
    )
    return ranked[0]


def _resolve_selected_candidate(
    *,
    requested: str,
    candidates: list[CandidateMetrics],
    task_type: str,
) -> CandidateMetrics:
    if requested == "auto":
        return _select_best_candidate(candidates, task_type=task_type)
    for item in candidates:
        if item.model_family == requested:
            return item
    return _select_best_candidate(candidates, task_type=task_type)


def _candidate_validation_rank(*, item: CandidateMetrics, task_type: str) -> tuple[float, ...]:
    metrics = item.validation_metrics
    if is_classification_task(task_type):
        if task_type in {"fraud_detection", "anomaly_detection"}:
            return (
                -float(metrics.get("pr_auc", 0.0)),
                -float(metrics.get("recall", 0.0)),
                -float(metrics.get("f1", 0.0)),
                -float(metrics.get("accuracy", 0.0)),
                float(metrics.get("log_loss", float("inf"))),
            )
        return (
            -float(metrics.get("f1", 0.0)),
            -float(metrics.get("accuracy", 0.0)),
            -float(metrics.get("precision", 0.0)),
            -float(metrics.get("recall", 0.0)),
            float(metrics.get("log_loss", float("inf"))),
        )
    return (
        float(metrics.get("mae", float("inf"))),
        float(metrics.get("rmse", float("inf"))),
    )


def _save_selected_model_state(*, model: Any, run_dir: Path, model_name: str) -> Path:
    filename = "model_state.json" if model_name == "linear_ridge" else f"{model_name}_state.json"
    path = Path(run_dir) / filename
    return model.save(path)


def _best_params_payload(
    *,
    selected_model_name: str,
    linear_rows_used: int,
    lagged_model: LaggedLinearSurrogate | None,
    lagged_tree_model: LaggedTreeEnsembleSurrogate | None,
    tree_model: BaggedTreeEnsembleSurrogate | None,
    boosted_tree_model: BoostedTreeEnsembleSurrogate | None,
    requested: str,
) -> dict[str, Any]:
    if selected_model_name == "linear_ridge":
        return {
            "requested_model_family": requested,
            "ridge": 1e-8,
            "training_rows_used": int(linear_rows_used),
        }
    if selected_model_name == "lagged_linear" and lagged_model is not None:
        return {
            "requested_model_family": requested,
            "ridge": float(lagged_model.ridge),
            "lag_horizon_samples": int(lagged_model.lag_horizon),
            "training_feature_count": int(len(lagged_model._lagged_feature_columns)),
        }
    if selected_model_name == "lagged_tree_ensemble" and lagged_tree_model is not None:
        return {
            "requested_model_family": requested,
            "lag_horizon_samples": int(lagged_tree_model.lag_horizon),
            "n_estimators": int(lagged_tree_model.n_estimators),
            "max_depth": int(lagged_tree_model.max_depth),
            "min_leaf": int(lagged_tree_model.min_leaf),
            "training_feature_count": int(len(lagged_tree_model._lagged_feature_columns)),
        }
    if selected_model_name == "boosted_tree_ensemble" and boosted_tree_model is not None:
        return {
            "requested_model_family": requested,
            "n_estimators": int(boosted_tree_model.n_estimators),
            "learning_rate": float(boosted_tree_model.learning_rate),
            "max_depth": int(boosted_tree_model.max_depth),
            "min_leaf": int(boosted_tree_model.min_leaf),
        }
    if tree_model is not None:
        return {
            "requested_model_family": requested,
            "n_estimators": int(tree_model.n_estimators),
            "max_depth": int(tree_model.max_depth),
            "min_leaf": int(tree_model.min_leaf),
        }
    return {"requested_model_family": requested}


def _build_regression_professional_analysis(
    *,
    selected_model_name: str,
    selected_model_obj: Any,
    selected_candidate: CandidateMetrics,
    prepared_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    target_column: str,
    lag_horizon_samples: int | None,
) -> dict[str, Any]:
    train = selected_candidate.train_metrics
    validation = selected_candidate.validation_metrics
    test = selected_candidate.test_metrics
    train_mae = _metric_or_none(train, "mae")
    val_mae = _metric_or_none(validation, "mae")
    train_r2 = _metric_or_none(train, "r2")
    val_r2 = _metric_or_none(validation, "r2")
    test_r2 = _metric_or_none(test, "r2")

    diagnostics: list[str] = []
    risk_flags: list[str] = []
    if train_mae is not None and val_mae is not None:
        ratio = val_mae / max(train_mae, 1e-9)
        diagnostics.append(
            f"Generalization gap (MAE): train={train_mae:.4f}, val={val_mae:.4f}, ratio={ratio:.3f}."
        )
        if ratio > 1.25:
            risk_flags.append("overfitting_risk_high")
    if train_r2 is not None and val_r2 is not None:
        diagnostics.append(
            f"Generalization gap (R2): train={train_r2:.4f}, val={val_r2:.4f}, delta={train_r2 - val_r2:.4f}."
        )
    if test_r2 is not None and test_r2 < 0.50:
        risk_flags.append("underfitting_or_coverage_gap")

    feature_eval, y_true, y_pred = _extract_regression_eval_payload(
        selected_model_name=selected_model_name,
        selected_model_obj=selected_model_obj,
        prepared_frames=prepared_frames,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    feedback = analyze_model_performance(
        y_true=y_true,
        y_pred=y_pred,
        feature_frame=feature_eval,
        top_k_regions=3,
        trajectory_budget=3,
    )
    feedback_payload = feedback.to_dict()
    residual = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    diagnostics.append(
        f"Residual check: mean={float(np.mean(residual)):.4f}, std={float(np.std(residual)):.4f}, abs_max={float(np.max(np.abs(residual))):.4f}."
    )

    suggestions = [
        str(item.get("rationale", "")).strip()
        for item in feedback_payload.get("trajectory_suggestions", [])
        if isinstance(item, dict) and str(item.get("rationale", "")).strip()
    ]
    summary = (
        f"{feedback_payload.get('summary', 'Regression evaluation complete.')} "
        f"Selected model `{selected_model_name}` with lag_horizon={int(lag_horizon_samples or 0)}."
    )
    return {
        "summary": summary.strip(),
        "diagnostics": diagnostics,
        "risk_flags": risk_flags,
        "high_error_regions": feedback_payload.get("bad_regions", []),
        "suggestions": suggestions[:3],
    }


def _build_classification_professional_analysis(
    *,
    selected_model_name: str,
    selected_model_obj: Any,
    selected_candidate: CandidateMetrics,
    prepared_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    target_column: str,
    task_type: str,
) -> dict[str, Any]:
    train = selected_candidate.train_metrics
    validation = selected_candidate.validation_metrics
    test = selected_candidate.test_metrics
    train_f1 = _metric_or_none(train, "f1")
    val_f1 = _metric_or_none(validation, "f1")
    train_recall = _metric_or_none(train, "recall")
    val_recall = _metric_or_none(validation, "recall")
    test_recall = _metric_or_none(test, "recall")
    test_pr_auc = _metric_or_none(test, "pr_auc")

    diagnostics: list[str] = []
    risk_flags: list[str] = []
    if train_f1 is not None and val_f1 is not None:
        diagnostics.append(
            f"Generalization gap (F1): train={train_f1:.4f}, val={val_f1:.4f}, delta={train_f1 - val_f1:.4f}."
        )
        if train_f1 > max(val_f1 + 0.10, val_f1 * 1.10):
            risk_flags.append("overfitting_risk_high")
    if train_recall is not None and val_recall is not None:
        diagnostics.append(
            f"Minority recall stability: train={train_recall:.4f}, val={val_recall:.4f}."
        )

    feature_eval, y_true, y_pred, probs, class_labels, decision_threshold = _extract_classification_eval_payload(
        selected_model_name=selected_model_name,
        selected_model_obj=selected_model_obj,
        selected_candidate=selected_candidate,
        prepared_frames=prepared_frames,
        feature_columns=feature_columns,
        target_column=target_column,
    )
    regions = _analyze_classification_error_regions(
        feature_frame=feature_eval,
        y_true=y_true,
        y_pred=y_pred,
        max_regions=3,
    )
    suggestions: list[str] = []
    for region in regions:
        suggestions.append(
            f"Increase labeled coverage in {region['feature']} range "
            f"[{region['lower_bound']:.3g}, {region['upper_bound']:.3g}] where error rate is "
            f"{region['error_rate']:.3f} (lift {region['error_lift']:.3f})."
        )
    if task_type in {"fraud_detection", "anomaly_detection"}:
        if test_recall is not None and test_recall < 0.70:
            risk_flags.append("minority_recall_below_target")
            suggestions.append("Collect more positive-event sequences and hard negatives near the decision boundary.")
        if test_pr_auc is not None and test_pr_auc < 0.35:
            risk_flags.append("precision_recall_separation_weak")
            suggestions.append("Add discriminative predictors and targeted event windows to improve PR-AUC.")

    if not suggestions:
        suggestions.append(
            "Collect additional boundary-crossing examples in underrepresented operating regions before increasing model complexity."
        )

    avg_conf = float(np.mean(np.max(probs, axis=1))) if probs.size else float("nan")
    diagnostics.append(
        f"Prediction confidence: mean_max_probability={avg_conf:.4f}, classes={len(class_labels)}, decision_threshold={decision_threshold:.2f}."
    )
    summary = (
        f"Classification evaluation complete for `{selected_model_name}`: "
        f"test_f1={_metric_or_none(test, 'f1') if _metric_or_none(test, 'f1') is not None else float('nan'):.4f}, "
        f"test_accuracy={_metric_or_none(test, 'accuracy') if _metric_or_none(test, 'accuracy') is not None else float('nan'):.4f}, "
        f"test_recall={_metric_or_none(test, 'recall') if _metric_or_none(test, 'recall') is not None else float('nan'):.4f}."
    )
    return {
        "summary": summary,
        "diagnostics": diagnostics,
        "risk_flags": risk_flags,
        "high_error_regions": regions,
        "suggestions": suggestions[:3],
    }


def _extract_regression_eval_payload(
    *,
    selected_model_name: str,
    selected_model_obj: Any,
    prepared_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if selected_model_name in {"lagged_linear", "lagged_tree_ensemble"} and hasattr(selected_model_obj, "_lagged_frame"):
        context = pd.concat([prepared_frames["train"], prepared_frames["validation"]], ignore_index=True)
        eval_frame = selected_model_obj._lagged_frame(
            frame=prepared_frames["test"],
            context_frame=context,
        )
        delegate = selected_model_obj._require_delegate()
        model_features = list(getattr(selected_model_obj, "_lagged_feature_columns", []))
        y_true = eval_frame[target_column].to_numpy(dtype=float)
        y_pred = delegate.predict_dataframe(eval_frame)
        return eval_frame[model_features].copy(), y_true, np.asarray(y_pred, dtype=float)

    eval_frame = prepared_frames["test"]
    y_true = eval_frame[target_column].to_numpy(dtype=float)
    y_pred = selected_model_obj.predict_dataframe(eval_frame)
    return eval_frame[feature_columns].copy(), y_true, np.asarray(y_pred, dtype=float)


def _extract_classification_eval_payload(
    *,
    selected_model_name: str,
    selected_model_obj: Any,
    selected_candidate: CandidateMetrics,
    prepared_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, list[str], list[str], np.ndarray, list[str], float]:
    if selected_model_name in {"lagged_logistic_regression", "lagged_tree_classifier"} and hasattr(selected_model_obj, "_lagged_frame"):
        context = pd.concat([prepared_frames["train"], prepared_frames["validation"]], ignore_index=True)
        eval_frame = selected_model_obj._lagged_frame(
            frame=prepared_frames["test"],
            context_frame=context,
        )
        delegate = selected_model_obj._require_delegate()
        model_features = list(getattr(selected_model_obj, "_lagged_feature_columns", []))
        probabilities = delegate.predict_proba_dataframe(eval_frame)
    else:
        eval_frame = prepared_frames["test"]
        model_features = list(feature_columns)
        probabilities = selected_model_obj.predict_proba_dataframe(eval_frame)
    class_labels = [str(item) for item in getattr(selected_model_obj, "class_labels", [])]
    y_true = [str(item) for item in eval_frame[target_column].tolist()]
    positive_label = _minority_label_token(frame=prepared_frames["train"], target_column=target_column)
    decision_threshold = float(selected_candidate.test_metrics.get("decision_threshold", 0.5))
    y_pred = _predict_labels_from_probabilities(
        probabilities=probabilities,
        class_labels=class_labels,
        positive_label=positive_label,
        decision_threshold=decision_threshold,
    )
    return eval_frame[model_features].copy(), y_true, y_pred, probabilities, class_labels, decision_threshold


def _predict_labels_from_probabilities(
    *,
    probabilities: np.ndarray,
    class_labels: list[str],
    positive_label: str | None,
    decision_threshold: float,
) -> list[str]:
    if probabilities.size == 0 or not class_labels:
        return []
    if len(class_labels) == 2 and positive_label in class_labels:
        pos_idx = class_labels.index(str(positive_label))
        neg_idx = 1 - pos_idx
        out: list[str] = []
        for row in probabilities:
            if float(row[pos_idx]) >= float(decision_threshold):
                out.append(class_labels[pos_idx])
            else:
                out.append(class_labels[neg_idx])
        return out
    labels = np.asarray(class_labels, dtype=object)
    return [str(item) for item in labels[np.argmax(probabilities, axis=1)]]


def _analyze_classification_error_regions(
    *,
    feature_frame: pd.DataFrame,
    y_true: list[str],
    y_pred: list[str],
    max_regions: int = 3,
) -> list[dict[str, Any]]:
    if feature_frame.empty or not y_true or len(y_true) != len(y_pred):
        return []
    error = np.asarray([1.0 if truth != pred else 0.0 for truth, pred in zip(y_true, y_pred)], dtype=float)
    overall_error = float(np.mean(error))
    minimum_region_size = max(8, int(0.05 * len(error)))
    regions: list[dict[str, Any]] = []
    for feature in feature_frame.columns:
        numeric = pd.to_numeric(feature_frame[feature], errors="coerce")
        valid = numeric.notna()
        if int(valid.sum()) < minimum_region_size:
            continue
        values = numeric[valid].to_numpy(dtype=float)
        errors = error[valid.to_numpy()]
        boundaries = np.unique(np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]))
        if len(boundaries) < 2:
            continue
        for idx in range(len(boundaries) - 1):
            low = float(boundaries[idx])
            high = float(boundaries[idx + 1])
            if idx == len(boundaries) - 2:
                mask = (values >= low) & (values <= high)
            else:
                mask = (values >= low) & (values < high)
            count = int(np.sum(mask))
            if count < minimum_region_size:
                continue
            region_error = float(np.mean(errors[mask]))
            lift = region_error - overall_error
            if lift <= 0.01:
                continue
            regions.append(
                {
                    "feature": str(feature),
                    "lower_bound": low,
                    "upper_bound": high,
                    "sample_count": count,
                    "error_rate": region_error,
                    "error_lift": lift,
                }
            )
    regions.sort(key=lambda item: float(item.get("error_lift", 0.0)), reverse=True)
    return regions[: max(int(max_regions), 1)]


def _metric_or_none(metrics: dict[str, float], key: str) -> float | None:
    try:
        if key not in metrics:
            return None
        return float(metrics[key])
    except (TypeError, ValueError):
        return None


def _resolve_lag_horizon(
    *,
    data_mode: str,
    requested: str,
    lag_horizon_samples: int | None,
    split: DatasetSplit,
) -> int | None:
    if data_mode != "time_series":
        return None
    max_safe = max(1, min(12, int(split.train_indices.size) - 2))
    if max_safe <= 0:
        return None
    if lag_horizon_samples is not None:
        return max(1, min(int(lag_horizon_samples), max_safe))
    if requested in {"lagged_linear", "lagged_logistic_regression"}:
        return min(4, max_safe)
    return min(3, max_safe)


def _infer_data_mode(*, frame: pd.DataFrame, timestamp_column: str | None) -> str:
    if not timestamp_column or timestamp_column not in frame.columns:
        return "steady_state"
    numeric = pd.to_numeric(frame[timestamp_column], errors="coerce")
    valid = numeric.dropna()
    if len(valid) >= 8:
        diffs = valid.diff().dropna()
        if int((diffs > 0).sum()) >= max(5, int(len(diffs) * 0.7)):
            return "time_series"
    parsed = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if float(parsed.notna().mean()) >= 0.8:
        diffs_dt = parsed.dropna().diff().dt.total_seconds().dropna()
        if int((diffs_dt > 0).sum()) >= max(5, int(len(diffs_dt) * 0.7)):
            return "time_series"
    return "steady_state"


def _prepare_split_safe_frames(
    *,
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    target_column: str,
    missing_data_strategy: str,
    fill_constant_value: float | None,
) -> dict[str, Any]:
    coerced: dict[str, pd.DataFrame] = {}
    for name, frame in split_frames.items():
        subset = frame[feature_columns + [target_column]].copy()
        for col in feature_columns + [target_column]:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")
        subset = subset.dropna(subset=[target_column]).reset_index(drop=True)
        coerced[name] = subset

    strategy = missing_data_strategy.strip().lower()
    effective_strategy = strategy
    fill_values: dict[str, float] = {}
    if strategy in {"fill_median", "median"}:
        train = coerced["train"]
        for col in feature_columns:
            median = float(train[col].median()) if train[col].notna().any() else 0.0
            fill_values[col] = median
        for name in list(coerced.keys()):
            coerced[name] = coerced[name].fillna(fill_values)
        effective_strategy = "fill_median_train_only"
    elif strategy in {"fill_constant", "constant"}:
        value = 0.0 if fill_constant_value is None else float(fill_constant_value)
        fill_values = {col: value for col in feature_columns}
        for name in list(coerced.keys()):
            coerced[name] = coerced[name].fillna(fill_values)
        effective_strategy = "fill_constant_train_policy"
    else:
        if strategy == "keep":
            effective_strategy = "keep_requested_drop_remaining_for_model"
        for name in list(coerced.keys()):
            coerced[name] = coerced[name].dropna().reset_index(drop=True)

    for name in list(coerced.keys()):
        if strategy in {"fill_median", "median", "fill_constant", "constant"}:
            coerced[name] = coerced[name].dropna().reset_index(drop=True)

    if coerced["train"].shape[0] < 6 or coerced["validation"].shape[0] < 2 or coerced["test"].shape[0] < 2:
        raise ValueError(
            "Split-safe preprocessing left too few rows in one or more splits. "
            "Reduce missingness, change strategy, or use more data."
        )

    return {
        "frames": coerced,
        "preprocessing": {
            "missing_data_strategy_requested": strategy,
            "missing_data_strategy_effective": effective_strategy,
            "fill_values": fill_values,
            "rows_after": {name: int(part.shape[0]) for name, part in coerced.items()},
        },
    }


def _prepare_split_safe_frames_classification(
    *,
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    target_column: str,
    missing_data_strategy: str,
    fill_constant_value: float | None,
) -> dict[str, Any]:
    coerced: dict[str, pd.DataFrame] = {}
    for name, frame in split_frames.items():
        subset = frame[feature_columns + [target_column]].copy()
        for col in feature_columns:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")
        target = subset[target_column]
        target = target.where(target.notna(), None)
        target = target.map(lambda value: None if value is None else str(value).strip())
        subset[target_column] = target
        subset = subset[(subset[target_column].notna()) & (subset[target_column] != "")].reset_index(drop=True)
        coerced[name] = subset

    strategy = missing_data_strategy.strip().lower()
    effective_strategy = strategy
    fill_values: dict[str, float] = {}
    if strategy in {"fill_median", "median"}:
        train = coerced["train"]
        for col in feature_columns:
            median = float(train[col].median()) if train[col].notna().any() else 0.0
            fill_values[col] = median
        for name in list(coerced.keys()):
            for col, value in fill_values.items():
                coerced[name][col] = coerced[name][col].fillna(value)
        effective_strategy = "fill_median_train_only"
    elif strategy in {"fill_constant", "constant"}:
        value = 0.0 if fill_constant_value is None else float(fill_constant_value)
        fill_values = {col: value for col in feature_columns}
        for name in list(coerced.keys()):
            for col, fill_value in fill_values.items():
                coerced[name][col] = coerced[name][col].fillna(fill_value)
        effective_strategy = "fill_constant_train_policy"
    else:
        if strategy == "keep":
            effective_strategy = "keep_requested_drop_remaining_for_model"
        for name in list(coerced.keys()):
            coerced[name] = coerced[name].dropna(subset=feature_columns).reset_index(drop=True)

    for name in list(coerced.keys()):
        if strategy in {"fill_median", "median", "fill_constant", "constant"}:
            coerced[name] = coerced[name].dropna(subset=feature_columns).reset_index(drop=True)

    if coerced["train"].shape[0] < 6 or coerced["validation"].shape[0] < 2 or coerced["test"].shape[0] < 2:
        raise ValueError(
            "Split-safe preprocessing left too few rows in one or more splits. "
            "Reduce missingness, change strategy, or use more data."
        )

    train_labels = {str(item) for item in coerced["train"][target_column]}
    unseen = set()
    for part_name in ("validation", "test"):
        unseen |= {
            str(item)
            for item in coerced[part_name][target_column]
            if str(item) not in train_labels
        }
    if unseen:
        raise ValueError(
            "Split-safe classification preprocessing produced labels in validation/test "
            "that were not present in the training split. Collect more examples per class."
        )

    return {
        "frames": coerced,
        "preprocessing": {
            "missing_data_strategy_requested": strategy,
            "missing_data_strategy_effective": effective_strategy,
            "fill_values": fill_values,
            "rows_after": {name: int(part.shape[0]) for name, part in coerced.items()},
        },
    }


def _prepare_xy(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    _require_columns(frame, feature_columns + [target_column])
    subset = frame[feature_columns + [target_column]].copy().dropna()
    x = subset[feature_columns].to_numpy(dtype=float)
    y = subset[target_column].to_numpy(dtype=float)
    return x, y


def _lagged_feature_names(*, feature_columns: list[str], lag_horizon: int) -> list[str]:
    names: list[str] = []
    for col in feature_columns:
        names.append(f"{col}__t")
        for lag in range(1, int(lag_horizon) + 1):
            names.append(f"{col}__lag{lag}")
    return names


def _build_lagged_design_frame(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    lag_horizon: int,
    context_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    required = list(feature_columns) + [target_column]
    _require_columns(frame, required)
    if context_frame is not None:
        _require_columns(context_frame, required)
    context_len = int(len(context_frame)) if context_frame is not None else 0
    if context_frame is not None and context_len > 0:
        combined = pd.concat(
            [context_frame[required], frame[required]],
            ignore_index=True,
        )
    else:
        combined = frame[required].copy().reset_index(drop=True)

    out = pd.DataFrame(index=combined.index)
    for col in feature_columns:
        numeric = pd.to_numeric(combined[col], errors="coerce")
        out[f"{col}__t"] = numeric
        for lag in range(1, int(lag_horizon) + 1):
            out[f"{col}__lag{lag}"] = numeric.shift(lag)
    out[target_column] = pd.to_numeric(combined[target_column], errors="coerce")
    out["_is_current"] = False
    out.loc[context_len:, "_is_current"] = True
    out = out.dropna()
    out = out[out["_is_current"]].drop(columns=["_is_current"]).reset_index(drop=True)
    return out


def _prepare_feature_matrix(*, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    _require_columns(frame, feature_columns)
    subset = frame[feature_columns].copy()
    if subset.isna().any().any():
        raise ValueError("Feature frame contains missing values after preprocessing.")
    return subset.to_numpy(dtype=float)


def _prepare_target_vector(*, frame: pd.DataFrame, target_column: str) -> np.ndarray:
    _require_columns(frame, [target_column])
    target = frame[target_column].copy()
    if target.isna().any():
        raise ValueError("Target column contains missing values after preprocessing.")
    return target.to_numpy(dtype=float)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _fit_regression_tree(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    depth: int,
    max_depth: int,
    min_leaf: int,
) -> dict[str, Any]:
    leaf_value = float(np.mean(y_train)) if y_train.size > 0 else 0.0
    if (
        depth >= max_depth
        or y_train.size < max(2 * int(min_leaf), 8)
        or float(np.var(y_train)) <= 1e-12
    ):
        return {"leaf": True, "value": leaf_value}

    best_feature = -1
    best_threshold = float("nan")
    best_score = float("inf")
    n_features = int(x_train.shape[1]) if x_train.ndim == 2 else 0
    for feature_idx in range(n_features):
        values = x_train[:, feature_idx]
        finite_values = values[np.isfinite(values)]
        if finite_values.size < max(2 * int(min_leaf), 8):
            continue
        thresholds = _candidate_thresholds(finite_values)
        for threshold in thresholds:
            left_mask = values <= threshold
            right_mask = ~left_mask
            left_count = int(np.sum(left_mask))
            right_count = int(np.sum(right_mask))
            if left_count < min_leaf or right_count < min_leaf:
                continue
            score = _sum_squared_error(y_train[left_mask]) + _sum_squared_error(y_train[right_mask])
            if score < best_score:
                best_score = score
                best_feature = feature_idx
                best_threshold = float(threshold)

    if best_feature < 0 or not math.isfinite(best_threshold):
        return {"leaf": True, "value": leaf_value}

    values = x_train[:, best_feature]
    left_mask = values <= best_threshold
    right_mask = ~left_mask
    return {
        "leaf": False,
        "value": leaf_value,
        "feature_index": int(best_feature),
        "threshold": float(best_threshold),
        "left": _fit_regression_tree(
            x_train=x_train[left_mask],
            y_train=y_train[left_mask],
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf=min_leaf,
        ),
        "right": _fit_regression_tree(
            x_train=x_train[right_mask],
            y_train=y_train[right_mask],
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf=min_leaf,
        ),
    }


def _predict_regression_tree_batch(*, tree: dict[str, Any], x: np.ndarray) -> np.ndarray:
    pred = np.empty(x.shape[0], dtype=float)
    for idx in range(x.shape[0]):
        pred[idx] = _predict_regression_tree_row(tree=tree, row=x[idx])
    return pred


def _predict_regression_tree_row(*, tree: dict[str, Any], row: np.ndarray) -> float:
    node = tree
    while not bool(node.get("leaf", False)):
        feature_index = int(node["feature_index"])
        threshold = float(node["threshold"])
        node = node["left"] if float(row[feature_index]) <= threshold else node["right"]
    return float(node.get("value", 0.0))


def _candidate_thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values)
    if unique.size <= 1:
        return np.array([], dtype=float)
    if unique.size <= 12:
        return ((unique[:-1] + unique[1:]) / 2.0).astype(float)
    quantiles = np.linspace(0.1, 0.9, 9)
    candidates = np.unique(np.quantile(values, quantiles))
    lower = float(np.min(values))
    upper = float(np.max(values))
    candidates = candidates[(candidates > lower) & (candidates < upper)]
    return candidates.astype(float)


def _sum_squared_error(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    centered = values - float(np.mean(values))
    return float(np.sum(np.square(centered)))
