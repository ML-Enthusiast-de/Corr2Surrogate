"""Inference workflows for persisted Agent 2 model artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from corr2surrogate.core.json_utils import write_json
from corr2surrogate.ingestion.csv_loader import load_tabular_data
from .baselines import IncrementalLinearSurrogate
from .checkpoints import ModelCheckpointStore
from .classifiers import (
    BaggedTreeClassifierSurrogate,
    BoostedTreeClassifierSurrogate,
    LogisticClassificationSurrogate,
)
from .evaluation import classification_metrics, regression_metrics
from .normalization import MinMaxNormalizer
from .training import (
    BaggedTreeEnsembleSurrogate,
    BoostedTreeEnsembleSurrogate,
    LaggedLinearSurrogate,
    LaggedLogisticClassificationSurrogate,
    LaggedTreeClassifierSurrogate,
    LaggedTreeEnsembleSurrogate,
)


_CLASSIFICATION_MODEL_NAMES = {
    "logistic_regression",
    "lagged_logistic_regression",
    "bagged_tree_classifier",
    "boosted_tree_classifier",
    "lagged_tree_classifier",
}

_LAGGED_MODEL_NAMES = {
    "lagged_linear",
    "lagged_tree_ensemble",
    "lagged_logistic_regression",
    "lagged_tree_classifier",
}


def run_inference_from_artifacts(
    *,
    data_path: str,
    checkpoint_id: str | None = None,
    run_dir: str | None = None,
    sheet_name: str | int | None = None,
    header_row: int | None = None,
    data_start_row: int | None = None,
    delimiter: str | None = None,
    decision_threshold: float | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Run inference from a persisted checkpoint or run directory."""
    resolved = _resolve_artifacts(checkpoint_id=checkpoint_id, run_dir=run_dir)
    model_params = _load_model_params(resolved["model_params_path"])
    model = _load_model_from_state(
        state_path=resolved["model_state_path"],
        model_name_hint=resolved["model_name"],
    )

    loaded = load_tabular_data(
        data_path,
        sheet_name=sheet_name,
        header_row=header_row,
        data_start_row=data_start_row,
        delimiter=delimiter,
    )
    frame = loaded.frame.copy()
    feature_columns = [str(value) for value in resolved["feature_columns"]]
    target_column = str(resolved["target_column"])
    _require_columns(frame=frame, columns=feature_columns)

    preprocessing_info = _extract_preprocessing_info(model_params=model_params)
    prepared = _prepare_inference_frame(
        frame=frame,
        feature_columns=feature_columns,
        missing_data_strategy=preprocessing_info["missing_data_strategy"],
        fill_values=preprocessing_info["fill_values"],
    )
    prepared_frame = prepared["frame"]

    normalizer = _load_normalizer(model_params=model_params)
    if normalizer is not None:
        transformed_frame = normalizer.transform_features(prepared_frame.copy())
    else:
        transformed_frame = prepared_frame.copy()

    predict_frame = transformed_frame.copy()
    if resolved["model_name"] in _LAGGED_MODEL_NAMES and target_column not in predict_frame.columns:
        predict_frame[target_column] = 0.0
    elif resolved["model_name"] in _LAGGED_MODEL_NAMES:
        predict_frame[target_column] = 0.0

    predictions_raw, probabilities, class_labels = _predict(
        model=model,
        predict_frame=predict_frame,
        model_name=resolved["model_name"],
    )
    if predictions_raw.size == 0:
        raise ValueError("Inference produced no predictions after preprocessing.")

    if resolved["model_name"] in _LAGGED_MODEL_NAMES:
        aligned_source = predict_frame.tail(len(predictions_raw))
        lag_warmup_rows = int(len(predict_frame) - len(predictions_raw))
    else:
        aligned_source = predict_frame
        lag_warmup_rows = 0

    source_rows = aligned_source["__source_row"].to_numpy(dtype=int)
    aligned_raw = (
        prepared_frame.set_index("__source_row")
        .loc[source_rows]
        .reset_index()
    )

    task_type = _extract_task_type(model_params=model_params)
    is_classification = resolved["model_name"] in _CLASSIFICATION_MODEL_NAMES
    threshold_used = _resolve_decision_threshold(
        decision_threshold=decision_threshold,
        model_params=model_params,
        is_classification=is_classification,
    )

    eval_payload = _evaluate_predictions(
        aligned_frame=aligned_raw,
        target_column=target_column,
        model_name=resolved["model_name"],
        predictions=predictions_raw,
        probabilities=probabilities,
        class_labels=class_labels,
        task_type=task_type,
        decision_threshold=threshold_used,
    )

    denormalized = False
    if (not is_classification) and normalizer is not None and normalizer.target_column == target_column:
        try:
            predictions = np.asarray(
                normalizer.inverse_transform_target(predictions_raw),
                dtype=float,
            )
            denormalized = True
        except Exception:
            predictions = np.asarray(predictions_raw)
    else:
        predictions = np.asarray(predictions_raw)

    diagnostics = _compute_shift_diagnostics(
        aligned_frame=aligned_raw,
        feature_columns=feature_columns,
        normalizer=normalizer,
    )
    recommendations = _build_recommendations(
        diagnostics=diagnostics,
        evaluation=eval_payload.get("metrics"),
        task_type=task_type,
        is_classification=is_classification,
    )

    prediction_frame = _build_prediction_frame(
        source_rows=source_rows,
        predictions=predictions,
        probabilities=probabilities,
        class_labels=class_labels,
        threshold_used=threshold_used,
        target_column=target_column,
        task_type=task_type,
    )
    report_path, predictions_path = _resolve_output_paths(
        data_path=data_path,
        output_path=output_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(predictions_path, index=False)

    payload = {
        "status": "ok",
        "checkpoint_id": resolved["checkpoint_id"],
        "run_dir": resolved["run_dir"],
        "model_name": resolved["model_name"],
        "task_type": task_type,
        "data_path": str(Path(data_path)),
        "target_column": target_column,
        "feature_columns": feature_columns,
        "rows_input": int(len(frame)),
        "rows_after_preprocessing": int(len(prepared_frame)),
        "dropped_rows_missing_features": int(prepared["dropped_rows"]),
        "prediction_count": int(len(predictions)),
        "lag_warmup_rows": int(lag_warmup_rows),
        "normalization": {
            "enabled": bool(normalizer is not None),
            "normalizer_path": str(model_params.get("normalizer_path", "")) if normalizer is not None else "",
            "target_denormalized": denormalized,
        },
        "decision_threshold_used": threshold_used,
        "evaluation": eval_payload,
        "ood_summary": diagnostics["ood_summary"],
        "drift_summary": diagnostics["drift_summary"],
        "recommendations": recommendations,
        "predictions_path": str(predictions_path),
        "report_path": str(report_path),
        "predictions_preview": prediction_frame.head(5).to_dict(orient="records"),
    }
    write_json(report_path, payload, indent=2)
    return payload


def _resolve_artifacts(*, checkpoint_id: str | None, run_dir: str | None) -> dict[str, Any]:
    if checkpoint_id:
        store = ModelCheckpointStore()
        checkpoint = store.load_checkpoint(checkpoint_id)
        model_params_path = Path(checkpoint.run_dir) / "model_params.json"
        return {
            "checkpoint_id": checkpoint.checkpoint_id,
            "run_dir": str(Path(checkpoint.run_dir)),
            "model_name": str(checkpoint.model_name),
            "model_state_path": str(Path(checkpoint.model_state_path)),
            "model_params_path": str(model_params_path),
            "target_column": str(checkpoint.target_column),
            "feature_columns": [str(value) for value in checkpoint.feature_columns],
        }
    if not run_dir:
        raise ValueError("Provide either checkpoint_id or run_dir for inference.")
    resolved_run_dir = Path(run_dir)
    model_params_path = resolved_run_dir / "model_params.json"
    model_params = _load_model_params(model_params_path)
    model_name = str(model_params.get("model_name", "")).strip()
    if not model_name:
        raise ValueError(f"Missing model_name in {model_params_path}.")
    state_path = _infer_state_path(run_dir=resolved_run_dir, model_name=model_name)
    return {
        "checkpoint_id": "",
        "run_dir": str(resolved_run_dir),
        "model_name": model_name,
        "model_state_path": str(state_path),
        "model_params_path": str(model_params_path),
        "target_column": str(model_params.get("target_column", "")),
        "feature_columns": [str(value) for value in model_params.get("feature_columns", [])],
    }


def _load_model_params(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Model params file must decode to an object: {path}")
    return payload


def _infer_state_path(*, run_dir: Path, model_name: str) -> Path:
    if model_name == "linear_ridge":
        direct = run_dir / "model_state.json"
        if direct.is_file():
            return direct
    named = run_dir / f"{model_name}_state.json"
    if named.is_file():
        return named
    fallback = run_dir / "model_state.json"
    if fallback.is_file():
        return fallback
    for candidate in sorted(run_dir.glob("*_state.json")):
        return candidate
    raise FileNotFoundError(f"No model state file found in {run_dir}.")


def _extract_preprocessing_info(*, model_params: dict[str, Any]) -> dict[str, Any]:
    extra = model_params.get("extra")
    if not isinstance(extra, dict):
        extra = {}
    preprocessing = extra.get("preprocessing")
    if not isinstance(preprocessing, dict):
        preprocessing = {}
    requested = str(preprocessing.get("missing_data_strategy_requested", "keep")).strip() or "keep"
    fill_values = preprocessing.get("fill_values")
    if not isinstance(fill_values, dict):
        fill_values = {}
    return {
        "missing_data_strategy": requested,
        "fill_values": {str(k): float(v) for k, v in fill_values.items() if _is_number(v)},
    }


def _prepare_inference_frame(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    missing_data_strategy: str,
    fill_values: dict[str, float],
) -> dict[str, Any]:
    working = frame.copy()
    working["__source_row"] = np.arange(len(working), dtype=int)
    for col in feature_columns:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    strategy = missing_data_strategy.lower().strip()
    if strategy == "fill_median":
        for col in feature_columns:
            median = working[col].median(skipna=True)
            if pd.notna(median):
                working[col] = working[col].fillna(float(median))
    elif strategy == "fill_constant":
        for col in feature_columns:
            if col in fill_values:
                working[col] = working[col].fillna(float(fill_values[col]))
            else:
                working[col] = working[col].fillna(0.0)

    before = int(len(working))
    mask = working[feature_columns].notna().all(axis=1)
    working = working.loc[mask].reset_index(drop=True)
    dropped_rows = before - int(len(working))
    if working.empty:
        raise ValueError("No usable rows left for inference after missing-data handling.")
    return {"frame": working, "dropped_rows": dropped_rows}


def _load_normalizer(*, model_params: dict[str, Any]) -> MinMaxNormalizer | None:
    path = str(model_params.get("normalizer_path", "")).strip()
    if not path:
        return None
    normalizer_path = Path(path)
    if not normalizer_path.is_file():
        return None
    return MinMaxNormalizer.load(normalizer_path)


def _load_model_from_state(*, state_path: str | Path, model_name_hint: str | None) -> Any:
    payload = json.loads(Path(state_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Model state must decode to an object: {state_path}")
    model_name = str(payload.get("model_name", "")).strip() or str(model_name_hint or "").strip()
    if not model_name:
        raise ValueError("Unable to resolve model_name from model state.")

    if model_name == "linear_ridge":
        return IncrementalLinearSurrogate.from_state_dict(payload)
    if model_name == "bagged_tree_ensemble":
        model = BaggedTreeEnsembleSurrogate(
            feature_columns=[str(v) for v in payload.get("feature_columns", [])],
            target_column=str(payload.get("target_column", "")),
            n_estimators=int(payload.get("n_estimators", 12)),
            max_depth=int(payload.get("max_depth", 4)),
            min_leaf=int(payload.get("min_leaf", 6)),
        )
        model._estimators = list(payload.get("estimators", []))
        return model
    if model_name == "boosted_tree_ensemble":
        model = BoostedTreeEnsembleSurrogate(
            feature_columns=[str(v) for v in payload.get("feature_columns", [])],
            target_column=str(payload.get("target_column", "")),
            n_estimators=int(payload.get("n_estimators", 28)),
            learning_rate=float(payload.get("learning_rate", 0.25)),
            max_depth=int(payload.get("max_depth", 3)),
            min_leaf=int(payload.get("min_leaf", 5)),
        )
        model._bias = float(payload.get("bias", 0.0))
        model._estimators = list(payload.get("estimators", []))
        return model
    if model_name == "lagged_linear":
        model = LaggedLinearSurrogate(
            feature_columns=[str(v) for v in payload.get("feature_columns", [])],
            target_column=str(payload.get("target_column", "")),
            lag_horizon=int(payload.get("lag_horizon", 3)),
            ridge=float(payload.get("ridge", 1e-8)),
        )
        delegate = IncrementalLinearSurrogate.from_state_dict(
            payload.get("linear_state", {}),
        )
        model._delegate = delegate
        if isinstance(payload.get("lagged_feature_columns"), list):
            model._lagged_feature_columns = [str(v) for v in payload["lagged_feature_columns"]]
        return model
    if model_name == "lagged_tree_ensemble":
        model = LaggedTreeEnsembleSurrogate(
            feature_columns=[str(v) for v in payload.get("feature_columns", [])],
            target_column=str(payload.get("target_column", "")),
            lag_horizon=int(payload.get("lag_horizon", 3)),
            n_estimators=int(payload.get("n_estimators", 12)),
            max_depth=int(payload.get("max_depth", 4)),
            min_leaf=int(payload.get("min_leaf", 6)),
        )
        tree_state = payload.get("tree_state", {})
        delegate = BaggedTreeEnsembleSurrogate(
            feature_columns=[str(v) for v in tree_state.get("feature_columns", [])],
            target_column=str(tree_state.get("target_column", model.target_column)),
            n_estimators=int(tree_state.get("n_estimators", model.n_estimators)),
            max_depth=int(tree_state.get("max_depth", model.max_depth)),
            min_leaf=int(tree_state.get("min_leaf", model.min_leaf)),
        )
        delegate._estimators = list(tree_state.get("estimators", []))
        model._delegate = delegate
        if isinstance(payload.get("lagged_feature_columns"), list):
            model._lagged_feature_columns = [str(v) for v in payload["lagged_feature_columns"]]
        return model
    if model_name == "logistic_regression":
        model = LogisticClassificationSurrogate(
            feature_columns=[str(v) for v in payload.get("feature_columns", [])],
            target_column=str(payload.get("target_column", "")),
            learning_rate=float(payload.get("learning_rate", 0.25)),
            epochs=int(payload.get("epochs", 350)),
            l2=float(payload.get("l2", 1e-4)),
        )
        model.class_labels = [str(v) for v in payload.get("class_labels", [])]
        model._weights = np.asarray(payload.get("weights", []), dtype=float)
        return model
    if model_name == "bagged_tree_classifier":
        model = BaggedTreeClassifierSurrogate(
            feature_columns=[str(v) for v in payload.get("feature_columns", [])],
            target_column=str(payload.get("target_column", "")),
            n_estimators=int(payload.get("n_estimators", 12)),
            max_depth=int(payload.get("max_depth", 4)),
            min_leaf=int(payload.get("min_leaf", 6)),
        )
        model.class_labels = [str(v) for v in payload.get("class_labels", [])]
        model._estimators = list(payload.get("estimators", []))
        return model
    if model_name == "boosted_tree_classifier":
        model = BoostedTreeClassifierSurrogate(
            feature_columns=[str(v) for v in payload.get("feature_columns", [])],
            target_column=str(payload.get("target_column", "")),
            n_estimators=int(payload.get("n_estimators", 24)),
            learning_rate=float(payload.get("learning_rate", 0.6)),
            max_depth=int(payload.get("max_depth", 3)),
            min_leaf=int(payload.get("min_leaf", 5)),
        )
        model.class_labels = [str(v) for v in payload.get("class_labels", [])]
        model._estimators = list(payload.get("estimators", []))
        return model
    if model_name == "lagged_logistic_regression":
        model = LaggedLogisticClassificationSurrogate(
            feature_columns=[str(v) for v in payload.get("feature_columns", [])],
            target_column=str(payload.get("target_column", "")),
            lag_horizon=int(payload.get("lag_horizon", 3)),
            learning_rate=float(payload.get("learning_rate", 0.25)),
            epochs=int(payload.get("epochs", 350)),
            l2=float(payload.get("l2", 1e-4)),
        )
        logistic_state = payload.get("logistic_state", {})
        delegate = LogisticClassificationSurrogate(
            feature_columns=[str(v) for v in logistic_state.get("feature_columns", [])],
            target_column=str(logistic_state.get("target_column", model.target_column)),
            learning_rate=float(logistic_state.get("learning_rate", model.learning_rate)),
            epochs=int(logistic_state.get("epochs", model.epochs)),
            l2=float(logistic_state.get("l2", model.l2)),
        )
        delegate.class_labels = [str(v) for v in logistic_state.get("class_labels", [])]
        delegate._weights = np.asarray(logistic_state.get("weights", []), dtype=float)
        model._delegate = delegate
        if isinstance(payload.get("lagged_feature_columns"), list):
            model._lagged_feature_columns = [str(v) for v in payload["lagged_feature_columns"]]
        return model
    if model_name == "lagged_tree_classifier":
        model = LaggedTreeClassifierSurrogate(
            feature_columns=[str(v) for v in payload.get("feature_columns", [])],
            target_column=str(payload.get("target_column", "")),
            lag_horizon=int(payload.get("lag_horizon", 3)),
            n_estimators=int(payload.get("n_estimators", 12)),
            max_depth=int(payload.get("max_depth", 4)),
            min_leaf=int(payload.get("min_leaf", 6)),
        )
        tree_state = payload.get("tree_state", {})
        delegate = BaggedTreeClassifierSurrogate(
            feature_columns=[str(v) for v in tree_state.get("feature_columns", [])],
            target_column=str(tree_state.get("target_column", model.target_column)),
            n_estimators=int(tree_state.get("n_estimators", model.n_estimators)),
            max_depth=int(tree_state.get("max_depth", model.max_depth)),
            min_leaf=int(tree_state.get("min_leaf", model.min_leaf)),
        )
        delegate.class_labels = [str(v) for v in tree_state.get("class_labels", [])]
        delegate._estimators = list(tree_state.get("estimators", []))
        model._delegate = delegate
        if isinstance(payload.get("lagged_feature_columns"), list):
            model._lagged_feature_columns = [str(v) for v in payload["lagged_feature_columns"]]
        return model
    raise ValueError(f"Unsupported model_name in state file: {model_name}")


def _predict(*, model: Any, predict_frame: pd.DataFrame, model_name: str) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    if model_name in _CLASSIFICATION_MODEL_NAMES:
        probabilities = np.asarray(model.predict_proba_dataframe(predict_frame), dtype=float)
        class_labels = [str(value) for value in getattr(model, "class_labels", [])]
        if not class_labels:
            class_labels = [str(i) for i in range(probabilities.shape[1])]
        labels = _labels_from_probabilities(
            probabilities=probabilities,
            class_labels=class_labels,
            decision_threshold=None,
            positive_label=None,
        )
        return np.asarray(labels, dtype=object), probabilities, class_labels
    predictions = np.asarray(model.predict_dataframe(predict_frame), dtype=float)
    return predictions, None, []


def _evaluate_predictions(
    *,
    aligned_frame: pd.DataFrame,
    target_column: str,
    model_name: str,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    class_labels: list[str],
    task_type: str,
    decision_threshold: float | None,
) -> dict[str, Any]:
    if target_column not in aligned_frame.columns:
        return {
            "available": False,
            "reason": f"Target column '{target_column}' not present in inference dataset.",
        }
    if model_name in _CLASSIFICATION_MODEL_NAMES and probabilities is not None:
        y_true = [str(item).strip() for item in aligned_frame[target_column].tolist()]
        if any(not value for value in y_true):
            return {"available": False, "reason": "Target contains empty labels; skipped evaluation."}
        positive_label = _resolve_positive_label(y_true=y_true, class_labels=class_labels, task_type=task_type)
        try:
            metrics = classification_metrics(
                y_true=y_true,
                probabilities=probabilities,
                class_labels=class_labels,
                positive_label=positive_label,
                decision_threshold=decision_threshold if len(class_labels) == 2 else None,
            ).to_dict()
        except ValueError as exc:
            return {"available": False, "reason": f"Classification evaluation skipped: {exc}"}
        labels = _labels_from_probabilities(
            probabilities=probabilities,
            class_labels=class_labels,
            decision_threshold=decision_threshold if len(class_labels) == 2 else None,
            positive_label=positive_label,
        )
        metrics["decision_threshold"] = float(decision_threshold) if decision_threshold is not None else None
        metrics["positive_label"] = str(positive_label) if positive_label is not None else None
        return {
            "available": True,
            "metrics": metrics,
            "y_true_preview": y_true[:5],
            "y_pred_preview": labels[:5],
        }

    y_true = pd.to_numeric(aligned_frame[target_column], errors="coerce")
    valid = y_true.notna()
    if not bool(valid.any()):
        return {"available": False, "reason": "Target contains no numeric values; skipped evaluation."}
    y_true_arr = y_true.loc[valid].to_numpy(dtype=float)
    y_pred_arr = np.asarray(predictions, dtype=float)[valid.to_numpy()]
    metrics = regression_metrics(y_true=y_true_arr, y_pred=y_pred_arr).to_dict()
    return {
        "available": True,
        "metrics": metrics,
        "y_true_preview": y_true_arr[:5].tolist(),
        "y_pred_preview": y_pred_arr[:5].tolist(),
    }


def _resolve_positive_label(*, y_true: list[str], class_labels: list[str], task_type: str) -> str | None:
    if not class_labels or len(class_labels) != 2:
        return None
    if task_type in {"fraud_detection", "anomaly_detection"} and "1" in class_labels:
        return "1"
    counts: dict[str, int] = {}
    for token in y_true:
        counts[token] = counts.get(token, 0) + 1
    usable = [token for token in class_labels if token in counts]
    if not usable:
        return "1" if "1" in class_labels else class_labels[1]
    usable.sort(key=lambda token: counts[token])
    return usable[0]


def _resolve_decision_threshold(
    *,
    decision_threshold: float | None,
    model_params: dict[str, Any],
    is_classification: bool,
) -> float | None:
    if not is_classification:
        return None
    if decision_threshold is not None:
        return float(max(1e-6, min(0.999999, decision_threshold)))
    best_params = model_params.get("best_params")
    if isinstance(best_params, dict) and _is_number(best_params.get("decision_threshold")):
        value = float(best_params["decision_threshold"])
        return float(max(1e-6, min(0.999999, value)))
    return 0.5


def _labels_from_probabilities(
    *,
    probabilities: np.ndarray,
    class_labels: list[str],
    decision_threshold: float | None,
    positive_label: str | None,
) -> list[str]:
    if probabilities.size == 0:
        return []
    if len(class_labels) == 2 and decision_threshold is not None and positive_label in class_labels:
        pos_idx = class_labels.index(str(positive_label))
        neg_idx = 1 - pos_idx
        out: list[str] = []
        for row in probabilities:
            out.append(class_labels[pos_idx] if float(row[pos_idx]) >= float(decision_threshold) else class_labels[neg_idx])
        return out
    labels = np.asarray(class_labels, dtype=object)
    return [str(item) for item in labels[np.argmax(probabilities, axis=1)]]


def _compute_shift_diagnostics(
    *,
    aligned_frame: pd.DataFrame,
    feature_columns: list[str],
    normalizer: MinMaxNormalizer | None,
) -> dict[str, Any]:
    if normalizer is None:
        return {
            "ood_summary": {
                "overall_ood_fraction": 0.0,
                "features_flagged": 0,
                "details": [],
                "note": "Normalizer state unavailable; OOD range checks skipped.",
            },
            "drift_summary": {
                "overall_drift_score": 0.0,
                "features_flagged": 0,
                "details": [],
                "note": "Normalizer state unavailable; drift checks skipped.",
            },
        }
    ood_details: list[dict[str, Any]] = []
    drift_details: list[dict[str, Any]] = []
    overall_fraction = 0.0
    overall_drift = 0.0
    state = normalizer._feature_state
    for col in feature_columns:
        if col not in state:
            continue
        numeric = pd.to_numeric(aligned_frame[col], errors="coerce")
        valid = numeric.dropna()
        if valid.empty:
            continue
        minimum = float(state[col].minimum)
        maximum = float(state[col].maximum)
        span = float(max(state[col].span, 1e-12))
        below = int((valid < minimum).sum())
        above = int((valid > maximum).sum())
        ood_fraction = float((below + above) / max(int(len(valid)), 1))
        mean_value = float(valid.mean())
        train_mid = 0.5 * (minimum + maximum)
        drift_score = float(abs(mean_value - train_mid) / span)
        ood_details.append(
            {
                "feature": col,
                "ood_fraction": ood_fraction,
                "below_min_count": below,
                "above_max_count": above,
                "training_min": minimum,
                "training_max": maximum,
                "inference_mean": mean_value,
            }
        )
        drift_details.append(
            {
                "feature": col,
                "drift_score": drift_score,
                "training_midpoint": train_mid,
                "inference_mean": mean_value,
            }
        )
        overall_fraction = max(overall_fraction, ood_fraction)
        overall_drift = max(overall_drift, drift_score)
    ood_details.sort(key=lambda item: float(item.get("ood_fraction", 0.0)), reverse=True)
    drift_details.sort(key=lambda item: float(item.get("drift_score", 0.0)), reverse=True)
    return {
        "ood_summary": {
            "overall_ood_fraction": overall_fraction,
            "features_flagged": int(sum(1 for item in ood_details if float(item.get("ood_fraction", 0.0)) > 0.0)),
            "details": ood_details[:10],
        },
        "drift_summary": {
            "overall_drift_score": overall_drift,
            "features_flagged": int(sum(1 for item in drift_details if float(item.get("drift_score", 0.0)) >= 0.35)),
            "details": drift_details[:10],
        },
    }


def _build_recommendations(
    *,
    diagnostics: dict[str, Any],
    evaluation: dict[str, Any] | None,
    task_type: str,
    is_classification: bool,
) -> list[str]:
    out: list[str] = []
    ood_fraction = float((diagnostics.get("ood_summary") or {}).get("overall_ood_fraction", 0.0))
    drift_score = float((diagnostics.get("drift_summary") or {}).get("overall_drift_score", 0.0))
    if ood_fraction >= 0.05:
        out.append(
            "Incoming data exceeds training feature ranges for at least one signal; retraining with expanded operating envelopes is recommended."
        )
    if drift_score >= 0.50:
        out.append(
            "Feature distribution drift is high relative to the training envelope; validate model outputs before deployment and schedule refresh training."
        )
    if isinstance(evaluation, dict):
        if is_classification:
            f1 = _safe_float(evaluation.get("f1"))
            recall = _safe_float(evaluation.get("recall"))
            pr_auc = _safe_float(evaluation.get("pr_auc"))
            if task_type in {"fraud_detection", "anomaly_detection"}:
                if recall < 0.70 or pr_auc < 0.35:
                    out.append(
                        "Fraud/anomaly quality is below target (recall/pr_auc); collect more positive-event and hard-negative windows."
                    )
            elif f1 < 0.75:
                out.append(
                    "Classification F1 is below target; expand feature coverage or compare boosted/lagged classifier families."
                )
        else:
            r2 = _safe_float(evaluation.get("r2"))
            if r2 < 0.70:
                out.append(
                    "Regression fit is below target (R2<0.70); collect denser trajectories in high-error regions and reassess feature/lag design."
                )
    if not out:
        out.append(
            "No immediate retraining trigger detected; continue monitoring drift/OOD and re-run inference diagnostics on each new batch."
        )
    return out


def _build_prediction_frame(
    *,
    source_rows: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    class_labels: list[str],
    threshold_used: float | None,
    target_column: str,
    task_type: str,
) -> pd.DataFrame:
    frame = pd.DataFrame({"source_row": source_rows.astype(int)})
    if probabilities is None:
        frame["prediction"] = np.asarray(predictions, dtype=float)
        return frame
    positive_label = _resolve_positive_label(
        y_true=[],
        class_labels=class_labels,
        task_type=task_type,
    )
    labels = _labels_from_probabilities(
        probabilities=probabilities,
        class_labels=class_labels,
        decision_threshold=threshold_used if len(class_labels) == 2 else None,
        positive_label=positive_label,
    )
    frame["predicted_label"] = labels
    for idx, label in enumerate(class_labels):
        safe_label = re.sub(r"[^A-Za-z0-9_]+", "_", label).strip("_") or f"class_{idx}"
        frame[f"probability_{safe_label}"] = probabilities[:, idx]
    frame["target_column"] = target_column
    if threshold_used is not None:
        frame["decision_threshold_used"] = float(threshold_used)
    return frame


def _extract_task_type(*, model_params: dict[str, Any]) -> str:
    extra = model_params.get("extra")
    if isinstance(extra, dict):
        task_profile = extra.get("task_profile")
        if isinstance(task_profile, dict):
            task_type = str(task_profile.get("task_type", "")).strip()
            if task_type:
                return task_type
    return "regression"


def _resolve_output_paths(*, data_path: str, output_path: str | None) -> tuple[Path, Path]:
    if output_path:
        report_path = Path(output_path)
        predictions_path = report_path.with_name(f"{report_path.stem}_predictions.csv")
        return report_path, predictions_path
    stem = _slugify(Path(data_path).stem)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = Path("reports") / "inference" / stem
    report_path = base / f"inference_{timestamp}.json"
    predictions_path = base / f"inference_{timestamp}_predictions.csv"
    return report_path, predictions_path


def _require_columns(*, frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns for inference: {', '.join(missing)}")


def _slugify(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "dataset"


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")
