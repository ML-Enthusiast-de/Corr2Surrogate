"""Evaluation utilities for surrogate regression and classification models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    """Common regression metrics."""

    mae: float
    rmse: float
    r2: float
    mape: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Common classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    log_loss: float
    n_samples: int
    roc_auc: float | None = None
    pr_auc: float | None = None
    positive_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


def regression_metrics(
    *,
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
) -> RegressionMetrics:
    """Compute deterministic regression metrics."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have same shape.")
    if true.size == 0:
        raise ValueError("At least one sample is required.")

    error = pred - true
    abs_error = np.abs(error)
    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(np.square(error))))

    centered = true - np.mean(true)
    denom = float(np.sum(np.square(centered)))
    if denom == 0.0:
        r2 = 1.0 if np.allclose(pred, true) else 0.0
    else:
        sse = float(np.sum(np.square(error)))
        r2 = 1.0 - (sse / denom)

    non_zero = np.abs(true) > 1e-12
    if np.any(non_zero):
        mape = float(np.mean(np.abs((pred[non_zero] - true[non_zero]) / true[non_zero])) * 100.0)
    else:
        mape = 0.0

    return RegressionMetrics(
        mae=mae,
        rmse=rmse,
        r2=float(r2),
        mape=mape,
        n_samples=int(true.size),
    )


def classification_metrics(
    *,
    y_true: np.ndarray | list[Any],
    probabilities: np.ndarray | list[list[float]],
    class_labels: list[Any],
    positive_label: Any | None = None,
    decision_threshold: float | None = None,
) -> ClassificationMetrics:
    """Compute deterministic classification metrics for binary or multiclass tasks."""
    labels = [str(item) for item in class_labels]
    if not labels:
        raise ValueError("class_labels cannot be empty.")
    proba = np.asarray(probabilities, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    if proba.shape[1] != len(labels):
        raise ValueError("probabilities column count must match class_labels.")
    normalized = _normalize_probabilities(proba)
    raw_true = np.asarray([str(item) for item in y_true], dtype=object)
    if raw_true.shape[0] != normalized.shape[0]:
        raise ValueError("y_true and probabilities must have same row count.")
    if raw_true.size == 0:
        raise ValueError("At least one sample is required.")

    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    try:
        true_idx = np.asarray([label_to_idx[str(item)] for item in raw_true], dtype=int)
    except KeyError as exc:
        raise ValueError(f"y_true contains unknown label {exc}") from exc
    pred_idx = np.argmax(normalized, axis=1).astype(int)
    accuracy = float(np.mean(pred_idx == true_idx))
    log_loss = _multiclass_log_loss(true_idx=true_idx, probabilities=normalized)

    if len(labels) == 2:
        counts = np.bincount(true_idx, minlength=2)
        if positive_label is not None and str(positive_label) in label_to_idx:
            positive_index = int(label_to_idx[str(positive_label)])
        else:
            positive_index = int(np.argmin(counts))
        negative_index = 1 - positive_index
        y_true_bin = (true_idx == positive_index).astype(int)
        if decision_threshold is None:
            y_pred_bin = (pred_idx == positive_index).astype(int)
        else:
            threshold = float(max(1e-6, min(0.999999, decision_threshold)))
            positive_scores = normalized[:, positive_index]
            y_pred_bin = (positive_scores >= threshold).astype(int)
            pred_idx = np.where(y_pred_bin == 1, positive_index, negative_index).astype(int)
        precision = _safe_divide(
            float(np.sum((y_pred_bin == 1) & (y_true_bin == 1))),
            float(np.sum(y_pred_bin == 1)),
        )
        recall = _safe_divide(
            float(np.sum((y_pred_bin == 1) & (y_true_bin == 1))),
            float(np.sum(y_true_bin == 1)),
        )
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        positive_scores = normalized[:, positive_index]
        roc_auc = _binary_roc_auc(y_true=y_true_bin, y_score=positive_scores)
        pr_auc = _binary_pr_auc(y_true=y_true_bin, y_score=positive_scores)
        positive_rate = float(np.mean(y_true_bin))
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            log_loss=log_loss,
            n_samples=int(raw_true.size),
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            positive_rate=positive_rate,
        )

    precision, recall, f1 = _macro_prf(
        true_idx=true_idx,
        pred_idx=pred_idx,
        n_classes=len(labels),
    )
    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        log_loss=log_loss,
        n_samples=int(raw_true.size),
    )


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0)
    row_sums = np.sum(clipped, axis=1, keepdims=True)
    row_sums[row_sums <= 1e-12] = 1.0
    return clipped / row_sums


def _multiclass_log_loss(*, true_idx: np.ndarray, probabilities: np.ndarray) -> float:
    chosen = probabilities[np.arange(true_idx.size), true_idx]
    chosen = np.clip(chosen, 1e-9, 1.0)
    return float(-np.mean(np.log(chosen)))


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator / denominator)


def _binary_roc_auc(*, y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return 0.0
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=float)
    positive_rank_sum = float(np.sum(ranks[y_true == 1]))
    auc = (positive_rank_sum - (positives * (positives + 1) / 2.0)) / (positives * negatives)
    return float(max(0.0, min(1.0, auc)))


def _binary_pr_auc(*, y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return 0.0
    order = np.argsort(-y_score)
    true_sorted = y_true[order]
    tp = 0.0
    fp = 0.0
    recall_prev = 0.0
    area = 0.0
    for label in true_sorted:
        if int(label) == 1:
            tp += 1.0
        else:
            fp += 1.0
        recall = tp / positives
        precision = tp / max(tp + fp, 1.0)
        area += (recall - recall_prev) * precision
        recall_prev = recall
    return float(max(0.0, min(1.0, area)))


def _macro_prf(*, true_idx: np.ndarray, pred_idx: np.ndarray, n_classes: int) -> tuple[float, float, float]:
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for class_idx in range(n_classes):
        tp = float(np.sum((pred_idx == class_idx) & (true_idx == class_idx)))
        fp = float(np.sum((pred_idx == class_idx) & (true_idx != class_idx)))
        fn = float(np.sum((pred_idx != class_idx) & (true_idx == class_idx)))
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2.0 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return (
        float(np.mean(precisions)) if precisions else 0.0,
        float(np.mean(recalls)) if recalls else 0.0,
        float(np.mean(f1s)) if f1s else 0.0,
    )
