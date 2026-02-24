"""Evaluation utilities for surrogate regression models."""

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
