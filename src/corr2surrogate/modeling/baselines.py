"""Baseline surrogate models with local save/load/retrain support."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from corr2surrogate.core.json_utils import write_json
from .evaluation import regression_metrics


@dataclass(frozen=True)
class IncrementalLinearState:
    """Serializable state for incremental linear surrogate."""

    feature_columns: list[str]
    target_column: str
    ridge: float
    n_samples_seen: int
    xtx: list[list[float]]
    xty: list[float]
    coefficients: list[float]
    intercept: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IncrementalLinearSurrogate:
    """Linear surrogate with sufficient-statistics retraining.

    This model can be resumed and retrained with additional data without
    requiring all historical raw samples.
    """

    def __init__(
        self,
        *,
        feature_columns: list[str],
        target_column: str,
        ridge: float = 1e-8,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.ridge = float(ridge)
        self.n_samples_seen = 0
        self._xtx: np.ndarray | None = None
        self._xty: np.ndarray | None = None
        self._weights: np.ndarray | None = None

    def fit_dataframe(self, frame: pd.DataFrame) -> int:
        """Fit from scratch on provided dataframe."""
        self.n_samples_seen = 0
        self._xtx = None
        self._xty = None
        self._weights = None
        return self.update_from_dataframe(frame)

    def update_from_dataframe(self, frame: pd.DataFrame) -> int:
        """Incrementally update model state with additional samples."""
        x, y = _prepare_numeric_training_arrays(
            frame=frame,
            feature_columns=self.feature_columns,
            target_column=self.target_column,
        )
        if x.shape[0] == 0:
            raise ValueError("No valid numeric rows available for training.")

        design = np.column_stack([np.ones(x.shape[0]), x])
        xtx_new = design.T @ design
        xty_new = design.T @ y

        if self._xtx is None or self._xty is None:
            self._xtx = xtx_new
            self._xty = xty_new
        else:
            self._xtx = self._xtx + xtx_new
            self._xty = self._xty + xty_new

        self.n_samples_seen += int(x.shape[0])
        self._recompute_weights()
        return int(x.shape[0])

    def predict_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict target values for provided dataframe."""
        self._require_fitted()
        x = _prepare_feature_array(frame=frame, feature_columns=self.feature_columns)
        design = np.column_stack([np.ones(x.shape[0]), x])
        pred = design @ self._weights
        return pred.astype(float)

    def evaluate_dataframe(self, frame: pd.DataFrame) -> dict[str, float]:
        """Evaluate model on dataframe containing target and features."""
        y_true = _prepare_target_array(frame=frame, target_column=self.target_column)
        y_pred = self.predict_dataframe(frame)
        metrics = regression_metrics(y_true=y_true, y_pred=y_pred).to_dict()
        return {k: float(v) for k, v in metrics.items()}

    def state_dict(self) -> dict[str, Any]:
        """Serialize model state."""
        self._require_fitted()
        weights = self._weights.astype(float)
        state = IncrementalLinearState(
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            ridge=self.ridge,
            n_samples_seen=int(self.n_samples_seen),
            xtx=self._xtx.astype(float).tolist(),
            xty=self._xty.astype(float).tolist(),
            intercept=float(weights[0]),
            coefficients=weights[1:].astype(float).tolist(),
        )
        return state.to_dict()

    def save(self, path: str | Path) -> Path:
        """Persist model state as JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return write_json(output, self.state_dict(), indent=2)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "IncrementalLinearSurrogate":
        """Rebuild model from serialized state."""
        model = cls(
            feature_columns=list(state["feature_columns"]),
            target_column=str(state["target_column"]),
            ridge=float(state.get("ridge", 1e-8)),
        )
        model.n_samples_seen = int(state.get("n_samples_seen", 0))
        model._xtx = np.asarray(state["xtx"], dtype=float)
        model._xty = np.asarray(state["xty"], dtype=float)
        weights = np.asarray(
            [state["intercept"], *list(state.get("coefficients", []))],
            dtype=float,
        )
        model._weights = weights
        return model

    @classmethod
    def load(cls, path: str | Path) -> "IncrementalLinearSurrogate":
        """Load model state JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_state_dict(payload)

    def _recompute_weights(self) -> None:
        self._require_statistics()
        regularized = self._xtx.copy()
        regularized[1:, 1:] += np.eye(regularized.shape[0] - 1) * self.ridge
        self._weights = np.linalg.solve(regularized, self._xty)

    def _require_statistics(self) -> None:
        if self._xtx is None or self._xty is None:
            raise RuntimeError("Model statistics are not initialized.")

    def _require_fitted(self) -> None:
        if self._weights is None:
            raise RuntimeError("Model has not been fitted yet.")


def _prepare_numeric_training_arrays(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    _require_columns(frame, feature_columns + [target_column])
    subset = frame[feature_columns + [target_column]].copy()
    for column in feature_columns + [target_column]:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    subset = subset.dropna()
    x = subset[feature_columns].to_numpy(dtype=float)
    y = subset[target_column].to_numpy(dtype=float)
    return x, y


def _prepare_feature_array(*, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    _require_columns(frame, feature_columns)
    subset = frame[feature_columns].copy()
    for column in feature_columns:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    if subset.isna().any().any():
        raise ValueError("Feature frame contains non-numeric or missing values.")
    return subset.to_numpy(dtype=float)


def _prepare_target_array(*, frame: pd.DataFrame, target_column: str) -> np.ndarray:
    _require_columns(frame, [target_column])
    target = pd.to_numeric(frame[target_column], errors="coerce")
    if target.isna().any():
        raise ValueError("Target column contains non-numeric or missing values.")
    return target.to_numpy(dtype=float)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
