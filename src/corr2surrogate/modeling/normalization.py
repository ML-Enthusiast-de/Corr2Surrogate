"""Normalization utilities with persisted state for training and inference."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from corr2surrogate.core.json_utils import write_json


@dataclass(frozen=True)
class MinMaxColumnState:
    """Persisted min-max parameters for one column."""

    minimum: float
    maximum: float
    span: float
    constant: bool


class MinMaxNormalizer:
    """Min-max normalization with reversible target scaling."""

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        low, high = feature_range
        if low >= high:
            raise ValueError("feature_range must be (low, high) with low < high.")
        self.feature_range = (float(low), float(high))
        self.feature_columns: list[str] = []
        self.target_column: str | None = None
        self._feature_state: dict[str, MinMaxColumnState] = {}
        self._target_state: MinMaxColumnState | None = None

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        feature_columns: list[str],
        target_column: str | None = None,
    ) -> "MinMaxNormalizer":
        """Fit column-wise min-max stats from a dataframe."""
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        self.feature_columns = feature_columns
        self.target_column = target_column
        self._feature_state = {
            col: _fit_column(frame[col], column_name=col) for col in feature_columns
        }
        self._target_state = (
            _fit_column(frame[target_column], column_name=target_column)
            if target_column is not None
            else None
        )
        return self

    def transform_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with fitted feature columns normalized."""
        self._require_fitted()
        transformed = frame.copy()
        for col in self.feature_columns:
            transformed[col] = _transform_series(
                transformed[col],
                self._feature_state[col],
                feature_range=self.feature_range,
            )
        return transformed

    def transform_target(self, values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
        """Normalize target values with fitted target parameters."""
        self._require_fitted(require_target=True)
        if isinstance(values, pd.Series):
            return _transform_series(values, self._target_state, feature_range=self.feature_range)
        array = np.asarray(values, dtype=float)
        return _transform_array(array, self._target_state, feature_range=self.feature_range)

    def inverse_transform_target(
        self, values: pd.Series | np.ndarray
    ) -> pd.Series | np.ndarray:
        """De-normalize target predictions back to original engineering units."""
        self._require_fitted(require_target=True)
        if isinstance(values, pd.Series):
            return _inverse_transform_series(
                values, self._target_state, feature_range=self.feature_range
            )
        array = np.asarray(values, dtype=float)
        return _inverse_transform_array(
            array, self._target_state, feature_range=self.feature_range
        )

    def state_dict(self) -> dict[str, Any]:
        """Return serializable normalizer state."""
        self._require_fitted()
        return {
            "method": "minmax",
            "feature_range": list(self.feature_range),
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "feature_state": {k: asdict(v) for k, v in self._feature_state.items()},
            "target_state": asdict(self._target_state) if self._target_state else None,
        }

    def save(self, path: str | Path) -> Path:
        """Persist normalizer state as JSON."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return write_json(output, self.state_dict(), indent=2)

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "MinMaxNormalizer":
        """Reconstruct normalizer from serialized state."""
        feature_range = tuple(state["feature_range"])
        normalizer = cls(feature_range=feature_range)
        normalizer.feature_columns = list(state["feature_columns"])
        normalizer.target_column = state.get("target_column")
        normalizer._feature_state = {
            key: MinMaxColumnState(**value)
            for key, value in state["feature_state"].items()
        }
        target_state = state.get("target_state")
        normalizer._target_state = (
            MinMaxColumnState(**target_state) if target_state is not None else None
        )
        return normalizer

    @classmethod
    def load(cls, path: str | Path) -> "MinMaxNormalizer":
        """Load normalizer state from JSON."""
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_state_dict(state)

    def _require_fitted(self, *, require_target: bool = False) -> None:
        if not self._feature_state:
            raise RuntimeError("Normalizer is not fitted.")
        if require_target and self._target_state is None:
            raise RuntimeError("Target column was not fitted for this normalizer.")


def _fit_column(series: pd.Series, *, column_name: str) -> MinMaxColumnState:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        raise ValueError(f"Column '{column_name}' has no numeric values to normalize.")
    minimum = float(numeric.min())
    maximum = float(numeric.max())
    span = maximum - minimum
    constant = span == 0.0
    return MinMaxColumnState(
        minimum=minimum,
        maximum=maximum,
        span=1.0 if constant else span,
        constant=constant,
    )


def _transform_series(
    series: pd.Series,
    state: MinMaxColumnState,
    *,
    feature_range: tuple[float, float],
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    transformed = _transform_array(numeric.to_numpy(dtype=float), state, feature_range=feature_range)
    return pd.Series(transformed, index=series.index, name=series.name)


def _inverse_transform_series(
    series: pd.Series,
    state: MinMaxColumnState,
    *,
    feature_range: tuple[float, float],
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    restored = _inverse_transform_array(
        numeric.to_numpy(dtype=float), state, feature_range=feature_range
    )
    return pd.Series(restored, index=series.index, name=series.name)


def _transform_array(
    values: np.ndarray,
    state: MinMaxColumnState,
    *,
    feature_range: tuple[float, float],
) -> np.ndarray:
    low, high = feature_range
    scaled = (values - state.minimum) / state.span
    return scaled * (high - low) + low


def _inverse_transform_array(
    values: np.ndarray,
    state: MinMaxColumnState,
    *,
    feature_range: tuple[float, float],
) -> np.ndarray:
    low, high = feature_range
    scaled = (values - low) / (high - low)
    return scaled * state.span + state.minimum
