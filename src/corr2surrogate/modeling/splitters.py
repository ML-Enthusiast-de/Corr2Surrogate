"""Data splitting strategies for modeling workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    """Deterministic split indices for train/validation/test."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    strategy: str
    data_mode: str

    def to_dict(self) -> dict[str, object]:
        return {
            "train_size": int(self.train_indices.size),
            "validation_size": int(self.validation_indices.size),
            "test_size": int(self.test_indices.size),
            "strategy": self.strategy,
            "data_mode": self.data_mode,
        }


def build_train_validation_test_split(
    *,
    n_rows: int,
    data_mode: str,
) -> DatasetSplit:
    """Build deterministic split indices for tabular or time-series data."""
    if n_rows < 12:
        raise ValueError("At least 12 rows are required for train/validation/test split.")
    mode = data_mode.strip().lower()
    if mode == "time_series":
        return _build_time_series_split(n_rows=n_rows)
    return _build_steady_state_split(n_rows=n_rows)


def _build_time_series_split(*, n_rows: int) -> DatasetSplit:
    train_end = max(6, int(round(n_rows * 0.70)))
    val_end = max(train_end + 2, int(round(n_rows * 0.85)))
    train_end = min(train_end, n_rows - 4)
    val_end = min(val_end, n_rows - 2)
    if train_end < 6 or (val_end - train_end) < 2 or (n_rows - val_end) < 2:
        train_end = max(6, n_rows - 4)
        val_end = n_rows - 2
    if train_end < 6 or (val_end - train_end) < 2 or (n_rows - val_end) < 2:
        raise ValueError("Unable to create a valid time-series split with current row count.")
    return DatasetSplit(
        train_indices=np.arange(0, train_end, dtype=int),
        validation_indices=np.arange(train_end, val_end, dtype=int),
        test_indices=np.arange(val_end, n_rows, dtype=int),
        strategy="blocked_time_order_70_15_15",
        data_mode="time_series",
    )


def _build_steady_state_split(*, n_rows: int) -> DatasetSplit:
    idx = np.arange(n_rows, dtype=int)
    remainder = idx % 20
    test_mask = remainder < 3
    val_mask = (remainder >= 3) & (remainder < 6)
    train_mask = ~(test_mask | val_mask)
    if int(np.sum(test_mask)) < 2 or int(np.sum(val_mask)) < 2 or int(np.sum(train_mask)) < 6:
        split_a = max(6, int(round(n_rows * 0.70)))
        split_b = max(split_a + 2, int(round(n_rows * 0.85)))
        split_a = min(split_a, n_rows - 4)
        split_b = min(split_b, n_rows - 2)
        train_idx = idx[:split_a]
        val_idx = idx[split_a:split_b]
        test_idx = idx[split_b:]
        if train_idx.size < 6 or val_idx.size < 2 or test_idx.size < 2:
            raise ValueError("Unable to create a valid steady-state split with current row count.")
        return DatasetSplit(
            train_indices=train_idx,
            validation_indices=val_idx,
            test_indices=test_idx,
            strategy="deterministic_70_15_15_fallback",
            data_mode="steady_state",
        )
    return DatasetSplit(
        train_indices=idx[train_mask],
        validation_indices=idx[val_mask],
        test_indices=idx[test_mask],
        strategy="deterministic_modulo_70_15_15",
        data_mode="steady_state",
    )
