"""Deterministic quality checks for tabular sensor data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityCheckResult:
    """Summary of dataset quality checks."""

    rows: int
    columns: int
    completeness_score: float
    missing_fraction_by_column: dict[str, float]
    duplicate_rows: int
    constant_columns: list[str]
    outlier_count_by_column: dict[str, int]
    extreme_outlier_columns: list[str]
    timestamp_column: str | None
    invalid_timestamps: int
    duplicate_timestamps: int
    monotonic_timestamp: bool | None
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_quality_checks(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
    outlier_z_threshold: float = 4.0,
) -> QualityCheckResult:
    """Run basic quality checks needed before correlation/modeling."""
    rows = int(len(frame))
    columns = int(len(frame.columns))
    total_cells = max(rows * max(columns, 1), 1)
    missing_count = int(frame.isna().sum().sum())
    completeness = float(max(0.0, 1.0 - (missing_count / total_cells)))

    missing_fraction = {
        col: float(frame[col].isna().mean()) for col in frame.columns
    }
    duplicate_rows = int(frame.duplicated().sum())

    numeric_columns = _numeric_columns(frame, exclude={timestamp_column} if timestamp_column else set())
    constant_columns = [
        col
        for col in numeric_columns
        if _coerce_numeric_like(frame[col]).nunique(dropna=True) <= 1
    ]
    outlier_counts = {
        col: _count_outliers(
            _coerce_numeric_like(frame[col]),
            z_threshold=outlier_z_threshold,
        )
        for col in numeric_columns
    }
    extreme_columns = [
        col
        for col, count in outlier_counts.items()
        if count > max(5, int(0.02 * rows))
    ]

    ts_col = _resolve_timestamp_column(frame, explicit=timestamp_column)
    invalid_ts = 0
    duplicate_ts = 0
    monotonic_ts: bool | None = None
    warnings: list[str] = []

    if ts_col is not None:
        ts_series = frame[ts_col]
        numeric_ts = _coerce_numeric_like(ts_series)
        numeric_ratio = float(numeric_ts.notna().mean()) if len(ts_series) > 0 else 0.0
        if numeric_ratio >= 0.95:
            valid_ts = numeric_ts.dropna()
            invalid_ts = int(numeric_ts.isna().sum())
            duplicate_ts = int(valid_ts.duplicated().sum())
            monotonic_ts = bool(valid_ts.is_monotonic_increasing)
        else:
            parsed_ts = pd.to_datetime(ts_series, errors="coerce")
            valid_ts = parsed_ts.dropna()
            invalid_ts = int(parsed_ts.isna().sum())
            duplicate_ts = int(valid_ts.duplicated().sum())
            monotonic_ts = bool(valid_ts.is_monotonic_increasing)
        if invalid_ts > 0:
            warnings.append(
                f"Timestamp column '{ts_col}' has {invalid_ts} invalid timestamps."
            )
        if duplicate_ts > 0:
            warnings.append(
                f"Timestamp column '{ts_col}' has {duplicate_ts} duplicate timestamps."
            )
        if monotonic_ts is False:
            warnings.append(
                f"Timestamp column '{ts_col}' is not monotonic increasing."
            )

    if duplicate_rows > 0:
        warnings.append(f"Dataset contains {duplicate_rows} duplicate rows.")
    if extreme_columns:
        warnings.append(
            "Signals with many outliers: " + ", ".join(extreme_columns)
        )
    if completeness < 0.95:
        warnings.append(
            f"Completeness score is {completeness:.3f}. Missing data treatment may be required."
        )
    if constant_columns:
        warnings.append(
            "Constant/near-constant numeric columns: " + ", ".join(constant_columns)
        )

    return QualityCheckResult(
        rows=rows,
        columns=columns,
        completeness_score=completeness,
        missing_fraction_by_column=missing_fraction,
        duplicate_rows=duplicate_rows,
        constant_columns=constant_columns,
        outlier_count_by_column=outlier_counts,
        extreme_outlier_columns=extreme_columns,
        timestamp_column=ts_col,
        invalid_timestamps=invalid_ts,
        duplicate_timestamps=duplicate_ts,
        monotonic_timestamp=monotonic_ts,
        warnings=warnings,
    )


def _count_outliers(series: pd.Series, *, z_threshold: float) -> int:
    clean = series.dropna()
    if len(clean) < 8:
        return 0
    values = clean.to_numpy(dtype=float)

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 1e-12:
        robust_z = 0.6745 * (values - median) / mad
        robust_mask = np.abs(robust_z) > z_threshold
    else:
        robust_mask = np.zeros_like(values, dtype=bool)

    q1 = float(np.quantile(values, 0.25))
    q3 = float(np.quantile(values, 0.75))
    iqr = q3 - q1
    if iqr > 1e-12:
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        iqr_mask = (values < low) | (values > high)
    else:
        iqr_mask = np.zeros_like(values, dtype=bool)

    return int(np.sum(robust_mask | iqr_mask))


def _numeric_columns(frame: pd.DataFrame, *, exclude: set[str]) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in exclude:
            continue
        numeric = _coerce_numeric_like(frame[col])
        if numeric.notna().sum() >= 3:
            cols.append(col)
    return cols


def _coerce_numeric_like(series: pd.Series) -> pd.Series:
    direct = pd.to_numeric(series, errors="coerce")
    if len(series) == 0:
        return direct
    direct_ratio = float(direct.notna().mean())
    if direct_ratio >= 0.95:
        return direct

    text = series.astype("string")
    normalized = text.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    comma_decimal = pd.to_numeric(normalized, errors="coerce")
    if float(comma_decimal.notna().mean()) > direct_ratio:
        return comma_decimal
    return direct


def _resolve_timestamp_column(frame: pd.DataFrame, *, explicit: str | None) -> str | None:
    if explicit and explicit in frame.columns:
        return explicit
    candidates = [
        col
        for col in frame.columns
        if any(token in col.lower() for token in ("time", "timestamp", "date"))
    ]
    for col in candidates:
        parsed = pd.to_datetime(frame[col], errors="coerce")
        if float(parsed.notna().mean()) >= 0.8:
            return col
    return None
