"""Sensor-level diagnostics and trust scoring for lab datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SensorDiagnostic:
    """Diagnostics for one numeric signal."""

    signal: str
    sample_count: int
    missing_fraction: float
    saturation_fraction: float
    quantization_fraction: float
    drift_slope_normalized: float
    dropout_run_fraction: float
    stuck_run_fraction: float
    trust_score: float
    flags: list[str]


@dataclass(frozen=True)
class SensorDiagnosticsSummary:
    """Aggregate diagnostics output."""

    timestamp_column: str | None
    timestamp_jitter_cv: float
    flagged_signals: list[str]
    diagnostics: list[SensorDiagnostic]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_column": self.timestamp_column,
            "timestamp_jitter_cv": self.timestamp_jitter_cv,
            "flagged_signals": self.flagged_signals,
            "diagnostics": [asdict(item) for item in self.diagnostics],
            "warnings": self.warnings,
        }


def run_sensor_diagnostics(
    frame: pd.DataFrame,
    *,
    timestamp_column: str | None = None,
    max_signals: int = 250,
) -> SensorDiagnosticsSummary:
    """Compute signal diagnostics and reliability indicators."""
    numeric_cols = _numeric_columns(frame, exclude={timestamp_column} if timestamp_column else set())
    diagnostics: list[SensorDiagnostic] = []
    warnings: list[str] = []
    for col in numeric_cols[:max_signals]:
        series = pd.to_numeric(frame[col], errors="coerce")
        diag = _diagnose_signal(str(col), series)
        diagnostics.append(diag)

    diagnostics.sort(key=lambda item: item.trust_score)
    flagged = [item.signal for item in diagnostics if item.flags]
    if flagged:
        warnings.append(
            "Low-trust diagnostic flags for signals: " + ", ".join(flagged[:20])
        )

    jitter_cv = _timestamp_jitter_cv(frame, timestamp_column=timestamp_column)
    if np.isfinite(jitter_cv) and jitter_cv > 0.25:
        warnings.append(
            f"Timestamp jitter is high (coefficient of variation={jitter_cv:.3f})."
        )

    return SensorDiagnosticsSummary(
        timestamp_column=timestamp_column,
        timestamp_jitter_cv=float(jitter_cv) if np.isfinite(jitter_cv) else float("nan"),
        flagged_signals=flagged,
        diagnostics=diagnostics,
        warnings=warnings,
    )


def _diagnose_signal(signal: str, series: pd.Series) -> SensorDiagnostic:
    n = int(len(series))
    missing_fraction = float(series.isna().mean()) if n > 0 else 1.0
    clean = series.dropna().to_numpy(dtype=float)
    if len(clean) < 8:
        return SensorDiagnostic(
            signal=signal,
            sample_count=n,
            missing_fraction=missing_fraction,
            saturation_fraction=float("nan"),
            quantization_fraction=float("nan"),
            drift_slope_normalized=float("nan"),
            dropout_run_fraction=_max_nan_run_fraction(series),
            stuck_run_fraction=float("nan"),
            trust_score=0.0,
            flags=["insufficient_data"],
        )

    min_v = float(np.min(clean))
    max_v = float(np.max(clean))
    scale = max(max_v - min_v, 1e-12)
    eps = 0.01 * scale
    sat_low = np.mean(clean <= (min_v + eps))
    sat_high = np.mean(clean >= (max_v - eps))
    saturation_fraction = float(sat_low + sat_high)

    diffs = np.diff(clean)
    quantization_fraction = float(0.0)
    if len(diffs) >= 8:
        rounded = np.round(np.abs(diffs), decimals=8)
        rounded = rounded[rounded > 0]
        if len(rounded) > 0:
            values, counts = np.unique(rounded, return_counts=True)
            mode_count = int(np.max(counts))
            quantization_fraction = float(mode_count / len(rounded))

    drift = _normalized_drift(clean)
    dropout_run = _max_nan_run_fraction(series)
    stuck_run = _max_stuck_run_fraction(series)

    trust = 1.0
    trust -= min(0.35, saturation_fraction)
    trust -= min(0.20, max(0.0, quantization_fraction - 0.5))
    trust -= min(0.20, abs(drift))
    trust -= min(0.15, dropout_run)
    trust -= min(0.20, stuck_run)
    trust = float(max(0.0, min(1.0, trust)))

    flags: list[str] = []
    if saturation_fraction >= 0.15:
        flags.append("saturation")
    if quantization_fraction >= 0.75:
        flags.append("quantized")
    if abs(drift) >= 0.10:
        flags.append("drift")
    if dropout_run >= 0.10:
        flags.append("dropout_runs")
    if stuck_run >= 0.10:
        flags.append("stuck_segments")

    return SensorDiagnostic(
        signal=signal,
        sample_count=n,
        missing_fraction=missing_fraction,
        saturation_fraction=saturation_fraction,
        quantization_fraction=quantization_fraction,
        drift_slope_normalized=float(drift),
        dropout_run_fraction=dropout_run,
        stuck_run_fraction=stuck_run,
        trust_score=trust,
        flags=flags,
    )


def _numeric_columns(frame: pd.DataFrame, *, exclude: set[str]) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if str(col) in exclude:
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if int(numeric.notna().sum()) >= 8 and int(numeric.nunique(dropna=True)) > 1:
            cols.append(str(col))
    return cols


def _normalized_drift(values: np.ndarray) -> float:
    n = len(values)
    if n < 8:
        return float("nan")
    x = np.arange(n, dtype=float)
    y = values.astype(float)
    y_std = float(np.std(y))
    if y_std <= 1e-12:
        return 0.0
    x_center = x - np.mean(x)
    slope = float(np.dot(x_center, y - np.mean(y)) / max(np.dot(x_center, x_center), 1e-12))
    return float((slope * n) / y_std)


def _max_nan_run_fraction(series: pd.Series) -> float:
    mask = series.isna().to_numpy(dtype=bool)
    if len(mask) == 0:
        return 0.0
    max_run = 0
    run = 0
    for item in mask:
        if item:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return float(max_run / len(mask))


def _max_stuck_run_fraction(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return 0.0
    max_run = 1
    run = 1
    for idx in range(1, n):
        prev = values[idx - 1]
        curr = values[idx]
        if np.isnan(prev) or np.isnan(curr):
            run = 1
            continue
        if abs(curr - prev) <= 1e-12:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return float(max_run / n)


def _timestamp_jitter_cv(frame: pd.DataFrame, *, timestamp_column: str | None) -> float:
    if not timestamp_column or timestamp_column not in frame.columns:
        return float("nan")
    series = frame[timestamp_column]
    parsed = pd.to_datetime(series, errors="coerce")
    if float(parsed.notna().mean()) >= 0.8:
        diffs = parsed.dropna().diff().dt.total_seconds().dropna().to_numpy(dtype=float)
    else:
        numeric = pd.to_numeric(series, errors="coerce")
        diffs = numeric.dropna().diff().dropna().to_numpy(dtype=float)
    diffs = diffs[diffs > 0]
    if len(diffs) < 5:
        return float("nan")
    mean = float(np.mean(diffs))
    std = float(np.std(diffs))
    if mean <= 1e-12:
        return float("nan")
    return float(std / mean)

