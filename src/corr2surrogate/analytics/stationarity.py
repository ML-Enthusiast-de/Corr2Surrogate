"""Heuristic stationarity assessment for sensor time series."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StationaritySignalResult:
    """Stationarity diagnostics for one signal."""

    signal: str
    sample_count: int
    status: str
    confidence: float
    mean_shift_z: float
    variance_ratio: float
    slope_normalized: float
    reason: str


@dataclass(frozen=True)
class StationaritySummary:
    """Aggregate stationarity summary for analyzed signals."""

    analyzed_signals: int
    stationary_signals: int
    non_stationary_signals: int
    inconclusive_signals: int
    results: list[StationaritySignalResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyzed_signals": self.analyzed_signals,
            "stationary_signals": self.stationary_signals,
            "non_stationary_signals": self.non_stationary_signals,
            "inconclusive_signals": self.inconclusive_signals,
            "results": [asdict(item) for item in self.results],
        }


def assess_stationarity(
    frame: pd.DataFrame,
    *,
    signal_columns: list[str] | None = None,
    min_samples: int = 30,
) -> StationaritySummary:
    """Assess stationarity using robust heuristics for practical use."""
    columns = signal_columns or _numeric_columns(frame)
    results: list[StationaritySignalResult] = []

    for signal in columns:
        series = pd.to_numeric(frame[signal], errors="coerce").dropna()
        results.append(_analyze_series(signal=signal, values=series, min_samples=min_samples))

    stationary = sum(1 for item in results if item.status == "stationary")
    non_stationary = sum(1 for item in results if item.status == "non_stationary")
    inconclusive = sum(1 for item in results if item.status == "inconclusive")

    return StationaritySummary(
        analyzed_signals=len(results),
        stationary_signals=stationary,
        non_stationary_signals=non_stationary,
        inconclusive_signals=inconclusive,
        results=results,
    )


def _analyze_series(
    *,
    signal: str,
    values: pd.Series,
    min_samples: int,
) -> StationaritySignalResult:
    n = int(len(values))
    if n < min_samples:
        return StationaritySignalResult(
            signal=signal,
            sample_count=n,
            status="inconclusive",
            confidence=0.25,
            mean_shift_z=0.0,
            variance_ratio=1.0,
            slope_normalized=0.0,
            reason=f"Not enough samples ({n} < {min_samples}).",
        )

    arr = values.to_numpy(dtype=float)
    std_all = float(np.std(arr))
    if std_all < 1e-12:
        return StationaritySignalResult(
            signal=signal,
            sample_count=n,
            status="stationary",
            confidence=0.95,
            mean_shift_z=0.0,
            variance_ratio=1.0,
            slope_normalized=0.0,
            reason="Signal is effectively constant.",
        )

    split = max(5, n // 3)
    first = arr[:split]
    last = arr[-split:]

    mean_shift_z = float(abs(np.mean(last) - np.mean(first)) / (std_all + 1e-12))
    var_first = float(np.var(first))
    var_last = float(np.var(last))
    variance_ratio = float(var_last / (var_first + 1e-12))

    t = np.arange(n, dtype=float)
    slope = float(np.polyfit(t, arr, 1)[0])
    slope_normalized = float(abs(slope) / (std_all + 1e-12))

    stationary = (
        mean_shift_z < 0.5
        and 0.5 <= variance_ratio <= 2.0
        and slope_normalized < 0.01
    )

    status = "stationary" if stationary else "non_stationary"
    confidence = _stationarity_confidence(
        mean_shift_z=mean_shift_z,
        variance_ratio=variance_ratio,
        slope_normalized=slope_normalized,
        stationary=stationary,
    )
    reason = _stationarity_reason(
        mean_shift_z=mean_shift_z,
        variance_ratio=variance_ratio,
        slope_normalized=slope_normalized,
        stationary=stationary,
    )
    return StationaritySignalResult(
        signal=signal,
        sample_count=n,
        status=status,
        confidence=confidence,
        mean_shift_z=mean_shift_z,
        variance_ratio=variance_ratio,
        slope_normalized=slope_normalized,
        reason=reason,
    )


def _stationarity_confidence(
    *,
    mean_shift_z: float,
    variance_ratio: float,
    slope_normalized: float,
    stationary: bool,
) -> float:
    ratio_penalty = min(abs(np.log(max(variance_ratio, 1e-12))), 2.0) / 2.0
    shift_penalty = min(mean_shift_z / 1.5, 1.0)
    slope_penalty = min(slope_normalized / 0.05, 1.0)
    aggregate = float((ratio_penalty + shift_penalty + slope_penalty) / 3.0)
    if stationary:
        return float(max(0.5, 1.0 - aggregate))
    return float(max(0.5, aggregate))


def _stationarity_reason(
    *,
    mean_shift_z: float,
    variance_ratio: float,
    slope_normalized: float,
    stationary: bool,
) -> str:
    if stationary:
        return (
            f"Stable mean/variance trend (mean_shift_z={mean_shift_z:.3f}, "
            f"variance_ratio={variance_ratio:.3f}, slope_norm={slope_normalized:.4f})."
        )
    triggers: list[str] = []
    if mean_shift_z >= 0.5:
        triggers.append(f"mean drift ({mean_shift_z:.3f})")
    if variance_ratio < 0.5 or variance_ratio > 2.0:
        triggers.append(f"variance shift ({variance_ratio:.3f})")
    if slope_normalized >= 0.01:
        triggers.append(f"trend slope ({slope_normalized:.4f})")
    return "Likely non-stationary due to " + ", ".join(triggers) + "."


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().sum() >= 5:
            cols.append(col)
    return cols
