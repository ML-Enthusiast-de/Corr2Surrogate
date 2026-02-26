"""Trajectory recommendations for targeted data collection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExperimentRecommendation:
    """Concrete suggestion for next data collection trajectory."""

    target_signal: str
    priority: int
    trajectory_type: str
    required_signals: list[str]
    suggestion: str
    rationale: str
    score: float


def recommend_data_trajectories(
    *,
    frame: pd.DataFrame,
    correlations: dict[str, Any],
    sensor_diagnostics: dict[str, Any] | None = None,
    max_recommendations: int = 12,
) -> list[ExperimentRecommendation]:
    """Build actionable data-acquisition recommendations."""
    target_analyses = list((correlations or {}).get("target_analyses", []))
    diagnostics_lookup = _diagnostics_lookup(sensor_diagnostics or {})
    recs: list[ExperimentRecommendation] = []

    for analysis in target_analyses:
        target = str(analysis.get("target_signal", "unknown"))
        predictors = list(analysis.get("predictor_results", []))[:5]
        if not predictors:
            recs.append(
                ExperimentRecommendation(
                    target_signal=target,
                    priority=1,
                    trajectory_type="baseline_extension",
                    required_signals=[],
                    suggestion=(
                        f"Collect a broader operating envelope for `{target}` with synchronized "
                        "core process signals."
                    ),
                    rationale="No stable predictors found in current data.",
                    score=0.95,
                )
            )
            continue

        for rank, predictor in enumerate(predictors, start=1):
            pred_name = str(predictor.get("predictor_signal", "n/a"))
            best_abs = _safe_float(predictor.get("best_abs_score"))
            stability = _safe_float(predictor.get("stability_score"))
            gain_potential = max(0.0, 0.85 - best_abs)
            instability = max(0.0, 0.6 - stability) if np.isfinite(stability) else 0.25
            trust_penalty = 0.0
            diag = diagnostics_lookup.get(pred_name)
            if diag is not None:
                trust_penalty = max(0.0, 0.8 - _safe_float(diag.get("trust_score")))
            rec_score = min(1.0, gain_potential + instability + trust_penalty + 0.05 * rank)
            if rec_score < 0.25:
                continue

            low, high = _quantile_range(frame, pred_name)
            if np.isfinite(low) and np.isfinite(high):
                sweep = (
                    f"Sweep `{pred_name}` across [{low:.3g}, {high:.3g}] "
                    "with slow ramps and short hold segments; record full transient response."
                )
            else:
                sweep = (
                    f"Excite `{pred_name}` through low/mid/high regimes with reproducible ramps "
                    "and hold segments; record full transient response."
                )

            rationale_parts = [
                f"target-predictor association currently {best_abs:.3f}",
            ]
            if np.isfinite(stability):
                rationale_parts.append(f"window stability={stability:.3f}")
            if diag is not None and diag.get("flags"):
                rationale_parts.append(f"sensor diagnostics flagged: {diag.get('flags')}")

            recs.append(
                ExperimentRecommendation(
                    target_signal=target,
                    priority=rank,
                    trajectory_type="single_factor_sweep",
                    required_signals=[pred_name],
                    suggestion=sweep,
                    rationale="; ".join(rationale_parts),
                    score=float(rec_score),
                )
            )

    recs.sort(key=lambda item: item.score, reverse=True)
    return recs[:max_recommendations]


def recommendations_to_dict(items: list[ExperimentRecommendation]) -> list[dict[str, Any]]:
    return [asdict(item) for item in items]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _quantile_range(frame: pd.DataFrame, signal: str) -> tuple[float, float]:
    if signal not in frame.columns:
        return float("nan"), float("nan")
    values = pd.to_numeric(frame[signal], errors="coerce").dropna()
    if len(values) < 8:
        return float("nan"), float("nan")
    return float(values.quantile(0.10)), float(values.quantile(0.90))


def _diagnostics_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = list(payload.get("diagnostics", []))
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("signal", "")).strip()
        if key:
            lookup[key] = row
    return lookup

