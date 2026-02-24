"""Performance diagnostics and data-trajectory recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import RegressionMetrics, regression_metrics


@dataclass(frozen=True)
class BadRegion:
    """Region where prediction error is elevated."""

    feature: str
    lower_bound: float
    upper_bound: float
    sample_count: int
    mean_abs_error: float


@dataclass(frozen=True)
class TrajectorySuggestion:
    """Recommended data-collection trajectory for lab/testbench."""

    title: str
    rationale: str
    trajectory_type: str
    operating_region: dict[str, float]


@dataclass(frozen=True)
class PerformanceFeedback:
    """Comprehensive post-evaluation feedback package."""

    summary: str
    metrics: RegressionMetrics
    bad_regions: list[BadRegion]
    trajectory_suggestions: list[TrajectorySuggestion]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "metrics": self.metrics.to_dict(),
            "bad_regions": [asdict(item) for item in self.bad_regions],
            "trajectory_suggestions": [asdict(item) for item in self.trajectory_suggestions],
        }


def analyze_model_performance(
    *,
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    feature_frame: pd.DataFrame,
    top_k_regions: int = 3,
    trajectory_budget: int = 3,
) -> PerformanceFeedback:
    """Analyze model errors and suggest high-value additional data collection."""
    metrics = regression_metrics(y_true=y_true, y_pred=y_pred)
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if feature_frame.shape[0] != true.shape[0]:
        raise ValueError("feature_frame row count must match y_true/y_pred length.")

    abs_error = np.abs(pred - true)
    numeric_features = [
        col
        for col in feature_frame.columns
        if pd.to_numeric(feature_frame[col], errors="coerce").notna().sum() >= 5
    ]
    regions = _identify_bad_regions(
        feature_frame=feature_frame,
        abs_error=abs_error,
        feature_columns=numeric_features,
        top_k_regions=top_k_regions,
    )
    suggestions = _build_trajectory_suggestions(
        regions=regions,
        budget=trajectory_budget,
        metrics=metrics,
    )
    summary = _build_summary(metrics=metrics, regions=regions)
    return PerformanceFeedback(
        summary=summary,
        metrics=metrics,
        bad_regions=regions,
        trajectory_suggestions=suggestions,
    )


def _identify_bad_regions(
    *,
    feature_frame: pd.DataFrame,
    abs_error: np.ndarray,
    feature_columns: list[str],
    top_k_regions: int,
) -> list[BadRegion]:
    region_scores: list[BadRegion] = []
    n_rows = int(len(abs_error))
    minimum_region_size = max(5, int(0.05 * n_rows))

    for feature in feature_columns:
        numeric = pd.to_numeric(feature_frame[feature], errors="coerce")
        valid = numeric.notna()
        if int(valid.sum()) < minimum_region_size:
            continue
        values = numeric[valid].to_numpy(dtype=float)
        errors = abs_error[valid.to_numpy()]
        quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
        boundaries = np.unique(quantiles)
        if len(boundaries) < 2:
            continue
        for idx in range(len(boundaries) - 1):
            low = float(boundaries[idx])
            high = float(boundaries[idx + 1])
            if idx == len(boundaries) - 2:
                mask = (values >= low) & (values <= high)
            else:
                mask = (values >= low) & (values < high)
            count = int(np.sum(mask))
            if count < minimum_region_size:
                continue
            mean_abs_error = float(np.mean(errors[mask]))
            region_scores.append(
                BadRegion(
                    feature=feature,
                    lower_bound=low,
                    upper_bound=high,
                    sample_count=count,
                    mean_abs_error=mean_abs_error,
                )
            )

    region_scores.sort(key=lambda item: item.mean_abs_error, reverse=True)
    return region_scores[: max(int(top_k_regions), 1)]


def _build_trajectory_suggestions(
    *,
    regions: list[BadRegion],
    budget: int,
    metrics: RegressionMetrics,
) -> list[TrajectorySuggestion]:
    suggestions: list[TrajectorySuggestion] = []
    remaining = max(int(budget), 1)

    if not regions:
        suggestions.append(
            TrajectorySuggestion(
                title="General coverage expansion",
                rationale=(
                    "No dominant bad region was isolated from current data. "
                    "Collect trajectories spanning low/mid/high operating ranges and include dynamics."
                ),
                trajectory_type="coverage_expansion",
                operating_region={},
            )
        )
        return suggestions[:remaining]

    for region in regions:
        if remaining <= 0:
            break
        suggestions.append(
            TrajectorySuggestion(
                title=f"Dense sweep for {region.feature}",
                rationale=(
                    f"Highest local error appears in {region.feature} range "
                    f"[{region.lower_bound:.3g}, {region.upper_bound:.3g}] "
                    f"with mean abs error {region.mean_abs_error:.4f}."
                ),
                trajectory_type="dense_sweep",
                operating_region={
                    "feature_min": region.lower_bound,
                    "feature_max": region.upper_bound,
                },
            )
        )
        remaining -= 1

    if remaining > 0 and regions:
        worst = regions[0]
        suggestions.append(
            TrajectorySuggestion(
                title="Transient boundary ramps",
                rationale=(
                    f"Model error may spike around regime boundaries near {worst.feature}. "
                    "Run ramp-up/ramp-down trajectories crossing the high-error range."
                ),
                trajectory_type="transient_ramp",
                operating_region={
                    "feature_min": worst.lower_bound,
                    "feature_max": worst.upper_bound,
                },
            )
        )
        remaining -= 1

    if remaining > 0 and metrics.r2 < 0.8:
        suggestions.append(
            TrajectorySuggestion(
                title="Coverage expansion campaign",
                rationale=(
                    "Global fit is limited. Expand operating coverage with low/high extremes "
                    "and mixed-condition trajectories."
                ),
                trajectory_type="coverage_expansion",
                operating_region={},
            )
        )

    return suggestions[: max(int(budget), 1)]


def _build_summary(*, metrics: RegressionMetrics, regions: list[BadRegion]) -> str:
    if not regions:
        return (
            f"Model metrics: MAE={metrics.mae:.4f}, RMSE={metrics.rmse:.4f}, "
            f"R2={metrics.r2:.4f}. No dominant bad region identified."
        )
    worst = regions[0]
    return (
        f"Model metrics: MAE={metrics.mae:.4f}, RMSE={metrics.rmse:.4f}, "
        f"R2={metrics.r2:.4f}. Highest local error is on {worst.feature} in "
        f"[{worst.lower_bound:.3g}, {worst.upper_bound:.3g}]."
    )
