"""Correlation analysis and feature-engineering opportunity discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Any

import numpy as np
import pandas as pd

from .ranking import CandidateSignal


@dataclass(frozen=True)
class PairCorrelationResult:
    """Correlation scores between one target and one predictor."""

    target_signal: str
    predictor_signal: str
    sample_count: int
    pearson: float
    spearman: float
    kendall: float
    distance_corr: float
    best_lag: int
    lagged_pearson: float
    best_method: str
    best_abs_score: float
    pearson_ci_low: float = float("nan")
    pearson_ci_high: float = float("nan")
    pearson_pvalue: float = float("nan")
    spearman_ci_low: float = float("nan")
    spearman_ci_high: float = float("nan")
    spearman_pvalue: float = float("nan")
    stability_score: float = float("nan")
    confounder_signal: str = ""
    partial_pearson: float = float("nan")
    conditional_mi: float = float("nan")


@dataclass(frozen=True)
class FeatureEngineeringOpportunity:
    """Transformation candidate that improves target association."""

    target_signal: str
    base_signal: str
    transformation: str
    expression: str
    score_abs: float
    gain_over_raw: float
    notes: str


@dataclass(frozen=True)
class TargetCorrelationAnalysis:
    """Per-target correlation analysis results."""

    target_signal: str
    predictor_results: list[PairCorrelationResult]
    top_predictors: list[str]
    feature_opportunities: list[FeatureEngineeringOpportunity]


@dataclass(frozen=True)
class CorrelationAnalysisBundle:
    """Aggregate correlation analysis output."""

    data_mode: str
    timestamp_column: str | None
    target_analyses: list[TargetCorrelationAnalysis]

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_mode": self.data_mode,
            "timestamp_column": self.timestamp_column,
            "target_analyses": [
                {
                    "target_signal": item.target_signal,
                    "predictor_results": [asdict(p) for p in item.predictor_results],
                    "top_predictors": item.top_predictors,
                    "feature_opportunities": [asdict(op) for op in item.feature_opportunities],
                }
                for item in self.target_analyses
            ],
        }


def run_correlation_analysis(
    frame: pd.DataFrame,
    *,
    target_signals: list[str] | None = None,
    predictor_signals_by_target: dict[str, list[str]] | None = None,
    timestamp_column: str | None = None,
    max_lag: int = 8,
    min_samples: int = 8,
    include_feature_engineering: bool = True,
    feature_gain_threshold: float = 0.05,
    top_k_predictors: int = 10,
    feature_scan_predictors: int = 10,
    feature_pair_predictors: int = 5,
    max_feature_opportunities: int = 20,
    confidence_top_k: int = 10,
    bootstrap_rounds: int = 40,
    stability_windows: int = 4,
) -> CorrelationAnalysisBundle:
    """Run multi-technique correlation analysis for each target."""
    ts_col = _resolve_timestamp_column(frame, explicit=timestamp_column)
    data_mode = _infer_data_mode(frame, timestamp_column=ts_col)
    numeric = _numeric_signal_columns(frame, exclude={ts_col} if ts_col else set())

    targets = _resolve_targets(numeric_columns=numeric, target_signals=target_signals)
    analyses: list[TargetCorrelationAnalysis] = []

    for target in targets:
        predictors = _resolve_predictors(
            numeric_columns=numeric,
            target=target,
            predictor_signals_by_target=predictor_signals_by_target,
        )
        pair_results: list[PairCorrelationResult] = []
        for predictor in predictors:
            pair = _pairwise_result(
                frame=frame,
                target=target,
                predictor=predictor,
                data_mode=data_mode,
                max_lag=max_lag,
                min_samples=min_samples,
            )
            if pair is not None:
                pair_results.append(pair)

        pair_results.sort(key=lambda item: item.best_abs_score, reverse=True)
        if pair_results and confidence_top_k > 0:
            pair_results = _augment_top_pairs_with_confidence_and_confounding(
                frame=frame,
                target=target,
                pair_results=pair_results,
                top_k=min(confidence_top_k, len(pair_results)),
                bootstrap_rounds=bootstrap_rounds,
                stability_windows=stability_windows,
                data_mode=data_mode,
                min_samples=min_samples,
            )
        top_predictors = [item.predictor_signal for item in pair_results[:top_k_predictors]]
        opportunities: list[FeatureEngineeringOpportunity] = []
        if include_feature_engineering:
            opportunities = discover_feature_engineering_opportunities(
                frame=frame,
                target_signal=target,
                predictor_results=pair_results,
                data_mode=data_mode,
                gain_threshold=feature_gain_threshold,
                max_lag=max_lag,
                max_opportunities=max_feature_opportunities,
                max_predictors_to_scan=feature_scan_predictors,
                max_pair_predictors=feature_pair_predictors,
            )
        analyses.append(
            TargetCorrelationAnalysis(
                target_signal=target,
                predictor_results=pair_results,
                top_predictors=top_predictors,
                feature_opportunities=opportunities,
            )
        )

    return CorrelationAnalysisBundle(
        data_mode=data_mode,
        timestamp_column=ts_col,
        target_analyses=analyses,
    )


def build_candidate_signals_from_correlations(
    bundle: CorrelationAnalysisBundle,
    *,
    required_signals_top_k: int = 3,
) -> list[CandidateSignal]:
    """Convert correlation results into ranking candidates."""
    candidates: list[CandidateSignal] = []
    for analysis in bundle.target_analyses:
        predictors = analysis.predictor_results
        if predictors:
            top_scores = [item.best_abs_score for item in predictors[:3]]
            top_primary = predictors[0].best_abs_score
            score = float(0.6 * top_primary + 0.4 * float(np.mean(top_scores)))
        else:
            score = 0.0

        feature_bonus = min(0.15, 0.03 * len(analysis.feature_opportunities))
        adjusted = max(0.0, min(1.0, score + feature_bonus))
        required = analysis.top_predictors[:required_signals_top_k]
        note = (
            "No usable predictors."
            if not predictors
            else f"Top predictor: {predictors[0].predictor_signal} via {predictors[0].best_method}."
        )
        candidates.append(
            CandidateSignal(
                target_signal=analysis.target_signal,
                base_score=adjusted,
                required_signals=required,
                notes=note,
            )
        )
    return candidates


def discover_feature_engineering_opportunities(
    *,
    frame: pd.DataFrame,
    target_signal: str,
    predictor_results: list[PairCorrelationResult],
    data_mode: str,
    gain_threshold: float = 0.05,
    max_lag: int = 8,
    max_opportunities: int = 20,
    max_predictors_to_scan: int = 10,
    max_pair_predictors: int = 5,
) -> list[FeatureEngineeringOpportunity]:
    """Discover transformations with stronger target association than raw signal."""
    opportunities: list[FeatureEngineeringOpportunity] = []
    if not predictor_results:
        return opportunities

    target_series = pd.to_numeric(frame[target_signal], errors="coerce")
    top_predictors = predictor_results[: max(1, max_predictors_to_scan)]

    for item in top_predictors:
        predictor = item.predictor_signal
        raw = pd.to_numeric(frame[predictor], errors="coerce")
        base = abs(_safe_corr(target_series, raw, method="pearson"))
        transforms = _univariate_transforms(raw, data_mode=data_mode, max_lag=max_lag)
        for name, transformed in transforms:
            score = abs(_safe_corr(target_series, transformed, method="pearson"))
            gain = score - base
            if score >= 0.2 and gain >= gain_threshold:
                opportunities.append(
                    FeatureEngineeringOpportunity(
                        target_signal=target_signal,
                        base_signal=predictor,
                        transformation=name,
                        expression=f"{name}({predictor})",
                        score_abs=float(score),
                        gain_over_raw=float(gain),
                        notes=f"Improves abs Pearson from {base:.3f} to {score:.3f}.",
                    )
                )

    if len(top_predictors) >= 2:
        pair_cols = [
            item.predictor_signal for item in top_predictors[: max(2, max_pair_predictors)]
        ]
        for a, b in combinations(pair_cols, 2):
            sa = pd.to_numeric(frame[a], errors="coerce")
            sb = pd.to_numeric(frame[b], errors="coerce")
            target = target_series
            base = max(
                abs(_safe_corr(target, sa, method="pearson")),
                abs(_safe_corr(target, sb, method="pearson")),
            )
            product = sa * sb
            ratio = sa / (sb.replace(0, np.nan))

            product_score = abs(_safe_corr(target, product, method="pearson"))
            if product_score >= 0.2 and (product_score - base) >= gain_threshold:
                opportunities.append(
                    FeatureEngineeringOpportunity(
                        target_signal=target_signal,
                        base_signal=f"{a},{b}",
                        transformation="product",
                        expression=f"{a}*{b}",
                        score_abs=float(product_score),
                        gain_over_raw=float(product_score - base),
                        notes="Pair interaction improves target association.",
                    )
                )

            ratio_score = abs(_safe_corr(target, ratio, method="pearson"))
            if ratio_score >= 0.2 and (ratio_score - base) >= gain_threshold:
                opportunities.append(
                    FeatureEngineeringOpportunity(
                        target_signal=target_signal,
                        base_signal=f"{a},{b}",
                        transformation="ratio",
                        expression=f"{a}/({b}+eps)",
                        score_abs=float(ratio_score),
                        gain_over_raw=float(ratio_score - base),
                        notes="Ratio feature improves target association.",
                    )
                )

    opportunities.sort(key=lambda item: item.gain_over_raw, reverse=True)
    return opportunities[:max_opportunities]


def _pairwise_result(
    *,
    frame: pd.DataFrame,
    target: str,
    predictor: str,
    data_mode: str,
    max_lag: int,
    min_samples: int,
) -> PairCorrelationResult | None:
    target_series = pd.to_numeric(frame[target], errors="coerce")
    predictor_series = pd.to_numeric(frame[predictor], errors="coerce")
    aligned = pd.concat([target_series, predictor_series], axis=1).dropna()
    if len(aligned) < min_samples:
        return None

    x = aligned.iloc[:, 0]
    y = aligned.iloc[:, 1]
    pearson = _safe_corr(x, y, method="pearson")
    spearman = _safe_corr(x, y, method="spearman")
    kendall = _safe_corr(x, y, method="kendall")
    distance_corr = _distance_correlation(x.to_numpy(dtype=float), y.to_numpy(dtype=float))

    best_lag = 0
    lagged = float("nan")
    if data_mode == "time_series":
        best_lag, lagged = _best_lagged_pearson(
            target_series=target_series,
            predictor_series=predictor_series,
            max_lag=max_lag,
            min_samples=min_samples,
        )

    methods = {
        "pearson": pearson,
        "spearman": spearman,
        "kendall": kendall,
        "distance_corr": distance_corr,
        "lagged_pearson": lagged,
    }
    best_method, best_score = _best_method(methods)
    return PairCorrelationResult(
        target_signal=target,
        predictor_signal=predictor,
        sample_count=int(len(aligned)),
        pearson=float(pearson),
        spearman=float(spearman),
        kendall=float(kendall),
        distance_corr=float(distance_corr),
        best_lag=int(best_lag),
        lagged_pearson=float(lagged),
        best_method=best_method,
        best_abs_score=float(best_score),
    )


def _best_method(methods: dict[str, float]) -> tuple[str, float]:
    valid = {k: abs(v) for k, v in methods.items() if np.isfinite(v)}
    if not valid:
        return "none", 0.0
    best = max(valid.items(), key=lambda item: item[1])
    return best[0], float(best[1])


def _best_lagged_pearson(
    *,
    target_series: pd.Series,
    predictor_series: pd.Series,
    max_lag: int,
    min_samples: int,
) -> tuple[int, float]:
    best_lag = 0
    best_score = float("nan")
    best_abs = -1.0
    for lag in range(1, max_lag + 1):
        shifted = predictor_series.shift(lag)
        aligned = pd.concat([target_series, shifted], axis=1).dropna()
        if len(aligned) < min_samples:
            continue
        score = _safe_corr(aligned.iloc[:, 0], aligned.iloc[:, 1], method="pearson")
        if np.isfinite(score) and abs(score) > best_abs:
            best_abs = abs(score)
            best_score = score
            best_lag = lag
    return best_lag, best_score


def _distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = x.shape[0]
    if n < 3:
        return float("nan")
    max_samples = 600
    if n > max_samples:
        indices = np.linspace(0, n - 1, max_samples, dtype=int)
        x = x[indices]
        y = y[indices]
        n = max_samples

    a = np.abs(x - x.T)
    b = np.abs(y - y.T)
    A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()

    dcov2 = float(np.mean(A * B))
    dvarx2 = float(np.mean(A * A))
    dvary2 = float(np.mean(B * B))
    if dvarx2 <= 1e-12 or dvary2 <= 1e-12:
        return 0.0
    dcor = np.sqrt(max(dcov2, 0.0)) / np.sqrt(np.sqrt(dvarx2 * dvary2))
    return float(max(0.0, min(1.0, dcor)))


def _univariate_transforms(
    series: pd.Series,
    *,
    data_mode: str,
    max_lag: int,
) -> list[tuple[str, pd.Series]]:
    eps = 1e-9
    transforms: list[tuple[str, pd.Series]] = [
        ("signed_log", np.sign(series) * np.log1p(np.abs(series))),
        ("square", np.square(series)),
        ("sqrt_abs", np.sqrt(np.abs(series))),
        ("inverse", 1.0 / (series + eps)),
    ]
    if data_mode == "time_series":
        transforms.append(("delta", series.diff()))
        for lag in range(1, min(max_lag, 3) + 1):
            transforms.append((f"lag{lag}", series.shift(lag)))
    return transforms


def _safe_corr(a: pd.Series, b: pd.Series, *, method: str) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < 3:
        return float("nan")
    x = pd.to_numeric(aligned.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(aligned.iloc[:, 1], errors="coerce")
    aligned_numeric = pd.concat([x, y], axis=1).dropna()
    if len(aligned_numeric) < 3:
        return float("nan")
    x = aligned_numeric.iloc[:, 0]
    y = aligned_numeric.iloc[:, 1]

    if method == "pearson":
        return _pearson_numeric(x.to_numpy(dtype=float), y.to_numpy(dtype=float))
    if method == "spearman":
        xr = x.rank(method="average")
        yr = y.rank(method="average")
        return _pearson_numeric(xr.to_numpy(dtype=float), yr.to_numpy(dtype=float))
    if method == "kendall":
        try:
            from scipy.stats import kendalltau  # type: ignore

            value = kendalltau(x.to_numpy(dtype=float), y.to_numpy(dtype=float))[0]
            return float(value) if value is not None else float("nan")
        except Exception:
            # Fallback for local environments without scipy.
            xr = x.rank(method="average")
            yr = y.rank(method="average")
            return _pearson_numeric(xr.to_numpy(dtype=float), yr.to_numpy(dtype=float))

    value = x.corr(y, method=method)
    return float(value) if value is not None else float("nan")


def _augment_top_pairs_with_confidence_and_confounding(
    *,
    frame: pd.DataFrame,
    target: str,
    pair_results: list[PairCorrelationResult],
    top_k: int,
    bootstrap_rounds: int,
    stability_windows: int,
    data_mode: str,
    min_samples: int,
) -> list[PairCorrelationResult]:
    if top_k <= 0:
        return pair_results
    updated = list(pair_results)
    top_predictors = [item.predictor_signal for item in pair_results[:top_k]]
    if not top_predictors:
        return updated
    confounder_by_predictor: dict[str, str] = {}
    fallback = top_predictors[0]
    for predictor in top_predictors:
        alternatives = [name for name in top_predictors if name != predictor]
        confounder_by_predictor[predictor] = alternatives[0] if alternatives else fallback

    for idx in range(top_k):
        base = updated[idx]
        x = pd.to_numeric(frame[target], errors="coerce")
        y = pd.to_numeric(frame[base.predictor_signal], errors="coerce")
        aligned_xy = pd.concat([x, y], axis=1).dropna()
        if len(aligned_xy) < min_samples:
            continue
        x_arr = aligned_xy.iloc[:, 0].to_numpy(dtype=float)
        y_arr = aligned_xy.iloc[:, 1].to_numpy(dtype=float)

        pearson_ci_low, pearson_ci_high, pearson_pvalue = _bootstrap_ci_and_pvalue(
            x_arr, y_arr, method="pearson", rounds=bootstrap_rounds
        )
        spearman_ci_low, spearman_ci_high, spearman_pvalue = _bootstrap_ci_and_pvalue(
            x_arr, y_arr, method="spearman", rounds=bootstrap_rounds
        )
        stability_score = _window_stability_score(
            x_arr, y_arr, windows=stability_windows, data_mode=data_mode
        )

        confounder = confounder_by_predictor.get(base.predictor_signal, "")
        partial_pearson = float("nan")
        conditional_mi = float("nan")
        if confounder and confounder in frame.columns and confounder != base.predictor_signal:
            z = pd.to_numeric(frame[confounder], errors="coerce")
            aligned_xyz = pd.concat([x, y, z], axis=1).dropna()
            if len(aligned_xyz) >= min_samples:
                xv = aligned_xyz.iloc[:, 0].to_numpy(dtype=float)
                yv = aligned_xyz.iloc[:, 1].to_numpy(dtype=float)
                zv = aligned_xyz.iloc[:, 2].to_numpy(dtype=float)
                partial_pearson = _partial_correlation(xv, yv, zv)
                conditional_mi = _conditional_mi_gaussian(xv, yv, zv)

        updated[idx] = PairCorrelationResult(
            target_signal=base.target_signal,
            predictor_signal=base.predictor_signal,
            sample_count=base.sample_count,
            pearson=base.pearson,
            spearman=base.spearman,
            kendall=base.kendall,
            distance_corr=base.distance_corr,
            best_lag=base.best_lag,
            lagged_pearson=base.lagged_pearson,
            best_method=base.best_method,
            best_abs_score=base.best_abs_score,
            pearson_ci_low=pearson_ci_low,
            pearson_ci_high=pearson_ci_high,
            pearson_pvalue=pearson_pvalue,
            spearman_ci_low=spearman_ci_low,
            spearman_ci_high=spearman_ci_high,
            spearman_pvalue=spearman_pvalue,
            stability_score=stability_score,
            confounder_signal=confounder,
            partial_pearson=partial_pearson,
            conditional_mi=conditional_mi,
        )
    return updated


def _bootstrap_ci_and_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    rounds: int,
) -> tuple[float, float, float]:
    if len(x) < 4 or len(y) < 4:
        return float("nan"), float("nan"), float("nan")
    observed = _corr_array(x, y, method=method)
    if not math.isfinite(observed):
        return float("nan"), float("nan"), float("nan")

    if rounds <= 0:
        pvalue = _approx_corr_pvalue(observed, n=len(x))
        return float("nan"), float("nan"), pvalue

    rng = np.random.default_rng(42)
    vals: list[float] = []
    n = len(x)
    for _ in range(rounds):
        indices = rng.integers(0, n, size=n)
        val = _corr_array(x[indices], y[indices], method=method)
        if math.isfinite(val):
            vals.append(float(val))
    if len(vals) < 5:
        return float("nan"), float("nan"), _approx_corr_pvalue(observed, n=len(x))

    low = float(np.quantile(vals, 0.025))
    high = float(np.quantile(vals, 0.975))
    return low, high, _approx_corr_pvalue(observed, n=len(x))


def _corr_array(x: np.ndarray, y: np.ndarray, *, method: str) -> float:
    xs = pd.Series(x)
    ys = pd.Series(y)
    return _safe_corr(xs, ys, method=method)


def _approx_corr_pvalue(rho: float, *, n: int) -> float:
    if n < 4 or not math.isfinite(rho):
        return float("nan")
    clipped = max(min(rho, 0.999999), -0.999999)
    z = abs(np.arctanh(clipped)) * math.sqrt(max(n - 3, 1))
    p = 2.0 * (1.0 - _normal_cdf(z))
    return float(max(0.0, min(1.0, p)))


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _window_stability_score(
    x: np.ndarray,
    y: np.ndarray,
    *,
    windows: int,
    data_mode: str,
) -> float:
    n = len(x)
    if n < 20 or windows < 2:
        return float("nan")
    order_x = x
    order_y = y
    if data_mode != "time_series":
        rng = np.random.default_rng(7)
        idx = rng.permutation(n)
        order_x = x[idx]
        order_y = y[idx]
    splits = np.array_split(np.arange(n), windows)
    vals: list[float] = []
    for split in splits:
        if len(split) < 4:
            continue
        corr = _corr_array(order_x[split], order_y[split], method="pearson")
        if math.isfinite(corr):
            vals.append(abs(corr))
    if len(vals) < 2:
        return float("nan")
    mean = float(np.mean(vals))
    std = float(np.std(vals))
    if mean <= 1e-8:
        return 0.0
    score = 1.0 - min(1.0, std / mean)
    return float(max(0.0, min(1.0, score)))


def _partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    if len(x) < 4:
        return float("nan")
    zc = z - np.mean(z)
    denom = float(np.dot(zc, zc))
    if denom <= 1e-12:
        return _pearson_numeric(x, y)
    bx = float(np.dot(x - np.mean(x), zc) / denom)
    by = float(np.dot(y - np.mean(y), zc) / denom)
    rx = x - (np.mean(x) + bx * zc)
    ry = y - (np.mean(y) + by * zc)
    return _pearson_numeric(rx, ry)


def _conditional_mi_gaussian(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    rho = _partial_correlation(x, y, z)
    if not math.isfinite(rho):
        return float("nan")
    clipped = max(min(abs(rho), 0.999999), 0.0)
    return float(max(0.0, -0.5 * math.log(max(1e-12, 1.0 - clipped * clipped))))


def _pearson_numeric(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return float("nan")
    return corr


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


def _infer_data_mode(frame: pd.DataFrame, *, timestamp_column: str | None) -> str:
    if timestamp_column and timestamp_column in frame.columns:
        parsed = pd.to_datetime(frame[timestamp_column], errors="coerce")
        valid_ratio = float(parsed.notna().mean())
        monotonic = bool(parsed.dropna().is_monotonic_increasing)
        if valid_ratio >= 0.8 and monotonic:
            return "time_series"
    return "steady_state"


def _numeric_signal_columns(frame: pd.DataFrame, *, exclude: set[str]) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in exclude:
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if int(numeric.notna().sum()) >= 8 and numeric.nunique(dropna=True) > 1:
            cols.append(col)
    return cols


def _resolve_targets(
    *,
    numeric_columns: list[str],
    target_signals: list[str] | None,
) -> list[str]:
    if target_signals:
        return [signal for signal in target_signals if signal in numeric_columns]
    return list(numeric_columns)


def _resolve_predictors(
    *,
    numeric_columns: list[str],
    target: str,
    predictor_signals_by_target: dict[str, list[str]] | None,
) -> list[str]:
    if not predictor_signals_by_target:
        return [signal for signal in numeric_columns if signal != target]
    specific = predictor_signals_by_target.get(target)
    wildcard = predictor_signals_by_target.get("*")
    selected = specific if specific is not None else wildcard
    if selected is None:
        return [signal for signal in numeric_columns if signal != target]
    return [signal for signal in selected if signal in numeric_columns and signal != target]
