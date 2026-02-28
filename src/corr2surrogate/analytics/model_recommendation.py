"""Lightweight model-family recommendation probes for Agent 1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from .correlations import CorrelationAnalysisBundle


@dataclass(frozen=True)
class ProbeModelScore:
    """Validation score for one cheap probe model."""

    model_family: str
    mae: float
    rmse: float
    r2: float
    relative_mae_gain_vs_linear: float
    train_samples: int
    validation_samples: int
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetModelRecommendation:
    """Agent 1 recommendation for Agent 2 model search order."""

    target_signal: str
    recommended_model_family: str
    priority_model_families: list[str]
    probe_predictor_signals: list[str]
    probe_complete_rows: int
    best_probe_model_family: str
    best_probe_mae: float
    best_probe_rmse: float
    best_probe_r2: float
    best_probe_gain_vs_linear: float
    recommendation_statement: str
    recommendation_confidence: str
    recommendation_confidence_score: float
    rationale: str
    tree_model_worth_testing: bool
    sequence_model_worth_testing: bool
    interaction_gain: float
    regime_strength: float
    residual_nonlinearity_score: float
    lag_benefit: float
    target_autocorr_lag1: float
    candidate_models: list[ProbeModelScore]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_models"] = [item.to_dict() for item in self.candidate_models]
        return payload


@dataclass(frozen=True)
class ModelStrategySummary:
    """Aggregate model-strategy recommendations."""

    data_mode: str
    target_recommendations: list[TargetModelRecommendation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_mode": self.data_mode,
            "target_recommendations": [item.to_dict() for item in self.target_recommendations],
        }


def recommend_model_strategies(
    *,
    frame: pd.DataFrame,
    correlations: CorrelationAnalysisBundle,
    max_lag: int,
    max_predictors: int = 4,
) -> ModelStrategySummary:
    """Run cheap probes and recommend which model families Agent 2 should try first."""
    targets: list[TargetModelRecommendation] = []
    for analysis in correlations.target_analyses:
        targets.append(
            _recommend_for_target(
                frame=frame,
                analysis=analysis,
                data_mode=correlations.data_mode,
                max_lag=max_lag,
                max_predictors=max_predictors,
            )
        )
    return ModelStrategySummary(
        data_mode=correlations.data_mode,
        target_recommendations=targets,
    )


def _recommend_for_target(
    *,
    frame: pd.DataFrame,
    analysis: Any,
    data_mode: str,
    max_lag: int,
    max_predictors: int,
) -> TargetModelRecommendation:
    target_signal = str(analysis.target_signal)
    predictor_cols = [
        str(item.predictor_signal)
        for item in list(analysis.predictor_results)[: max(1, int(max_predictors))]
        if str(item.predictor_signal).strip()
    ]
    if not predictor_cols:
        return TargetModelRecommendation(
            target_signal=target_signal,
            recommended_model_family="insufficient_signal_support",
            priority_model_families=["collect_more_data"],
            probe_predictor_signals=[],
            probe_complete_rows=0,
            best_probe_model_family="none",
            best_probe_mae=float("nan"),
            best_probe_rmse=float("nan"),
            best_probe_r2=float("nan"),
            best_probe_gain_vs_linear=0.0,
            recommendation_statement=(
                f"For target `{target_signal}`, no probe-model recommendation is possible because "
                "correlation screening did not yield usable predictors."
            ),
            recommendation_confidence="low",
            recommendation_confidence_score=0.0,
            rationale="No usable predictors survived correlation screening.",
            tree_model_worth_testing=False,
            sequence_model_worth_testing=False,
            interaction_gain=0.0,
            regime_strength=0.0,
            residual_nonlinearity_score=0.0,
            lag_benefit=0.0,
            target_autocorr_lag1=float("nan"),
            candidate_models=[],
        )

    numeric_cols = [target_signal, *predictor_cols]
    numeric = pd.DataFrame(
        {
            col: pd.to_numeric(frame[col], errors="coerce")
            for col in numeric_cols
            if col in frame.columns
        }
    ).dropna()
    if len(numeric) < 20:
        return TargetModelRecommendation(
            target_signal=target_signal,
            recommended_model_family="insufficient_data_probe",
            priority_model_families=["linear_ridge", "collect_more_data"],
            probe_predictor_signals=list(predictor_cols),
            probe_complete_rows=int(len(numeric)),
            best_probe_model_family="none",
            best_probe_mae=float("nan"),
            best_probe_rmse=float("nan"),
            best_probe_r2=float("nan"),
            best_probe_gain_vs_linear=0.0,
            recommendation_statement=(
                f"For target `{target_signal}`, collect more complete rows before using the "
                "model-family recommendation for Agent 2."
            ),
            recommendation_confidence="low",
            recommendation_confidence_score=0.1,
            rationale="Too few complete numeric rows for reliable probe-model screening.",
            tree_model_worth_testing=False,
            sequence_model_worth_testing=False,
            interaction_gain=0.0,
            regime_strength=0.0,
            residual_nonlinearity_score=0.0,
            lag_benefit=0.0,
            target_autocorr_lag1=_autocorr_lag1(numeric[target_signal].to_numpy(dtype=float)),
            candidate_models=[],
        )

    train_idx, val_idx = _build_train_validation_split(
        n_rows=len(numeric),
        data_mode=data_mode,
    )
    train = numeric.iloc[train_idx].reset_index(drop=True)
    val = numeric.iloc[val_idx].reset_index(drop=True)
    if len(train) < 12 or len(val) < 6:
        midpoint = max(12, int(len(numeric) * 0.8))
        midpoint = min(midpoint, len(numeric) - 6)
        train = numeric.iloc[:midpoint].reset_index(drop=True)
        val = numeric.iloc[midpoint:].reset_index(drop=True)
    if len(train) < 12 or len(val) < 6:
        return TargetModelRecommendation(
            target_signal=target_signal,
            recommended_model_family="insufficient_data_probe",
            priority_model_families=["linear_ridge", "collect_more_data"],
            probe_predictor_signals=list(predictor_cols),
            probe_complete_rows=int(len(numeric)),
            best_probe_model_family="none",
            best_probe_mae=float("nan"),
            best_probe_rmse=float("nan"),
            best_probe_r2=float("nan"),
            best_probe_gain_vs_linear=0.0,
            recommendation_statement=(
                f"For target `{target_signal}`, the probe split was too small for a reliable "
                "quick recommendation. Collect more data or reduce missingness first."
            ),
            recommendation_confidence="low",
            recommendation_confidence_score=0.1,
            rationale="Probe split left too few rows for train/validation screening.",
            tree_model_worth_testing=False,
            sequence_model_worth_testing=False,
            interaction_gain=0.0,
            regime_strength=0.0,
            residual_nonlinearity_score=0.0,
            lag_benefit=0.0,
            target_autocorr_lag1=_autocorr_lag1(numeric[target_signal].to_numpy(dtype=float)),
            candidate_models=[],
        )

    y_train = train[target_signal].to_numpy(dtype=float)
    y_val = val[target_signal].to_numpy(dtype=float)
    x_train = train[predictor_cols].to_numpy(dtype=float)
    x_val = val[predictor_cols].to_numpy(dtype=float)

    linear_pred = _ridge_predict(x_train, y_train, x_val)
    linear_metrics = _regression_metrics(y_val, linear_pred)
    baseline_mae = linear_metrics["mae"]

    candidates: list[ProbeModelScore] = [
        _probe_score(
            model_family="linear_ridge",
            metrics=linear_metrics,
            baseline_mae=baseline_mae,
            train_samples=len(train),
            validation_samples=len(val),
            notes="Closed-form ridge baseline on top ranked predictors.",
        )
    ]

    interaction_train, interaction_names = _build_interaction_features(
        train[predictor_cols],
        max_base_features=min(3, len(predictor_cols)),
    )
    interaction_val = _apply_interaction_features(
        val[predictor_cols],
        feature_names=interaction_names,
    )
    interaction_pred = _ridge_predict(
        interaction_train.to_numpy(dtype=float),
        y_train,
        interaction_val.to_numpy(dtype=float),
    )
    interaction_metrics = _regression_metrics(y_val, interaction_pred)
    candidates.append(
        _probe_score(
            model_family="interaction_ridge",
            metrics=interaction_metrics,
            baseline_mae=baseline_mae,
            train_samples=len(train),
            validation_samples=len(val),
            notes="Adds squares and pairwise products of top predictors.",
        )
    )

    tree_metrics = _tree_probe_metrics(
        train=train,
        val=val,
        target_signal=target_signal,
        predictor_cols=predictor_cols,
    )
    candidates.append(
        _probe_score(
            model_family="tiny_tree_probe",
            metrics=tree_metrics,
            baseline_mae=baseline_mae,
            train_samples=len(train),
            validation_samples=len(val),
            notes="Depth-limited regression tree on the top ranked predictors.",
        )
    )

    lag_benefit = 0.0
    if data_mode == "time_series" and int(max_lag) > 0:
        lagged = _lagged_probe_metrics(
            series=numeric,
            target_signal=target_signal,
            predictor_cols=predictor_cols,
            lag_horizon=min(max(int(max_lag), 1), 6),
        )
        if lagged is not None:
            lag_benefit = max(
                0.0,
                (baseline_mae - lagged["metrics"]["mae"]) / max(baseline_mae, 1e-12),
            )
            candidates.append(
                _probe_score(
                    model_family="lagged_linear",
                    metrics=lagged["metrics"],
                    baseline_mae=baseline_mae,
                    train_samples=int(lagged["train_samples"]),
                    validation_samples=int(lagged["validation_samples"]),
                    notes=(
                        f"Current + lagged predictor windows up to {lagged['lag_horizon']} samples."
                    ),
                )
            )

    residual_nonlinearity_score = _residual_nonlinearity_score(
        residuals=y_val - linear_pred,
        val=val,
        predictor_cols=predictor_cols,
    )
    regime_strength = _regime_strength(
        train=train,
        target_signal=target_signal,
        reference_signal=predictor_cols[0],
    )
    target_autocorr = _autocorr_lag1(numeric[target_signal].to_numpy(dtype=float))

    best_tree_probe_mae = min(
        interaction_metrics["mae"],
        tree_metrics["mae"],
    )
    interaction_gain = max(
        0.0,
        (baseline_mae - best_tree_probe_mae) / max(baseline_mae, 1e-12),
    )
    tree_model_worth_testing = bool(
        interaction_gain >= 0.05
        or regime_strength >= 0.30
        or residual_nonlinearity_score >= 0.20
    )

    best_candidate = min(
        candidates,
        key=lambda item: item.mae if math.isfinite(item.mae) else float("inf"),
    )
    best_candidate_gain = max(0.0, float(best_candidate.relative_mae_gain_vs_linear))
    sequence_model_worth_testing = bool(
        data_mode == "time_series"
        and lag_benefit >= 0.05
        and target_autocorr >= 0.30
        and best_candidate.r2 < 0.75
        and residual_nonlinearity_score >= 0.10
    )

    recommended_model_family, priority = _select_model_family(
        data_mode=data_mode,
        best_probe=best_candidate.model_family,
        best_probe_gain=best_candidate_gain,
        tree_model_worth_testing=tree_model_worth_testing,
        sequence_model_worth_testing=sequence_model_worth_testing,
        lag_benefit=lag_benefit,
    )
    rationale = _build_rationale(
        recommended_model_family=recommended_model_family,
        best_candidate=best_candidate,
        interaction_gain=interaction_gain,
        regime_strength=regime_strength,
        residual_nonlinearity_score=residual_nonlinearity_score,
        lag_benefit=lag_benefit,
        target_autocorr_lag1=target_autocorr,
    )
    recommendation_statement = _build_recommendation_statement(
        target_signal=target_signal,
        recommended_model_family=recommended_model_family,
        best_candidate=best_candidate,
        tree_model_worth_testing=tree_model_worth_testing,
        sequence_model_worth_testing=sequence_model_worth_testing,
    )
    confidence_label, confidence_score = _recommendation_confidence(
        recommended_model_family=recommended_model_family,
        best_candidate=best_candidate,
        interaction_gain=interaction_gain,
        regime_strength=regime_strength,
        residual_nonlinearity_score=residual_nonlinearity_score,
        lag_benefit=lag_benefit,
        tree_model_worth_testing=tree_model_worth_testing,
        sequence_model_worth_testing=sequence_model_worth_testing,
        probe_complete_rows=int(len(numeric)),
    )

    return TargetModelRecommendation(
        target_signal=target_signal,
        recommended_model_family=recommended_model_family,
        priority_model_families=priority,
        probe_predictor_signals=list(predictor_cols),
        probe_complete_rows=int(len(numeric)),
        best_probe_model_family=best_candidate.model_family,
        best_probe_mae=float(best_candidate.mae),
        best_probe_rmse=float(best_candidate.rmse),
        best_probe_r2=float(best_candidate.r2),
        best_probe_gain_vs_linear=float(best_candidate_gain),
        recommendation_statement=recommendation_statement,
        recommendation_confidence=confidence_label,
        recommendation_confidence_score=float(confidence_score),
        rationale=rationale,
        tree_model_worth_testing=tree_model_worth_testing,
        sequence_model_worth_testing=sequence_model_worth_testing,
        interaction_gain=float(interaction_gain),
        regime_strength=float(regime_strength),
        residual_nonlinearity_score=float(residual_nonlinearity_score),
        lag_benefit=float(lag_benefit),
        target_autocorr_lag1=float(target_autocorr),
        candidate_models=candidates,
    )


def _probe_score(
    *,
    model_family: str,
    metrics: dict[str, float],
    baseline_mae: float,
    train_samples: int,
    validation_samples: int,
    notes: str,
) -> ProbeModelScore:
    mae = float(metrics.get("mae", float("nan")))
    gain = (
        max(0.0, (baseline_mae - mae) / max(baseline_mae, 1e-12))
        if math.isfinite(baseline_mae) and math.isfinite(mae)
        else 0.0
    )
    if model_family == "linear_ridge":
        gain = 0.0
    return ProbeModelScore(
        model_family=model_family,
        mae=mae,
        rmse=float(metrics.get("rmse", float("nan"))),
        r2=float(metrics.get("r2", float("nan"))),
        relative_mae_gain_vs_linear=float(gain),
        train_samples=int(train_samples),
        validation_samples=int(validation_samples),
        notes=notes,
    )


def _build_train_validation_split(*, n_rows: int, data_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if n_rows < 10:
        idx = np.arange(n_rows, dtype=int)
        return idx[: max(n_rows - 2, 1)], idx[max(n_rows - 2, 1) :]
    if data_mode == "time_series":
        split = max(8, int(n_rows * 0.8))
        split = min(split, n_rows - 2)
        return np.arange(split, dtype=int), np.arange(split, n_rows, dtype=int)
    val_mask = np.arange(n_rows, dtype=int) % 5 == 0
    if int(np.sum(val_mask)) < 6:
        split = max(8, int(n_rows * 0.8))
        split = min(split, n_rows - 2)
        return np.arange(split, dtype=int), np.arange(split, n_rows, dtype=int)
    train_idx = np.where(~val_mask)[0]
    val_idx = np.where(val_mask)[0]
    return train_idx, val_idx


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> np.ndarray:
    if x_train.ndim != 2 or x_val.ndim != 2:
        raise ValueError("x_train and x_val must be 2D.")
    design_train = np.column_stack([np.ones(x_train.shape[0]), x_train])
    design_val = np.column_stack([np.ones(x_val.shape[0]), x_val])
    regularized = design_train.T @ design_train
    if regularized.shape[0] > 1:
        regularized[1:, 1:] += np.eye(regularized.shape[0] - 1) * float(ridge)
    rhs = design_train.T @ y_train
    try:
        weights = np.linalg.solve(regularized, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(regularized) @ rhs
    return (design_val @ weights).astype(float)


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.size == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    error = pred - true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    centered = true - np.mean(true)
    denom = float(np.sum(np.square(centered)))
    if denom <= 1e-12:
        r2 = 1.0 if np.allclose(true, pred) else 0.0
    else:
        r2 = 1.0 - float(np.sum(np.square(error))) / denom
    return {"mae": mae, "rmse": rmse, "r2": float(r2)}


def _build_interaction_features(
    frame: pd.DataFrame,
    *,
    max_base_features: int,
) -> tuple[pd.DataFrame, list[tuple[str, str | None]]]:
    limited_cols = list(frame.columns[: max(1, int(max_base_features))])
    out = pd.DataFrame(index=frame.index)
    feature_names: list[tuple[str, str | None]] = []
    for col in frame.columns:
        out[str(col)] = pd.to_numeric(frame[col], errors="coerce")
        feature_names.append((str(col), None))
    for col in limited_cols:
        name = f"{col}__square"
        out[name] = np.square(pd.to_numeric(frame[col], errors="coerce"))
        feature_names.append((str(col), "square"))
    for idx, a in enumerate(limited_cols):
        for b in limited_cols[idx + 1 :]:
            name = f"{a}__x__{b}"
            out[name] = (
                pd.to_numeric(frame[a], errors="coerce")
                * pd.to_numeric(frame[b], errors="coerce")
            )
            feature_names.append((f"{a}*{b}", "product"))
    return out, feature_names


def _apply_interaction_features(
    frame: pd.DataFrame,
    *,
    feature_names: list[tuple[str, str | None]],
) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for base, transform in feature_names:
        if transform is None:
            out[base] = pd.to_numeric(frame[base], errors="coerce")
            continue
        if transform == "square":
            col = base
            out[f"{col}__square"] = np.square(pd.to_numeric(frame[col], errors="coerce"))
            continue
        if transform == "product":
            left, right = base.split("*", 1)
            out[f"{left}__x__{right}"] = (
                pd.to_numeric(frame[left], errors="coerce")
                * pd.to_numeric(frame[right], errors="coerce")
            )
    return out


def _tree_probe_metrics(
    *,
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_signal: str,
    predictor_cols: list[str],
) -> dict[str, float]:
    features = predictor_cols[: max(1, min(4, len(predictor_cols)))]
    if not features:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    x_train = train[features].to_numpy(dtype=float)
    y_train = train[target_signal].to_numpy(dtype=float)
    x_val = val[features].to_numpy(dtype=float)
    y_val = val[target_signal].to_numpy(dtype=float)
    min_leaf = max(4, min(12, int(len(train) * 0.08)))
    tree = _fit_tiny_regression_tree(
        x_train=x_train,
        y_train=y_train,
        depth=0,
        max_depth=3,
        min_leaf=min_leaf,
    )
    pred = _predict_tiny_regression_tree(tree=tree, x=x_val)
    return _regression_metrics(y_val, pred)


def _fit_tiny_regression_tree(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    depth: int,
    max_depth: int,
    min_leaf: int,
) -> dict[str, Any]:
    leaf_value = float(np.mean(y_train)) if len(y_train) > 0 else 0.0
    if (
        depth >= max_depth
        or len(y_train) < max(2 * int(min_leaf), 8)
        or float(np.var(y_train)) <= 1e-12
    ):
        return {"leaf": True, "value": leaf_value}

    best_feature = -1
    best_threshold = float("nan")
    best_score = float("inf")
    n_features = int(x_train.shape[1]) if x_train.ndim == 2 else 0
    for feature_idx in range(n_features):
        values = x_train[:, feature_idx]
        finite_values = values[np.isfinite(values)]
        if finite_values.size < max(2 * int(min_leaf), 8):
            continue
        thresholds = _candidate_thresholds(finite_values)
        for threshold in thresholds:
            left_mask = values <= threshold
            right_mask = ~left_mask
            left_count = int(np.sum(left_mask))
            right_count = int(np.sum(right_mask))
            if left_count < min_leaf or right_count < min_leaf:
                continue
            score = _sum_squared_error(y_train[left_mask]) + _sum_squared_error(y_train[right_mask])
            if score < best_score:
                best_score = score
                best_feature = feature_idx
                best_threshold = float(threshold)

    if best_feature < 0 or not math.isfinite(best_threshold):
        return {"leaf": True, "value": leaf_value}

    values = x_train[:, best_feature]
    left_mask = values <= best_threshold
    right_mask = ~left_mask
    return {
        "leaf": False,
        "value": leaf_value,
        "feature_index": int(best_feature),
        "threshold": float(best_threshold),
        "left": _fit_tiny_regression_tree(
            x_train=x_train[left_mask],
            y_train=y_train[left_mask],
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf=min_leaf,
        ),
        "right": _fit_tiny_regression_tree(
            x_train=x_train[right_mask],
            y_train=y_train[right_mask],
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf=min_leaf,
        ),
    }


def _predict_tiny_regression_tree(*, tree: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("x must be 2D.")
    pred = np.empty(x.shape[0], dtype=float)
    for idx in range(x.shape[0]):
        pred[idx] = _predict_tiny_regression_tree_row(tree=tree, row=x[idx])
    return pred


def _predict_tiny_regression_tree_row(*, tree: dict[str, Any], row: np.ndarray) -> float:
    node = tree
    while not bool(node.get("leaf", False)):
        feature_index = int(node["feature_index"])
        threshold = float(node["threshold"])
        if float(row[feature_index]) <= threshold:
            node = dict(node["left"])
        else:
            node = dict(node["right"])
    return float(node.get("value", 0.0))


def _candidate_thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values)
    if unique.size <= 1:
        return np.array([], dtype=float)
    if unique.size <= 12:
        return ((unique[:-1] + unique[1:]) / 2.0).astype(float)
    quantiles = np.linspace(0.1, 0.9, 9)
    candidates = np.unique(np.quantile(values, quantiles))
    lower = float(np.min(values))
    upper = float(np.max(values))
    candidates = candidates[(candidates > lower) & (candidates < upper)]
    return candidates.astype(float)


def _sum_squared_error(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    centered = values - float(np.mean(values))
    return float(np.sum(np.square(centered)))


def _digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if len(edges) <= 2:
        return np.zeros(len(values), dtype=int)
    bins = np.searchsorted(edges[1:-1], values, side="right")
    return bins.astype(int)


def _lagged_probe_metrics(
    *,
    series: pd.DataFrame,
    target_signal: str,
    predictor_cols: list[str],
    lag_horizon: int,
) -> dict[str, Any] | None:
    if lag_horizon <= 0:
        return None
    lag_frame = pd.DataFrame(index=series.index)
    for col in predictor_cols:
        numeric = pd.to_numeric(series[col], errors="coerce")
        lag_frame[f"{col}__t"] = numeric
        for lag in range(1, lag_horizon + 1):
            lag_frame[f"{col}__lag{lag}"] = numeric.shift(lag)
    lag_frame[target_signal] = pd.to_numeric(series[target_signal], errors="coerce")
    lag_frame = lag_frame.dropna().reset_index(drop=True)
    if len(lag_frame) < 30:
        return None
    split = max(12, int(len(lag_frame) * 0.8))
    split = min(split, len(lag_frame) - 6)
    if split < 12 or (len(lag_frame) - split) < 6:
        return None
    train = lag_frame.iloc[:split]
    val = lag_frame.iloc[split:]
    feature_cols = [col for col in lag_frame.columns if col != target_signal]
    pred = _ridge_predict(
        train[feature_cols].to_numpy(dtype=float),
        train[target_signal].to_numpy(dtype=float),
        val[feature_cols].to_numpy(dtype=float),
    )
    return {
        "metrics": _regression_metrics(val[target_signal].to_numpy(dtype=float), pred),
        "train_samples": int(len(train)),
        "validation_samples": int(len(val)),
        "lag_horizon": int(lag_horizon),
    }


def _residual_nonlinearity_score(
    *,
    residuals: np.ndarray,
    val: pd.DataFrame,
    predictor_cols: list[str],
) -> float:
    if residuals.size < 6 or not predictor_cols:
        return 0.0
    scores: list[float] = []
    primary = pd.to_numeric(val[predictor_cols[0]], errors="coerce").to_numpy(dtype=float)
    scores.append(abs(_safe_corr(residuals, np.square(primary))))
    if len(predictor_cols) >= 2:
        secondary = pd.to_numeric(val[predictor_cols[1]], errors="coerce").to_numpy(dtype=float)
        scores.append(abs(_safe_corr(residuals, primary * secondary)))
    finite = [item for item in scores if math.isfinite(item)]
    if not finite:
        return 0.0
    return float(max(finite))


def _regime_strength(
    *,
    train: pd.DataFrame,
    target_signal: str,
    reference_signal: str,
) -> float:
    values = pd.to_numeric(train[reference_signal], errors="coerce").to_numpy(dtype=float)
    target = pd.to_numeric(train[target_signal], errors="coerce").to_numpy(dtype=float)
    if len(values) < 12:
        return 0.0
    edges = np.unique(np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]))
    bins = _digitize(values, edges)
    design = np.column_stack([np.ones(len(values)), values])
    gram = design.T @ design
    rhs = design.T @ target
    try:
        weights = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(gram) @ rhs
    residuals = target - (design @ weights)
    means: list[float] = []
    for idx in np.unique(bins):
        mask = bins == idx
        if int(np.sum(mask)) < 3:
            continue
        means.append(float(np.mean(residuals[mask])))
    if len(means) < 2:
        return 0.0
    target_std = float(np.std(target))
    if target_std <= 1e-12:
        return 0.0
    return float((max(means) - min(means)) / target_std)


def _autocorr_lag1(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 3:
        return float("nan")
    return _safe_corr(arr[1:], arr[:-1])


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else float("nan")


def _select_model_family(
    *,
    data_mode: str,
    best_probe: str,
    best_probe_gain: float,
    tree_model_worth_testing: bool,
    sequence_model_worth_testing: bool,
    lag_benefit: float,
) -> tuple[str, list[str]]:
    if data_mode == "time_series":
        if sequence_model_worth_testing:
            return (
                "sequence_model_candidate",
                ["lagged_linear", "tree_ensemble_candidate", "sequence_model_candidate"],
            )
        if best_probe == "lagged_linear" and lag_benefit >= 0.03:
            if tree_model_worth_testing:
                return (
                    "lagged_linear",
                    ["lagged_linear", "tree_ensemble_candidate", "linear_ridge"],
                )
            return ("lagged_linear", ["lagged_linear", "linear_ridge"])
        if best_probe in {"interaction_ridge", "tiny_tree_probe"} and best_probe_gain >= 0.03:
            return (
                "tree_ensemble_candidate",
                ["linear_ridge", "tree_ensemble_candidate", "lagged_linear"],
            )
        if tree_model_worth_testing:
            if lag_benefit >= 0.02:
                return (
                    "lagged_linear",
                    ["lagged_linear", "tree_ensemble_candidate", "linear_ridge"],
                )
            return (
                "linear_ridge",
                ["linear_ridge", "lagged_linear", "tree_ensemble_candidate"],
            )
        return ("linear_ridge", ["linear_ridge", "lagged_linear"])

    if best_probe in {"interaction_ridge", "tiny_tree_probe"} and best_probe_gain >= 0.03:
        return (
            "tree_ensemble_candidate",
            ["linear_ridge", "tree_ensemble_candidate"],
        )
    if tree_model_worth_testing:
        return ("linear_ridge", ["linear_ridge", "tree_ensemble_candidate"])
    if best_probe == "interaction_ridge" and best_probe_gain >= 0.02:
        return ("interaction_ridge", ["linear_ridge", "interaction_ridge"])
    return ("linear_ridge", ["linear_ridge", "tree_ensemble_candidate"])


def _build_recommendation_statement(
    *,
    target_signal: str,
    recommended_model_family: str,
    best_candidate: ProbeModelScore,
    tree_model_worth_testing: bool,
    sequence_model_worth_testing: bool,
) -> str:
    base = (
        f"For target `{target_signal}`, start with `{recommended_model_family}`. "
        f"The strongest quick screen was `{best_candidate.model_family}` "
        f"(MAE={best_candidate.mae:.4f}, R2={best_candidate.r2:.3f}, "
        f"gain_vs_linear={best_candidate.relative_mae_gain_vs_linear:.3f})."
        if math.isfinite(best_candidate.mae) and math.isfinite(best_candidate.r2)
        else f"For target `{target_signal}`, start with `{recommended_model_family}`."
    )
    if recommended_model_family == "tree_ensemble_candidate":
        return (
            base
            + " A tree-based family is recommended next because nonlinear or regime-like "
            "structure appears material even after the linear baseline."
        )
    if recommended_model_family == "lagged_linear":
        tail = " Lagged effects improved the quick validation screen enough to justify a temporal tabular baseline first."
        if tree_model_worth_testing:
            tail += " Tree ensembles are still worth testing after that baseline."
        return base + tail
    if recommended_model_family == "sequence_model_candidate":
        return (
            base
            + " Sequence models are worth testing only after simpler lagged/tabular baselines, "
            "because meaningful temporal residual structure remains."
        )
    if recommended_model_family == "interaction_ridge":
        return (
            base
            + " Simple explicit nonlinear interactions appear helpful, but the evidence is not "
            "strong enough yet to skip interpretable tabular baselines."
        )
    tail = " The linear baseline remains the strongest or most reliable first step."
    if tree_model_worth_testing:
        tail += " Tree models are a secondary follow-up, not the primary recommendation."
    if sequence_model_worth_testing:
        tail += " Sequence models are worth revisiting only if lagged baselines still leave residual structure."
    return base + tail


def _recommendation_confidence(
    *,
    recommended_model_family: str,
    best_candidate: ProbeModelScore,
    interaction_gain: float,
    regime_strength: float,
    residual_nonlinearity_score: float,
    lag_benefit: float,
    tree_model_worth_testing: bool,
    sequence_model_worth_testing: bool,
    probe_complete_rows: int,
) -> tuple[str, float]:
    score = 0.10
    if probe_complete_rows >= 40:
        score += 0.10
    if probe_complete_rows >= 100:
        score += 0.10
    if math.isfinite(best_candidate.r2):
        score += min(0.20, max(0.0, best_candidate.r2) * 0.20)
    if math.isfinite(best_candidate.relative_mae_gain_vs_linear):
        score += min(0.20, max(0.0, best_candidate.relative_mae_gain_vs_linear) * 2.5)

    if recommended_model_family == "linear_ridge":
        score += 0.20 if best_candidate.model_family == "linear_ridge" else 0.05
        if max(interaction_gain, lag_benefit) < 0.03 and residual_nonlinearity_score < 0.15:
            score += 0.15
    elif recommended_model_family == "lagged_linear":
        score += min(0.25, max(0.0, lag_benefit) * 3.0)
        if tree_model_worth_testing:
            score += 0.05
    elif recommended_model_family == "tree_ensemble_candidate":
        score += min(0.20, max(0.0, interaction_gain) * 2.0)
        score += min(0.10, max(0.0, regime_strength) * 0.25)
        score += min(0.10, max(0.0, residual_nonlinearity_score) * 0.30)
    elif recommended_model_family == "sequence_model_candidate":
        score += min(0.20, max(0.0, lag_benefit) * 2.5)
        if sequence_model_worth_testing:
            score += 0.10
    elif recommended_model_family in {"insufficient_data_probe", "insufficient_signal_support"}:
        score = min(score, 0.20)

    score = float(max(0.0, min(1.0, score)))
    if score >= 0.70:
        return "high", score
    if score >= 0.45:
        return "medium", score
    return "low", score


def _build_rationale(
    *,
    recommended_model_family: str,
    best_candidate: ProbeModelScore,
    interaction_gain: float,
    regime_strength: float,
    residual_nonlinearity_score: float,
    lag_benefit: float,
    target_autocorr_lag1: float,
) -> str:
    parts = [
        f"best_probe={best_candidate.model_family}",
        f"probe_mae={best_candidate.mae:.4f}" if math.isfinite(best_candidate.mae) else "probe_mae=n/a",
        f"probe_r2={best_candidate.r2:.3f}" if math.isfinite(best_candidate.r2) else "probe_r2=n/a",
        f"interaction_gain={interaction_gain:.3f}",
        f"regime_strength={regime_strength:.3f}",
        f"residual_nonlinearity={residual_nonlinearity_score:.3f}",
    ]
    if math.isfinite(target_autocorr_lag1):
        parts.append(f"target_autocorr_lag1={target_autocorr_lag1:.3f}")
    if lag_benefit > 0.0:
        parts.append(f"lag_benefit={lag_benefit:.3f}")
    guidance = {
        "linear_ridge": "Start with a simple interpretable baseline; escalate only if residual structure remains.",
        "interaction_ridge": "Simple nonlinear interactions help; test explicit engineered features before heavier models.",
        "tree_ensemble_candidate": "Piecewise/interacting behavior is likely; prioritize tree ensembles after the linear baseline.",
        "lagged_linear": "Temporal memory is useful but still structured enough for a lagged tabular baseline.",
        "sequence_model_candidate": "Lag structure and residual dynamics suggest testing sequence models after lagged/tabular baselines.",
        "insufficient_data_probe": "Probe screening was not reliable; collect more complete data first.",
        "insufficient_signal_support": "No stable predictor support was available from Agent 1.",
    }
    return f"{guidance.get(recommended_model_family, '')} " + "; ".join(parts)
