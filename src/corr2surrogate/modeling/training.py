"""Split-safe training and model comparison helpers for Agent 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from corr2surrogate.core.json_utils import write_json
from corr2surrogate.persistence.artifact_store import ArtifactStore
from .baselines import IncrementalLinearSurrogate
from .checkpoints import ModelCheckpointStore
from .evaluation import regression_metrics
from .normalization import MinMaxNormalizer
from .splitters import DatasetSplit, build_train_validation_test_split


@dataclass(frozen=True)
class CandidateMetrics:
    """Train/validation/test metrics for one candidate model."""

    model_family: str
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_candidate_model_family(requested_model_family: str) -> str | None:
    """Normalize user-facing model names to implemented candidate identifiers."""
    normalized = requested_model_family.strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "auto": "auto",
        "recommended": "auto",
        "linear_ridge": "linear_ridge",
        "ridge": "linear_ridge",
        "linear": "linear_ridge",
        "incremental_linear_surrogate": "linear_ridge",
        "bagged_tree_ensemble": "bagged_tree_ensemble",
        "tree_ensemble": "bagged_tree_ensemble",
        "tree_ensemble_candidate": "bagged_tree_ensemble",
        "tree": "bagged_tree_ensemble",
        "extra_trees": "bagged_tree_ensemble",
        "hist_gradient_boosting": "bagged_tree_ensemble",
    }
    return aliases.get(normalized)


def train_surrogate_candidates(
    *,
    frame: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    requested_model_family: str,
    timestamp_column: str | None = None,
    normalize: bool = True,
    missing_data_strategy: str = "fill_median",
    fill_constant_value: float | None = None,
    compare_against_baseline: bool = True,
    run_id: str | None = None,
    checkpoint_tag: str | None = None,
    data_references: list[str] | None = None,
) -> dict[str, Any]:
    """Train a split-safe linear baseline and optional nonlinear comparator."""
    if not feature_columns:
        raise ValueError("feature_columns cannot be empty.")
    _require_columns(frame, list(feature_columns) + [target_column])

    data_mode = _infer_data_mode(frame=frame, timestamp_column=timestamp_column)
    split = build_train_validation_test_split(n_rows=len(frame), data_mode=data_mode)
    split_frames = {
        "train": frame.iloc[split.train_indices].reset_index(drop=True),
        "validation": frame.iloc[split.validation_indices].reset_index(drop=True),
        "test": frame.iloc[split.test_indices].reset_index(drop=True),
    }
    prepared = _prepare_split_safe_frames(
        split_frames=split_frames,
        feature_columns=feature_columns,
        target_column=target_column,
        missing_data_strategy=missing_data_strategy,
        fill_constant_value=fill_constant_value,
    )
    prepared_frames = prepared["frames"]
    preprocessing = prepared["preprocessing"]

    normalizer: MinMaxNormalizer | None = None
    if normalize:
        normalizer = MinMaxNormalizer()
        normalizer.fit(prepared_frames["train"], feature_columns=feature_columns)
        prepared_frames = {
            name: normalizer.transform_features(part)
            for name, part in prepared_frames.items()
        }

    linear_model = IncrementalLinearSurrogate(
        feature_columns=feature_columns,
        target_column=target_column,
    )
    linear_rows_used = linear_model.fit_dataframe(prepared_frames["train"])
    linear_candidate = _candidate_metrics_from_model(
        model_family="linear_ridge",
        model=linear_model,
        frames=prepared_frames,
        notes="Split-safe ridge baseline with train-only preprocessing.",
    )

    requested = normalize_candidate_model_family(requested_model_family)
    if requested is None:
        raise ValueError(
            "Requested model is not implemented. "
            "Supported: auto, linear_ridge/ridge/linear, "
            "bagged_tree_ensemble/tree/tree_ensemble/extra_trees/hist_gradient_boosting."
        )

    candidates: list[CandidateMetrics] = [linear_candidate]
    tree_model: BaggedTreeEnsembleSurrogate | None = None
    if compare_against_baseline or requested == "bagged_tree_ensemble":
        tree_model = BaggedTreeEnsembleSurrogate(
            feature_columns=feature_columns,
            target_column=target_column,
        )
        tree_rows_used = tree_model.fit_dataframe(prepared_frames["train"])
        tree_notes = (
            "Bagged depth-limited regression trees as the first nonlinear local baseline. "
            f"Train rows used={tree_rows_used}."
        )
        candidates.append(
            _candidate_metrics_from_model(
                model_family="bagged_tree_ensemble",
                model=tree_model,
                frames=prepared_frames,
                notes=tree_notes,
            )
        )

    best_by_validation = _select_best_candidate(candidates)
    selected_candidate = _resolve_selected_candidate(
        requested=requested,
        candidates=candidates,
    )
    selected_model_name = selected_candidate.model_family
    selected_model_obj: Any
    if selected_model_name == "linear_ridge":
        selected_model_obj = linear_model
    elif selected_model_name == "bagged_tree_ensemble" and tree_model is not None:
        selected_model_obj = tree_model
    else:
        raise RuntimeError("Selected model object is unavailable.")

    artifact_store = ArtifactStore()
    run_dir = artifact_store.create_run_dir(run_id=run_id)
    normalizer_path: str | None = None
    if normalizer is not None:
        normalizer_path = str(artifact_store.save_normalizer(run_dir=run_dir, normalizer=normalizer))

    model_state_path = _save_selected_model_state(
        model=selected_model_obj,
        run_dir=run_dir,
        model_name=selected_model_name,
    )
    params_path = artifact_store.save_model_params(
        run_dir=run_dir,
        model_name=selected_model_name,
        best_params=_best_params_payload(
            selected_model_name=selected_model_name,
            linear_rows_used=linear_rows_used,
            tree_model=tree_model,
            requested=requested,
        ),
        metrics=selected_candidate.test_metrics,
        feature_columns=feature_columns,
        target_column=target_column,
        split_strategy=split.strategy,
        normalizer_path=normalizer_path,
        extra={
            "requested_model_family": requested,
            "selected_model_family": selected_candidate.model_family,
            "best_validation_model_family": best_by_validation.model_family,
            "data_mode": data_mode,
            "split": split.to_dict(),
            "preprocessing": preprocessing,
            "comparison": [item.to_dict() for item in candidates],
        },
    )
    checkpoint_store = ModelCheckpointStore()
    checkpoint = checkpoint_store.create_checkpoint(
        model_name=selected_model_name,
        run_dir=run_dir,
        model_state_path=model_state_path,
        target_column=target_column,
        feature_columns=feature_columns,
        metrics=selected_candidate.test_metrics,
        data_references=list(data_references or []),
        notes=checkpoint_tag or "",
        tags=[checkpoint_tag] if checkpoint_tag else [],
    )

    return {
        "status": "ok",
        "data_mode": data_mode,
        "requested_model_family": requested,
        "selected_model_family": selected_candidate.model_family,
        "best_validation_model_family": best_by_validation.model_family,
        "comparison": [item.to_dict() for item in candidates],
        "split": split.to_dict(),
        "preprocessing": preprocessing,
        "normalization": {
            "enabled": bool(normalizer is not None),
            "method": "minmax" if normalizer is not None else "none",
            "normalizer_path": normalizer_path,
        },
        "checkpoint_id": checkpoint.checkpoint_id,
        "run_dir": str(run_dir),
        "model_state_path": str(model_state_path),
        "model_params_path": str(params_path),
        "rows_used": int(prepared_frames["train"].shape[0]),
        "selected_metrics": {
            "train": selected_candidate.train_metrics,
            "validation": selected_candidate.validation_metrics,
            "test": selected_candidate.test_metrics,
        },
    }


class BaggedTreeEnsembleSurrogate:
    """Small local regression tree ensemble without external dependencies."""

    def __init__(
        self,
        *,
        feature_columns: list[str],
        target_column: str,
        n_estimators: int = 12,
        max_depth: int = 4,
        min_leaf: int = 6,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_leaf = int(min_leaf)
        self._estimators: list[dict[str, Any]] = []

    def fit_dataframe(self, frame: pd.DataFrame) -> int:
        x, y = _prepare_xy(frame=frame, feature_columns=self.feature_columns, target_column=self.target_column)
        if x.shape[0] == 0:
            raise ValueError("No valid numeric rows available for tree training.")
        self._estimators = []
        feature_count = x.shape[1]
        feature_subsample = max(1, int(math.sqrt(feature_count)))
        for seed in range(self.n_estimators):
            rng = np.random.default_rng(100 + seed)
            row_idx = rng.integers(0, x.shape[0], size=x.shape[0])
            feat_idx = np.sort(
                rng.choice(feature_count, size=feature_subsample, replace=False)
            ).astype(int)
            tree = _fit_regression_tree(
                x_train=x[row_idx][:, feat_idx],
                y_train=y[row_idx],
                depth=0,
                max_depth=self.max_depth,
                min_leaf=min(self.min_leaf, max(2, x.shape[0] // 8)),
            )
            self._estimators.append(
                {
                    "feature_indices": feat_idx.tolist(),
                    "tree": tree,
                }
            )
        return int(x.shape[0])

    def predict_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._estimators:
            raise RuntimeError("Tree ensemble has not been fitted yet.")
        x = _prepare_feature_matrix(frame=frame, feature_columns=self.feature_columns)
        preds = []
        for estimator in self._estimators:
            feat_idx = np.asarray(estimator["feature_indices"], dtype=int)
            pred = _predict_regression_tree_batch(
                tree=estimator["tree"],
                x=x[:, feat_idx],
            )
            preds.append(pred)
        return np.mean(np.vstack(preds), axis=0).astype(float)

    def evaluate_dataframe(self, frame: pd.DataFrame) -> dict[str, float]:
        y_true = _prepare_target_vector(frame=frame, target_column=self.target_column)
        y_pred = self.predict_dataframe(frame)
        return regression_metrics(y_true=y_true, y_pred=y_pred).to_dict()

    def state_dict(self) -> dict[str, Any]:
        if not self._estimators:
            raise RuntimeError("Tree ensemble has not been fitted yet.")
        return {
            "model_name": "bagged_tree_ensemble",
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_leaf": self.min_leaf,
            "estimators": self._estimators,
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return write_json(output, self.state_dict(), indent=2)


def _candidate_metrics_from_model(
    *,
    model_family: str,
    model: Any,
    frames: dict[str, pd.DataFrame],
    notes: str,
) -> CandidateMetrics:
    return CandidateMetrics(
        model_family=model_family,
        train_metrics={k: float(v) for k, v in model.evaluate_dataframe(frames["train"]).items()},
        validation_metrics={
            k: float(v) for k, v in model.evaluate_dataframe(frames["validation"]).items()
        },
        test_metrics={k: float(v) for k, v in model.evaluate_dataframe(frames["test"]).items()},
        notes=notes,
    )


def _select_best_candidate(candidates: list[CandidateMetrics]) -> CandidateMetrics:
    ranked = sorted(
        candidates,
        key=lambda item: (
            float(item.validation_metrics.get("mae", float("inf"))),
            0 if item.model_family == "linear_ridge" else 1,
        ),
    )
    return ranked[0]


def _resolve_selected_candidate(
    *,
    requested: str,
    candidates: list[CandidateMetrics],
) -> CandidateMetrics:
    if requested == "auto":
        return _select_best_candidate(candidates)
    for item in candidates:
        if item.model_family == requested:
            return item
    return _select_best_candidate(candidates)


def _save_selected_model_state(*, model: Any, run_dir: Path, model_name: str) -> Path:
    filename = "model_state.json" if model_name == "linear_ridge" else f"{model_name}_state.json"
    path = Path(run_dir) / filename
    return model.save(path)


def _best_params_payload(
    *,
    selected_model_name: str,
    linear_rows_used: int,
    tree_model: BaggedTreeEnsembleSurrogate | None,
    requested: str,
) -> dict[str, Any]:
    if selected_model_name == "linear_ridge":
        return {
            "requested_model_family": requested,
            "ridge": 1e-8,
            "training_rows_used": int(linear_rows_used),
        }
    if tree_model is not None:
        return {
            "requested_model_family": requested,
            "n_estimators": int(tree_model.n_estimators),
            "max_depth": int(tree_model.max_depth),
            "min_leaf": int(tree_model.min_leaf),
        }
    return {"requested_model_family": requested}


def _infer_data_mode(*, frame: pd.DataFrame, timestamp_column: str | None) -> str:
    if not timestamp_column or timestamp_column not in frame.columns:
        return "steady_state"
    numeric = pd.to_numeric(frame[timestamp_column], errors="coerce")
    valid = numeric.dropna()
    if len(valid) >= 8:
        diffs = valid.diff().dropna()
        if int((diffs > 0).sum()) >= max(5, int(len(diffs) * 0.7)):
            return "time_series"
    parsed = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if float(parsed.notna().mean()) >= 0.8:
        diffs_dt = parsed.dropna().diff().dt.total_seconds().dropna()
        if int((diffs_dt > 0).sum()) >= max(5, int(len(diffs_dt) * 0.7)):
            return "time_series"
    return "steady_state"


def _prepare_split_safe_frames(
    *,
    split_frames: dict[str, pd.DataFrame],
    feature_columns: list[str],
    target_column: str,
    missing_data_strategy: str,
    fill_constant_value: float | None,
) -> dict[str, Any]:
    coerced: dict[str, pd.DataFrame] = {}
    for name, frame in split_frames.items():
        subset = frame[feature_columns + [target_column]].copy()
        for col in feature_columns + [target_column]:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")
        subset = subset.dropna(subset=[target_column]).reset_index(drop=True)
        coerced[name] = subset

    strategy = missing_data_strategy.strip().lower()
    effective_strategy = strategy
    fill_values: dict[str, float] = {}
    if strategy in {"fill_median", "median"}:
        train = coerced["train"]
        for col in feature_columns:
            median = float(train[col].median()) if train[col].notna().any() else 0.0
            fill_values[col] = median
        for name in list(coerced.keys()):
            coerced[name] = coerced[name].fillna(fill_values)
        effective_strategy = "fill_median_train_only"
    elif strategy in {"fill_constant", "constant"}:
        value = 0.0 if fill_constant_value is None else float(fill_constant_value)
        fill_values = {col: value for col in feature_columns}
        for name in list(coerced.keys()):
            coerced[name] = coerced[name].fillna(fill_values)
        effective_strategy = "fill_constant_train_policy"
    else:
        if strategy == "keep":
            effective_strategy = "keep_requested_drop_remaining_for_model"
        for name in list(coerced.keys()):
            coerced[name] = coerced[name].dropna().reset_index(drop=True)

    for name in list(coerced.keys()):
        if strategy in {"fill_median", "median", "fill_constant", "constant"}:
            coerced[name] = coerced[name].dropna().reset_index(drop=True)

    if coerced["train"].shape[0] < 6 or coerced["validation"].shape[0] < 2 or coerced["test"].shape[0] < 2:
        raise ValueError(
            "Split-safe preprocessing left too few rows in one or more splits. "
            "Reduce missingness, change strategy, or use more data."
        )

    return {
        "frames": coerced,
        "preprocessing": {
            "missing_data_strategy_requested": strategy,
            "missing_data_strategy_effective": effective_strategy,
            "fill_values": fill_values,
            "rows_after": {name: int(part.shape[0]) for name, part in coerced.items()},
        },
    }


def _prepare_xy(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    _require_columns(frame, feature_columns + [target_column])
    subset = frame[feature_columns + [target_column]].copy().dropna()
    x = subset[feature_columns].to_numpy(dtype=float)
    y = subset[target_column].to_numpy(dtype=float)
    return x, y


def _prepare_feature_matrix(*, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    _require_columns(frame, feature_columns)
    subset = frame[feature_columns].copy()
    if subset.isna().any().any():
        raise ValueError("Feature frame contains missing values after preprocessing.")
    return subset.to_numpy(dtype=float)


def _prepare_target_vector(*, frame: pd.DataFrame, target_column: str) -> np.ndarray:
    _require_columns(frame, [target_column])
    target = frame[target_column].copy()
    if target.isna().any():
        raise ValueError("Target column contains missing values after preprocessing.")
    return target.to_numpy(dtype=float)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _fit_regression_tree(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    depth: int,
    max_depth: int,
    min_leaf: int,
) -> dict[str, Any]:
    leaf_value = float(np.mean(y_train)) if y_train.size > 0 else 0.0
    if (
        depth >= max_depth
        or y_train.size < max(2 * int(min_leaf), 8)
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
        "left": _fit_regression_tree(
            x_train=x_train[left_mask],
            y_train=y_train[left_mask],
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf=min_leaf,
        ),
        "right": _fit_regression_tree(
            x_train=x_train[right_mask],
            y_train=y_train[right_mask],
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf=min_leaf,
        ),
    }


def _predict_regression_tree_batch(*, tree: dict[str, Any], x: np.ndarray) -> np.ndarray:
    pred = np.empty(x.shape[0], dtype=float)
    for idx in range(x.shape[0]):
        pred[idx] = _predict_regression_tree_row(tree=tree, row=x[idx])
    return pred


def _predict_regression_tree_row(*, tree: dict[str, Any], row: np.ndarray) -> float:
    node = tree
    while not bool(node.get("leaf", False)):
        feature_index = int(node["feature_index"])
        threshold = float(node["threshold"])
        node = node["left"] if float(row[feature_index]) <= threshold else node["right"]
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
