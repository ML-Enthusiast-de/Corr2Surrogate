"""Small local classifier baselines without external ML dependencies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from corr2surrogate.core.json_utils import write_json
from .evaluation import classification_metrics


class LogisticClassificationSurrogate:
    """One-vs-rest logistic classifier trained with simple gradient descent."""

    def __init__(
        self,
        *,
        feature_columns: list[str],
        target_column: str,
        learning_rate: float = 0.25,
        epochs: int = 350,
        l2: float = 1e-4,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.class_labels: list[str] = []
        self._weights: np.ndarray | None = None

    def fit_dataframe(self, frame: pd.DataFrame) -> int:
        x = _prepare_feature_matrix(frame=frame, feature_columns=self.feature_columns)
        y = _prepare_label_vector(frame=frame, target_column=self.target_column)
        if x.shape[0] == 0:
            raise ValueError("No valid rows available for classifier training.")
        class_labels = sorted({label for label in y})
        if len(class_labels) < 2:
            raise ValueError("Classification requires at least two target classes.")
        x_bias = _with_bias(x)
        weights: list[np.ndarray] = []
        for label in class_labels:
            target = (y == label).astype(float)
            current = np.zeros(x_bias.shape[1], dtype=float)
            for _ in range(self.epochs):
                logits = x_bias @ current
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
                gradient = (x_bias.T @ (probs - target)) / x_bias.shape[0]
                regularization = self.l2 * current
                regularization[0] = 0.0
                current -= self.learning_rate * (gradient + regularization)
            weights.append(current)
        self.class_labels = class_labels
        self._weights = np.vstack(weights)
        return int(x.shape[0])

    def predict_proba_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        weights = self._require_weights()
        x = _prepare_feature_matrix(frame=frame, feature_columns=self.feature_columns)
        logits = _with_bias(x) @ weights.T
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        row_sums = np.sum(probs, axis=1, keepdims=True)
        row_sums[row_sums <= 1e-12] = 1.0
        return probs / row_sums

    def predict_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba_dataframe(frame)
        labels = np.asarray(self.class_labels, dtype=object)
        return labels[np.argmax(probs, axis=1)]

    def evaluate_dataframe(self, frame: pd.DataFrame) -> dict[str, float]:
        y_true = _prepare_label_vector(frame=frame, target_column=self.target_column)
        probs = self.predict_proba_dataframe(frame)
        return classification_metrics(
            y_true=y_true,
            probabilities=probs,
            class_labels=self.class_labels,
        ).to_dict()

    def state_dict(self) -> dict[str, Any]:
        weights = self._require_weights()
        return {
            "model_name": "logistic_regression",
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "learning_rate": float(self.learning_rate),
            "epochs": int(self.epochs),
            "l2": float(self.l2),
            "class_labels": list(self.class_labels),
            "weights": weights.tolist(),
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return write_json(output, self.state_dict(), indent=2)

    def _require_weights(self) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Logistic classifier has not been fitted yet.")
        return self._weights


class BaggedTreeClassifierSurrogate:
    """Small bagged classification tree ensemble."""

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
        self.class_labels: list[str] = []
        self._estimators: list[dict[str, Any]] = []

    def fit_dataframe(self, frame: pd.DataFrame) -> int:
        x = _prepare_feature_matrix(frame=frame, feature_columns=self.feature_columns)
        y_labels = _prepare_label_vector(frame=frame, target_column=self.target_column)
        if x.shape[0] == 0:
            raise ValueError("No valid rows available for tree-classifier training.")
        self.class_labels = sorted({label for label in y_labels})
        if len(self.class_labels) < 2:
            raise ValueError("Classification requires at least two target classes.")
        label_to_idx = {label: idx for idx, label in enumerate(self.class_labels)}
        y = np.asarray([label_to_idx[label] for label in y_labels], dtype=int)

        self._estimators = []
        feature_count = x.shape[1]
        feature_subsample = max(1, int(math.sqrt(feature_count)))
        for seed in range(self.n_estimators):
            rng = np.random.default_rng(700 + seed)
            row_idx = rng.integers(0, x.shape[0], size=x.shape[0])
            feat_idx = np.sort(
                rng.choice(feature_count, size=feature_subsample, replace=False)
            ).astype(int)
            tree = _fit_classification_tree(
                x_train=x[row_idx][:, feat_idx],
                y_train=y[row_idx],
                depth=0,
                max_depth=self.max_depth,
                min_leaf=min(self.min_leaf, max(2, x.shape[0] // 8)),
                n_classes=len(self.class_labels),
            )
            self._estimators.append(
                {
                    "feature_indices": feat_idx.tolist(),
                    "tree": tree,
                }
            )
        return int(x.shape[0])

    def predict_proba_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._estimators:
            raise RuntimeError("Tree classifier has not been fitted yet.")
        x = _prepare_feature_matrix(frame=frame, feature_columns=self.feature_columns)
        probs = []
        for estimator in self._estimators:
            feat_idx = np.asarray(estimator["feature_indices"], dtype=int)
            pred = _predict_classification_tree_batch(
                tree=estimator["tree"],
                x=x[:, feat_idx],
                n_classes=len(self.class_labels),
            )
            probs.append(pred)
        stacked = np.mean(np.stack(probs, axis=0), axis=0)
        row_sums = np.sum(stacked, axis=1, keepdims=True)
        row_sums[row_sums <= 1e-12] = 1.0
        return stacked / row_sums

    def predict_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba_dataframe(frame)
        labels = np.asarray(self.class_labels, dtype=object)
        return labels[np.argmax(probs, axis=1)]

    def evaluate_dataframe(self, frame: pd.DataFrame) -> dict[str, float]:
        y_true = _prepare_label_vector(frame=frame, target_column=self.target_column)
        probs = self.predict_proba_dataframe(frame)
        return classification_metrics(
            y_true=y_true,
            probabilities=probs,
            class_labels=self.class_labels,
        ).to_dict()

    def state_dict(self) -> dict[str, Any]:
        if not self._estimators:
            raise RuntimeError("Tree classifier has not been fitted yet.")
        return {
            "model_name": "bagged_tree_classifier",
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "n_estimators": int(self.n_estimators),
            "max_depth": int(self.max_depth),
            "min_leaf": int(self.min_leaf),
            "class_labels": list(self.class_labels),
            "estimators": self._estimators,
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return write_json(output, self.state_dict(), indent=2)


class BoostedTreeClassifierSurrogate:
    """Simple SAMME-style boosted tree classifier with local decision trees."""

    def __init__(
        self,
        *,
        feature_columns: list[str],
        target_column: str,
        n_estimators: int = 24,
        learning_rate: float = 0.6,
        max_depth: int = 3,
        min_leaf: int = 5,
    ) -> None:
        if not feature_columns:
            raise ValueError("feature_columns cannot be empty.")
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_leaf = int(min_leaf)
        self.class_labels: list[str] = []
        self._estimators: list[dict[str, Any]] = []

    def fit_dataframe(self, frame: pd.DataFrame) -> int:
        x = _prepare_feature_matrix(frame=frame, feature_columns=self.feature_columns)
        y_labels = _prepare_label_vector(frame=frame, target_column=self.target_column)
        if x.shape[0] == 0:
            raise ValueError("No valid rows available for boosted classifier training.")
        self.class_labels = sorted({label for label in y_labels})
        if len(self.class_labels) < 2:
            raise ValueError("Classification requires at least two target classes.")

        label_to_idx = {label: idx for idx, label in enumerate(self.class_labels)}
        y = np.asarray([label_to_idx[label] for label in y_labels], dtype=int)
        n_rows = int(x.shape[0])
        n_classes = int(len(self.class_labels))
        feature_count = int(x.shape[1])
        feature_subsample = max(1, int(feature_count))
        weights = np.full(n_rows, 1.0 / n_rows, dtype=float)
        self._estimators = []

        for step in range(self.n_estimators):
            rng = np.random.default_rng(900 + step)
            row_idx = rng.choice(n_rows, size=n_rows, replace=True, p=weights).astype(int)
            feat_idx = np.sort(
                rng.choice(feature_count, size=feature_subsample, replace=False)
            ).astype(int)
            tree = _fit_classification_tree(
                x_train=x[row_idx][:, feat_idx],
                y_train=y[row_idx],
                depth=0,
                max_depth=self.max_depth,
                min_leaf=min(self.min_leaf, max(2, n_rows // 10)),
                n_classes=n_classes,
            )
            prob = _predict_classification_tree_batch(
                tree=tree,
                x=x[:, feat_idx],
                n_classes=n_classes,
            )
            pred = np.argmax(prob, axis=1).astype(int)
            miss = (pred != y).astype(float)
            weighted_error = float(np.sum(weights * miss))
            weighted_error = min(max(weighted_error, 1e-6), 1.0 - 1e-6)

            if n_classes > 2:
                alpha = self.learning_rate * (
                    math.log((1.0 - weighted_error) / weighted_error) + math.log(n_classes - 1.0)
                )
            else:
                alpha = self.learning_rate * 0.5 * math.log((1.0 - weighted_error) / weighted_error)

            if not math.isfinite(alpha):
                continue
            weights *= np.exp(alpha * miss)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0.0 or not math.isfinite(weight_sum):
                break
            weights /= weight_sum
            self._estimators.append(
                {
                    "feature_indices": feat_idx.tolist(),
                    "tree": tree,
                    "alpha": float(alpha),
                }
            )
            if weighted_error <= 1e-4:
                break

        if not self._estimators:
            raise RuntimeError("Boosted classifier failed to build any valid weak learner.")
        return n_rows

    def predict_proba_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._estimators:
            raise RuntimeError("Boosted classifier has not been fitted yet.")
        x = _prepare_feature_matrix(frame=frame, feature_columns=self.feature_columns)
        n_classes = len(self.class_labels)
        scores = np.zeros((x.shape[0], n_classes), dtype=float)
        for estimator in self._estimators:
            feat_idx = np.asarray(estimator["feature_indices"], dtype=int)
            alpha = float(estimator.get("alpha", 1.0))
            probs = _predict_classification_tree_batch(
                tree=estimator["tree"],
                x=x[:, feat_idx],
                n_classes=n_classes,
            )
            pred_idx = np.argmax(probs, axis=1)
            scores[np.arange(x.shape[0]), pred_idx] += alpha
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(shifted, -40.0, 40.0))
        denom = np.sum(exp_scores, axis=1, keepdims=True)
        denom[denom <= 1e-12] = 1.0
        return exp_scores / denom

    def predict_dataframe(self, frame: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba_dataframe(frame)
        labels = np.asarray(self.class_labels, dtype=object)
        return labels[np.argmax(probs, axis=1)]

    def evaluate_dataframe(self, frame: pd.DataFrame) -> dict[str, float]:
        y_true = _prepare_label_vector(frame=frame, target_column=self.target_column)
        probs = self.predict_proba_dataframe(frame)
        return classification_metrics(
            y_true=y_true,
            probabilities=probs,
            class_labels=self.class_labels,
        ).to_dict()

    def state_dict(self) -> dict[str, Any]:
        if not self._estimators:
            raise RuntimeError("Boosted classifier has not been fitted yet.")
        return {
            "model_name": "boosted_tree_classifier",
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "n_estimators": int(self.n_estimators),
            "learning_rate": float(self.learning_rate),
            "max_depth": int(self.max_depth),
            "min_leaf": int(self.min_leaf),
            "class_labels": list(self.class_labels),
            "estimators": self._estimators,
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        return write_json(output, self.state_dict(), indent=2)


def _prepare_feature_matrix(*, frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    subset = frame[feature_columns].copy()
    if subset.isna().any().any():
        raise ValueError("Feature frame contains missing values after preprocessing.")
    return subset.to_numpy(dtype=float)


def _prepare_label_vector(*, frame: pd.DataFrame, target_column: str) -> np.ndarray:
    target = frame[target_column]
    if target.isna().any():
        raise ValueError("Target column contains missing values after preprocessing.")
    normalized = [str(item).strip() for item in target]
    if any(not item for item in normalized):
        raise ValueError("Target column contains empty labels after preprocessing.")
    return np.asarray(normalized, dtype=object)


def _with_bias(x: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((x.shape[0], 1), dtype=float), x])


def _fit_classification_tree(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    depth: int,
    max_depth: int,
    min_leaf: int,
    n_classes: int,
) -> dict[str, Any]:
    class_counts = np.bincount(y_train, minlength=n_classes).astype(float)
    probs = class_counts / max(float(np.sum(class_counts)), 1.0)
    if (
        depth >= max_depth
        or y_train.size < max(2 * int(min_leaf), 8)
        or int(np.unique(y_train).size) <= 1
    ):
        return {"leaf": True, "probs": probs.tolist()}

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
            score = _gini_impurity(y_train[left_mask], n_classes) + _gini_impurity(
                y_train[right_mask], n_classes
            )
            if score < best_score:
                best_score = score
                best_feature = feature_idx
                best_threshold = float(threshold)

    if best_feature < 0 or not math.isfinite(best_threshold):
        return {"leaf": True, "probs": probs.tolist()}

    values = x_train[:, best_feature]
    left_mask = values <= best_threshold
    right_mask = ~left_mask
    return {
        "leaf": False,
        "probs": probs.tolist(),
        "feature_index": int(best_feature),
        "threshold": float(best_threshold),
        "left": _fit_classification_tree(
            x_train=x_train[left_mask],
            y_train=y_train[left_mask],
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf=min_leaf,
            n_classes=n_classes,
        ),
        "right": _fit_classification_tree(
            x_train=x_train[right_mask],
            y_train=y_train[right_mask],
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf=min_leaf,
            n_classes=n_classes,
        ),
    }


def _predict_classification_tree_batch(
    *,
    tree: dict[str, Any],
    x: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    pred = np.empty((x.shape[0], n_classes), dtype=float)
    for idx in range(x.shape[0]):
        pred[idx] = _predict_classification_tree_row(tree=tree, row=x[idx], n_classes=n_classes)
    return pred


def _predict_classification_tree_row(
    *,
    tree: dict[str, Any],
    row: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    node = tree
    while not bool(node.get("leaf", False)):
        feature_index = int(node["feature_index"])
        threshold = float(node["threshold"])
        node = node["left"] if float(row[feature_index]) <= threshold else node["right"]
    probs = np.asarray(node.get("probs", [0.0] * n_classes), dtype=float)
    if probs.shape[0] != n_classes:
        fixed = np.zeros(n_classes, dtype=float)
        limit = min(n_classes, probs.shape[0])
        fixed[:limit] = probs[:limit]
        probs = fixed
    total = float(np.sum(probs))
    if total <= 1e-12:
        probs = np.full(n_classes, 1.0 / max(n_classes, 1), dtype=float)
    else:
        probs = probs / total
    return probs


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


def _gini_impurity(values: np.ndarray, n_classes: int) -> float:
    if values.size == 0:
        return 0.0
    counts = np.bincount(values, minlength=n_classes).astype(float)
    probs = counts / float(np.sum(counts))
    return float(values.size) * (1.0 - float(np.sum(np.square(probs))))
