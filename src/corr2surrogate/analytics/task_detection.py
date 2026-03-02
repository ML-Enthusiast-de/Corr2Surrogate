"""Task-profile inference helpers shared by Agent 1 and Agent 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_TASK_TYPES = {
    "auto",
    "regression",
    "binary_classification",
    "multiclass_classification",
    "fraud_detection",
    "anomaly_detection",
}

CLASSIFICATION_TASK_TYPES = {
    "binary_classification",
    "multiclass_classification",
    "fraud_detection",
    "anomaly_detection",
}

_BOOLEAN_LABELS = {
    "0",
    "1",
    "true",
    "false",
    "yes",
    "no",
    "y",
    "n",
}

_FRAUD_KEYWORDS = ("fraud", "chargeback", "scam", "abuse")
_ANOMALY_KEYWORDS = ("anomaly", "anomal", "outlier", "fault", "alert", "attack")


@dataclass(frozen=True)
class TaskProfile:
    """Detected or user-forced task profile for one target signal."""

    target_signal: str
    task_type: str
    task_family: str
    override_applied: bool
    row_count: int
    class_count: int
    class_balance: dict[str, int]
    minority_class_fraction: float | None
    positive_class_label: str | None
    recommended_split_strategy: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_task_type_hint(value: str | None) -> str | None:
    """Normalize user-provided task hints to supported task identifiers."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": None,
        "auto": "auto",
        "regression": "regression",
        "regress": "regression",
        "classification": "binary_classification",
        "binary": "binary_classification",
        "binary_classification": "binary_classification",
        "multiclass": "multiclass_classification",
        "multiclass_classification": "multiclass_classification",
        "fraud": "fraud_detection",
        "fraud_detection": "fraud_detection",
        "anomaly": "anomaly_detection",
        "anomaly_detection": "anomaly_detection",
    }
    return aliases.get(normalized)


def is_classification_task(task_type: str | None) -> bool:
    """Return whether the normalized task type is classification-like."""
    return str(task_type or "").strip() in CLASSIFICATION_TASK_TYPES


def assess_task_profiles(
    *,
    frame: pd.DataFrame,
    target_columns: list[str],
    data_mode: str,
    task_type_hint: str | None = None,
) -> list[TaskProfile]:
    """Infer task profiles for the requested target columns."""
    profiles: list[TaskProfile] = []
    for target_column in target_columns:
        if target_column not in frame.columns:
            continue
        profiles.append(
            assess_task_profile(
                frame=frame,
                target_column=target_column,
                data_mode=data_mode,
                task_type_hint=task_type_hint,
            )
        )
    return profiles


def assess_task_profile(
    *,
    frame: pd.DataFrame,
    target_column: str,
    data_mode: str,
    task_type_hint: str | None = None,
) -> TaskProfile:
    """Infer whether a target is regression or classification-like."""
    if target_column not in frame.columns:
        raise ValueError(f"Target column `{target_column}` is not present in the dataset.")
    series = frame[target_column]
    non_null = series.dropna()
    override = normalize_task_type_hint(task_type_hint)
    if override and override != "auto":
        return _build_profile_from_type(
            target_column=target_column,
            task_type=override,
            data_mode=data_mode,
            series=non_null,
            override_applied=True,
            rationale=f"User override forced task type `{override}`.",
        )
    return _infer_task_profile(
        target_column=target_column,
        data_mode=data_mode,
        series=non_null,
    )


def _infer_task_profile(
    *,
    target_column: str,
    data_mode: str,
    series: pd.Series,
) -> TaskProfile:
    row_count = int(len(series))
    if row_count == 0:
        return _build_profile_from_type(
            target_column=target_column,
            task_type="regression",
            data_mode=data_mode,
            series=series,
            override_applied=False,
            rationale="No non-null target rows were available, so regression is the safe default.",
        )

    label_stats = _label_statistics(series)
    numeric_ratio = _numeric_ratio(series)
    integer_like = _looks_integer_like(series)
    bool_like = _looks_boolean_like(series)
    class_count = int(label_stats["class_count"])
    distinct_ratio = class_count / max(row_count, 1)
    name_lower = target_column.strip().lower()
    fraud_named = any(token in name_lower for token in _FRAUD_KEYWORDS)
    anomaly_named = any(token in name_lower for token in _ANOMALY_KEYWORDS)
    discrete_threshold = max(3, min(16, int(round(math.sqrt(row_count) * 0.6))))
    classification_like = bool_like or (
        integer_like and class_count <= discrete_threshold and distinct_ratio <= 0.12
    )

    if class_count == 2 and classification_like:
        task_type = "binary_classification"
        if fraud_named:
            task_type = "fraud_detection"
        elif anomaly_named:
            task_type = "anomaly_detection"
        rationale = (
            f"Detected 2 discrete target states with class balance {label_stats['class_balance']}."
        )
        return _build_profile_from_type(
            target_column=target_column,
            task_type=task_type,
            data_mode=data_mode,
            series=series,
            override_applied=False,
            rationale=rationale,
        )

    if class_count >= 3 and classification_like:
        task_type = "multiclass_classification"
        rationale = (
            f"Detected {class_count} discrete target states with distinct_ratio={distinct_ratio:.3f}."
        )
        return _build_profile_from_type(
            target_column=target_column,
            task_type=task_type,
            data_mode=data_mode,
            series=series,
            override_applied=False,
            rationale=rationale,
        )

    if (fraud_named or anomaly_named) and class_count <= discrete_threshold and class_count >= 2:
        task_type = "fraud_detection" if fraud_named else "anomaly_detection"
        rationale = (
            "Target name suggests a rare-event label, and the observed label cardinality is low "
            f"({class_count} states)."
        )
        return _build_profile_from_type(
            target_column=target_column,
            task_type=task_type,
            data_mode=data_mode,
            series=series,
            override_applied=False,
            rationale=rationale,
        )

    rationale = (
        "Target behaves like a continuous signal "
        f"(numeric_ratio={numeric_ratio:.3f}, distinct_states={class_count})."
    )
    return _build_profile_from_type(
        target_column=target_column,
        task_type="regression",
        data_mode=data_mode,
        series=series,
        override_applied=False,
        rationale=rationale,
    )


def _build_profile_from_type(
    *,
    target_column: str,
    task_type: str,
    data_mode: str,
    series: pd.Series,
    override_applied: bool,
    rationale: str,
) -> TaskProfile:
    labels = _label_statistics(series)
    recommended_split_strategy = _recommended_split_strategy(
        data_mode=data_mode,
        task_type=task_type,
    )
    positive_class_label: str | None = None
    minority_fraction: float | None = None
    if is_classification_task(task_type) and labels["class_balance"]:
        ordered = sorted(
            labels["class_balance"].items(),
            key=lambda item: (item[1], str(item[0])),
        )
        positive_class_label = str(ordered[0][0])
        minority_fraction = float(ordered[0][1]) / max(int(labels["row_count"]), 1)
        if task_type in {"fraud_detection", "anomaly_detection"} and minority_fraction is not None:
            rationale = (
                f"{rationale} Minority class `{positive_class_label}` covers "
                f"{minority_fraction:.3%} of labeled rows."
            )
    return TaskProfile(
        target_signal=target_column,
        task_type=task_type,
        task_family="classification" if is_classification_task(task_type) else "regression",
        override_applied=override_applied,
        row_count=int(labels["row_count"]),
        class_count=int(labels["class_count"]),
        class_balance={str(k): int(v) for k, v in labels["class_balance"].items()},
        minority_class_fraction=minority_fraction,
        positive_class_label=positive_class_label,
        recommended_split_strategy=recommended_split_strategy,
        rationale=rationale,
    )


def _recommended_split_strategy(*, data_mode: str, task_type: str) -> str:
    mode = str(data_mode).strip().lower()
    if mode == "time_series":
        return "blocked_time_order_70_15_15"
    if is_classification_task(task_type):
        return "stratified_deterministic_modulo_70_15_15"
    return "deterministic_modulo_70_15_15"


def _label_statistics(series: pd.Series) -> dict[str, Any]:
    clean = series.dropna()
    if clean.empty:
        return {"row_count": 0, "class_count": 0, "class_balance": {}}
    labels = clean.map(_normalize_label_token)
    counts = labels.value_counts(dropna=False)
    return {
        "row_count": int(labels.shape[0]),
        "class_count": int(counts.shape[0]),
        "class_balance": {str(index): int(value) for index, value in counts.items()},
    }


def _normalize_label_token(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else "missing"
    if math.isfinite(numeric) and abs(numeric - round(numeric)) <= 1e-9:
        return str(int(round(numeric)))
    return str(numeric)


def _numeric_ratio(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.notna().mean())


def _looks_integer_like(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    rounded = np.round(numeric.to_numpy(dtype=float))
    close_fraction = float(np.mean(np.isclose(numeric.to_numpy(dtype=float), rounded, atol=1e-9)))
    return close_fraction >= 0.98


def _looks_boolean_like(series: pd.Series) -> bool:
    lowered = {str(value).strip().lower() for value in series.dropna()}
    return bool(lowered) and lowered.issubset(_BOOLEAN_LABELS)
