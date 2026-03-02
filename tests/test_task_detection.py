import pandas as pd

from corr2surrogate.analytics.task_detection import (
    assess_task_profile,
    assess_task_profiles,
    normalize_task_type_hint,
)


def test_assess_task_profile_detects_binary_classification() -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.8, 0.9, 0.3, 0.7],
            "class_label": [0, 0, 1, 1, 0, 1],
        }
    )
    profile = assess_task_profile(
        frame=frame,
        target_column="class_label",
        data_mode="steady_state",
    )
    assert profile.task_type == "binary_classification"
    assert profile.task_family == "classification"
    assert profile.recommended_split_strategy == "stratified_deterministic_modulo_70_15_15"


def test_assess_task_profile_detects_fraud_by_name_and_imbalance() -> None:
    labels = [0] * 95 + [1] * 5
    frame = pd.DataFrame(
        {
            "score": [idx / 100.0 for idx in range(100)],
            "fraud_flag": labels,
        }
    )
    profile = assess_task_profile(
        frame=frame,
        target_column="fraud_flag",
        data_mode="steady_state",
    )
    assert profile.task_type == "fraud_detection"
    assert profile.positive_class_label == "1"
    assert profile.minority_class_fraction == 0.05


def test_assess_task_profile_respects_override() -> None:
    frame = pd.DataFrame(
        {
            "signal": [0.0, 0.1, 0.2, 0.3],
            "class_label": [0, 1, 0, 1],
        }
    )
    profile = assess_task_profile(
        frame=frame,
        target_column="class_label",
        data_mode="steady_state",
        task_type_hint="regression",
    )
    assert profile.task_type == "regression"
    assert profile.override_applied is True


def test_assess_task_profiles_skips_missing_targets() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]})
    profiles = assess_task_profiles(
        frame=frame,
        target_columns=["missing", "y"],
        data_mode="steady_state",
    )
    assert [item.target_signal for item in profiles] == ["y"]


def test_normalize_task_type_hint_handles_aliases() -> None:
    assert normalize_task_type_hint("classification") == "binary_classification"
    assert normalize_task_type_hint("fraud") == "fraud_detection"
    assert normalize_task_type_hint("task") is None
