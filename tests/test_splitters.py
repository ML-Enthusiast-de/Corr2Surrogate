import numpy as np

from corr2surrogate.modeling.splitters import build_train_validation_test_split


def test_build_train_validation_test_split_keeps_time_series_order() -> None:
    split = build_train_validation_test_split(
        n_rows=40,
        data_mode="time_series",
        task_type="binary_classification",
        stratify_labels=[0, 1] * 20,
    )
    assert split.strategy == "blocked_time_order_70_15_15"
    assert int(split.train_indices.max()) < int(split.validation_indices.min())
    assert int(split.validation_indices.max()) < int(split.test_indices.min())


def test_build_train_validation_test_split_stratifies_classification_labels() -> None:
    labels = np.array([0] * 80 + [1] * 20, dtype=int)
    split = build_train_validation_test_split(
        n_rows=100,
        data_mode="steady_state",
        task_type="fraud_detection",
        stratify_labels=labels,
    )
    assert split.strategy == "stratified_deterministic_modulo_70_15_15"

    def positive_fraction(indices: np.ndarray) -> float:
        subset = labels[indices]
        return float(np.mean(subset == 1))

    train_rate = positive_fraction(split.train_indices)
    val_rate = positive_fraction(split.validation_indices)
    test_rate = positive_fraction(split.test_indices)
    assert abs(train_rate - val_rate) < 0.08
    assert abs(train_rate - test_rate) < 0.08
