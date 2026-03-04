from pathlib import Path

from corr2surrogate.orchestration.default_tools import build_default_registry
def test_train_surrogate_candidates_tool_runs_split_safe_linear_and_tree(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "train_compare.csv"
    rows = ["time,x1,x2,y"]
    for idx in range(60):
        x1 = idx / 10.0
        x2 = 1.0 if idx % 2 == 0 else 0.0
        y = 1.8 * x1 + 0.6 * x2 + 0.2
        rows.append(f"{idx},{x1:.4f},{x2:.4f},{y:.4f}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    registry = build_default_registry()
    result = registry.execute(
        "train_surrogate_candidates",
        {
            "data_path": str(csv_path),
            "target_column": "y",
            "feature_columns": ["x1", "x2"],
            "requested_model_family": "auto",
            "timestamp_column": "time",
            "run_id": "compare_run",
        },
    )
    assert result.status == "ok"
    payload = result.output
    assert payload["status"] == "ok"
    assert payload["split"]["strategy"]
    assert payload["split"]["train_size"] > payload["split"]["validation_size"] >= 2
    assert payload["normalization"]["enabled"] is True
    assert payload["normalization"]["normalizer_path"]
    families = [row["model_family"] for row in payload["comparison"]]
    assert "linear_ridge" in families
    assert "lagged_linear" in families
    assert "bagged_tree_ensemble" in families
    assert payload["selected_model_family"] in {
        "linear_ridge",
        "lagged_linear",
        "bagged_tree_ensemble",
    }
    assert isinstance(payload["selected_hyperparameters"], dict)
    assert payload["selected_hyperparameters"]
    assert payload["checkpoint_id"]
    assert Path(payload["run_dir"]).is_dir()
    assert Path(payload["model_params_path"]).is_file()


def test_train_surrogate_candidates_tool_honors_requested_tree_model(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "train_tree.csv"
    rows = ["x1,x2,y"]
    for idx in range(80):
        x1 = -1.0 + (2.0 * idx / 79.0)
        x2 = 1.0 if idx % 3 == 0 else -1.0
        y = 2.5 if (x1 > 0.2 and x2 > 0) else -1.2
        rows.append(f"{x1:.5f},{x2:.5f},{y:.5f}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    registry = build_default_registry()
    result = registry.execute(
        "train_surrogate_candidates",
        {
            "data_path": str(csv_path),
            "target_column": "y",
            "feature_columns": ["x1", "x2"],
            "requested_model_family": "tree_ensemble_candidate",
            "run_id": "tree_run",
        },
    )
    assert result.status == "ok"
    payload = result.output
    assert payload["requested_model_family"] == "bagged_tree_ensemble"
    assert payload["selected_model_family"] == "bagged_tree_ensemble"
    assert payload["selected_metrics"]["test"]["r2"] > 0.0


def test_train_surrogate_candidates_tool_supports_lagged_linear_time_series(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "train_lagged.csv"
    rows = ["time,x,y"]
    base = [0.7, -1.4, 1.1, 0.3, -0.9, 1.6, -0.2, 0.8]
    values = [base[idx % len(base)] for idx in range(120)]
    for idx in range(120):
        x = values[idx]
        y = 0.0 if idx < 3 else (1.4 * values[idx - 3] - 0.2)
        rows.append(f"{idx},{x:.5f},{y:.5f}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    registry = build_default_registry()
    result = registry.execute(
        "train_surrogate_candidates",
        {
            "data_path": str(csv_path),
            "target_column": "y",
            "feature_columns": ["x"],
            "requested_model_family": "lagged_linear",
            "timestamp_column": "time",
            "lag_horizon_samples": 3,
            "run_id": "lagged_run",
        },
    )
    assert result.status == "ok"
    payload = result.output
    families = [row["model_family"] for row in payload["comparison"]]
    assert "lagged_linear" in families
    assert payload["requested_model_family"] == "lagged_linear"
    assert payload["selected_model_family"] == "lagged_linear"
    assert payload["lag_horizon_samples"] == 3
    assert payload["rows_used"] > 0
    assert payload["selected_metrics"]["test"]["r2"] > 0.8


def test_train_surrogate_candidates_auto_can_select_temporal_lagged_model_when_best(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "train_auto_lagged.csv"
    rows = ["time,x,y"]
    base = [1.2, -0.6, 0.9, -1.1, 0.4, 1.8, -0.3]
    values = [base[idx % len(base)] for idx in range(140)]
    for idx in range(140):
        x = values[idx]
        y = 0.5 if idx < 2 else (2.0 * values[idx - 2] + 0.1)
        rows.append(f"{idx},{x:.5f},{y:.5f}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    registry = build_default_registry()
    result = registry.execute(
        "train_surrogate_candidates",
        {
            "data_path": str(csv_path),
            "target_column": "y",
            "feature_columns": ["x"],
            "requested_model_family": "auto",
            "timestamp_column": "time",
            "lag_horizon_samples": 2,
            "run_id": "auto_lagged_run",
        },
    )
    assert result.status == "ok"
    payload = result.output
    assert payload["best_validation_model_family"] in {"lagged_linear", "lagged_tree_ensemble"}
    assert payload["selected_model_family"] in {"lagged_linear", "lagged_tree_ensemble"}
    assert payload["selected_metrics"]["test"]["r2"] > 0.85


def test_train_surrogate_candidates_tool_supports_lagged_tree_ensemble(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "train_lagged_tree.csv"
    rows = ["time,x,y"]
    base = [0.2, 0.9, -0.4, 1.4, -0.8, 0.6, 1.1, -0.2]
    values = [base[idx % len(base)] for idx in range(150)]
    for idx in range(150):
        x = values[idx]
        if idx < 2:
            y = -1.0
        else:
            y = 2.0 if values[idx - 2] > 0.5 else -1.5
        rows.append(f"{idx},{x:.5f},{y:.5f}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    registry = build_default_registry()
    result = registry.execute(
        "train_surrogate_candidates",
        {
            "data_path": str(csv_path),
            "target_column": "y",
            "feature_columns": ["x"],
            "requested_model_family": "lagged_tree_ensemble",
            "timestamp_column": "time",
            "lag_horizon_samples": 2,
            "run_id": "lagged_tree_run",
        },
    )
    assert result.status == "ok"
    payload = result.output
    families = [row["model_family"] for row in payload["comparison"]]
    assert "lagged_tree_ensemble" in families
    assert payload["requested_model_family"] == "lagged_tree_ensemble"
    assert payload["selected_model_family"] == "lagged_tree_ensemble"
    assert payload["selected_metrics"]["test"]["r2"] > 0.75


def test_train_surrogate_candidates_supports_binary_classification_targets(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "classification.csv"
    rows = ["feature_a,feature_b,class_label"]
    for idx in range(80):
        a = idx / 79.0
        b = 1 if idx % 3 == 0 else 0
        label = 1 if (a > 0.55 or b == 1) else 0
        rows.append(f"{a:.5f},{b},{label}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    registry = build_default_registry()
    result = registry.execute(
        "train_surrogate_candidates",
        {
            "data_path": str(csv_path),
            "target_column": "class_label",
            "feature_columns": ["feature_a", "feature_b"],
            "requested_model_family": "auto",
            "run_id": "classification_run",
        },
    )
    assert result.status == "ok"
    payload = result.output
    assert payload["task_profile"]["task_type"] == "binary_classification"
    assert payload["split"]["strategy"] == "stratified_deterministic_modulo_70_15_15"
    families = [row["model_family"] for row in payload["comparison"]]
    assert "logistic_regression" in families
    assert "bagged_tree_classifier" in families
    assert payload["selected_model_family"] in {"logistic_regression", "bagged_tree_classifier"}
    assert payload["selected_metrics"]["test"]["f1"] >= 0.70
    assert "r2" not in payload["selected_metrics"]["test"]
    assert payload["selected_hyperparameters"]["threshold_policy"] == "auto"


def test_train_surrogate_candidates_supports_fraud_detection_targets(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "fraud.csv"
    rows = ["amount_norm,device_risk,velocity_score,fraud_flag"]
    for idx in range(160):
        fraud = 1 if idx % 10 == 0 else 0
        if fraud == 1:
            amount = 0.88 + ((idx % 3) * 0.03)
            device = 0.92
            velocity = 0.90
        else:
            amount = 0.05 + ((idx % 8) * 0.07)
            device = 0.10 + ((idx % 5) * 0.08)
            velocity = 0.08 + ((idx % 6) * 0.06)
        rows.append(f"{amount:.5f},{device:.5f},{velocity:.5f},{fraud}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    registry = build_default_registry()
    result = registry.execute(
        "train_surrogate_candidates",
        {
            "data_path": str(csv_path),
            "target_column": "fraud_flag",
            "feature_columns": ["amount_norm", "device_risk", "velocity_score"],
            "requested_model_family": "auto",
            "run_id": "fraud_run",
        },
    )
    assert result.status == "ok"
    payload = result.output
    assert payload["task_profile"]["task_type"] == "fraud_detection"
    assert payload["split"]["strategy"] == "stratified_deterministic_modulo_70_15_15"
    assert payload["selected_model_family"] in {"logistic_regression", "bagged_tree_classifier"}
    test_metrics = payload["selected_metrics"]["test"]
    assert test_metrics["recall"] >= 0.70
    assert test_metrics["pr_auc"] >= 0.35


def test_train_surrogate_candidates_supports_lagged_binary_classification(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "classification_lagged.csv"
    rows = ["time,feature_x,class_label"]
    base = [0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.85, 0.15]
    values = [base[idx % len(base)] for idx in range(120)]
    for idx in range(120):
        x = values[idx]
        label = 0 if idx < 2 else int(values[idx - 2] > 0.5)
        rows.append(f"{idx},{x:.5f},{label}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    registry = build_default_registry()
    result = registry.execute(
        "train_surrogate_candidates",
        {
            "data_path": str(csv_path),
            "target_column": "class_label",
            "feature_columns": ["feature_x"],
            "requested_model_family": "lagged_logistic_regression",
            "timestamp_column": "time",
            "lag_horizon_samples": 2,
            "run_id": "classification_lagged_run",
        },
    )
    assert result.status == "ok"
    payload = result.output
    assert payload["task_profile"]["task_type"] == "binary_classification"
    assert payload["requested_model_family"] == "lagged_logistic_regression"
    assert payload["selected_model_family"] == "lagged_logistic_regression"
    assert payload["lag_horizon_samples"] == 2
    assert payload["selected_hyperparameters"]["threshold_policy"] == "auto"
    assert payload["selected_hyperparameters"]["lag_horizon_samples"] == 2
    assert payload["selected_metrics"]["test"]["f1"] >= 0.75


def test_train_surrogate_candidates_honors_binary_threshold_policy(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "classification_threshold.csv"
    rows = ["feature_a,feature_b,class_label"]
    for idx in range(100):
        a = idx / 99.0
        b = 1 if idx % 4 == 0 else 0
        label = 1 if (a > 0.62 or b == 1) else 0
        rows.append(f"{a:.5f},{b},{label}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    registry = build_default_registry()
    result = registry.execute(
        "train_surrogate_candidates",
        {
            "data_path": str(csv_path),
            "target_column": "class_label",
            "feature_columns": ["feature_a", "feature_b"],
            "requested_model_family": "logistic_regression",
            "threshold_policy": "favor_recall",
            "run_id": "classification_threshold_run",
        },
    )
    assert result.status == "ok"
    payload = result.output
    assert payload["selected_model_family"] == "logistic_regression"
    assert payload["selected_hyperparameters"]["threshold_policy"] == "favor_recall"
    assert 0.0 < payload["selected_hyperparameters"]["decision_threshold"] < 1.0
