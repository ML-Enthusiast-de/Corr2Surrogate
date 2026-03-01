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
    assert "bagged_tree_ensemble" in families
    assert payload["selected_model_family"] in {"linear_ridge", "bagged_tree_ensemble"}
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
