from pathlib import Path

from corr2surrogate.orchestration.default_tools import build_default_registry


def test_default_tools_train_resume_and_analyze(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    registry = build_default_registry()

    data1 = tmp_path / "train.csv"
    data1.write_text(
        "\n".join(
            [
                "x1,x2,y",
                "0,1,1",
                "1,1,3",
                "2,1,5",
                "3,1,7",
                "4,1,9",
            ]
        ),
        encoding="utf-8",
    )
    train_result = registry.execute(
        "train_incremental_linear_surrogate",
        {
            "data_path": str(data1),
            "target_column": "y",
            "feature_columns": ["x1", "x2"],
            "run_id": "run_train",
        },
    )
    assert train_result.status == "ok"
    checkpoint_id = train_result.output["checkpoint_id"]
    assert checkpoint_id

    data2 = tmp_path / "extra.csv"
    data2.write_text(
        "\n".join(
            [
                "x1,x2,y",
                "5,1,11",
                "6,1,13",
            ]
        ),
        encoding="utf-8",
    )
    resume_result = registry.execute(
        "resume_incremental_linear_surrogate",
        {
            "checkpoint_id": checkpoint_id,
            "additional_data_path": str(data2),
            "run_id": "run_resume",
        },
    )
    assert resume_result.status == "ok"
    assert resume_result.output["new_checkpoint_id"]

    analyze_result = registry.execute(
        "analyze_model_checkpoint_performance",
        {
            "checkpoint_id": resume_result.output["new_checkpoint_id"],
            "data_path": str(data2),
            "top_k_regions": 1,
            "trajectory_budget": 1,
        },
    )
    assert analyze_result.status == "ok"
    assert analyze_result.output["feedback"]["trajectory_suggestions"]
