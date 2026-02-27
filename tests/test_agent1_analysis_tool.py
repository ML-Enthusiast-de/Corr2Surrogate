from pathlib import Path

from corr2surrogate.orchestration.default_tools import build_default_registry


def test_run_agent1_analysis_tool_end_to_end(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "\n".join(
            [
                "time,sensor_a,sensor_b,target",
                "0,1.0,0.1,2.0",
                "1,1.5,0.1,3.0",
                "2,2.0,0.2,4.0",
                "3,2.5,0.2,5.0",
                "4,3.0,0.3,6.0",
                "5,3.5,0.3,7.0",
                "6,4.0,0.4,8.0",
                "7,4.5,0.4,9.0",
                "8,5.0,0.5,10.0",
            ]
        ),
        encoding="utf-8",
    )

    registry = build_default_registry()
    result = registry.execute(
        "run_agent1_analysis",
        {
            "data_path": str(csv_path),
            "timestamp_column": "time",
            "target_signals": ["target"],
            "forced_requests": [
                {
                    "target_signal": "target",
                    "predictor_signals": ["sensor_a", "sensor_b"],
                    "user_reason": "lab requirement",
                }
            ],
            "user_hypotheses": [
                {
                    "target_signal": "target",
                    "predictor_signals": ["sensor_a"],
                    "user_reason": "expected strong coupling",
                }
            ],
            "feature_hypotheses": [
                {
                    "target_signal": "target",
                    "base_signal": "sensor_a",
                    "transformation": "rate_change",
                    "user_reason": "dynamic bench behavior",
                }
            ],
            "save_report": True,
            "run_id": "agent1_test_run",
        },
    )
    assert result.status == "ok"
    payload = result.output
    assert payload["status"] == "ok"
    assert payload["target_count"] == 1
    assert payload["ranking"]
    assert payload["forced_requests"]
    assert payload["user_hypotheses"]
    assert payload["feature_hypotheses"]
    assert payload["report_path"] is not None
    assert payload["lineage_path"]
    assert payload["artifact_paths"]["artifact_dir"]
    report_path = Path(payload["report_path"])
    assert report_path.is_file()
    assert report_path.parent.name == "data"
    assert report_path.name == "agent1_test_run.md"
    markdown = payload["report_markdown"]
    assert "Preprocessing Decisions" in markdown
    assert "User Hypotheses" in markdown
    assert "Agentic Planning" in markdown
    assert "Sensor Diagnostics" in markdown
    assert "Experiment Recommendations" in markdown
    assert "Split leakage risk (missing-data plan)" in markdown
    assert "Correlation Details (Top 10 Predictors per Target)" in markdown
    assert "| Category | Target | Rank | Predictor | Correlation Type | Strength |" in markdown
    assert "Feature Engineering Opportunities (Top 10)" in markdown
    assert "User Hypothesis Checks" in markdown


def test_run_agent1_analysis_tool_handles_nan_and_row_coverage_strategy(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "nan_lengths.csv"
    csv_path.write_text(
        "\n".join(
            [
                "time,sensor_a,sensor_b,target",
                "0,1.0,10.0,2.0",
                "1,1.2,,2.3",
                "2,1.4,10.8,2.7",
                "3,,11.2,3.0",
                "4,1.9,11.5,3.6",
                "5,2.2,,4.1",
                "6,2.5,12.1,4.8",
                "7,2.8,12.6,5.3",
                "8,,13.0,6.0",
                "9,3.5,13.5,6.9",
                "10,3.9,13.9,7.8",
                "11,4.2,,8.4",
                "12,4.6,14.7,9.1",
                "13,5.0,15.0,9.9",
            ]
        ),
        encoding="utf-8",
    )

    registry = build_default_registry()
    result = registry.execute(
        "run_agent1_analysis",
        {
            "data_path": str(csv_path),
            "timestamp_column": "time",
            "target_signals": ["target"],
            "missing_data_strategy": "fill_median",
            "row_coverage_strategy": "drop_sparse_rows",
            "sparse_row_min_fraction": 0.75,
            "max_samples": 10,
            "sample_selection": "uniform",
            "save_report": False,
        },
    )
    assert result.status == "ok"
    payload = result.output
    assert payload["status"] == "ok"
    preprocessing = payload["preprocessing"]
    assert preprocessing["sample_plan"]["applied"] is True
    assert preprocessing["missing_data_plan"]["applied"] is True
    assert preprocessing["missing_data_plan"]["split_leakage_risk"] == "high"
    assert "split first" in preprocessing["missing_data_plan"]["recommended_split_safe_policy"].lower()
    assert preprocessing["row_coverage_plan"]["applied"] is True
    assert preprocessing["final_rows"] <= preprocessing["initial_rows"]


def test_run_agent1_analysis_respects_user_preprocessing_when_strategy_search_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "nan_user_locked.csv"
    csv_path.write_text(
        "\n".join(
            [
                "time,sensor_a,sensor_b,target",
                "0,1.0,10.0,2.0",
                "1,1.2,,2.3",
                "2,1.4,10.8,2.7",
                "3,,11.2,3.0",
                "4,1.9,11.5,3.6",
                "5,2.2,,4.1",
                "6,2.5,12.1,4.8",
                "7,2.8,12.6,5.3",
                "8,,13.0,6.0",
                "9,3.5,13.5,6.9",
                "10,3.9,13.9,7.8",
                "11,4.2,,8.4",
                "12,4.6,14.7,9.1",
                "13,5.0,15.0,9.9",
            ]
        ),
        encoding="utf-8",
    )

    registry = build_default_registry()
    result = registry.execute(
        "run_agent1_analysis",
        {
            "data_path": str(csv_path),
            "timestamp_column": "time",
            "target_signals": ["target"],
            "missing_data_strategy": "drop_rows",
            "row_coverage_strategy": "keep",
            "enable_strategy_search": True,
            "strategy_search_candidates": 4,
            "save_report": False,
        },
    )
    assert result.status == "ok"
    payload = result.output
    assert payload["status"] == "ok"
    preprocessing = payload["preprocessing"]
    assert preprocessing["missing_data_plan"]["strategy"] == "drop_rows"
    assert preprocessing["missing_data_plan"]["split_leakage_risk"] == "low"
    assert preprocessing["missing_data_plan"]["applied"] is True
    assert payload["critic_decision"]["selected_candidate_id"] == "user_plan"
    assert payload["critic_decision"]["planner_recommended_candidate_id"] in {
        "user_plan",
        "fill_median_trim",
        "fill_constant_keep",
        "drop_rows_drop_sparse",
        "keep_keep",
    }
