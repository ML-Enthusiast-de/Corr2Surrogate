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
    assert payload["report_path"] is not None
    report_path = Path(payload["report_path"])
    assert report_path.is_file()
    assert report_path.parent.name == "data"
    assert report_path.name == "agent1_test_run.md"
