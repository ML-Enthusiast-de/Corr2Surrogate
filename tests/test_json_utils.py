import json
from pathlib import Path

from corr2surrogate.analytics.reporting import save_agent1_artifacts
from corr2surrogate.core.json_utils import dumps_json


def test_dumps_json_converts_nonfinite_to_null() -> None:
    payload = {
        "nan_value": float("nan"),
        "pos_inf": float("inf"),
        "neg_inf": float("-inf"),
        "nested": {"inner": float("nan")},
    }
    encoded = dumps_json(payload, indent=2)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    decoded = json.loads(encoded)
    assert decoded["nan_value"] is None
    assert decoded["pos_inf"] is None
    assert decoded["neg_inf"] is None
    assert decoded["nested"]["inner"] is None


def test_save_agent1_artifacts_writes_strict_json(tmp_path: Path) -> None:
    structured = {
        "correlations": {
            "target_analyses": [
                {
                    "target_signal": "target",
                    "predictor_results": [{"predictor_signal": "p1", "best_abs_score": float("nan")}],
                    "feature_opportunities": [],
                    "hypothesis_pair_checks": [],
                    "hypothesis_feature_checks": [],
                }
            ]
        },
        "experiment_recommendations": [{"score": float("inf")}],
        "sensor_diagnostics": {"diagnostics": [{"trust_score": float("-inf")}]},
        "planner_trace": [{"score": float("nan")}],
    }
    paths = save_agent1_artifacts(
        structured=structured,
        data_path="demo.csv",
        reports_dir=tmp_path,
        run_id="agent1_json_strict",
    )
    json_path = Path(paths["json_path"])
    content = json_path.read_text(encoding="utf-8")
    assert "NaN" not in content
    assert "Infinity" not in content
    decoded = json.loads(content)
    assert decoded["correlations"]["target_analyses"][0]["predictor_results"][0]["best_abs_score"] is None
    assert decoded["experiment_recommendations"][0]["score"] is None
    assert decoded["sensor_diagnostics"]["diagnostics"][0]["trust_score"] is None
    assert decoded["planner_trace"][0]["score"] is None
