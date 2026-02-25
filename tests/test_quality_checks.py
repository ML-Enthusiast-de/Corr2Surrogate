import pandas as pd

from corr2surrogate.analytics.quality_checks import run_quality_checks


def test_run_quality_checks_detects_duplicates_missing_and_outliers() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00",
                "2026-01-01T00:00:01",
                "2026-01-01T00:00:01",
                "2026-01-01T00:00:03",
                "2026-01-01T00:00:04",
            ],
            "sensor_a": [1.0, 1.1, 1.1, None, 25.0],
            "sensor_b": [0.0, 0.1, 0.1, 0.2, 0.3],
        }
    )
    result = run_quality_checks(frame, timestamp_column="timestamp")
    payload = result.to_dict()
    assert payload["duplicate_rows"] >= 1
    assert payload["duplicate_timestamps"] >= 1
    assert payload["missing_fraction_by_column"]["sensor_a"] > 0.0
    assert "sensor_a" in payload["outlier_count_by_column"]
