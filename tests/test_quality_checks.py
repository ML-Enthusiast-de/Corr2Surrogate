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


def test_run_quality_checks_numeric_time_column_has_no_false_duplicate_timestamps() -> None:
    frame = pd.DataFrame(
        {
            "time": [0.0, 0.000127, 0.000254, 0.000382, 0.000509],
            "sensor_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = run_quality_checks(frame, timestamp_column="time")
    payload = result.to_dict()
    assert payload["duplicate_timestamps"] == 0
    assert payload["invalid_timestamps"] == 0
    assert payload["monotonic_timestamp"] is True
    assert not any("duplicate timestamps" in item for item in payload["warnings"])


def test_run_quality_checks_comma_decimal_time_column_has_no_false_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "time": ["0,0", "0,000127", "0,000254", "0,000382", "0,000509"],
            "sensor_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = run_quality_checks(frame, timestamp_column="time")
    payload = result.to_dict()
    assert payload["duplicate_timestamps"] == 0
    assert payload["invalid_timestamps"] == 0
    assert payload["monotonic_timestamp"] is True
    assert not any("duplicate timestamps" in item for item in payload["warnings"])
