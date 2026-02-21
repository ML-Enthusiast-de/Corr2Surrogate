import pandas as pd

from corr2surrogate.ingestion.csv_loader import _infer_header_and_data_start


def test_infer_header_identifies_textual_header_row() -> None:
    raw = pd.DataFrame(
        [
            ["metadata", None, None],
            ["time_s", "sensor_a", "sensor_b"],
            [0.0, 1.0, 2.0],
            [1.0, 1.5, 2.5],
        ]
    )
    preview = raw.head(4)
    inferred = _infer_header_and_data_start(preview, raw, confidence_threshold=0.70)
    assert inferred.header_row == 1
    assert inferred.data_start_row == 2


def test_infer_header_flags_low_confidence_when_rows_look_numeric() -> None:
    raw = pd.DataFrame(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )
    inferred = _infer_header_and_data_start(raw, raw, confidence_threshold=0.70)
    assert inferred.needs_user_confirmation
