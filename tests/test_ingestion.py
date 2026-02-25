import pandas as pd

from corr2surrogate.ingestion.csv_loader import _infer_header_and_data_start, load_tabular_data


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


def test_load_tabular_data_xlsx_works_with_mixed_numeric_and_text(tmp_path) -> None:
    xlsx_path = tmp_path / "mixed_signals.xlsx"
    raw = pd.DataFrame(
        [
            ["time", "sensor_a", "mode"],
            [0, 1.5, "idle"],
            [1, 2.0, "load"],
            [2, 3.1, "idle"],
        ]
    )
    raw.to_excel(xlsx_path, header=False, index=False)

    loaded = load_tabular_data(str(xlsx_path))
    frame = loaded.frame
    assert frame.shape == (3, 3)
    assert frame["time"].dtype.kind in {"i", "u", "f"}
    assert frame["sensor_a"].dtype.kind in {"i", "u", "f"}
    assert frame["mode"].dtype.kind in {"O", "U", "S"}
    assert frame["mode"].iloc[0] == "idle"


def test_infer_header_prefers_dense_signal_row_over_sparse_description_row() -> None:
    raw = pd.DataFrame(
        [
            ["time", "sig_a", "time", "sig_b", "time", "sig_c"],
            ["ms", "-", "ms", "-", "ms", "-"],
            [None, "Sensor A description", None, "Sensor B description", None, "Sensor C description"],
            [0.0, 1.0, 0.0, 2.0, 0.0, 3.0],
            [1.0, 1.1, 1.0, 2.1, 1.0, 3.1],
        ]
    )
    preview = raw.head(5)
    inferred = _infer_header_and_data_start(preview, raw, confidence_threshold=0.70)
    assert inferred.header_row == 0
