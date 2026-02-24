from pathlib import Path

import numpy as np
import pandas as pd

from corr2surrogate.modeling.baselines import IncrementalLinearSurrogate
from corr2surrogate.modeling.checkpoints import ModelCheckpointStore


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": [0.0, 1.0, 2.0, 3.0, 4.0],
            "x2": [1.0, 1.0, 1.0, 1.0, 1.0],
            "y": [1.0, 3.0, 5.0, 7.0, 9.0],
        }
    )


def test_incremental_surrogate_fit_save_load_update(tmp_path: Path) -> None:
    train = _sample_frame()
    model = IncrementalLinearSurrogate(feature_columns=["x1", "x2"], target_column="y")
    rows = model.fit_dataframe(train)
    assert rows == len(train)

    pred = model.predict_dataframe(train)
    assert pred.shape[0] == len(train)
    assert np.allclose(pred, train["y"].to_numpy(dtype=float), atol=1e-5)

    state_path = tmp_path / "model_state.json"
    model.save(state_path)
    loaded = IncrementalLinearSurrogate.load(state_path)
    extra = pd.DataFrame({"x1": [5.0, 6.0], "x2": [1.0, 1.0], "y": [11.0, 13.0]})
    added = loaded.update_from_dataframe(extra)
    assert added == 2
    pred_extra = loaded.predict_dataframe(extra)
    assert np.allclose(pred_extra, extra["y"].to_numpy(dtype=float), atol=1e-4)


def test_checkpoint_store_create_load_list(tmp_path: Path) -> None:
    store = ModelCheckpointStore(base_dir=tmp_path / "checkpoints")
    checkpoint = store.create_checkpoint(
        model_name="incremental_linear_surrogate",
        run_dir=tmp_path / "run_1",
        model_state_path=tmp_path / "run_1" / "model_state.json",
        target_column="y",
        feature_columns=["x1", "x2"],
        metrics={"mae": 0.1, "rmse": 0.12, "r2": 0.98, "mape": 2.0, "n_samples": 10},
        data_references=["data/train.csv"],
        notes="initial",
    )
    loaded = store.load_checkpoint(checkpoint.checkpoint_id)
    assert loaded.checkpoint_id == checkpoint.checkpoint_id

    listed = store.list_checkpoints(model_name="incremental_linear_surrogate", limit=10)
    assert listed
    assert listed[0].checkpoint_id == checkpoint.checkpoint_id
