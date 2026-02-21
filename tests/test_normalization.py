import numpy as np
import pandas as pd

from corr2surrogate.modeling.normalization import MinMaxNormalizer


def test_minmax_normalization_and_inverse_target() -> None:
    frame = pd.DataFrame(
        {
            "x1": [0.0, 5.0, 10.0],
            "x2": [10.0, 20.0, 30.0],
            "y": [100.0, 150.0, 200.0],
        }
    )
    norm = MinMaxNormalizer(feature_range=(0.0, 1.0)).fit(
        frame, feature_columns=["x1", "x2"], target_column="y"
    )

    transformed = norm.transform_features(frame)
    assert transformed["x1"].iloc[0] == 0.0
    assert transformed["x1"].iloc[-1] == 1.0

    y_scaled = norm.transform_target(frame["y"])
    y_restored = norm.inverse_transform_target(y_scaled)
    assert np.allclose(y_restored.to_numpy(), frame["y"].to_numpy())
