import numpy as np
import pandas as pd

from corr2surrogate.analytics.correlations import run_correlation_analysis
from corr2surrogate.analytics.experiment_design import recommend_data_trajectories


def test_recommend_data_trajectories_returns_actionable_items() -> None:
    rng = np.random.default_rng(4)
    n = 220
    x = np.linspace(0.0, 4.0, n)
    y = 0.4 * x + rng.normal(0.0, 0.8, n)
    z = rng.normal(0.0, 1.0, n)
    frame = pd.DataFrame({"time": np.arange(n), "x": x, "z": z, "y": y})
    corr = run_correlation_analysis(
        frame,
        target_signals=["y"],
        timestamp_column="time",
        bootstrap_rounds=10,
        confidence_top_k=3,
    )
    recs = recommend_data_trajectories(
        frame=frame,
        correlations=corr.to_dict(),
        sensor_diagnostics={"diagnostics": []},
    )
    assert recs
    assert recs[0].target_signal == "y"
    assert "suggestion" in recs[0].__dict__
