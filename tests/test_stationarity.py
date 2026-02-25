import numpy as np
import pandas as pd

from corr2surrogate.analytics.stationarity import assess_stationarity


def test_assess_stationarity_flags_trend_as_non_stationary() -> None:
    n = 120
    frame = pd.DataFrame(
        {
            "stationary_sig": np.random.default_rng(7).normal(0.0, 1.0, n),
            "trend_sig": np.linspace(0.0, 10.0, n) + np.random.default_rng(11).normal(0.0, 0.3, n),
        }
    )
    summary = assess_stationarity(frame, signal_columns=["stationary_sig", "trend_sig"])
    by_signal = {item.signal: item for item in summary.results}
    assert by_signal["trend_sig"].status == "non_stationary"
    assert by_signal["stationary_sig"].status in {"stationary", "non_stationary"}
