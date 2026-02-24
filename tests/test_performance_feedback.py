import numpy as np
import pandas as pd

from corr2surrogate.modeling.performance_feedback import analyze_model_performance


def test_analyze_model_performance_returns_regions_and_suggestions() -> None:
    frame = pd.DataFrame(
        {
            "temperature": np.linspace(20, 100, 80),
            "pressure": np.linspace(1, 5, 80),
        }
    )
    y_true = 0.5 * frame["temperature"].to_numpy() + 3.0
    y_pred = y_true.copy()
    y_pred[60:] += 8.0

    feedback = analyze_model_performance(
        y_true=y_true,
        y_pred=y_pred,
        feature_frame=frame,
        top_k_regions=2,
        trajectory_budget=2,
    )
    payload = feedback.to_dict()
    assert "summary" in payload
    assert payload["bad_regions"]
    assert payload["trajectory_suggestions"]
