import numpy as np
import pandas as pd

from corr2surrogate.analytics.correlations import (
    build_candidate_signals_from_correlations,
    run_correlation_analysis,
)


def test_run_correlation_analysis_identifies_top_predictor() -> None:
    rng = np.random.default_rng(42)
    n = 150
    x1 = np.linspace(0.0, 10.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    y = 3.0 * x1 + rng.normal(0.0, 0.2, n)
    frame = pd.DataFrame({"time": np.arange(n), "x1": x1, "x2": x2, "y": y})

    bundle = run_correlation_analysis(
        frame,
        target_signals=["y"],
        timestamp_column="time",
        max_lag=4,
    )
    analysis = bundle.target_analyses[0]
    assert analysis.top_predictors
    assert analysis.top_predictors[0] == "x1"

    candidates = build_candidate_signals_from_correlations(bundle)
    assert candidates
    assert candidates[0].target_signal == "y"
    assert candidates[0].base_score > 0.5


def test_feature_engineering_opportunity_detects_square_relation() -> None:
    rng = np.random.default_rng(9)
    x = np.linspace(-3.0, 3.0, 180)
    y = np.square(x) + rng.normal(0.0, 0.05, len(x))
    frame = pd.DataFrame({"x": x, "y": y})

    bundle = run_correlation_analysis(
        frame,
        target_signals=["y"],
        predictor_signals_by_target={"y": ["x"]},
        include_feature_engineering=True,
        feature_gain_threshold=0.05,
    )
    opportunities = bundle.target_analyses[0].feature_opportunities
    assert any(item.transformation == "square" for item in opportunities)
