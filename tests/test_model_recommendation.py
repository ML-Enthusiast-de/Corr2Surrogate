import numpy as np
import pandas as pd

from corr2surrogate.analytics import recommend_model_strategies, run_correlation_analysis


def test_recommend_model_strategies_prefers_linear_for_linear_signal() -> None:
    rng = np.random.default_rng(7)
    n = 240
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    target = 2.0 * x1 - 0.7 * x2 + rng.normal(0.0, 0.05, size=n)
    frame = pd.DataFrame(
        {
            "sensor_a": x1,
            "sensor_b": x2,
            "target": target,
        }
    )
    correlations = run_correlation_analysis(frame=frame, target_signals=["target"], max_lag=4)
    summary = recommend_model_strategies(frame=frame, correlations=correlations, max_lag=4)
    rec = summary.target_recommendations[0]
    assert rec.recommended_model_family == "linear_ridge"
    assert rec.priority_model_families[0] == "linear_ridge"
    assert set(rec.probe_predictor_signals) == {"sensor_a", "sensor_b"}
    assert rec.best_probe_model_family in {"linear_ridge", "interaction_ridge"}
    assert rec.recommendation_confidence in {"medium", "high"}
    assert "For target `target`, start with `linear_ridge`." in rec.recommendation_statement


def test_recommend_model_strategies_flags_tree_candidate_for_interactions() -> None:
    rng = np.random.default_rng(11)
    n = 320
    x1 = rng.uniform(-1.0, 1.0, size=n)
    x2 = rng.uniform(-1.0, 1.0, size=n)
    x3 = rng.normal(0.0, 0.2, size=n)
    target = np.where((x1 > 0.1) & (x2 > 0.1), 1.0, -0.8) + 0.1 * x3
    target = target + rng.normal(0.0, 0.03, size=n)
    frame = pd.DataFrame(
        {
            "sensor_a": x1,
            "sensor_b": x2,
            "sensor_c": x3,
            "target": target,
        }
    )
    correlations = run_correlation_analysis(frame=frame, target_signals=["target"], max_lag=4)
    summary = recommend_model_strategies(frame=frame, correlations=correlations, max_lag=4)
    rec = summary.target_recommendations[0]
    assert rec.tree_model_worth_testing is True
    assert rec.recommended_model_family == "tree_ensemble_candidate"
    assert "tree_ensemble_candidate" in rec.priority_model_families
    assert any(item.model_family == "tiny_tree_probe" for item in rec.candidate_models)
    assert rec.best_probe_model_family in {"tiny_tree_probe", "interaction_ridge"}
    assert "For target `target`, start with `tree_ensemble_candidate`." in rec.recommendation_statement
