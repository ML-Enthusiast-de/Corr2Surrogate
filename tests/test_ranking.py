from corr2surrogate.analytics.ranking import CandidateSignal, rank_surrogate_candidates


def test_dependency_aware_ranking_penalizes_virtual_dependencies() -> None:
    candidates = [
        CandidateSignal(target_signal="virtual_a", base_score=0.9, required_signals=["sensor_x"]),
        CandidateSignal(
            target_signal="virtual_b",
            base_score=0.9,
            required_signals=["virtual_a"],
        ),
    ]
    ranked = rank_surrogate_candidates(
        candidates=candidates,
        physically_available_signals=["sensor_x"],
        non_virtualizable_signals=["sensor_x"],
    )
    by_target = {item.target_signal: item for item in ranked}
    assert by_target["virtual_b"].adjusted_score < by_target["virtual_a"].adjusted_score
    assert "virtual_a" in by_target["virtual_b"].blocked_virtual_dependencies


def test_dependency_aware_ranking_marks_missing_physical_path_infeasible() -> None:
    candidates = [
        CandidateSignal(
            target_signal="virtual_c",
            base_score=0.8,
            required_signals=["virtual_b"],
        ),
    ]
    ranked = rank_surrogate_candidates(
        candidates=candidates,
        physically_available_signals=[],
        non_virtualizable_signals=[],
        enforce_physical_dependency=True,
    )
    assert not ranked[0].feasible


def test_dependency_aware_ranking_without_constraints_does_not_assume_virtual() -> None:
    candidates = [
        CandidateSignal(
            target_signal="virtual_c",
            base_score=0.8,
            required_signals=["virtual_b"],
        ),
    ]
    ranked = rank_surrogate_candidates(
        candidates=candidates,
        enforce_physical_dependency=True,
    )
    assert ranked[0].feasible
    assert ranked[0].missing_physical_dependencies == []
    assert "feasibility is unverified" in ranked[0].rationale.lower()
