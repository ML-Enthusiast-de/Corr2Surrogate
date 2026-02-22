from corr2surrogate.analytics.ranking import ForcedModelingDirective, RankedSignal
from corr2surrogate.orchestration.workflow import (
    build_modeling_directives,
    evaluate_training_iteration,
    prepare_ingestion_step,
)


def test_prepare_ingestion_returns_sheet_options(monkeypatch) -> None:
    from corr2surrogate.ingestion.csv_loader import SheetSelectionRequiredError

    def fake_loader(*args, **kwargs):
        raise SheetSelectionRequiredError(["A", "B"])

    monkeypatch.setattr("corr2surrogate.orchestration.workflow.load_tabular_data", fake_loader)
    result = prepare_ingestion_step(path="dummy.xlsx")
    assert result.status == "needs_user_input"
    assert result.options == ["A", "B"]


def test_build_modeling_directives_includes_forced_requests() -> None:
    ranked = [
        RankedSignal(
            target_signal="t1",
            base_score=0.9,
            adjusted_score=0.9,
            required_signals=["s1"],
            blocked_virtual_dependencies=[],
            missing_physical_dependencies=[],
            feasible=True,
            rationale="ok",
        ),
    ]
    forced = [
        ForcedModelingDirective(
            target_signal="t_forced",
            predictor_signals=["s2", "s3"],
            user_reason="domain requirement",
        )
    ]
    directives = build_modeling_directives(ranked_signals=ranked, forced_requests=forced)
    assert directives[0].target_signal == "t_forced"
    assert directives[0].force_run_regardless_of_correlation
    assert directives[1].target_signal == "t1"


def test_evaluate_training_iteration_requests_retry_when_unmet() -> None:
    decision = evaluate_training_iteration(
        metrics={"val_mae": 1.2, "train_mae": 0.7, "n_samples": 200},
        acceptance_criteria={"val_mae": 1.0},
        attempt=1,
        max_attempts=3,
    )
    assert decision.should_continue
    assert "val_mae" in decision.unmet_criteria
    assert decision.recommendations
