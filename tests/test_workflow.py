from corr2surrogate.orchestration.workflow import prepare_ingestion_step


def test_prepare_ingestion_returns_sheet_options(monkeypatch) -> None:
    from corr2surrogate.ingestion.csv_loader import SheetSelectionRequiredError

    def fake_loader(*args, **kwargs):
        raise SheetSelectionRequiredError(["A", "B"])

    monkeypatch.setattr("corr2surrogate.orchestration.workflow.load_tabular_data", fake_loader)
    result = prepare_ingestion_step(path="dummy.xlsx")
    assert result.status == "needs_user_input"
    assert result.options == ["A", "B"]
