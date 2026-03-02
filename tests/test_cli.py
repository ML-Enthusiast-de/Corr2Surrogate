from dataclasses import dataclass
from pathlib import Path
import json

from corr2surrogate.ui.cli import (
    _build_analysis_interpretation_prompt,
    _extract_top3_correlations_global,
    _format_top3_correlations_line,
    _interpretation_mentions_top3,
    _suggest_default_analysis_targets,
    main,
)


@dataclass
class _DummyResult:
    output: dict


class _DummyRegistry:
    def __init__(self) -> None:
        self.last_tool_name = ""
        self.last_args = {}

    def execute(self, tool_name, arguments):
        self.last_tool_name = tool_name
        self.last_args = arguments
        return _DummyResult(output={"status": "ok"})


class _SessionRegistry:
    def __init__(self, scripted_outputs):
        self.calls = []
        self._scripted_outputs = scripted_outputs

    def execute(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        outputs = self._scripted_outputs.get(tool_name, [])
        if outputs:
            return _DummyResult(output=outputs.pop(0))
        return _DummyResult(output={"status": "ok"})


class _RaisingSessionRegistry(_SessionRegistry):
    def __init__(self, scripted_outputs, *, raise_on_tool: str, message: str) -> None:
        super().__init__(scripted_outputs)
        self._raise_on_tool = raise_on_tool
        self._raise_message = message

    def execute(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        if tool_name == self._raise_on_tool:
            raise ValueError(self._raise_message)
        outputs = self._scripted_outputs.get(tool_name, [])
        if outputs:
            return _DummyResult(output=outputs.pop(0))
        return _DummyResult(output={"status": "ok"})


def _stub_llm_interpretation(**_kwargs):
    return {"event": {"message": "LLM interpretation summary."}}


def test_cli_run_agent1_analysis_omits_none_fields(monkeypatch) -> None:
    registry = _DummyRegistry()
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    exit_code = main(
        [
            "run-agent1-analysis",
            "--data-path",
            "dummy.csv",
        ]
    )
    assert exit_code == 0
    assert registry.last_tool_name == "run_agent1_analysis"
    assert "sheet_name" not in registry.last_args
    assert "timestamp_column" not in registry.last_args
    assert "target_signals" not in registry.last_args
    assert "max_samples" not in registry.last_args
    assert "fill_constant_value" not in registry.last_args
    assert "row_range_start" not in registry.last_args
    assert "row_range_end" not in registry.last_args


def test_cli_run_agent1_analysis_prints_strict_json_for_nonfinite_values(
    monkeypatch, capsys
) -> None:
    class _NonFiniteRegistry:
        def execute(self, _tool_name, _arguments):
            return _DummyResult(
                output={
                    "status": "ok",
                    "score_nan": float("nan"),
                    "score_pos_inf": float("inf"),
                    "score_neg_inf": float("-inf"),
                    "nested": {"value": float("nan")},
                }
            )

    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: _NonFiniteRegistry())
    exit_code = main(
        [
            "run-agent1-analysis",
            "--data-path",
            "dummy.csv",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "NaN" not in output
    assert "Infinity" not in output
    payload = json.loads(output)
    assert payload["score_nan"] is None
    assert payload["score_pos_inf"] is None
    assert payload["score_neg_inf"] is None
    assert payload["nested"]["value"] is None


def test_suggest_default_analysis_targets_prefers_label_like_columns() -> None:
    suggested = _suggest_default_analysis_targets(
        available_signals=["amount_norm", "device_risk", "fraud_flag"],
        default_count=5,
    )
    assert suggested == ["fraud_flag"]


def test_cli_setup_local_llm_invokes_setup(monkeypatch) -> None:
    captured = {}

    def fake_setup_local_llm(**kwargs):
        captured.update(kwargs)
        return {"ready": True}

    monkeypatch.setattr("corr2surrogate.ui.cli.setup_local_llm", fake_setup_local_llm)
    exit_code = main(
        [
            "setup-local-llm",
            "--provider",
            "llama_cpp",
            "--no-download-model",
            "--no-start-runtime",
        ]
    )
    assert exit_code == 0
    assert captured["provider"] == "llama_cpp"
    assert captured["download_model"] is False
    assert captured["start_runtime"] is False


def test_cli_setup_local_llm_handles_runtime_error(monkeypatch, capsys) -> None:
    def fake_setup_local_llm(**_kwargs):
        raise RuntimeError("policy-blocked")

    monkeypatch.setattr("corr2surrogate.ui.cli.setup_local_llm", fake_setup_local_llm)
    exit_code = main(
        [
            "setup-local-llm",
            "--provider",
            "openai",
        ]
    )
    assert exit_code == 1
    output = capsys.readouterr().out
    assert "\"ready\": false" in output.lower()
    assert "policy-blocked" in output


def test_cli_run_agent_session_basic_flow(monkeypatch) -> None:
    calls = []
    inputs = iter(["hello there", "/exit"])

    def fake_input(_prompt):
        return next(inputs)

    def fake_run_local_agent_once(*, agent, user_message, context, config_path):
        calls.append(
            {
                "agent": agent,
                "user_message": user_message,
                "context": context,
                "config_path": config_path,
            }
        )
        return {
            "event": {"message": "hi back"},
            "runtime": {"provider": "llama_cpp"},
        }

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", fake_run_local_agent_once)
    exit_code = main(
        [
            "run-agent-session",
            "--agent",
            "analyst",
            "--max-turns",
            "5",
        ]
    )
    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["agent"] == "analyst"
    assert calls[0]["user_message"] == "hello there"
    assert calls[0]["context"]["session_messages"] == []


def test_cli_run_agent_session_persists_messages_between_turns(monkeypatch) -> None:
    calls = []
    inputs = iter(["first question", "second question", "/exit"])

    def fake_input(_prompt):
        return next(inputs)

    def fake_run_local_agent_once(*, agent, user_message, context, config_path):
        calls.append(
            {
                "agent": agent,
                "user_message": user_message,
                "context": context,
                "config_path": config_path,
            }
        )
        return {
            "event": {"message": f"reply to {user_message}"},
            "runtime": {"provider": "llama_cpp"},
        }

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", fake_run_local_agent_once)
    exit_code = main(
        [
            "run-agent-session",
            "--agent",
            "analyst",
        ]
    )
    assert exit_code == 0
    assert len(calls) == 2
    second_context_messages = calls[1]["context"]["session_messages"]
    assert len(second_context_messages) == 2
    assert second_context_messages[0]["role"] == "user"
    assert second_context_messages[0]["content"] == "first question"
    assert second_context_messages[1]["role"] == "assistant"


def test_cli_run_agent_session_context_includes_last_five_user_prompts(monkeypatch) -> None:
    calls = []
    inputs = iter(
        [
            "q1",
            "q2",
            "q3",
            "q4",
            "q5",
            "q6",
            "/exit",
        ]
    )

    def fake_run_local_agent_once(*, agent, user_message, context, config_path):
        calls.append(
            {
                "user_message": user_message,
                "context": context,
            }
        )
        return {"event": {"message": f"reply to {user_message}"}}

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", fake_run_local_agent_once)
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert len(calls) == 6
    sixth_context = calls[5]["context"]
    assert sixth_context["recent_user_prompts"] == ["q1", "q2", "q3", "q4", "q5"]


def test_cli_run_agent_session_analyst_autopilot_runs_analysis(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "demo_data.xlsx"
    data_path.write_text("testdata", encoding="utf-8")
    inputs = iter([f"Analyze {data_path}", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": "Sheet1",
                    "available_sheets": ["Sheet1"],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 0.98,
                    "needs_user_confirmation": False,
                }
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "time_series",
                    "target_count": 8,
                    "candidate_count": 4,
                    "report_path": "reports/demo_data/agent1_20260225_120000.md",
                    "correlations": {
                        "target_analyses": [
                            {
                                "target_signal": "t1",
                                "predictor_results": [
                                    {
                                        "predictor_signal": "p1",
                                        "best_method": "pearson",
                                        "best_abs_score": 0.98,
                                        "sample_count": 100,
                                    },
                                    {
                                        "predictor_signal": "p2",
                                        "best_method": "spearman",
                                        "best_abs_score": 0.95,
                                        "sample_count": 100,
                                    },
                                    {
                                        "predictor_signal": "p3",
                                        "best_method": "distance_corr",
                                        "best_abs_score": 0.90,
                                        "sample_count": 100,
                                    },
                                ],
                            }
                        ]
                    },
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[0][0] == "prepare_ingestion_step"
    assert registry.calls[1][0] == "run_agent1_analysis"
    assert Path(registry.calls[1][1]["data_path"]).resolve() == data_path.resolve()
    output = capsys.readouterr().out
    assert "LLM interpretation:" in output
    assert "LLM interpretation summary." in output
    assert "Top 3 correlated predictors:" in output


def test_cli_run_agent_session_analyst_can_continue_directly_into_modeler(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "analyst_to_modeler.csv"
    data_path.write_text("A,B,C\n1,2,3\n", encoding="utf-8")
    handoff_path = tmp_path / "structured_report.json"
    handoff_path.write_text(
        json.dumps(
            {
                "data_path": str(data_path),
                "data_mode": "steady_state",
                "timestamp_column": None,
                "preprocessing": {
                    "missing_data_plan": {
                        "strategy": "fill_median",
                    }
                },
                "model_strategy_recommendations": {
                    "target_recommendations": [
                        {
                            "target_signal": "C",
                            "probe_predictor_signals": ["A", "B"],
                            "recommended_model_family": "linear_ridge",
                            "priority_model_families": ["linear_ridge", "bagged_tree_ensemble"],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    inputs = iter(
        [
            f"Analyze {data_path}",
            "y",
            "y",
            "",
            "",
            "",
            "/exit",
        ]
    )
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": None,
                    "available_sheets": [],
                    "row_count": 100,
                    "column_count": 3,
                    "signal_columns": ["A", "B", "C"],
                    "numeric_signal_columns": ["A", "B", "C"],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 0.99,
                    "needs_user_confirmation": False,
                },
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": None,
                    "available_sheets": [],
                    "row_count": 100,
                    "column_count": 3,
                    "signal_columns": ["A", "B", "C"],
                    "numeric_signal_columns": ["A", "B", "C"],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 0.99,
                    "needs_user_confirmation": False,
                },
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "steady_state",
                    "target_count": 1,
                    "candidate_count": 1,
                    "report_path": "reports/analyst_to_modeler/agent1_20260301_120000.md",
                    "artifact_paths": {"json_path": str(handoff_path)},
                }
            ],
            "train_surrogate_candidates": [
                {
                    "status": "ok",
                    "checkpoint_id": "ckpt_bridge_1",
                    "run_dir": "artifacts/run_bridge_1",
                    "selected_model_family": "linear_ridge",
                    "best_validation_model_family": "linear_ridge",
                    "lag_horizon_samples": 0,
                    "split": {
                        "strategy": "deterministic_modulo_70_15_15",
                        "train_size": 70,
                        "validation_size": 15,
                        "test_size": 15,
                    },
                    "preprocessing": {
                        "missing_data_strategy_requested": "fill_median",
                        "missing_data_strategy_effective": "fill_median_train_only",
                    },
                    "normalization": {"method": "minmax"},
                    "comparison": [
                        {
                            "model_family": "linear_ridge",
                            "validation_metrics": {"r2": 0.88, "mae": 0.12},
                            "test_metrics": {"r2": 0.86, "mae": 0.13},
                        }
                    ],
                    "selected_metrics": {
                        "train": {"mae": 0.10, "r2": 0.90},
                        "validation": {"mae": 0.12, "r2": 0.88},
                        "test": {"mae": 0.13, "r2": 0.86},
                    },
                    "rows_used": 70,
                }
            ],
            "evaluate_training_iteration": [
                {
                    "should_continue": False,
                    "attempt": 1,
                    "max_attempts": 3,
                    "unmet_criteria": [],
                    "recommendations": ["Quality criteria met. Proceed to artifact export."],
                    "summary": "Model meets all acceptance criteria.",
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert [call[0] for call in registry.calls] == [
        "prepare_ingestion_step",
        "run_agent1_analysis",
        "prepare_ingestion_step",
        "train_surrogate_candidates",
        "evaluate_training_iteration",
    ]
    output = capsys.readouterr().out
    assert "Structured handoff saved:" in output
    assert "Start Agent 2 modeling now from this handoff?" in output
    assert "Starting Agent 2 handoff-driven modeling." in output
    assert "Handoff contract:" in output
    assert "Model build complete:" in output


def test_cli_run_agent_session_analyst_autopilot_multi_sheet(monkeypatch, tmp_path: Path) -> None:
    data_path = tmp_path / "multi_sheet.xlsx"
    data_path.write_text("testdata", encoding="utf-8")
    inputs = iter([f"Analyze {data_path}", "2", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "needs_user_input",
                    "message": "Excel file has multiple sheets. Please choose one.",
                    "options": ["S1", "S2"],
                    "available_sheets": ["S1", "S2"],
                    "selected_sheet": None,
                    "header_row": None,
                    "data_start_row": None,
                    "header_confidence": None,
                    "needs_user_confirmation": False,
                },
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": "S2",
                    "available_sheets": ["S1", "S2"],
                    "header_row": 1,
                    "data_start_row": 2,
                    "header_confidence": 0.91,
                    "needs_user_confirmation": False,
                },
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "steady_state",
                    "target_count": 3,
                    "candidate_count": 1,
                    "report_path": "reports/multi_sheet/agent1_20260225_120000.md",
                }
            ],
        }
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "prepare_ingestion_step"
    assert registry.calls[1][1]["sheet_name"] == "S2"
    assert registry.calls[2][0] == "run_agent1_analysis"
    assert registry.calls[2][1]["sheet_name"] == "S2"


def test_cli_run_agent_session_autopilot_interpretation_runtime_error_is_suppressed(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "demo_data_runtime.xlsx"
    data_path.write_text("testdata", encoding="utf-8")
    inputs = iter([f"Analyze {data_path}", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": "Sheet1",
                    "available_sheets": ["Sheet1"],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 0.98,
                    "needs_user_confirmation": False,
                }
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "time_series",
                    "target_count": 1,
                    "candidate_count": 1,
                    "report_path": "reports/demo_data_runtime/agent1_20260225_120000.md",
                    "correlations": {
                        "target_analyses": [
                            {
                                "target_signal": "t1",
                                "predictor_results": [
                                    {
                                        "predictor_signal": "p1",
                                        "best_method": "pearson",
                                        "best_abs_score": 0.98,
                                        "sample_count": 100,
                                    },
                                    {
                                        "predictor_signal": "p2",
                                        "best_method": "spearman",
                                        "best_abs_score": 0.95,
                                        "sample_count": 100,
                                    },
                                    {
                                        "predictor_signal": "p3",
                                        "best_method": "distance_corr",
                                        "best_abs_score": 0.90,
                                        "sample_count": 100,
                                    },
                                ],
                            }
                        ]
                    },
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr(
        "corr2surrogate.ui.cli.run_local_agent_once",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "LLM interpretation unavailable for this turn." in output
    assert "Top 3 correlated predictors:" in output
    assert "I hit an internal runtime error in this step." not in output


def test_cli_run_agent_session_autopilot_interpretation_retries_with_compact_prompt(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "demo_data_interpret_retry.xlsx"
    data_path.write_text("testdata", encoding="utf-8")
    inputs = iter([f"Analyze {data_path}", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": "Sheet1",
                    "available_sheets": ["Sheet1"],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 0.98,
                    "needs_user_confirmation": False,
                }
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "time_series",
                    "target_count": 1,
                    "candidate_count": 1,
                    "report_path": "reports/demo_data_interpret_retry/agent1_20260225_120000.md",
                    "quality": {
                        "rows": 100,
                        "columns": 4,
                        "completeness_score": 0.99,
                        "warnings": [],
                    },
                    "correlations": {
                        "target_analyses": [
                            {
                                "target_signal": "t1",
                                "predictor_results": [
                                    {
                                        "predictor_signal": "p1",
                                        "best_method": "pearson",
                                        "best_abs_score": 0.98,
                                        "sample_count": 100,
                                    },
                                    {
                                        "predictor_signal": "p2",
                                        "best_method": "spearman",
                                        "best_abs_score": 0.95,
                                        "sample_count": 100,
                                    },
                                    {
                                        "predictor_signal": "p3",
                                        "best_method": "distance_corr",
                                        "best_abs_score": 0.90,
                                        "sample_count": 100,
                                    },
                                ],
                            }
                        ]
                    },
                }
            ],
        }
    )
    calls = {"n": 0}
    failure_text = (
        "I hit an internal runtime error in this step. "
        "The session is still active; you can retry, change inputs, or use /reset."
    )

    def fake_run_local_agent_once(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"event": {"message": failure_text}}
        return {"event": {"message": "Recovered interpretation on retry."}}

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", fake_run_local_agent_once)

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert calls["n"] >= 2
    assert "LLM interpretation:" in output
    assert "Recovered interpretation on retry." in output
    assert "LLM interpretation unavailable for this turn." not in output


def test_analysis_interpretation_prompt_contains_global_top3_correlations() -> None:
    analysis = {
        "data_mode": "time_series",
        "target_count": 1,
        "candidate_count": 1,
        "quality": {"rows": 100, "columns": 4, "completeness_score": 1.0, "warnings": []},
        "ranking": [],
        "correlations": {
            "target_analyses": [
                {
                    "target_signal": "target",
                    "predictor_results": [
                        {"predictor_signal": "sig_a", "best_method": "pearson", "best_abs_score": 0.91},
                        {"predictor_signal": "sig_b", "best_method": "spearman", "best_abs_score": 0.88},
                        {"predictor_signal": "sig_c", "best_method": "distance_corr", "best_abs_score": 0.85},
                    ],
                }
            ]
        },
    }
    top3 = _extract_top3_correlations_global(analysis)
    assert [item["predictor_signal"] for item in top3] == ["sig_a", "sig_b", "sig_c"]
    prompt = _build_analysis_interpretation_prompt(analysis)
    assert "top_3_correlated_predictors" in prompt
    assert "Top 3 correlated predictors:" in prompt
    assert "sig_a" in prompt
    assert "sig_b" in prompt
    assert "sig_c" in prompt
    line = _format_top3_correlations_line(top3)
    assert "sig_a->target" in line
    assert _interpretation_mentions_top3(
        interpretation="Top 3 correlated predictors: sig_a, sig_b, sig_c",
        top3=top3,
    )


def test_cli_run_agent_session_analyst_autopilot_multi_sheet_handles_small_talk(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "multi_sheet_chat.xlsx"
    data_path.write_text("testdata", encoding="utf-8")
    inputs = iter([f"Analyze {data_path}", "how are you?", "2", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "needs_user_input",
                    "message": "Excel file has multiple sheets. Please choose one.",
                    "options": ["S1", "S2"],
                    "available_sheets": ["S1", "S2"],
                    "selected_sheet": None,
                    "header_row": None,
                    "data_start_row": None,
                    "header_confidence": None,
                    "needs_user_confirmation": False,
                },
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": "S2",
                    "available_sheets": ["S1", "S2"],
                    "header_row": 1,
                    "data_start_row": 2,
                    "header_confidence": 0.91,
                    "needs_user_confirmation": False,
                },
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "steady_state",
                    "target_count": 3,
                    "candidate_count": 1,
                    "report_path": "reports/multi_sheet_chat/agent1_20260225_120000.md",
                }
            ],
        }
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr(
        "corr2surrogate.ui.cli.run_local_agent_once",
        lambda **kwargs: {"event": {"message": "LLM detour reply (sheet)."}},
    )

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "prepare_ingestion_step"
    assert registry.calls[1][1]["sheet_name"] == "S2"
    output = capsys.readouterr().out
    assert "LLM detour reply (sheet)." in output
    assert "To continue, please enter a sheet number/name" in output


def test_cli_run_agent_session_analyst_autopilot_header_override(monkeypatch, tmp_path: Path) -> None:
    data_path = tmp_path / "low_conf.xlsx"
    data_path.write_text("testdata", encoding="utf-8")
    inputs = iter([f"Analyze {data_path}", "n", "3,4", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Low confidence header inference.",
                    "options": [],
                    "selected_sheet": "S1",
                    "available_sheets": ["S1"],
                    "header_row": 1,
                    "data_start_row": 2,
                    "header_confidence": 0.52,
                    "needs_user_confirmation": True,
                },
                {
                    "status": "ok",
                    "message": "Header override accepted.",
                    "options": [],
                    "selected_sheet": "S1",
                    "available_sheets": ["S1"],
                    "header_row": 3,
                    "data_start_row": 4,
                    "header_confidence": 1.0,
                    "needs_user_confirmation": False,
                },
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "time_series",
                    "target_count": 2,
                    "candidate_count": 2,
                    "report_path": "reports/low_conf/agent1_20260225_120000.md",
                }
            ],
        }
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "prepare_ingestion_step"
    assert registry.calls[1][1]["header_row"] == 3
    assert registry.calls[1][1]["data_start_row"] == 4
    assert registry.calls[2][0] == "run_agent1_analysis"
    assert registry.calls[2][1]["header_row"] == 3
    assert registry.calls[2][1]["data_start_row"] == 4


def test_cli_run_agent_session_analyst_autopilot_header_confirmation_handles_small_talk(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "header_chat.xlsx"
    data_path.write_text("testdata", encoding="utf-8")
    inputs = iter([f"Analyze {data_path}", "hi", "Y", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Low confidence header inference.",
                    "options": [],
                    "selected_sheet": "S1",
                    "available_sheets": ["S1"],
                    "header_row": 1,
                    "data_start_row": 2,
                    "header_confidence": 0.52,
                    "needs_user_confirmation": True,
                },
                {
                    "status": "ok",
                    "message": "Header confirmed.",
                    "options": [],
                    "selected_sheet": "S1",
                    "available_sheets": ["S1"],
                    "header_row": 1,
                    "data_start_row": 2,
                    "header_confidence": 1.0,
                    "needs_user_confirmation": False,
                },
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "time_series",
                    "target_count": 2,
                    "candidate_count": 2,
                    "report_path": "reports/header_chat/agent1_20260225_120000.md",
                }
            ],
        }
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr(
        "corr2surrogate.ui.cli.run_local_agent_once",
        lambda **kwargs: {"event": {"message": "LLM detour reply (header confirm)."}},
    )

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "prepare_ingestion_step"
    assert registry.calls[1][1]["header_row"] == 1
    assert registry.calls[1][1]["data_start_row"] == 2
    output = capsys.readouterr().out
    assert "LLM detour reply (header confirm)." in output
    assert "To continue, reply with Y/Enter to keep inferred rows" in output


def test_cli_run_agent_session_analyst_autopilot_header_override_handles_small_talk(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "header_override_chat.xlsx"
    data_path.write_text("testdata", encoding="utf-8")
    inputs = iter([f"Analyze {data_path}", "n", "how are you?", "3,4", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Low confidence header inference.",
                    "options": [],
                    "selected_sheet": "S1",
                    "available_sheets": ["S1"],
                    "header_row": 1,
                    "data_start_row": 2,
                    "header_confidence": 0.52,
                    "needs_user_confirmation": True,
                },
                {
                    "status": "ok",
                    "message": "Header override accepted.",
                    "options": [],
                    "selected_sheet": "S1",
                    "available_sheets": ["S1"],
                    "header_row": 3,
                    "data_start_row": 4,
                    "header_confidence": 1.0,
                    "needs_user_confirmation": False,
                },
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "time_series",
                    "target_count": 2,
                    "candidate_count": 2,
                    "report_path": "reports/header_override_chat/agent1_20260225_120000.md",
                }
            ],
        }
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr(
        "corr2surrogate.ui.cli.run_local_agent_once",
        lambda **kwargs: {"event": {"message": "LLM detour reply (header override)."}},
    )

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "prepare_ingestion_step"
    assert registry.calls[1][1]["header_row"] == 3
    assert registry.calls[1][1]["data_start_row"] == 4
    output = capsys.readouterr().out
    assert "LLM detour reply (header override)." in output
    assert "To continue, please enter 'header_row,data_start_row'" in output


def test_cli_run_agent_session_analyst_target_prompt_accepts_list_command(
    monkeypatch, tmp_path: Path
) -> None:
    data_path = tmp_path / "wide.xlsx"
    data_path.write_text("testdata", encoding="utf-8")
    numeric_signals = [f"sig_{idx}" for idx in range(1, 46)]
    inputs = iter([f"Analyze {data_path}", "list the signal names", "sig_20", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": "S1",
                    "available_sheets": ["S1"],
                    "row_count": 100,
                    "column_count": 60,
                    "signal_columns": numeric_signals,
                    "numeric_signal_columns": numeric_signals,
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 0.95,
                    "needs_user_confirmation": False,
                }
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "steady_state",
                    "target_count": 1,
                    "candidate_count": 1,
                    "report_path": "reports/wide/agent1_20260225_120000.md",
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "run_agent1_analysis"
    assert registry.calls[1][1]["target_signals"] == ["sig_20"]


def test_cli_run_agent_session_analyst_target_prompt_accepts_hypothesis_command(
    monkeypatch, tmp_path: Path
) -> None:
    data_path = tmp_path / "wide_hypothesis.xlsx"
    data_path.write_text("testdata", encoding="utf-8")
    numeric_signals = [f"sig_{idx}" for idx in range(1, 46)]
    inputs = iter(
        [
            f"Analyze {data_path}",
            "hypothesis corr sig_20:sig_3,sig_4; feature sig_3->rate_change",
            "sig_20",
            "/exit",
        ]
    )
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": "S1",
                    "available_sheets": ["S1"],
                    "row_count": 100,
                    "column_count": 60,
                    "signal_columns": numeric_signals,
                    "numeric_signal_columns": numeric_signals,
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 0.95,
                    "needs_user_confirmation": False,
                }
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "steady_state",
                    "target_count": 1,
                    "candidate_count": 1,
                    "report_path": "reports/wide_hypothesis/agent1_20260225_120000.md",
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    args = registry.calls[1][1]
    assert args["target_signals"] == ["sig_20"]
    assert args["user_hypotheses"]
    assert args["feature_hypotheses"]
    assert args["feature_hypotheses"][0]["transformation"] == "rate_change"


def test_cli_run_agent_session_analyst_target_prompt_handles_small_talk(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "wide_chat.xlsx"
    data_path.write_text("testdata", encoding="utf-8")
    numeric_signals = [f"sig_{idx}" for idx in range(1, 46)]
    inputs = iter([f"Analyze {data_path}", "how are you?", "sig_2", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": "S1",
                    "available_sheets": ["S1"],
                    "row_count": 100,
                    "column_count": 60,
                    "signal_columns": numeric_signals,
                    "numeric_signal_columns": numeric_signals,
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 0.95,
                    "needs_user_confirmation": False,
                }
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "steady_state",
                    "target_count": 1,
                    "candidate_count": 1,
                    "report_path": "reports/wide_chat/agent1_20260225_120000.md",
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr(
        "corr2surrogate.ui.cli.run_local_agent_once",
        lambda **kwargs: {"event": {"message": "LLM detour reply (target)."}},
    )
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "run_agent1_analysis"
    assert registry.calls[1][1]["target_signals"] == ["sig_2"]
    output = capsys.readouterr().out
    assert "LLM detour reply (target)." in output
    assert "To continue, type 'list' to show names" in output


def test_cli_run_agent_session_rewrites_greeting_tool_error_fallback(monkeypatch, capsys) -> None:
    inputs = iter(["hi", "/exit"])
    calls = {"count": 0}

    def fake_input(_prompt):
        return next(inputs)

    def fake_run_local_agent_once(*, agent, user_message, context, config_path):
        calls["count"] += 1
        if context.get("chat_only"):
            return {"event": {"message": "LLM detour hello"}}
        return {
            "event": {
                "message": (
                    "I am stopping after repeated tool argument errors. "
                    "Please provide a clearer request."
                )
            }
        }

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", fake_run_local_agent_once)
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert calls["count"] >= 2
    output = capsys.readouterr().out
    assert "LLM detour hello" in output


def test_cli_run_agent_session_meaning_of_life_uses_llm_detour_on_fallback(
    monkeypatch, capsys
) -> None:
    inputs = iter(["what is the meaning of life", "/exit"])
    calls = {"chat_only": 0, "normal": 0}

    def fake_run_local_agent_once(*, agent, user_message, context, config_path):
        if context.get("chat_only"):
            calls["chat_only"] += 1
            return {"event": {"message": "Meaning is what you create through choices and responsibility."}}
        calls["normal"] += 1
        return {
            "event": {
                "message": (
                    "I am stopping after repeated tool argument errors. "
                    "Please provide a clearer request."
                )
            }
        }

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", fake_run_local_agent_once)
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert calls["normal"] == 1
    assert calls["chat_only"] >= 1
    output = capsys.readouterr().out
    assert "Meaning is what you create through choices and responsibility." in output


def test_cli_run_agent_session_greeting_uses_llm(monkeypatch, capsys) -> None:
    inputs = iter(["hello", "/exit"])
    calls = []
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    def fake_run_local_agent_once(*, agent, user_message, context, config_path):
        calls.append((agent, user_message))
        return {"event": {"message": "LLM says hello"}}

    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", fake_run_local_agent_once)
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert calls and calls[0][1] == "hello"
    output = capsys.readouterr().out
    assert "LLM says hello" in output


def test_cli_run_agent_session_how_are_you_uses_llm(monkeypatch, capsys) -> None:
    inputs = iter(["how are you", "/exit"])
    calls = []
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    def fake_run_local_agent_once(*, agent, user_message, context, config_path):
        calls.append((agent, user_message))
        return {"event": {"message": "LLM says I am fine."}}

    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", fake_run_local_agent_once)
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert calls and calls[0][1] == "how are you"
    output = capsys.readouterr().out
    assert "LLM says I am fine." in output


def test_cli_run_agent_session_awaiting_dataset_stage_reprompts_after_chat(
    monkeypatch, capsys
) -> None:
    inputs = iter(["how are you", "do you want a path?", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    def fake_run_local_agent_once(*, agent, user_message, context, config_path):
        if user_message.lower().startswith("how are you"):
            return {"event": {"message": "Hello! How can I assist you today?"}}
        return {
            "event": {
                "message": "Could you clarify what you mean by path?"
            }
        }

    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", fake_run_local_agent_once)
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Hello! How can I assist you today?" in output
    assert "Could you clarify what you mean by path?" in output
    assert "To continue, paste a CSV/XLSX path or type `default`" in output


def test_cli_run_agent_session_provider_connection_error_is_friendly(monkeypatch, capsys) -> None:
    inputs = iter(["what happened", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

    calls = {"count": 0}

    def _raise_provider_error(**_kwargs):
        calls["count"] += 1
        raise RuntimeError(
            "Provider connection error at http://127.0.0.1:8000/v1/chat/completions: "
            "<urlopen error [WinError 10061] connection refused>"
        )

    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _raise_provider_error)
    monkeypatch.setattr("corr2surrogate.ui.cli.setup_local_llm", lambda **_: {"ready": False})
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert calls["count"] == 2
    output = capsys.readouterr().out
    assert "Local LLM runtime is not reachable" in output
    assert "setup-local-llm" in output
    assert "WinError 10061" not in output


def test_cli_run_agent_session_analyst_autopilot_handles_internal_error(monkeypatch, capsys) -> None:
    inputs = iter(["hello", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: _SessionRegistry({}))
    monkeypatch.setattr(
        "corr2surrogate.ui.cli._run_analyst_autopilot_turn",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom with internal details")),
    )
    monkeypatch.setattr(
        "corr2surrogate.ui.cli.run_local_agent_once",
        lambda **_: (_ for _ in ()).throw(AssertionError("LLM path should not run for this test")),
    )

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "The session is still active" in output
    assert "Traceback" not in output


def test_cli_run_agent_session_analyst_autopilot_drops_none_optional_args(
    monkeypatch, tmp_path: Path
) -> None:
    data_path = tmp_path / "autopilot_drop_none.csv"
    data_path.write_text("testdata", encoding="utf-8")
    signals = ["time"] + [f"S{i}" for i in range(1, 72)]
    inputs = iter([f"Analyze {data_path}", "66", "y", "n", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": None,
                    "available_sheets": [],
                    "row_count": 7860,
                    "column_count": 72,
                    "signal_columns": signals,
                    "numeric_signal_columns": signals,
                    "timestamp_column_hint": "time",
                    "estimated_sample_period_seconds": 0.1,
                    "missing_overall_fraction": 0.0,
                    "columns_with_missing_count": 0,
                    "columns_with_missing": [],
                    "row_non_null_fraction_min": 1.0,
                    "row_non_null_fraction_median": 1.0,
                    "row_non_null_fraction_max": 1.0,
                    "potential_length_mismatch": False,
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 1.0,
                    "needs_user_confirmation": False,
                }
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "time_series",
                    "target_count": 1,
                    "candidate_count": 1,
                    "report_path": "reports/autopilot_drop_none/agent1_20260226_120000.md",
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "run_agent1_analysis"
    args = registry.calls[1][1]
    assert args["timestamp_column"] == "time"
    assert args["max_lag"] == 0
    assert "max_samples" not in args
    assert "fill_constant_value" not in args
    assert "row_range_start" not in args
    assert "row_range_end" not in args


def test_cli_run_agent_session_analyst_lag_prompt_sets_max_lag(monkeypatch, tmp_path: Path) -> None:
    data_path = tmp_path / "lag_flow.xlsx"
    data_path.write_text("testdata", encoding="utf-8")
    inputs = iter([f"Analyze {data_path}", "y", "y", "samples", "12", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": "S1",
                    "available_sheets": ["S1"],
                    "row_count": 100,
                    "column_count": 10,
                    "signal_columns": ["recorder_time", "sig_a", "sig_b"],
                    "numeric_signal_columns": ["sig_a", "sig_b"],
                    "timestamp_column_hint": "recorder_time",
                    "estimated_sample_period_seconds": 0.2,
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 0.95,
                    "needs_user_confirmation": False,
                }
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "time_series",
                    "target_count": 2,
                    "candidate_count": 1,
                    "report_path": "reports/lag_flow/agent1_20260226_120000.md",
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "run_agent1_analysis"
    assert registry.calls[1][1]["timestamp_column"] == "recorder_time"
    assert registry.calls[1][1]["max_lag"] == 12


def test_cli_run_agent_session_analyst_prompts_missing_and_length_handling(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "nan_len_flow.xlsx"
    data_path.write_text("testdata", encoding="utf-8")
    inputs = iter([f"Analyze {data_path}", "y", "fill_median", "trim_dense_window", "0.85", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": "S1",
                    "available_sheets": ["S1"],
                    "row_count": 120,
                    "column_count": 8,
                    "signal_columns": ["a", "b", "c"],
                    "numeric_signal_columns": ["a", "b", "c"],
                    "missing_overall_fraction": 0.21,
                    "columns_with_missing_count": 3,
                    "columns_with_missing": ["a", "b", "c"],
                    "row_non_null_fraction_min": 0.45,
                    "row_non_null_fraction_median": 0.88,
                    "row_non_null_fraction_max": 1.0,
                    "potential_length_mismatch": True,
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 0.95,
                    "needs_user_confirmation": False,
                }
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "steady_state",
                    "target_count": 2,
                    "candidate_count": 1,
                    "report_path": "reports/nan_len_flow/agent1_20260226_120000.md",
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "run_agent1_analysis"
    args = registry.calls[1][1]
    assert args["missing_data_strategy"] == "fill_median"
    assert args["row_coverage_strategy"] == "trim_dense_window"
    assert float(args["sparse_row_min_fraction"]) == 0.85
    output = capsys.readouterr().out
    assert "Leakage note" in output
    assert "split first" in output.lower()


def test_cli_run_agent_session_analyst_default_dataset_runs_autopilot(
    monkeypatch, tmp_path: Path
) -> None:
    default_path = tmp_path / "public_testbench_dataset_20k_minmax.csv"
    default_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    inputs = iter(["default", "y", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": None,
                    "available_sheets": [],
                    "row_count": 2,
                    "column_count": 2,
                    "signal_columns": ["a", "b"],
                    "numeric_signal_columns": ["a", "b"],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 0.95,
                    "needs_user_confirmation": False,
                }
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "steady_state",
                    "target_count": 2,
                    "candidate_count": 1,
                    "report_path": "reports/public_testbench_dataset_20k_minmax/agent1_20260226_120000.md",
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr(
        "corr2surrogate.ui.cli._resolve_default_public_dataset_path",
        lambda: default_path,
    )
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[0][0] == "prepare_ingestion_step"
    assert registry.calls[0][1]["path"] == str(default_path.resolve())
    assert registry.calls[1][0] == "run_agent1_analysis"
    assert registry.calls[1][1]["data_path"] == str(default_path.resolve())


def test_cli_run_agent_session_analyst_default_dataset_missing_is_reported(
    monkeypatch, capsys
) -> None:
    inputs = iter(["default", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(
        "corr2surrogate.ui.cli._resolve_default_public_dataset_path",
        lambda: None,
    )
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Default dataset is not available" in output


def test_cli_run_agent_session_prints_welcome_message(monkeypatch, capsys) -> None:
    inputs = iter(["/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Welcome to Corr2Surrogate" in output
    assert "Useful commands" in output
    assert "Dataset choice" in output


def test_cli_run_agent_session_prints_header_preview_on_confirmation(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "header_preview.xlsx"
    data_path.write_text("testdata", encoding="utf-8")
    inputs = iter([f"Analyze {data_path}", "Y", "y", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": "S1",
                    "available_sheets": ["S1"],
                    "row_count": 10,
                    "column_count": 55,
                    "signal_columns": ["sig_a", "sig_b", "sig_c"],
                    "numeric_signal_columns": ["sig_a", "sig_b"],
                    "header_row": 2,
                    "data_start_row": 3,
                    "header_confidence": 0.95,
                    "candidate_header_rows": [2, 1, 0],
                    "needs_user_confirmation": False,
                },
                {
                    "status": "ok",
                    "message": "Ingestion confirmed.",
                    "options": [],
                    "selected_sheet": "S1",
                    "available_sheets": ["S1"],
                    "row_count": 10,
                    "column_count": 55,
                    "signal_columns": ["sig_a", "sig_b", "sig_c"],
                    "numeric_signal_columns": ["sig_a", "sig_b"],
                    "header_row": 2,
                    "data_start_row": 3,
                    "header_confidence": 1.0,
                    "candidate_header_rows": [2, 1, 0],
                    "needs_user_confirmation": False,
                },
            ],
            "run_agent1_analysis": [
                {
                    "status": "ok",
                    "data_mode": "steady_state",
                    "target_count": 1,
                    "candidate_count": 1,
                    "report_path": "reports/header_preview/agent1_20260225_120000.md",
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Inferred header preview" in output
    assert "sig_a" in output


def test_cli_run_agent_session_modeler_direct_request_then_dataset(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "model_data.csv"
    data_path.write_text("A,B,C\n1,2,3\n", encoding="utf-8")
    inputs = iter(
        [
            "build model linear_ridge with inputs A,B and target C",
            str(data_path),
            "/exit",
        ]
    )
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": None,
                    "available_sheets": [],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 1.0,
                    "needs_user_confirmation": False,
                    "row_count": 10,
                    "column_count": 3,
                    "signal_columns": ["A", "B", "C"],
                    "numeric_signal_columns": ["A", "B", "C"],
                }
            ],
            "train_surrogate_candidates": [
                {
                    "status": "ok",
                    "checkpoint_id": "ckpt_linear_1",
                    "run_dir": "artifacts/run_linear_1",
                    "selected_model_family": "linear_ridge",
                    "best_validation_model_family": "bagged_tree_ensemble",
                    "split": {
                        "strategy": "blocked_time_order_70_15_15",
                        "train_size": 7,
                        "validation_size": 2,
                        "test_size": 2,
                    },
                    "preprocessing": {
                        "missing_data_strategy_requested": "fill_median",
                        "missing_data_strategy_effective": "fill_median_train_only",
                    },
                    "normalization": {"method": "minmax"},
                    "comparison": [
                        {
                            "model_family": "linear_ridge",
                            "validation_metrics": {"r2": 0.90, "mae": 0.14},
                            "test_metrics": {"r2": 0.91, "mae": 0.12},
                        },
                        {
                            "model_family": "bagged_tree_ensemble",
                            "validation_metrics": {"r2": 0.92, "mae": 0.11},
                            "test_metrics": {"r2": 0.89, "mae": 0.13},
                        },
                    ],
                    "selected_metrics": {
                        "test": {"r2": 0.91, "mae": 0.12},
                    },
                    "rows_used": 10,
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "modeler"])
    assert exit_code == 0
    assert registry.calls[0][0] == "prepare_ingestion_step"
    assert registry.calls[1][0] == "train_surrogate_candidates"
    assert registry.calls[1][1]["target_column"] == "C"
    assert registry.calls[1][1]["feature_columns"] == ["A", "B"]
    assert registry.calls[1][1]["requested_model_family"] == "linear_ridge"
    assert Path(registry.calls[1][1]["data_path"]).resolve() == data_path.resolve()
    output = capsys.readouterr().out
    assert "I parsed your model request." in output
    assert "Continuing with your pending model request." in output
    assert "Split-safe pipeline:" in output
    assert "Candidate `linear_ridge`" in output
    assert "Model build complete:" in output


def test_cli_run_agent_session_modeler_handoff_allows_override_prompts(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "model_data.csv"
    data_path.write_text("A,B,C\n1,2,3\n", encoding="utf-8")
    handoff_path = tmp_path / "structured_report.json"
    handoff_path.write_text(
        json.dumps(
            {
                "data_path": str(data_path),
                "data_mode": "steady_state",
                "timestamp_column": None,
                "preprocessing": {
                    "missing_data_plan": {
                        "strategy": "fill_median",
                    }
                },
                "model_strategy_recommendations": {
                    "target_recommendations": [
                        {
                            "target_signal": "C",
                            "probe_predictor_signals": ["A", "B"],
                            "recommended_model_family": "tree_ensemble_candidate",
                            "priority_model_families": [
                                "linear_ridge",
                                "tree_ensemble_candidate",
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    inputs = iter(
        [
            f"use handoff {handoff_path}",
            "",
            "",
            "",
            "/exit",
        ]
    )
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": None,
                    "available_sheets": [],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 1.0,
                    "needs_user_confirmation": False,
                    "row_count": 10,
                    "column_count": 3,
                    "signal_columns": ["A", "B", "C"],
                    "numeric_signal_columns": ["A", "B", "C"],
                }
            ],
            "train_surrogate_candidates": [
                {
                    "status": "ok",
                    "checkpoint_id": "ckpt_linear_2",
                    "run_dir": "artifacts/run_linear_2",
                    "selected_model_family": "bagged_tree_ensemble",
                    "best_validation_model_family": "bagged_tree_ensemble",
                    "split": {
                        "strategy": "deterministic_modulo_70_15_15",
                        "train_size": 7,
                        "validation_size": 2,
                        "test_size": 2,
                    },
                    "preprocessing": {
                        "missing_data_strategy_requested": "fill_median",
                        "missing_data_strategy_effective": "fill_median_train_only",
                    },
                    "normalization": {"method": "minmax"},
                    "comparison": [
                        {
                            "model_family": "linear_ridge",
                            "validation_metrics": {"r2": 0.82, "mae": 0.20},
                            "test_metrics": {"r2": 0.80, "mae": 0.21},
                        },
                        {
                            "model_family": "bagged_tree_ensemble",
                            "validation_metrics": {"r2": 0.90, "mae": 0.15},
                            "test_metrics": {"r2": 0.88, "mae": 0.16},
                        },
                    ],
                    "selected_metrics": {
                        "test": {"r2": 0.88, "mae": 0.16},
                    },
                    "rows_used": 10,
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "modeler"])
    assert exit_code == 0
    assert registry.calls[0][0] == "prepare_ingestion_step"
    assert registry.calls[1][0] == "train_surrogate_candidates"
    assert registry.calls[1][1]["target_column"] == "C"
    assert registry.calls[1][1]["feature_columns"] == ["A", "B"]
    assert registry.calls[1][1]["requested_model_family"] == "bagged_tree_ensemble"
    assert registry.calls[1][1]["normalize"] is True
    assert registry.calls[1][1]["missing_data_strategy"] == "fill_median"
    output = capsys.readouterr().out
    assert "Handoff contract:" in output
    assert "Handoff suggestion:" in output
    assert "Press Enter to use the recommended model `tree_ensemble_candidate`" in output
    assert "Candidate `bagged_tree_ensemble`" in output
    assert "Model build complete:" in output


def test_cli_run_agent_session_modeler_direct_lagged_request_passes_temporal_args(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "model_lagged.csv"
    data_path.write_text("time,A,B,C\n0,1,2,3\n", encoding="utf-8")
    inputs = iter(
        [
            "build model lagged_linear with inputs A,B and target C",
            str(data_path),
            "/exit",
        ]
    )
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": None,
                    "available_sheets": [],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 1.0,
                    "needs_user_confirmation": False,
                    "row_count": 20,
                    "column_count": 4,
                    "signal_columns": ["time", "A", "B", "C"],
                    "numeric_signal_columns": ["time", "A", "B", "C"],
                    "timestamp_column_hint": "time",
                }
            ],
            "train_surrogate_candidates": [
                {
                    "status": "ok",
                    "checkpoint_id": "ckpt_lagged_1",
                    "run_dir": "artifacts/run_lagged_1",
                    "selected_model_family": "lagged_linear",
                    "best_validation_model_family": "lagged_linear",
                    "lag_horizon_samples": 4,
                    "split": {
                        "strategy": "blocked_time_order_70_15_15",
                        "train_size": 14,
                        "validation_size": 3,
                        "test_size": 3,
                    },
                    "preprocessing": {
                        "missing_data_strategy_requested": "fill_median",
                        "missing_data_strategy_effective": "fill_median_train_only",
                    },
                    "normalization": {"method": "minmax"},
                    "comparison": [
                        {
                            "model_family": "linear_ridge",
                            "validation_metrics": {"r2": 0.30, "mae": 0.44},
                            "test_metrics": {"r2": 0.28, "mae": 0.46},
                        },
                        {
                            "model_family": "lagged_linear",
                            "validation_metrics": {"r2": 0.92, "mae": 0.09},
                            "test_metrics": {"r2": 0.90, "mae": 0.10},
                        },
                    ],
                    "selected_metrics": {
                        "test": {"r2": 0.90, "mae": 0.10},
                    },
                    "rows_used": 11,
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "modeler"])
    assert exit_code == 0
    assert registry.calls[1][0] == "train_surrogate_candidates"
    assert registry.calls[1][1]["requested_model_family"] == "lagged_linear"
    assert registry.calls[1][1]["timestamp_column"] == "time"
    output = capsys.readouterr().out
    assert "Timestamp context: using `time`" in output
    assert "Temporal feature plan: lag_horizon_samples=4." in output
    assert "Candidate `lagged_linear`" in output


def test_cli_run_agent_session_modeler_handoff_uses_structured_contract_defaults(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "model_handoff_lagged.csv"
    data_path.write_text("time,A,C\n0,1,3\n", encoding="utf-8")
    handoff_path = tmp_path / "structured_lagged_report.json"
    handoff_path.write_text(
        json.dumps(
            {
                "data_path": str(data_path),
                "data_mode": "time_series",
                "timestamp_column": "time",
                "preprocessing": {
                    "missing_data_plan": {
                        "strategy": "fill_constant",
                        "fill_constant_value": -1.0,
                    }
                },
                "model_strategy_recommendations": {
                    "target_recommendations": [
                        {
                            "target_signal": "C",
                            "probe_predictor_signals": ["A"],
                            "recommended_model_family": "lagged_linear",
                            "priority_model_families": [
                                "lagged_linear",
                                "tree_ensemble_candidate",
                                "linear_ridge",
                            ],
                            "recommendation_statement": "Start with lagged linear first.",
                            "candidate_models": [
                                {
                                    "model_family": "lagged_linear",
                                    "notes": "Current + lagged predictor windows up to 5 samples.",
                                }
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    inputs = iter([f"use handoff {handoff_path}", "", "", "", "/exit"])
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": None,
                    "available_sheets": [],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 1.0,
                    "needs_user_confirmation": False,
                    "row_count": 32,
                    "column_count": 3,
                    "signal_columns": ["time", "A", "C"],
                    "numeric_signal_columns": ["time", "A", "C"],
                    "timestamp_column_hint": "time",
                }
            ],
            "train_surrogate_candidates": [
                {
                    "status": "ok",
                    "checkpoint_id": "ckpt_lagged_handoff",
                    "run_dir": "artifacts/run_lagged_handoff",
                    "selected_model_family": "lagged_linear",
                    "best_validation_model_family": "lagged_linear",
                    "lag_horizon_samples": 5,
                    "split": {
                        "strategy": "blocked_time_order_70_15_15",
                        "train_size": 22,
                        "validation_size": 5,
                        "test_size": 5,
                    },
                    "preprocessing": {
                        "missing_data_strategy_requested": "fill_constant",
                        "missing_data_strategy_effective": "fill_constant_train_policy",
                    },
                    "normalization": {"method": "minmax"},
                    "comparison": [
                        {
                            "model_family": "linear_ridge",
                            "validation_metrics": {"r2": 0.41, "mae": 0.33},
                            "test_metrics": {"r2": 0.39, "mae": 0.35},
                        },
                        {
                            "model_family": "lagged_linear",
                            "validation_metrics": {"r2": 0.93, "mae": 0.07},
                            "test_metrics": {"r2": 0.91, "mae": 0.08},
                        },
                    ],
                    "selected_metrics": {
                        "test": {"r2": 0.91, "mae": 0.08},
                    },
                    "rows_used": 17,
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "modeler"])
    assert exit_code == 0
    assert registry.calls[1][0] == "train_surrogate_candidates"
    args = registry.calls[1][1]
    assert args["requested_model_family"] == "lagged_linear"
    assert args["timestamp_column"] == "time"
    assert args["missing_data_strategy"] == "fill_constant"
    assert float(args["fill_constant_value"]) == -1.0
    assert args["lag_horizon_samples"] == 5
    output = capsys.readouterr().out
    assert "Handoff contract:" in output
    assert "search_order=['lagged_linear', 'tree_ensemble_candidate', 'linear_ridge']" in output
    assert "Temporal feature plan: lag_horizon_samples=5." in output


def test_cli_run_agent_session_modeler_auto_retries_to_next_safe_family(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "model_auto_retry.csv"
    data_path.write_text("time,A,C\n0,1,3\n", encoding="utf-8")
    inputs = iter(
        [
            "build model auto with inputs A and target C",
            str(data_path),
            "/exit",
        ]
    )
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": None,
                    "available_sheets": [],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 1.0,
                    "needs_user_confirmation": False,
                    "row_count": 32,
                    "column_count": 3,
                    "signal_columns": ["time", "A", "C"],
                    "numeric_signal_columns": ["time", "A", "C"],
                    "timestamp_column_hint": "time",
                }
            ],
            "train_surrogate_candidates": [
                {
                    "status": "ok",
                    "checkpoint_id": "ckpt_auto_retry_1",
                    "run_dir": "artifacts/run_auto_retry_1",
                    "selected_model_family": "linear_ridge",
                    "best_validation_model_family": "lagged_linear",
                    "lag_horizon_samples": 3,
                    "split": {"strategy": "blocked_time_order_70_15_15", "train_size": 22, "validation_size": 5, "test_size": 5},
                    "preprocessing": {
                        "missing_data_strategy_requested": "fill_median",
                        "missing_data_strategy_effective": "fill_median_train_only",
                    },
                    "normalization": {"method": "minmax"},
                    "comparison": [
                        {
                            "model_family": "linear_ridge",
                            "validation_metrics": {"r2": 0.40, "mae": 0.30},
                            "test_metrics": {"r2": 0.38, "mae": 0.33},
                        },
                        {
                            "model_family": "lagged_linear",
                            "validation_metrics": {"r2": 0.86, "mae": 0.11},
                            "test_metrics": {"r2": 0.84, "mae": 0.12},
                        },
                    ],
                    "selected_metrics": {
                        "train": {"mae": 0.28, "r2": 0.42},
                        "validation": {"mae": 0.30, "r2": 0.40},
                        "test": {"mae": 0.33, "r2": 0.38},
                    },
                    "rows_used": 22,
                },
                {
                    "status": "ok",
                    "checkpoint_id": "ckpt_auto_retry_2",
                    "run_dir": "artifacts/run_auto_retry_2",
                    "selected_model_family": "lagged_linear",
                    "best_validation_model_family": "lagged_linear",
                    "lag_horizon_samples": 3,
                    "split": {"strategy": "blocked_time_order_70_15_15", "train_size": 22, "validation_size": 5, "test_size": 5},
                    "preprocessing": {
                        "missing_data_strategy_requested": "fill_median",
                        "missing_data_strategy_effective": "fill_median_train_only",
                    },
                    "normalization": {"method": "minmax"},
                    "comparison": [
                        {
                            "model_family": "linear_ridge",
                            "validation_metrics": {"r2": 0.40, "mae": 0.30},
                            "test_metrics": {"r2": 0.38, "mae": 0.33},
                        },
                        {
                            "model_family": "lagged_linear",
                            "validation_metrics": {"r2": 0.86, "mae": 0.11},
                            "test_metrics": {"r2": 0.84, "mae": 0.12},
                        },
                    ],
                    "selected_metrics": {
                        "train": {"mae": 0.08, "r2": 0.90},
                        "validation": {"mae": 0.11, "r2": 0.86},
                        "test": {"mae": 0.12, "r2": 0.84},
                    },
                    "rows_used": 19,
                },
            ],
            "evaluate_training_iteration": [
                {
                    "should_continue": True,
                    "attempt": 1,
                    "max_attempts": 2,
                    "unmet_criteria": ["r2"],
                    "recommendations": ["Try alternate architecture and tune window/lag features for unmet metrics: r2"],
                    "summary": "Acceptance criteria not met. Continuing optimization loop.",
                },
                {
                    "should_continue": False,
                    "attempt": 2,
                    "max_attempts": 2,
                    "unmet_criteria": [],
                    "recommendations": ["Quality criteria met. Proceed to artifact export."],
                    "summary": "Model meets all acceptance criteria.",
                },
            ],
        }
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "modeler"])
    assert exit_code == 0
    train_calls = [call for call in registry.calls if call[0] == "train_surrogate_candidates"]
    assert len(train_calls) == 2
    assert train_calls[0][1]["requested_model_family"] == "auto"
    assert train_calls[1][1]["requested_model_family"] == "lagged_linear"
    output = capsys.readouterr().out
    assert "Continuing bounded optimization loop with `lagged_linear` as the next retry." in output
    assert "Acceptance check: Model meets all acceptance criteria." in output


def test_cli_run_agent_session_modeler_explicit_model_does_not_auto_switch(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "model_locked.csv"
    data_path.write_text("time,A,C\n0,1,3\n", encoding="utf-8")
    inputs = iter(
        [
            "build model linear_ridge with inputs A and target C",
            str(data_path),
            "/exit",
        ]
    )
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": None,
                    "available_sheets": [],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 1.0,
                    "needs_user_confirmation": False,
                    "row_count": 32,
                    "column_count": 3,
                    "signal_columns": ["time", "A", "C"],
                    "numeric_signal_columns": ["time", "A", "C"],
                    "timestamp_column_hint": "time",
                }
            ],
            "train_surrogate_candidates": [
                {
                    "status": "ok",
                    "checkpoint_id": "ckpt_locked_1",
                    "run_dir": "artifacts/run_locked_1",
                    "selected_model_family": "linear_ridge",
                    "best_validation_model_family": "lagged_linear",
                    "lag_horizon_samples": 3,
                    "split": {"strategy": "blocked_time_order_70_15_15", "train_size": 22, "validation_size": 5, "test_size": 5},
                    "preprocessing": {
                        "missing_data_strategy_requested": "fill_median",
                        "missing_data_strategy_effective": "fill_median_train_only",
                    },
                    "normalization": {"method": "minmax"},
                    "comparison": [
                        {
                            "model_family": "linear_ridge",
                            "validation_metrics": {"r2": 0.40, "mae": 0.30},
                            "test_metrics": {"r2": 0.38, "mae": 0.33},
                        },
                        {
                            "model_family": "lagged_linear",
                            "validation_metrics": {"r2": 0.86, "mae": 0.11},
                            "test_metrics": {"r2": 0.84, "mae": 0.12},
                        },
                    ],
                    "selected_metrics": {
                        "train": {"mae": 0.28, "r2": 0.42},
                        "validation": {"mae": 0.30, "r2": 0.40},
                        "test": {"mae": 0.33, "r2": 0.38},
                    },
                    "rows_used": 22,
                }
            ],
            "evaluate_training_iteration": [
                {
                    "should_continue": True,
                    "attempt": 1,
                    "max_attempts": 2,
                    "unmet_criteria": ["r2"],
                    "recommendations": ["Try alternate architecture and tune window/lag features for unmet metrics: r2"],
                    "summary": "Acceptance criteria not met. Continuing optimization loop.",
                }
            ],
        }
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "modeler"])
    assert exit_code == 0
    train_calls = [call for call in registry.calls if call[0] == "train_surrogate_candidates"]
    assert len(train_calls) == 1
    output = capsys.readouterr().out
    assert "Architecture auto-switch is disabled because this model family was explicitly chosen by the user." in output


def test_cli_run_agent_session_modeler_keeps_best_attempt_when_retry_is_worse(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "model_retry_best.csv"
    data_path.write_text("A,B,C\n1,2,3\n", encoding="utf-8")
    inputs = iter(
        [
            "build model auto with inputs A,B and target C",
            str(data_path),
            "/exit",
        ]
    )
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": None,
                    "available_sheets": [],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 1.0,
                    "needs_user_confirmation": False,
                    "row_count": 120,
                    "column_count": 3,
                    "signal_columns": ["A", "B", "C"],
                    "numeric_signal_columns": ["A", "B", "C"],
                }
            ],
                "train_surrogate_candidates": [
                    {
                        "status": "ok",
                        "checkpoint_id": "ckpt_best_first",
                    "run_dir": "artifacts/run_best_first",
                    "selected_model_family": "bagged_tree_classifier",
                    "best_validation_model_family": "bagged_tree_classifier",
                    "split": {"strategy": "stratified_deterministic_modulo_70_15_15", "train_size": 84, "validation_size": 18, "test_size": 18},
                    "task_profile": {"task_type": "fraud_detection", "task_family": "classification", "recommended_split_strategy": "stratified_deterministic_modulo_70_15_15", "minority_class_fraction": 0.08},
                    "preprocessing": {
                        "missing_data_strategy_requested": "fill_median",
                        "missing_data_strategy_effective": "fill_median_train_only",
                    },
                    "normalization": {"method": "minmax"},
                    "comparison": [
                        {
                            "model_family": "bagged_tree_classifier",
                            "validation_metrics": {"f1": 0.78, "accuracy": 0.95, "precision": 0.88, "recall": 0.72, "pr_auc": 0.49, "log_loss": 0.20},
                            "test_metrics": {"f1": 0.76, "accuracy": 0.94, "precision": 0.86, "recall": 0.70, "pr_auc": 0.46, "log_loss": 0.21},
                        },
                        {
                            "model_family": "logistic_regression",
                            "validation_metrics": {"f1": 0.44, "accuracy": 0.90, "precision": 0.60, "recall": 0.35, "pr_auc": 0.25, "log_loss": 0.41},
                            "test_metrics": {"f1": 0.40, "accuracy": 0.89, "precision": 0.55, "recall": 0.32, "pr_auc": 0.22, "log_loss": 0.43},
                        },
                    ],
                    "selected_metrics": {
                        "train": {"f1": 0.82, "accuracy": 0.96, "precision": 0.90, "recall": 0.76, "pr_auc": 0.52, "log_loss": 0.18},
                        "validation": {"f1": 0.78, "accuracy": 0.95, "precision": 0.88, "recall": 0.72, "pr_auc": 0.49, "log_loss": 0.20},
                        "test": {"f1": 0.76, "accuracy": 0.94, "precision": 0.86, "recall": 0.70, "pr_auc": 0.46, "log_loss": 0.21},
                    },
                    "rows_used": 84,
                },
                    {
                        "status": "ok",
                        "checkpoint_id": "ckpt_best_second",
                        "run_dir": "artifacts/run_best_second",
                        "selected_model_family": "bagged_tree_classifier",
                        "best_validation_model_family": "bagged_tree_classifier",
                        "split": {"strategy": "stratified_deterministic_modulo_70_15_15", "train_size": 84, "validation_size": 18, "test_size": 18},
                        "task_profile": {"task_type": "fraud_detection", "task_family": "classification", "recommended_split_strategy": "stratified_deterministic_modulo_70_15_15", "minority_class_fraction": 0.08},
                    "preprocessing": {
                        "missing_data_strategy_requested": "fill_median",
                        "missing_data_strategy_effective": "fill_median_train_only",
                    },
                    "normalization": {"method": "minmax"},
                    "comparison": [
                        {
                            "model_family": "bagged_tree_classifier",
                            "validation_metrics": {"f1": 0.78, "accuracy": 0.95, "precision": 0.88, "recall": 0.72, "pr_auc": 0.49, "log_loss": 0.20},
                            "test_metrics": {"f1": 0.76, "accuracy": 0.94, "precision": 0.86, "recall": 0.70, "pr_auc": 0.46, "log_loss": 0.21},
                        },
                        {
                            "model_family": "logistic_regression",
                            "validation_metrics": {"f1": 0.44, "accuracy": 0.90, "precision": 0.60, "recall": 0.35, "pr_auc": 0.25, "log_loss": 0.41},
                            "test_metrics": {"f1": 0.40, "accuracy": 0.89, "precision": 0.55, "recall": 0.32, "pr_auc": 0.22, "log_loss": 0.43},
                            },
                        ],
                        "selected_metrics": {
                            "train": {"f1": 0.82, "accuracy": 0.96, "precision": 0.90, "recall": 0.76, "pr_auc": 0.52, "log_loss": 0.18},
                            "validation": {"f1": 0.78, "accuracy": 0.95, "precision": 0.88, "recall": 0.72, "pr_auc": 0.49, "log_loss": 0.20},
                            "test": {"f1": 0.76, "accuracy": 0.94, "precision": 0.86, "recall": 0.70, "pr_auc": 0.46, "log_loss": 0.21},
                        },
                        "rows_used": 84,
                    },
                    {
                        "status": "ok",
                        "checkpoint_id": "ckpt_retry_worse",
                        "run_dir": "artifacts/run_retry_worse",
                        "selected_model_family": "logistic_regression",
                        "best_validation_model_family": "bagged_tree_classifier",
                        "split": {"strategy": "stratified_deterministic_modulo_70_15_15", "train_size": 84, "validation_size": 18, "test_size": 18},
                        "task_profile": {"task_type": "fraud_detection", "task_family": "classification", "recommended_split_strategy": "stratified_deterministic_modulo_70_15_15", "minority_class_fraction": 0.08},
                        "preprocessing": {
                            "missing_data_strategy_requested": "fill_median",
                            "missing_data_strategy_effective": "fill_median_train_only",
                        },
                        "normalization": {"method": "minmax"},
                        "comparison": [
                            {
                                "model_family": "bagged_tree_classifier",
                                "validation_metrics": {"f1": 0.78, "accuracy": 0.95, "precision": 0.88, "recall": 0.72, "pr_auc": 0.49, "log_loss": 0.20},
                                "test_metrics": {"f1": 0.76, "accuracy": 0.94, "precision": 0.86, "recall": 0.70, "pr_auc": 0.46, "log_loss": 0.21},
                            },
                            {
                                "model_family": "logistic_regression",
                                "validation_metrics": {"f1": 0.44, "accuracy": 0.90, "precision": 0.60, "recall": 0.35, "pr_auc": 0.25, "log_loss": 0.41},
                                "test_metrics": {"f1": 0.40, "accuracy": 0.89, "precision": 0.55, "recall": 0.32, "pr_auc": 0.22, "log_loss": 0.43},
                            },
                        ],
                        "selected_metrics": {
                            "train": {"f1": 0.50, "accuracy": 0.91, "precision": 0.63, "recall": 0.41, "pr_auc": 0.28, "log_loss": 0.39},
                            "validation": {"f1": 0.44, "accuracy": 0.90, "precision": 0.60, "recall": 0.35, "pr_auc": 0.25, "log_loss": 0.41},
                            "test": {"f1": 0.40, "accuracy": 0.89, "precision": 0.55, "recall": 0.32, "pr_auc": 0.22, "log_loss": 0.43},
                        },
                        "rows_used": 84,
                    },
                ],
                "evaluate_training_iteration": [
                    {
                        "should_continue": True,
                        "attempt": 1,
                        "max_attempts": 3,
                        "unmet_criteria": ["recall", "pr_auc"],
                        "recommendations": ["Try alternate architecture and tune window/lag features for unmet metrics: recall, pr_auc"],
                        "summary": "Acceptance criteria not met. Continuing optimization loop.",
                    },
                    {
                        "should_continue": True,
                        "attempt": 2,
                        "max_attempts": 3,
                        "unmet_criteria": ["recall", "pr_auc"],
                        "recommendations": ["Try alternate architecture and tune window/lag features for unmet metrics: recall, pr_auc"],
                        "summary": "Acceptance criteria not met. Continuing optimization loop.",
                    },
                    {
                        "should_continue": False,
                        "attempt": 3,
                        "max_attempts": 3,
                        "unmet_criteria": ["recall", "pr_auc"],
                        "recommendations": ["Acceptance criteria not met and max attempts reached."],
                        "summary": "Acceptance criteria not met and max attempts reached.",
                    },
            ],
        }
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "modeler"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "selected_model=bagged_tree_classifier" in output
    assert "Checkpoint saved: ckpt_best_first" in output
    assert "Artifacts: artifacts/run_best_first" in output
    assert "ckpt_retry_worse" not in output


def test_cli_run_agent_session_modeler_invalid_handoff_is_safe(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    handoff_path = tmp_path / "bad_handoff.json"
    handoff_path.write_text(json.dumps({"foo": "bar"}), encoding="utf-8")
    inputs = iter([f"use handoff {handoff_path}", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)
    exit_code = main(["run-agent-session", "--agent", "modeler"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "I could not derive a valid Agent 2 handoff from that report." in output


def test_cli_run_agent_session_modeler_lagged_request_on_non_time_series_is_safe(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "steady_no_time.csv"
    data_path.write_text("A,C\n1,3\n", encoding="utf-8")
    inputs = iter(
        [
            "build model lagged_linear with inputs A and target C",
            str(data_path),
            "/exit",
        ]
    )
    registry = _RaisingSessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": None,
                    "available_sheets": [],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 1.0,
                    "needs_user_confirmation": False,
                    "row_count": 20,
                    "column_count": 2,
                    "signal_columns": ["A", "C"],
                    "numeric_signal_columns": ["A", "C"],
                    "timestamp_column_hint": None,
                }
            ]
        },
        raise_on_tool="train_surrogate_candidates",
        message="Lagged model families require time-series structure and a usable timestamp column.",
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)
    exit_code = main(["run-agent-session", "--agent", "modeler"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Lagged model families require time-series structure and a usable timestamp column." in output


def test_cli_run_agent_session_modeler_suppresses_runtime_fallback_as_fake_interpretation(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "model_interpret_fallback.csv"
    data_path.write_text("A,B,C\n1,2,3\n", encoding="utf-8")
    inputs = iter(
        [
            "build model linear_ridge with inputs A,B and target C",
            str(data_path),
            "/exit",
        ]
    )
    registry = _SessionRegistry(
        scripted_outputs={
            "prepare_ingestion_step": [
                {
                    "status": "ok",
                    "message": "Ingestion ready.",
                    "options": [],
                    "selected_sheet": None,
                    "available_sheets": [],
                    "header_row": 0,
                    "data_start_row": 1,
                    "header_confidence": 1.0,
                    "needs_user_confirmation": False,
                    "row_count": 20,
                    "column_count": 3,
                    "signal_columns": ["A", "B", "C"],
                    "numeric_signal_columns": ["A", "B", "C"],
                }
            ],
            "train_surrogate_candidates": [
                {
                    "status": "ok",
                    "checkpoint_id": "ckpt_interp_1",
                    "run_dir": "artifacts/run_interp_1",
                    "selected_model_family": "linear_ridge",
                    "best_validation_model_family": "linear_ridge",
                    "lag_horizon_samples": 0,
                    "selected_hyperparameters": {
                        "requested_model_family": "linear_ridge",
                        "ridge": 1e-8,
                        "training_rows_used": 14,
                    },
                    "split": {
                        "strategy": "deterministic_modulo_70_15_15",
                        "train_size": 14,
                        "validation_size": 3,
                        "test_size": 3,
                    },
                    "preprocessing": {
                        "missing_data_strategy_requested": "fill_median",
                        "missing_data_strategy_effective": "fill_median_train_only",
                    },
                    "normalization": {"method": "minmax"},
                    "comparison": [
                        {
                            "model_family": "linear_ridge",
                            "validation_metrics": {"r2": 0.88, "mae": 0.12},
                            "test_metrics": {"r2": 0.86, "mae": 0.13},
                        }
                    ],
                    "selected_metrics": {
                        "train": {"mae": 0.10, "r2": 0.90},
                        "validation": {"mae": 0.12, "r2": 0.88},
                        "test": {"mae": 0.13, "r2": 0.86},
                    },
                    "rows_used": 14,
                }
            ],
            "evaluate_training_iteration": [
                {
                    "should_continue": False,
                    "attempt": 1,
                    "max_attempts": 2,
                    "unmet_criteria": [],
                    "recommendations": ["Quality criteria met. Proceed to artifact export."],
                    "summary": "Model meets all acceptance criteria.",
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr(
        "corr2surrogate.ui.cli.run_local_agent_once",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    exit_code = main(["run-agent-session", "--agent", "modeler"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Selected hyperparameters:" in output
    assert "LLM interpretation unavailable for this turn. Using the deterministic model summary above." in output
    assert "agent> LLM interpretation:" not in output
    assert "I hit an internal runtime error. Please retry." not in output


def test_cli_run_agent_session_task_override_is_persisted(monkeypatch) -> None:
    calls = []
    inputs = iter(["task fraud_detection", "hello", "/exit"])

    def fake_run_local_agent_once(*, agent, user_message, context, config_path):
        calls.append(
            {
                "agent": agent,
                "user_message": user_message,
                "context": context,
                "config_path": config_path,
            }
        )
        return {"event": {"message": "reply"}}

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", fake_run_local_agent_once)
    exit_code = main(["run-agent-session", "--agent", "analyst", "--max-turns", "4"])
    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["context"]["task_type_override"] == "fraud_detection"


def test_cli_modeler_trains_classification_dataset_without_crashing(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "classification.csv"
    rows = ["feature_a,feature_b,class_label"]
    for idx in range(60):
        a = idx / 59.0
        b = 1 if idx % 4 == 0 else 0
        label = 1 if (a > 0.6 or b == 1) else 0
        rows.append(f"{a:.5f},{b},{label}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")

    inputs = iter(
        [
            str(csv_path),
            "build model auto with inputs feature_a,feature_b and target class_label",
            "/exit",
        ]
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "modeler", "--max-turns", "5"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Task profile: type=binary_classification" in output
    assert "Model build complete:" in output
    assert "test_f1=" in output
    assert "Selected hyperparameters:" in output


def test_cli_modeler_handoff_trains_fraud_classifier(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "fraud.csv"
    rows = ["amount_norm,device_risk,velocity_score,fraud_flag"]
    for idx in range(120):
        fraud = 1 if idx % 9 == 0 else 0
        if fraud == 1:
            amount = 0.90
            device = 0.94
            velocity = 0.91
        else:
            amount = 0.06 + ((idx % 7) * 0.08)
            device = 0.08 + ((idx % 4) * 0.09)
            velocity = 0.10 + ((idx % 5) * 0.07)
        rows.append(f"{amount:.5f},{device:.5f},{velocity:.5f},{fraud}")
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    handoff_path = tmp_path / "fraud_report.json"
    handoff_path.write_text(
        json.dumps(
            {
                "data_path": str(csv_path),
                "data_mode": "steady_state",
                "preprocessing": {
                    "missing_data_plan": {
                        "strategy": "fill_median",
                    }
                },
                "task_profiles": [
                    {
                        "target_signal": "fraud_flag",
                        "task_type": "fraud_detection",
                    }
                ],
                "model_strategy_recommendations": {
                    "target_recommendations": [
                        {
                            "target_signal": "fraud_flag",
                            "probe_predictor_signals": [
                                "amount_norm",
                                "device_risk",
                                "velocity_score",
                            ],
                            "recommended_model_family": "tree_ensemble_candidate",
                            "priority_model_families": [
                                "tree_ensemble_candidate",
                                "linear_ridge",
                            ],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    inputs = iter([f"use handoff {handoff_path}", "", "", "", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.run_local_agent_once", _stub_llm_interpretation)

    exit_code = main(["run-agent-session", "--agent", "modeler", "--max-turns", "6"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Handoff contract:" in output
    assert "Task profile: type=fraud_detection" in output
    assert "Model build complete:" in output
    assert "test_recall=" in output
    assert "test_pr_auc=" in output
