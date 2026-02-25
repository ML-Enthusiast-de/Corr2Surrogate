from dataclasses import dataclass
from pathlib import Path

from corr2surrogate.ui.cli import main


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


def test_cli_run_agent_session_analyst_autopilot_runs_analysis(monkeypatch, tmp_path: Path) -> None:
    data_path = tmp_path / "demo_data.xlsx"
    data_path.write_text("placeholder", encoding="utf-8")
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
                }
            ],
        }
    )

    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr("corr2surrogate.ui.cli.build_default_registry", lambda: registry)
    monkeypatch.setattr(
        "corr2surrogate.ui.cli.run_local_agent_once",
        lambda **_: (_ for _ in ()).throw(AssertionError("LLM path should not run for autopilot")),
    )

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[0][0] == "prepare_ingestion_step"
    assert registry.calls[1][0] == "run_agent1_analysis"
    assert Path(registry.calls[1][1]["data_path"]).resolve() == data_path.resolve()


def test_cli_run_agent_session_analyst_autopilot_multi_sheet(monkeypatch, tmp_path: Path) -> None:
    data_path = tmp_path / "multi_sheet.xlsx"
    data_path.write_text("placeholder", encoding="utf-8")
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

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "prepare_ingestion_step"
    assert registry.calls[1][1]["sheet_name"] == "S2"
    assert registry.calls[2][0] == "run_agent1_analysis"
    assert registry.calls[2][1]["sheet_name"] == "S2"


def test_cli_run_agent_session_analyst_autopilot_multi_sheet_handles_small_talk(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "multi_sheet_chat.xlsx"
    data_path.write_text("placeholder", encoding="utf-8")
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

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "prepare_ingestion_step"
    assert registry.calls[1][1]["sheet_name"] == "S2"
    output = capsys.readouterr().out
    assert "selecting an Excel sheet now" in output


def test_cli_run_agent_session_analyst_autopilot_header_override(monkeypatch, tmp_path: Path) -> None:
    data_path = tmp_path / "low_conf.xlsx"
    data_path.write_text("placeholder", encoding="utf-8")
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
    data_path.write_text("placeholder", encoding="utf-8")
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

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "prepare_ingestion_step"
    assert registry.calls[1][1]["header_row"] == 1
    assert registry.calls[1][1]["data_start_row"] == 2
    output = capsys.readouterr().out
    assert "confirming inferred rows now" in output


def test_cli_run_agent_session_analyst_autopilot_header_override_handles_small_talk(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "header_override_chat.xlsx"
    data_path.write_text("placeholder", encoding="utf-8")
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

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "prepare_ingestion_step"
    assert registry.calls[1][1]["header_row"] == 3
    assert registry.calls[1][1]["data_start_row"] == 4
    output = capsys.readouterr().out
    assert "choosing explicit row numbers now" in output


def test_cli_run_agent_session_analyst_target_prompt_accepts_list_command(
    monkeypatch, tmp_path: Path
) -> None:
    data_path = tmp_path / "wide.xlsx"
    data_path.write_text("placeholder", encoding="utf-8")
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

    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "run_agent1_analysis"
    assert registry.calls[1][1]["target_signals"] == ["sig_20"]


def test_cli_run_agent_session_analyst_target_prompt_handles_small_talk(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "wide_chat.xlsx"
    data_path.write_text("placeholder", encoding="utf-8")
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
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    assert registry.calls[1][0] == "run_agent1_analysis"
    assert registry.calls[1][1]["target_signals"] == ["sig_2"]
    output = capsys.readouterr().out
    assert "I am good and ready to continue" in output


def test_cli_run_agent_session_rewrites_greeting_tool_error_fallback(monkeypatch, capsys) -> None:
    inputs = iter(["hi", "/exit"])

    def fake_input(_prompt):
        return next(inputs)

    def fake_run_local_agent_once(*, agent, user_message, context, config_path):
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
    output = capsys.readouterr().out
    assert "Hi. I can chat and help with analysis." in output


def test_cli_run_agent_session_fast_greeting_bypasses_llm(monkeypatch, capsys) -> None:
    inputs = iter(["hello", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    monkeypatch.setattr(
        "corr2surrogate.ui.cli.run_local_agent_once",
        lambda **_: (_ for _ in ()).throw(AssertionError("LLM should not be called for greeting")),
    )
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Hi. I can chat and help with analysis." in output


def test_cli_run_agent_session_prints_welcome_message(monkeypatch, capsys) -> None:
    inputs = iter(["/exit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Welcome to Corr2Surrogate" in output
    assert "Useful commands" in output


def test_cli_run_agent_session_prints_header_preview_on_confirmation(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    data_path = tmp_path / "header_preview.xlsx"
    data_path.write_text("placeholder", encoding="utf-8")
    inputs = iter([f"Analyze {data_path}", "Y", "/exit"])
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
    exit_code = main(["run-agent-session", "--agent", "analyst"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Inferred header preview" in output
    assert "sig_a" in output
