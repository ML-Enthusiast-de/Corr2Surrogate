from pathlib import Path

from corr2surrogate.orchestration.harness_runner import (
    _extract_data_path_hints,
    run_local_agent_once,
)


def test_run_local_agent_once_returns_structured_payload(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "privacy:",
                "  local_only: true",
                "  api_calls_allowed: false",
                "  telemetry_allowed: false",
                "runtime:",
                "  provider: ollama",
                "  require_local_models: true",
                "  block_remote_endpoints: true",
                "  offline_mode: true",
                "  temperature: 0.0",
                "  timeout_seconds: 10",
                "  endpoints:",
                "    ollama: http://127.0.0.1:11434/api/chat",
                "  profiles:",
                "    small_cpu:",
                "      model: qwen-test",
                "      cpu_only: true",
                "      n_gpu_layers: 0",
                "      max_context: 4096",
                "  default_profile: small_cpu",
                "  fallback_order:",
                "    - small_cpu",
                "prompts:",
                "  analyst_system_path: ''",
                "  modeler_system_path: ''",
                "  extra_instructions: ''",
            ]
        ),
        encoding="utf-8",
    )

    def fake_call(self, *, history, context):
        return {"action": "respond", "message": "ok"}

    monkeypatch.setattr(
        "corr2surrogate.orchestration.local_provider.LocalLLMResponder.__call__",
        fake_call,
    )

    result = run_local_agent_once(
        agent="analyst",
        user_message="hello",
        context={"x": 1},
        config_path=str(config),
    )
    assert result["event"]["status"] == "respond"
    assert result["event"]["message"] == "ok"


def test_run_local_agent_once_handles_max_turns_without_crash(
    monkeypatch, tmp_path: Path
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "privacy:",
                "  local_only: true",
                "  api_calls_allowed: false",
                "  telemetry_allowed: false",
                "runtime:",
                "  provider: ollama",
                "  require_local_models: true",
                "  block_remote_endpoints: true",
                "  offline_mode: true",
                "  temperature: 0.0",
                "  timeout_seconds: 10",
                "  endpoints:",
                "    ollama: http://127.0.0.1:11434/api/chat",
                "  profiles:",
                "    small_cpu:",
                "      model: qwen-test",
                "      cpu_only: true",
                "      n_gpu_layers: 0",
                "      max_context: 4096",
                "  default_profile: small_cpu",
                "  fallback_order:",
                "    - small_cpu",
                "prompts:",
                "  analyst_system_path: ''",
                "  modeler_system_path: ''",
                "  extra_instructions: ''",
            ]
        ),
        encoding="utf-8",
    )

    def fake_call(self, *, history, context):
        return {
            "action": "tool_call",
            "tool_name": "unknown_tool",
            "arguments": {},
            "reason": "test",
        }

    monkeypatch.setattr(
        "corr2surrogate.orchestration.local_provider.LocalLLMResponder.__call__",
        fake_call,
    )

    result = run_local_agent_once(
        agent="analyst",
        user_message="hello",
        context={"x": 1},
        config_path=str(config),
    )
    assert result["event"]["status"] == "respond"
    assert "repeated tool argument errors" in result["event"]["message"]


def test_extract_data_path_hints() -> None:
    hints = _extract_data_path_hints("Analyze data/private/run1.csv and data/private/run2.xlsx")
    assert hints == ["data/private/run1.csv", "data/private/run2.xlsx"]
