from corr2surrogate.orchestration.agent_loop import AgentTurnEvent, AgentAction
from corr2surrogate.orchestration.local_provider import (
    LocalLLMResponder,
    LocalResponderConfig,
    _parse_action_payload,
)


def test_local_responder_ollama_path(monkeypatch) -> None:
    def fake_post(endpoint, payload, timeout_seconds):
        return {
            "message": {
                "content": '{"action":"respond","message":"ok"}',
            }
        }

    monkeypatch.setattr("corr2surrogate.orchestration.local_provider._http_post_json", fake_post)

    responder = LocalLLMResponder(
        config=LocalResponderConfig(
            provider="ollama",
            model="qwen-test",
            endpoint="http://127.0.0.1:11434/api/chat",
        ),
        system_prompt="test",
        tool_catalog=[],
    )
    output = responder(history=[], context={})
    assert isinstance(output, dict)
    assert output["action"] == "respond"
    assert output["message"] == "ok"


def test_local_responder_includes_recent_history(monkeypatch) -> None:
    captured = {}

    def fake_post(endpoint, payload, timeout_seconds):
        captured["payload"] = payload
        return {
            "message": {
                "content": '{"action":"respond","message":"done"}',
            }
        }

    monkeypatch.setattr("corr2surrogate.orchestration.local_provider._http_post_json", fake_post)

    responder = LocalLLMResponder(
        config=LocalResponderConfig(
            provider="ollama",
            model="qwen-test",
            endpoint="http://127.0.0.1:11434/api/chat",
        ),
        system_prompt="test",
        tool_catalog=[{"name": "prepare_ingestion_step", "description": "x", "risk_level": "low"}],
    )
    history = [
        AgentTurnEvent(
            turn=1,
            status="respond",
            action=AgentAction(action="respond", message="x"),
            message="x",
        )
    ]
    responder(history=history, context={"foo": "bar"})
    messages = captured["payload"]["messages"]
    assert messages[0]["role"] == "system"
    assert "tool_catalog" in messages[1]["content"]


def test_local_responder_chat_only_instruction_forces_respond(monkeypatch) -> None:
    captured = {}

    def fake_post(endpoint, payload, timeout_seconds):
        captured["payload"] = payload
        return {
            "message": {
                "content": '{"action":"respond","message":"chat-only-ok"}',
            }
        }

    monkeypatch.setattr("corr2surrogate.orchestration.local_provider._http_post_json", fake_post)

    responder = LocalLLMResponder(
        config=LocalResponderConfig(
            provider="ollama",
            model="qwen-test",
            endpoint="http://127.0.0.1:11434/api/chat",
        ),
        system_prompt="test",
        tool_catalog=[],
    )
    output = responder(history=[], context={"chat_only": True})
    assert isinstance(output, dict)
    assert output["action"] == "respond"
    prompt_payload = captured["payload"]["messages"][1]["content"]
    assert "Use action='respond' only" in prompt_payload


def test_parse_action_payload_wraps_plain_text_as_respond() -> None:
    parsed = _parse_action_payload("I need a file path to continue.")
    assert isinstance(parsed, dict)
    assert parsed["action"] == "respond"
    assert "file path" in parsed["message"]


def test_parse_action_payload_extracts_markdown_json() -> None:
    raw = "```json\n{\"action\":\"respond\",\"message\":\"ok\"}\n```"
    parsed = _parse_action_payload(raw)
    assert isinstance(parsed, dict)
    assert parsed["action"] == "respond"
    assert parsed["message"] == "ok"


def test_local_responder_openai_includes_bearer_header(monkeypatch) -> None:
    captured = {}

    def fake_post(endpoint, payload, timeout_seconds, headers=None):
        captured["headers"] = headers or {}
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"action":"respond","message":"api-ok"}',
                    }
                }
            ]
        }

    monkeypatch.setattr("corr2surrogate.orchestration.local_provider._http_post_json", fake_post)

    responder = LocalLLMResponder(
        config=LocalResponderConfig(
            provider="openai",
            model="gpt-4.1-mini",
            endpoint="https://api.openai.com/v1/chat/completions",
            auth_token="dummy-token",
        ),
        system_prompt="test",
        tool_catalog=[],
    )
    output = responder(history=[], context={})
    assert isinstance(output, dict)
    assert output["action"] == "respond"
    assert output["message"] == "api-ok"
    assert captured["headers"].get("Authorization") == "Bearer dummy-token"
