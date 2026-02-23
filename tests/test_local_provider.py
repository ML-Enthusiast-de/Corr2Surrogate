from corr2surrogate.orchestration.agent_loop import AgentTurnEvent, AgentAction
from corr2surrogate.orchestration.local_provider import LocalLLMResponder, LocalResponderConfig


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
