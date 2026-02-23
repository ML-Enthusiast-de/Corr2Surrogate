"""Local LLM responder adapters for Ollama and local OpenAI-compatible servers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .agent_loop import AgentTurnEvent


class LocalProviderError(RuntimeError):
    """Raised when the local LLM provider call fails."""


@dataclass(frozen=True)
class LocalResponderConfig:
    """Configuration for one local responder instance."""

    provider: str
    model: str
    endpoint: str
    temperature: float = 0.0
    max_context: int = 4096
    timeout_seconds: int = 120


class LocalLLMResponder:
    """Agent responder that calls a local LLM endpoint and returns one JSON action."""

    def __init__(
        self,
        *,
        config: LocalResponderConfig,
        system_prompt: str,
        tool_catalog: list[dict[str, str]] | None = None,
    ) -> None:
        self.config = config
        self.system_prompt = system_prompt.strip()
        self.tool_catalog = tool_catalog or []

    def __call__(
        self,
        *,
        history: list[AgentTurnEvent],
        context: dict[str, Any],
    ) -> dict[str, Any] | str:
        provider = self.config.provider.lower()
        messages = _build_messages(
            system_prompt=self.system_prompt,
            history=history,
            context=context,
            tool_catalog=self.tool_catalog,
        )

        if provider == "ollama":
            return _call_ollama(self.config, messages)
        if provider in {"llama.cpp", "llama_cpp"}:
            return _call_openai_compatible(self.config, messages)
        raise LocalProviderError(f"Unsupported provider '{self.config.provider}'.")


def _build_messages(
    *,
    system_prompt: str,
    history: list[AgentTurnEvent],
    context: dict[str, Any],
    tool_catalog: list[dict[str, str]],
) -> list[dict[str, str]]:
    history_payload = [event.to_dict() for event in history[-8:]]
    user_prompt = {
        "instruction": (
            "Return exactly one JSON action. Do not include markdown. "
            "Use action='tool_call' or action='respond'."
        ),
        "tool_catalog": tool_catalog,
        "context": context,
        "recent_history": history_payload,
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
    ]


def _call_ollama(
    config: LocalResponderConfig, messages: list[dict[str, str]]
) -> dict[str, Any] | str:
    payload = {
        "model": config.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_ctx": config.max_context,
        },
    }
    data = _http_post_json(config.endpoint, payload, timeout_seconds=config.timeout_seconds)
    content = ((data.get("message") or {}).get("content") or "").strip()
    if not content:
        raise LocalProviderError("Ollama response did not include message content.")
    return _parse_action_payload(content)


def _call_openai_compatible(
    config: LocalResponderConfig, messages: list[dict[str, str]]
) -> dict[str, Any] | str:
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": 700,
    }
    data = _http_post_json(config.endpoint, payload, timeout_seconds=config.timeout_seconds)
    choices = data.get("choices") or []
    if not choices:
        raise LocalProviderError("OpenAI-compatible response has no choices.")
    message = (choices[0].get("message") or {}).get("content", "").strip()
    if not message:
        raise LocalProviderError("OpenAI-compatible response has empty message content.")
    return _parse_action_payload(message)


def _http_post_json(
    endpoint: str, payload: dict[str, Any], *, timeout_seconds: int
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
        raise LocalProviderError(
            f"Provider HTTP error {exc.code} at {endpoint}: {detail}"
        ) from exc
    except URLError as exc:
        raise LocalProviderError(f"Provider connection error at {endpoint}: {exc}") from exc
    except TimeoutError as exc:
        raise LocalProviderError(f"Provider request timed out for {endpoint}.") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LocalProviderError("Provider returned non-JSON response.") from exc


def _parse_action_payload(text: str) -> dict[str, Any] | str:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        snippet = text[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return text
