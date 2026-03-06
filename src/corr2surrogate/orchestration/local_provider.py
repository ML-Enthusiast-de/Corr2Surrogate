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
    auth_token: str | None = None
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
        if provider == "openai":
            if not self.config.auth_token:
                raise LocalProviderError(
                    "Missing API key for OpenAI provider. Set C2S_API_KEY or OPENAI_API_KEY."
                )
            return _call_openai_compatible(
                self.config, messages, require_auth=True
            )
        if provider == "openai_compatible":
            return _call_openai_compatible(
                self.config,
                messages,
                require_auth=bool(self.config.auth_token),
            )
        raise LocalProviderError(f"Unsupported provider '{self.config.provider}'.")


def _build_messages(
    *,
    system_prompt: str,
    history: list[AgentTurnEvent],
    context: dict[str, Any],
    tool_catalog: list[dict[str, str]],
) -> list[dict[str, str]]:
    chat_only = bool(context.get("chat_only"))
    instruction = (
        "Return exactly one JSON action. Do not include markdown. "
        "Use action='tool_call' or action='respond'. "
        "If calling run_agent1_analysis, always include required field data_path. "
        "If context.path_hints exists, use the first entry as data_path."
    )
    if chat_only:
        instruction = (
            "Return exactly one JSON action. Do not include markdown. "
            "Use action='respond' only. Do not call tools. "
            "Answer conversationally and helpfully."
        )
    history_payload = [_sanitize_for_prompt(event.to_dict()) for event in history[-8:]]
    user_prompt = {
        "instruction": instruction,
        "tool_catalog": tool_catalog,
        "context": _sanitize_for_prompt(context),
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
    config: LocalResponderConfig,
    messages: list[dict[str, str]],
    *,
    require_auth: bool = False,
) -> dict[str, Any] | str:
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
    }
    headers: dict[str, str] | None = None
    if require_auth and config.auth_token:
        headers = {"Authorization": f"Bearer {config.auth_token}"}
    data = _http_post_json(
        config.endpoint,
        payload,
        timeout_seconds=config.timeout_seconds,
        headers=headers,
    )
    choices = data.get("choices") or []
    if not choices:
        raise LocalProviderError("OpenAI-compatible response has no choices.")
    message = (choices[0].get("message") or {}).get("content", "").strip()
    if not message:
        raise LocalProviderError("OpenAI-compatible response has empty message content.")
    return _parse_action_payload(message)


def _http_post_json(
    endpoint: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: int,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = Request(
        endpoint,
        data=body,
        headers=request_headers,
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
    except OSError as exc:
        lowered = str(exc).lower()
        if "timed out" in lowered or "timeout" in lowered:
            raise LocalProviderError(f"Provider request timed out for {endpoint}.") from exc
        raise LocalProviderError(f"Provider connection error at {endpoint}: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LocalProviderError("Provider returned non-JSON response.") from exc


def _parse_action_payload(text: str) -> dict[str, Any] | str:
    normalized = _strip_markdown_fences(text.strip())
    try:
        parsed = json.loads(normalized)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    parsed_snippet = _extract_first_json_object(normalized)
    if parsed_snippet is not None:
        return parsed_snippet

    if normalized:
        # Fail closed into a plain respond action to keep loop behavior deterministic.
        return {"action": "respond", "message": normalized[:2000]}
    return text


def _strip_markdown_fences(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    for start in range(len(text)):
        if text[start] != "{":
            continue
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
                continue
            if char == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : index + 1]
                    try:
                        parsed = json.loads(snippet)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
    return None


def _sanitize_for_prompt(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return "<truncated>"
    if isinstance(value, dict):
        skipped = {
            "report_markdown",
        }
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if str(key) in skipped:
                continue
            normalized[str(key)] = _sanitize_for_prompt(item, depth=depth + 1)
        return normalized
    if isinstance(value, list):
        return [_sanitize_for_prompt(item, depth=depth + 1) for item in value[:30]]
    if isinstance(value, str):
        if len(value) <= 1000:
            return value
        return value[:997] + "..."
    return value
