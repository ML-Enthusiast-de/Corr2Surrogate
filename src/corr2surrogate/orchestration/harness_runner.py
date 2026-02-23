"""High-level harness runner that wires policy, prompts, provider, and tools."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from corr2surrogate.agents.prompt_manager import load_system_prompt
from corr2surrogate.core.config import load_config

from .agent_loop import AgentLoop, AgentLoopLimits, AgentTurnEvent
from .default_tools import build_default_registry
from .local_provider import LocalLLMResponder, LocalResponderConfig
from .runtime_policy import apply_environment_overrides, load_runtime_policy


def run_local_agent_once(
    *,
    agent: str,
    user_message: str,
    context: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Run one local agent loop execution and return structured result."""
    config = load_config(config_path)
    policy = apply_environment_overrides(load_runtime_policy(config))
    runtime_cfg = config.get("runtime", {})
    endpoint = _resolve_endpoint(policy.provider, runtime_cfg)
    options = policy.runtime_options(endpoint=endpoint)

    prompts_cfg = config.get("prompts", {})
    override_key = "analyst_system_path" if agent.lower() == "analyst" else "modeler_system_path"
    override_path = prompts_cfg.get(override_key) or None
    extra_instructions = prompts_cfg.get("extra_instructions", "")
    prompt = load_system_prompt(
        agent=agent,
        override_path=override_path,
        extra_instructions=extra_instructions,
    )

    registry = build_default_registry()
    responder = LocalLLMResponder(
        config=LocalResponderConfig(
            provider=options["provider"],
            model=options["model"],
            endpoint=endpoint,
            temperature=float(runtime_cfg.get("temperature", 0.0)),
            max_context=int(options["max_context"]),
            timeout_seconds=int(runtime_cfg.get("timeout_seconds", 120)),
        ),
        system_prompt=prompt.content,
        tool_catalog=registry.list_tools(),
    )
    loop = AgentLoop(registry=registry, limits=AgentLoopLimits())
    start_context = dict(context or {})
    start_context["user_message"] = user_message
    final_event = loop.run(responder=responder, context=start_context)
    return {
        "agent": agent,
        "prompt_source": prompt.source,
        "runtime": options,
        "event": _event_to_dict(final_event),
        "history": [_event_to_dict(item) for item in loop.history],
    }


def _resolve_endpoint(provider: str, runtime_cfg: dict[str, Any]) -> str:
    provider_key = provider.lower()
    endpoints = runtime_cfg.get("endpoints", {})
    if provider_key in {"ollama"}:
        return str(endpoints.get("ollama", "http://127.0.0.1:11434/api/chat"))
    if provider_key in {"llama.cpp", "llama_cpp"}:
        return str(
            endpoints.get(
                "llama_cpp",
                "http://127.0.0.1:8000/v1/chat/completions",
            )
        )
    raise ValueError(f"Unsupported provider '{provider}'.")


def _event_to_dict(event: AgentTurnEvent) -> dict[str, Any]:
    payload = asdict(event)
    return payload
