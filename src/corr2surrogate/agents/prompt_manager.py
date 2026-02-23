"""Prompt loading utilities for agent behavior control."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path


class PromptError(RuntimeError):
    """Raised when prompt assets cannot be loaded."""


DEFAULT_PROMPT_FILES = {
    "analyst": "analyst_system.txt",
    "modeler": "modeler_system.txt",
}


@dataclass(frozen=True)
class PromptBundle:
    """Resolved prompt content and provenance."""

    agent: str
    content: str
    source: str


def load_system_prompt(
    *,
    agent: str,
    override_path: str | Path | None = None,
    extra_instructions: str = "",
) -> PromptBundle:
    """Load system prompt for one agent.

    Priority:
    1. explicit override path if provided
    2. packaged default prompt asset
    """
    normalized = agent.strip().lower()
    if override_path:
        path = Path(override_path).expanduser()
        if not path.is_file():
            raise PromptError(f"Prompt override not found: {override_path}")
        text = path.read_text(encoding="utf-8")
        merged = _merge_extra(text, extra_instructions)
        return PromptBundle(agent=normalized, content=merged, source=str(path))

    default_name = DEFAULT_PROMPT_FILES.get(normalized)
    if default_name is None:
        raise PromptError(
            f"Unsupported agent '{agent}'. Supported: {sorted(DEFAULT_PROMPT_FILES)}"
        )

    try:
        text = resources.files("corr2surrogate.agents.prompts").joinpath(default_name).read_text(
            encoding="utf-8"
        )
    except FileNotFoundError as exc:
        raise PromptError(f"Default prompt asset not found: {default_name}") from exc

    merged = _merge_extra(text, extra_instructions)
    return PromptBundle(
        agent=normalized,
        content=merged,
        source=f"package:corr2surrogate.agents.prompts/{default_name}",
    )


def _merge_extra(base: str, extra: str) -> str:
    if not extra.strip():
        return base.strip()
    return f"{base.strip()}\n\nAdditional run instructions:\n{extra.strip()}"
