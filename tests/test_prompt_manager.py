from pathlib import Path

from corr2surrogate.agents.prompt_manager import load_system_prompt


def test_load_default_prompt_bundle() -> None:
    bundle = load_system_prompt(agent="analyst")
    assert "Agent 1" in bundle.content
    assert bundle.source.startswith("package:")


def test_load_prompt_override(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("override prompt", encoding="utf-8")
    bundle = load_system_prompt(
        agent="analyst",
        override_path=prompt_file,
        extra_instructions="be concise",
    )
    assert "override prompt" in bundle.content
    assert "Additional run instructions" in bundle.content
