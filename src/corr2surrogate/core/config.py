"""Configuration loader utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration into a dictionary.

    Resolution order:
    1. explicit path argument
    2. `C2S_CONFIG_PATH` environment variable
    3. repository-relative default (`configs/default.yaml`)
    """
    env_path = os.getenv("C2S_CONFIG_PATH")
    requested = path if path is not None else env_path or "configs/default.yaml"
    config_path = _resolve_config_path(requested)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded


def _resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate

    if not candidate.is_absolute():
        project_root = Path(__file__).resolve().parents[3]
        alt = (project_root / candidate).resolve()
        if alt.is_file():
            return alt
    raise FileNotFoundError(f"Config file not found: {path}")
