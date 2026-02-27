"""Lineage and reproducibility metadata persistence."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from corr2surrogate.core.json_utils import write_json


class RunStore:
    """Store run-level lineage metadata for reproducibility."""

    def __init__(self, base_dir: str | Path = "reports") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_lineage(
        self,
        *,
        dataset_slug: str,
        run_id: str,
        data_path: str,
        analysis_config: dict[str, Any],
        selected_strategy: dict[str, Any] | None = None,
        planner_trace: list[dict[str, Any]] | None = None,
        critic_decision: dict[str, Any] | None = None,
        random_seed: int = 42,
    ) -> str:
        target_dir = self.base_dir / dataset_slug
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{run_id}.lineage.json"
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "dataset": self._dataset_fingerprint(data_path),
            "analysis_config": analysis_config,
            "selected_strategy": selected_strategy or {},
            "planner_trace": planner_trace or [],
            "critic_decision": critic_decision or {},
            "random_seed": int(random_seed),
            "runtime": self._runtime_info(),
        }
        write_json(path, payload, indent=2)
        return str(path)

    def _dataset_fingerprint(self, data_path: str) -> dict[str, Any]:
        file_path = Path(data_path)
        size = file_path.stat().st_size if file_path.exists() else None
        sha = self._sha256(file_path) if file_path.exists() else ""
        return {
            "path": str(file_path),
            "exists": bool(file_path.exists()),
            "size_bytes": size,
            "sha256": sha,
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _runtime_info() -> dict[str, Any]:
        git_commit = ""
        try:
            git_commit = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                .strip()
            )
        except Exception:
            git_commit = ""
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "git_commit": git_commit,
        }
