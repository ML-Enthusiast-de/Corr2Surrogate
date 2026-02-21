"""Artifact persistence for model metadata and normalization state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from corr2surrogate.modeling.normalization import MinMaxNormalizer


class ArtifactStore:
    """Simple run-based artifact store."""

    def __init__(self, base_dir: str | Path = "artifacts") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, run_id: str | None = None) -> Path:
        """Create a unique run directory."""
        if run_id is None:
            run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def save_model_params(
        self,
        *,
        run_dir: str | Path,
        model_name: str,
        best_params: dict[str, Any],
        metrics: dict[str, float],
        feature_columns: list[str],
        target_column: str,
        split_strategy: str,
        normalizer_path: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """Persist optimized model parameters and evaluation metadata."""
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "best_params": best_params,
            "metrics": metrics,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "split_strategy": split_strategy,
            "normalizer_path": normalizer_path,
            "extra": extra or {},
        }
        output = Path(run_dir) / "model_params.json"
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output

    def save_normalizer(
        self,
        *,
        run_dir: str | Path,
        normalizer: MinMaxNormalizer,
        filename: str = "normalization_state.json",
    ) -> Path:
        """Persist a fitted normalizer state for later inference de-normalization."""
        output = Path(run_dir) / filename
        return normalizer.save(output)

    @staticmethod
    def read_json(path: str | Path) -> dict[str, Any]:
        """Load a JSON artifact."""
        return json.loads(Path(path).read_text(encoding="utf-8"))
