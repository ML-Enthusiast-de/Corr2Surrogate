"""Model savepoint/checkpoint management."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from corr2surrogate.core.json_utils import write_json


@dataclass(frozen=True)
class ModelCheckpoint:
    """Serializable model savepoint metadata."""

    checkpoint_id: str
    created_at_utc: str
    parent_checkpoint_id: str | None
    model_name: str
    run_dir: str
    model_state_path: str
    target_column: str
    feature_columns: list[str]
    metrics: dict[str, float]
    data_references: list[str] = field(default_factory=list)
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrainPlan:
    """Structured plan for retraining from an existing savepoint."""

    checkpoint_id: str
    model_name: str
    target_column: str
    feature_columns: list[str]
    previous_data_references: list[str]
    additional_data_references: list[str]
    combined_data_reference_count: int
    recommended_action: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelCheckpointStore:
    """File-backed checkpoint store."""

    def __init__(self, base_dir: str | Path = "artifacts/checkpoints") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_checkpoint(
        self,
        *,
        model_name: str,
        run_dir: str | Path,
        model_state_path: str | Path,
        target_column: str,
        feature_columns: list[str],
        metrics: dict[str, float],
        parent_checkpoint_id: str | None = None,
        data_references: list[str] | None = None,
        notes: str = "",
        tags: list[str] | None = None,
    ) -> ModelCheckpoint:
        """Create and persist one checkpoint record."""
        checkpoint = ModelCheckpoint(
            checkpoint_id=_new_checkpoint_id(),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            parent_checkpoint_id=parent_checkpoint_id,
            model_name=model_name,
            run_dir=str(Path(run_dir)),
            model_state_path=str(Path(model_state_path)),
            target_column=target_column,
            feature_columns=list(feature_columns),
            metrics={k: float(v) for k, v in metrics.items()},
            data_references=list(data_references or []),
            notes=notes,
            tags=list(tags or []),
        )
        self._write_checkpoint_file(checkpoint)
        return checkpoint

    def load_checkpoint(self, checkpoint_id: str) -> ModelCheckpoint:
        """Load one checkpoint by id."""
        path = self.base_dir / f"{checkpoint_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return ModelCheckpoint(**payload)

    def list_checkpoints(
        self,
        *,
        model_name: str | None = None,
        target_column: str | None = None,
        limit: int = 50,
    ) -> list[ModelCheckpoint]:
        """List checkpoints sorted newest first."""
        items: list[ModelCheckpoint] = []
        for path in self.base_dir.glob("ckpt_*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            checkpoint = ModelCheckpoint(**payload)
            if model_name and checkpoint.model_name != model_name:
                continue
            if target_column and checkpoint.target_column != target_column:
                continue
            items.append(checkpoint)
        items.sort(key=lambda item: item.created_at_utc, reverse=True)
        return items[: max(int(limit), 1)]

    def latest_checkpoint(
        self,
        *,
        model_name: str | None = None,
        target_column: str | None = None,
    ) -> ModelCheckpoint | None:
        """Return latest checkpoint for optional filters."""
        items = self.list_checkpoints(model_name=model_name, target_column=target_column, limit=1)
        return items[0] if items else None

    def build_retrain_plan(
        self,
        *,
        checkpoint_id: str,
        additional_data_references: list[str],
        notes: str = "",
    ) -> RetrainPlan:
        """Build a retraining plan from an existing checkpoint."""
        checkpoint = self.load_checkpoint(checkpoint_id)
        combined = list(checkpoint.data_references) + list(additional_data_references)
        return RetrainPlan(
            checkpoint_id=checkpoint.checkpoint_id,
            model_name=checkpoint.model_name,
            target_column=checkpoint.target_column,
            feature_columns=list(checkpoint.feature_columns),
            previous_data_references=list(checkpoint.data_references),
            additional_data_references=list(additional_data_references),
            combined_data_reference_count=len(combined),
            recommended_action=(
                "Load model state, append new data, update model statistics, "
                "re-evaluate, and write a child checkpoint."
            ),
            notes=notes,
        )

    def _write_checkpoint_file(self, checkpoint: ModelCheckpoint) -> None:
        output = self.base_dir / f"{checkpoint.checkpoint_id}.json"
        write_json(output, checkpoint.to_dict(), indent=2)


def _new_checkpoint_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    token = uuid4().hex[:8]
    return f"ckpt_{stamp}_{token}"
