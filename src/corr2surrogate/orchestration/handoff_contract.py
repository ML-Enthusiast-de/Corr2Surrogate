"""Structured handoff contract between Agent 1 and Agent 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizationPlan:
    """Normalization decisions captured by Agent 1."""

    enabled: bool
    method: str
    feature_range: tuple[float, float]
    normalize_target: bool
    needs_user_confirmation: bool
    reason: str


@dataclass(frozen=True)
class Agent2Handoff:
    """Minimum machine-readable payload for model training."""

    dataset_profile: dict[str, Any]
    target_signal: str
    feature_signals: list[str]
    split_strategy: str
    acceptance_criteria: dict[str, float]
    normalization: NormalizationPlan
    user_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["normalization"]["feature_range"] = list(
            payload["normalization"]["feature_range"]
        )
        return payload
