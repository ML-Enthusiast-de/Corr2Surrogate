"""Structured handoff contract between Agent 1 and Agent 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
class ForcedModelingRequest:
    """Explicit user request to model a target with specific predictors."""

    target_signal: str
    predictor_signals: list[str]
    force_run_regardless_of_correlation: bool = True
    reason: str = ""


@dataclass(frozen=True)
class SystemKnowledge:
    """Domain knowledge injected by user for policy-aware modeling."""

    critical_signals: list[str] = field(default_factory=list)
    physically_required_signals: list[str] = field(default_factory=list)
    non_virtualizable_signals: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class AgenticLoopPolicy:
    """Retry policy when model quality does not meet acceptance criteria."""

    enabled: bool = True
    max_attempts: int = 3
    min_relative_improvement: float = 0.02
    allow_architecture_switch: bool = True
    allow_feature_set_expansion: bool = True
    suggest_more_data_when_stalled: bool = True


@dataclass(frozen=True)
class Agent2Handoff:
    """Machine-readable payload for model training and iteration control."""

    dataset_profile: dict[str, Any]
    target_signal: str
    feature_signals: list[str]
    split_strategy: str
    acceptance_criteria: dict[str, float]
    normalization: NormalizationPlan
    forced_modeling_requests: list[ForcedModelingRequest] = field(default_factory=list)
    dependency_map: dict[str, list[str]] = field(default_factory=dict)
    system_knowledge: SystemKnowledge = field(default_factory=SystemKnowledge)
    loop_policy: AgenticLoopPolicy = field(default_factory=AgenticLoopPolicy)
    user_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["normalization"]["feature_range"] = list(
            payload["normalization"]["feature_range"]
        )
        return payload
