"""Structured handoff contract between Agent 1 and Agent 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
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
    recommended_model_family: str = "linear_ridge"
    model_search_order: list[str] = field(default_factory=list)
    lag_horizon_samples: int = 0
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


def build_agent2_handoff_from_report_payload(payload: dict[str, Any]) -> Agent2Handoff | None:
    """Convert an Agent 1 structured report into a concrete Agent 2 handoff."""
    if not isinstance(payload, dict):
        return None
    target_rec = _primary_target_recommendation(payload)
    target_signal = str((target_rec or {}).get("target_signal", "")).strip()
    feature_signals = [
        str(item).strip()
        for item in ((target_rec or {}).get("probe_predictor_signals") or [])
        if str(item).strip()
    ]
    if not target_signal:
        target_signal = _fallback_target_from_report(payload)
    if not feature_signals:
        feature_signals = _fallback_features_from_report(payload, target_signal=target_signal)
    if not target_signal or not feature_signals:
        return None

    data_mode = str(payload.get("data_mode", "steady_state")).strip() or "steady_state"
    timestamp_column = str(payload.get("timestamp_column", "")).strip() or None
    recommended_model = str(
        (target_rec or {}).get("recommended_model_family", "linear_ridge")
    ).strip() or "linear_ridge"
    search_order = [
        str(item).strip()
        for item in ((target_rec or {}).get("priority_model_families") or [])
        if str(item).strip()
    ]
    lag_horizon = _derive_lag_horizon_samples(target_rec or {})
    preprocessing = payload.get("preprocessing") if isinstance(payload.get("preprocessing"), dict) else {}
    missing_plan = preprocessing.get("missing_data_plan") if isinstance(preprocessing.get("missing_data_plan"), dict) else {}
    missing_strategy = str(missing_plan.get("strategy", "keep")).strip() or "keep"
    fill_constant_value = missing_plan.get("fill_constant_value")

    dataset_profile = {
        "data_path": str(payload.get("data_path", "")).strip(),
        "data_mode": data_mode,
        "timestamp_column": timestamp_column,
        "missing_data_strategy": missing_strategy,
        "fill_constant_value": fill_constant_value,
    }
    split_strategy = (
        "blocked_time_order_70_15_15"
        if data_mode == "time_series"
        else "deterministic_modulo_70_15_15"
    )
    normalization = NormalizationPlan(
        enabled=True,
        method="minmax",
        feature_range=(0.0, 1.0),
        normalize_target=False,
        needs_user_confirmation=False,
        reason=(
            "Use min-max scaling fit on the training split only so saved inference artifacts "
            "can denormalize outputs consistently."
        ),
    )
    return Agent2Handoff(
        dataset_profile=dataset_profile,
        target_signal=target_signal,
        feature_signals=feature_signals,
        split_strategy=split_strategy,
        acceptance_criteria={"r2": 0.70},
        normalization=normalization,
        recommended_model_family=recommended_model,
        model_search_order=search_order,
        lag_horizon_samples=lag_horizon,
        forced_modeling_requests=_forced_requests_from_report(payload),
        user_notes=str((target_rec or {}).get("recommendation_statement", "")).strip(),
    )


def _primary_target_recommendation(payload: dict[str, Any]) -> dict[str, Any] | None:
    summary = payload.get("model_strategy_recommendations")
    if not isinstance(summary, dict):
        return None
    targets = summary.get("target_recommendations")
    if not isinstance(targets, list):
        return None
    for item in targets:
        if isinstance(item, dict):
            return item
    return None


def _fallback_target_from_report(payload: dict[str, Any]) -> str:
    correlations = payload.get("correlations")
    if isinstance(correlations, dict):
        for item in correlations.get("target_analyses", []):
            if isinstance(item, dict):
                target = str(item.get("target_signal", "")).strip()
                if target:
                    return target
    for item in payload.get("ranking", []):
        if isinstance(item, dict):
            target = str(item.get("target_signal", "")).strip()
            if target:
                return target
    return ""


def _fallback_features_from_report(payload: dict[str, Any], *, target_signal: str) -> list[str]:
    correlations = payload.get("correlations")
    features: list[str] = []
    if isinstance(correlations, dict):
        for item in correlations.get("target_analyses", []):
            if not isinstance(item, dict):
                continue
            if str(item.get("target_signal", "")).strip() != target_signal:
                continue
            predictor_rows = item.get("predictor_results")
            if isinstance(predictor_rows, list):
                for row in predictor_rows[:4]:
                    if not isinstance(row, dict):
                        continue
                    predictor = str(row.get("predictor_signal", "")).strip()
                    if predictor and predictor != target_signal and predictor not in features:
                        features.append(predictor)
            break
    return features


def _forced_requests_from_report(payload: dict[str, Any]) -> list[ForcedModelingRequest]:
    forced: list[ForcedModelingRequest] = []
    raw_forced = payload.get("forced_requests")
    if not isinstance(raw_forced, list):
        return forced
    for item in raw_forced:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target_signal", "")).strip()
        predictors = [
            str(value).strip()
            for value in item.get("predictor_signals", [])
            if str(value).strip()
        ]
        if not target or not predictors:
            continue
        forced.append(
            ForcedModelingRequest(
                target_signal=target,
                predictor_signals=predictors,
                force_run_regardless_of_correlation=True,
                reason=str(item.get("user_reason", item.get("reason", ""))).strip(),
            )
        )
    return forced


def _derive_lag_horizon_samples(target_rec: dict[str, Any]) -> int:
    if not isinstance(target_rec, dict):
        return 0
    if str(target_rec.get("recommended_model_family", "")).strip() != "lagged_linear":
        return 0
    for item in target_rec.get("candidate_models", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("model_family", "")).strip() != "lagged_linear":
            continue
        notes = str(item.get("notes", ""))
        match = re.search(r"up to\s+(\d+)\s+samples", notes, flags=re.IGNORECASE)
        if match:
            return max(1, int(match.group(1)))
    return 3
