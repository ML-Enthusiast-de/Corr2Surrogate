"""Dependency-aware ranking and user-forced modeling directives."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CandidateSignal:
    """A target signal candidate with baseline surrogateability score."""

    target_signal: str
    base_score: float
    required_signals: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(frozen=True)
class RankedSignal:
    """Final ranked signal including feasibility and dependency warnings."""

    target_signal: str
    base_score: float
    adjusted_score: float
    required_signals: list[str]
    blocked_virtual_dependencies: list[str]
    missing_physical_dependencies: list[str]
    feasible: bool
    rationale: str


@dataclass(frozen=True)
class ForcedModelingDirective:
    """A user-requested modeling task that bypasses correlation ranking gates."""

    target_signal: str
    predictor_signals: list[str]
    force_run_regardless_of_correlation: bool = True
    user_reason: str = ""


def rank_surrogate_candidates(
    *,
    candidates: list[CandidateSignal],
    physically_available_signals: list[str] | None = None,
    non_virtualizable_signals: list[str] | None = None,
    penalty_for_virtual_dependency: float = 0.25,
    enforce_physical_dependency: bool = True,
) -> list[RankedSignal]:
    """Rank candidates while accounting for virtual-dependency risk.

    A candidate is penalized when it requires other candidate targets as inputs.
    This prevents recommending virtualized signals that depend on each other
    without a stable physical anchor.
    """
    target_set = {c.target_signal for c in candidates}
    physical_constraints_provided = (
        physically_available_signals is not None
        or non_virtualizable_signals is not None
    )
    physical_set = set(physically_available_signals or [])
    non_virtualizable_set = set(non_virtualizable_signals or [])
    ranked: list[RankedSignal] = []

    for candidate in candidates:
        required = list(candidate.required_signals)
        blocked_virtual = sorted(set(required).intersection(target_set))
        physical_dependencies = sorted(
            set(required).intersection(physical_set.union(non_virtualizable_set))
        )
        missing_physical = (
            sorted(set(required) - set(physical_dependencies))
            if enforce_physical_dependency and required and physical_constraints_provided
            else []
        )

        adjusted = float(candidate.base_score)
        adjusted -= penalty_for_virtual_dependency * len(blocked_virtual)
        adjusted = max(0.0, min(1.0, adjusted))

        feasible = True
        reasons: list[str] = []

        if blocked_virtual:
            reasons.append(
                "Depends on signals also selected for virtualization: "
                + ", ".join(blocked_virtual)
            )
        if (
            enforce_physical_dependency
            and required
            and not physical_constraints_provided
        ):
            reasons.append(
                "Physical dependency constraints were not provided; "
                "feasibility is unverified."
            )
        if missing_physical:
            feasible = False
            reasons.append(
                "No stable physical dependency path for: " + ", ".join(missing_physical)
            )
        if not reasons:
            reasons.append("Feasible with currently known dependencies.")

        ranked.append(
            RankedSignal(
                target_signal=candidate.target_signal,
                base_score=candidate.base_score,
                adjusted_score=adjusted,
                required_signals=required,
                blocked_virtual_dependencies=blocked_virtual,
                missing_physical_dependencies=missing_physical,
                feasible=feasible,
                rationale=" ".join(reasons),
            )
        )

    ranked.sort(
        key=lambda item: (
            not item.feasible,
            -item.adjusted_score,
            item.target_signal.lower(),
        )
    )
    return ranked


def build_forced_directive(
    *,
    target_signal: str,
    predictor_signals: list[str],
    user_reason: str = "",
) -> ForcedModelingDirective:
    """Create a forced modeling directive from user-provided signal constraints."""
    if not predictor_signals:
        raise ValueError("predictor_signals cannot be empty for forced modeling.")
    return ForcedModelingDirective(
        target_signal=target_signal,
        predictor_signals=predictor_signals,
        force_run_regardless_of_correlation=True,
        user_reason=user_reason,
    )
