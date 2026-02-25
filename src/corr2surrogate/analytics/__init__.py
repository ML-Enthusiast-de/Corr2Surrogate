"""Analytics modules."""

from .correlations import (
    CorrelationAnalysisBundle,
    FeatureEngineeringOpportunity,
    PairCorrelationResult,
    TargetCorrelationAnalysis,
    build_candidate_signals_from_correlations,
    discover_feature_engineering_opportunities,
    run_correlation_analysis,
)
from .quality_checks import QualityCheckResult, run_quality_checks
from .ranking import (
    CandidateSignal,
    ForcedModelingDirective,
    RankedSignal,
    build_forced_directive,
    rank_surrogate_candidates,
)
from .reporting import build_agent1_report_payload, save_agent1_markdown_report
from .stationarity import StationaritySignalResult, StationaritySummary, assess_stationarity

__all__ = [
    "CandidateSignal",
    "CorrelationAnalysisBundle",
    "FeatureEngineeringOpportunity",
    "ForcedModelingDirective",
    "PairCorrelationResult",
    "QualityCheckResult",
    "RankedSignal",
    "StationaritySignalResult",
    "StationaritySummary",
    "TargetCorrelationAnalysis",
    "assess_stationarity",
    "build_agent1_report_payload",
    "build_candidate_signals_from_correlations",
    "build_forced_directive",
    "discover_feature_engineering_opportunities",
    "rank_surrogate_candidates",
    "run_correlation_analysis",
    "run_quality_checks",
    "save_agent1_markdown_report",
]
