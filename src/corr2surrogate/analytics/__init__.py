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
from .sensor_diagnostics import (
    SensorDiagnostic,
    SensorDiagnosticsSummary,
    run_sensor_diagnostics,
)
from .experiment_design import (
    ExperimentRecommendation,
    recommend_data_trajectories,
    recommendations_to_dict,
)
from .model_recommendation import (
    ModelStrategySummary,
    ProbeModelScore,
    TargetModelRecommendation,
    recommend_model_strategies,
)
from .ranking import (
    CandidateSignal,
    ForcedModelingDirective,
    RankedSignal,
    build_forced_directive,
    rank_surrogate_candidates,
)
from .reporting import (
    build_agent1_report_payload,
    save_agent1_artifacts,
    save_agent1_markdown_report,
)
from .stationarity import StationaritySignalResult, StationaritySummary, assess_stationarity
from .task_detection import (
    CLASSIFICATION_TASK_TYPES,
    SUPPORTED_TASK_TYPES,
    TaskProfile,
    assess_task_profile,
    assess_task_profiles,
    is_classification_task,
    normalize_task_type_hint,
)

__all__ = [
    "CandidateSignal",
    "CorrelationAnalysisBundle",
    "FeatureEngineeringOpportunity",
    "ForcedModelingDirective",
    "ModelStrategySummary",
    "PairCorrelationResult",
    "ProbeModelScore",
    "QualityCheckResult",
    "SensorDiagnostic",
    "SensorDiagnosticsSummary",
    "ExperimentRecommendation",
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
    "run_sensor_diagnostics",
    "recommend_data_trajectories",
    "recommendations_to_dict",
    "save_agent1_artifacts",
    "save_agent1_markdown_report",
    "TargetModelRecommendation",
    "TaskProfile",
    "recommend_model_strategies",
    "SUPPORTED_TASK_TYPES",
    "CLASSIFICATION_TASK_TYPES",
    "assess_task_profile",
    "assess_task_profiles",
    "is_classification_task",
    "normalize_task_type_hint",
]
