"""Default tool registration for the Corr2Surrogate harness."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from corr2surrogate.analytics import (
    assess_task_profiles,
    assess_stationarity,
    build_agent1_report_payload,
    build_candidate_signals_from_correlations,
    recommend_model_strategies,
    recommend_data_trajectories,
    recommendations_to_dict,
    run_correlation_analysis,
    run_quality_checks,
    run_sensor_diagnostics,
    save_agent1_artifacts,
    save_agent1_markdown_report,
)
from corr2surrogate.analytics.ranking import (
    ForcedModelingDirective,
    RankedSignal,
    build_forced_directive,
    rank_surrogate_candidates,
)
from corr2surrogate.ingestion import load_tabular_data
from corr2surrogate.modeling.baselines import IncrementalLinearSurrogate
from corr2surrogate.modeling.checkpoints import ModelCheckpointStore
from corr2surrogate.modeling.performance_feedback import analyze_model_performance
from corr2surrogate.modeling.training import train_surrogate_candidates
from corr2surrogate.persistence.artifact_store import ArtifactStore
from corr2surrogate.persistence.run_store import RunStore

from .tool_registry import ToolRegistry
from .workflow import (
    build_modeling_directives,
    evaluate_training_iteration,
    prepare_ingestion_step,
)


def build_default_registry() -> ToolRegistry:
    """Create the default tool registry for analyst/modeler loops."""
    registry = ToolRegistry()

    registry.register_function(
        name="prepare_ingestion_step",
        description="Load CSV/XLSX and return ingestion readiness status.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "sheet_name": {"type": "string"},
                "header_row": {"type": "integer"},
                "data_start_row": {"type": "integer"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        handler=_tool_prepare_ingestion_step,
        risk_level="low",
    )

    registry.register_function(
        name="run_agent1_analysis",
        description=(
            "Run full Agent 1 analysis: quality checks, stationarity, multi-technique "
            "correlations, feature opportunities, lightweight probe-model screening, "
            "model-family recommendations, and dependency-aware ranking."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "data_path": {"type": "string"},
                "sheet_name": {"type": "string"},
                "header_row": {"type": "integer"},
                "data_start_row": {"type": "integer"},
                "timestamp_column": {"type": "string"},
                "target_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "predictor_signals_by_target": {
                    "type": "object",
                    "additionalProperties": True,
                },
                "user_hypotheses": {"type": "array"},
                "feature_hypotheses": {"type": "array"},
                "forced_requests": {"type": "array"},
                "physically_available_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "non_virtualizable_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "max_lag": {"type": "integer"},
                "include_feature_engineering": {"type": "boolean"},
                "feature_gain_threshold": {"type": "number"},
                "top_k_predictors": {"type": "integer"},
                "feature_scan_predictors": {"type": "integer"},
                "max_feature_opportunities": {"type": "integer"},
                "confidence_top_k": {"type": "integer"},
                "bootstrap_rounds": {"type": "integer"},
                "stability_windows": {"type": "integer"},
                "task_type_hint": {"type": "string"},
                "max_samples": {"type": "integer"},
                "sample_selection": {"type": "string"},
                "missing_data_strategy": {"type": "string"},
                "fill_constant_value": {"type": "number"},
                "row_coverage_strategy": {"type": "string"},
                "sparse_row_min_fraction": {"type": "number"},
                "row_range_start": {"type": "integer"},
                "row_range_end": {"type": "integer"},
                "enable_strategy_search": {"type": "boolean"},
                "strategy_search_candidates": {"type": "integer"},
                "save_artifacts": {"type": "boolean"},
                "save_report": {"type": "boolean"},
                "run_id": {"type": "string"},
            },
            "required": ["data_path"],
            "additionalProperties": False,
        },
        handler=_tool_run_agent1_analysis,
        risk_level="low",
    )

    registry.register_function(
        name="evaluate_training_iteration",
        description="Evaluate one model attempt against acceptance criteria.",
        input_schema={
            "type": "object",
            "properties": {
                "metrics": {
                    "type": "object",
                    "additionalProperties": True,
                },
                "acceptance_criteria": {
                    "type": "object",
                    "additionalProperties": True,
                },
                "attempt": {"type": "integer"},
                "max_attempts": {"type": "integer"},
                "min_relative_improvement": {"type": "number"},
                "previous_best_score": {"type": "number"},
                "task_type_hint": {"type": "string"},
                "data_mode": {"type": "string"},
                "feature_columns": {"type": "array", "items": {"type": "string"}},
                "target_column": {"type": "string"},
                "lag_horizon_samples": {"type": "integer"},
            },
            "required": ["metrics", "acceptance_criteria", "attempt", "max_attempts"],
            "additionalProperties": False,
        },
        handler=_tool_evaluate_training_iteration,
        risk_level="low",
    )

    registry.register_function(
        name="build_modeling_directives",
        description=(
            "Merge ranked candidates and forced requests into actionable modeling directives."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "ranked_signals": {"type": "array"},
                "forced_requests": {"type": "array"},
            },
            "required": ["ranked_signals"],
            "additionalProperties": False,
        },
        handler=_tool_build_modeling_directives,
        risk_level="low",
    )

    registry.register_function(
        name="train_incremental_linear_surrogate",
        description=(
            "Train a baseline surrogate, create a savepoint, and persist model metadata."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "data_path": {"type": "string"},
                "target_column": {"type": "string"},
                "feature_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "sheet_name": {"type": "string"},
                "run_id": {"type": "string"},
                "checkpoint_tag": {"type": "string"},
            },
            "required": ["data_path", "target_column", "feature_columns"],
            "additionalProperties": False,
        },
        handler=_tool_train_incremental_linear_surrogate,
        risk_level="low",
    )

    registry.register_function(
        name="train_surrogate_candidates",
        description=(
            "Run split-safe train/validation/test training for regression or classification: "
            "linear/logistic baselines, lagged temporal baselines when applicable, and the first "
            "nonlinear tree baselines, compare them, and persist the selected model."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "data_path": {"type": "string"},
                "target_column": {"type": "string"},
                "feature_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "requested_model_family": {"type": "string"},
                "sheet_name": {"type": "string"},
                "timestamp_column": {"type": "string"},
                "run_id": {"type": "string"},
                "checkpoint_tag": {"type": "string"},
                "normalize": {"type": "boolean"},
                "missing_data_strategy": {"type": "string"},
                "fill_constant_value": {"type": "number"},
                "compare_against_baseline": {"type": "boolean"},
                "lag_horizon_samples": {"type": "integer"},
                "threshold_policy": {"type": "string"},
                "decision_threshold": {"type": "number"},
                "task_type_hint": {"type": "string"},
            },
            "required": ["data_path", "target_column", "feature_columns", "requested_model_family"],
            "additionalProperties": False,
        },
        handler=_tool_train_surrogate_candidates,
        risk_level="low",
    )

    registry.register_function(
        name="resume_incremental_linear_surrogate",
        description=(
            "Load a savepoint, add new data, retrain model statistics, and create child savepoint."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "checkpoint_id": {"type": "string"},
                "additional_data_path": {"type": "string"},
                "sheet_name": {"type": "string"},
                "run_id": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["checkpoint_id", "additional_data_path"],
            "additionalProperties": False,
        },
        handler=_tool_resume_incremental_linear_surrogate,
        risk_level="low",
    )

    registry.register_function(
        name="list_model_checkpoints",
        description="List available model savepoints/checkpoints.",
        input_schema={
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "target_column": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": [],
            "additionalProperties": False,
        },
        handler=_tool_list_model_checkpoints,
        risk_level="low",
    )

    registry.register_function(
        name="analyze_model_checkpoint_performance",
        description=(
            "Evaluate checkpoint performance on provided data and suggest new lab trajectories."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "checkpoint_id": {"type": "string"},
                "data_path": {"type": "string"},
                "sheet_name": {"type": "string"},
                "top_k_regions": {"type": "integer"},
                "trajectory_budget": {"type": "integer"},
            },
            "required": ["checkpoint_id", "data_path"],
            "additionalProperties": False,
        },
        handler=_tool_analyze_model_checkpoint_performance,
        risk_level="low",
    )

    return registry


def _tool_prepare_ingestion_step(
    *,
    path: str,
    sheet_name: str | None = None,
    header_row: int | None = None,
    data_start_row: int | None = None,
) -> dict[str, Any]:
    result = prepare_ingestion_step(
        path=path,
        sheet_name=sheet_name,
        header_row=header_row,
        data_start_row=data_start_row,
    )
    ingestion = result.ingestion_result
    inferred = ingestion.inferred_header if ingestion is not None else None
    available_sheets = list(ingestion.available_sheets) if ingestion is not None else list(
        result.options or []
    )
    signal_columns = list(ingestion.frame.columns) if ingestion is not None else []
    numeric_signal_columns = (
        _detect_numeric_signal_columns(ingestion.frame) if ingestion is not None else []
    )
    timestamp_hint = _infer_timestamp_column_hint(ingestion.frame) if ingestion is not None else None
    estimated_sample_period_seconds = (
        _estimate_sample_period_seconds(ingestion.frame, timestamp_hint)
        if ingestion is not None and timestamp_hint is not None
        else None
    )
    missing_overall_fraction = (
        float(ingestion.frame.isna().mean().mean()) if ingestion is not None else 0.0
    )
    missing_by_column = (
        ingestion.frame.isna().mean().sort_values(ascending=False) if ingestion is not None else None
    )
    columns_with_missing = (
        [str(col) for col, frac in missing_by_column.items() if float(frac) > 0.0][:30]
        if missing_by_column is not None
        else []
    )
    row_non_null_fraction = (
        ingestion.frame.notna().mean(axis=1) if ingestion is not None else pd.Series(dtype=float)
    )
    row_non_null_min = (
        float(row_non_null_fraction.min()) if len(row_non_null_fraction) > 0 else 1.0
    )
    row_non_null_median = (
        float(row_non_null_fraction.median()) if len(row_non_null_fraction) > 0 else 1.0
    )
    row_non_null_max = (
        float(row_non_null_fraction.max()) if len(row_non_null_fraction) > 0 else 1.0
    )
    potential_length_mismatch = bool(row_non_null_min < 0.999 and row_non_null_max > row_non_null_min)
    payload = {
        "status": result.status,
        "message": result.message,
        "options": result.options or [],
        "selected_sheet": ingestion.selected_sheet if ingestion is not None else None,
        "available_sheets": available_sheets,
        "row_count": int(len(ingestion.frame)) if ingestion is not None else 0,
        "column_count": int(len(signal_columns)) if ingestion is not None else 0,
        "signal_columns": signal_columns,
        "numeric_signal_columns": numeric_signal_columns,
        "timestamp_column_hint": timestamp_hint,
        "estimated_sample_period_seconds": estimated_sample_period_seconds,
        "missing_overall_fraction": missing_overall_fraction,
        "columns_with_missing": columns_with_missing,
        "columns_with_missing_count": int(len(columns_with_missing)),
        "row_non_null_fraction_min": row_non_null_min,
        "row_non_null_fraction_median": row_non_null_median,
        "row_non_null_fraction_max": row_non_null_max,
        "potential_length_mismatch": potential_length_mismatch,
        "header_row": inferred.header_row if inferred is not None else None,
        "data_start_row": inferred.data_start_row if inferred is not None else None,
        "header_confidence": inferred.confidence if inferred is not None else None,
        "candidate_header_rows": inferred.candidate_rows if inferred is not None else [],
        "needs_user_confirmation": (
            inferred.needs_user_confirmation if inferred is not None else False
        ),
    }
    return payload


def _tool_evaluate_training_iteration(
    *,
    metrics: dict[str, float],
    acceptance_criteria: dict[str, float],
    attempt: int,
    max_attempts: int,
    min_relative_improvement: float = 0.02,
    previous_best_score: float | None = None,
    task_type_hint: str | None = None,
    data_mode: str | None = None,
    feature_columns: list[str] | None = None,
    target_column: str | None = None,
    lag_horizon_samples: int | None = None,
) -> dict[str, Any]:
    result = evaluate_training_iteration(
        metrics=metrics,
        acceptance_criteria=acceptance_criteria,
        attempt=attempt,
        max_attempts=max_attempts,
        min_relative_improvement=min_relative_improvement,
        previous_best_score=previous_best_score,
        task_type_hint=task_type_hint,
        data_mode=data_mode,
        feature_columns=feature_columns,
        target_column=target_column,
        lag_horizon_samples=lag_horizon_samples,
    )
    return asdict(result)


def _tool_run_agent1_analysis(
    *,
    data_path: str,
    sheet_name: str | None = None,
    header_row: int | None = None,
    data_start_row: int | None = None,
    timestamp_column: str | None = None,
    target_signals: list[str] | None = None,
    predictor_signals_by_target: dict[str, list[str]] | None = None,
    user_hypotheses: list[dict[str, Any]] | None = None,
    feature_hypotheses: list[dict[str, Any]] | None = None,
    forced_requests: list[dict[str, Any]] | None = None,
    physically_available_signals: list[str] | None = None,
    non_virtualizable_signals: list[str] | None = None,
    max_lag: int = 8,
    include_feature_engineering: bool = True,
    feature_gain_threshold: float = 0.05,
    top_k_predictors: int = 10,
    feature_scan_predictors: int = 10,
    max_feature_opportunities: int = 20,
    confidence_top_k: int = 10,
    bootstrap_rounds: int = 40,
    stability_windows: int = 4,
    task_type_hint: str | None = None,
    max_samples: int | None = None,
    sample_selection: str = "uniform",
    missing_data_strategy: str = "keep",
    fill_constant_value: float | None = None,
    row_coverage_strategy: str = "keep",
    sparse_row_min_fraction: float = 0.8,
    row_range_start: int | None = None,
    row_range_end: int | None = None,
    enable_strategy_search: bool = True,
    strategy_search_candidates: int = 4,
    save_artifacts: bool = True,
    save_report: bool = True,
    run_id: str | None = None,
) -> dict[str, Any]:
    loaded = load_tabular_data(
        path=data_path,
        sheet_name=sheet_name,
        header_row=header_row,
        data_start_row=data_start_row,
    )
    source_frame = loaded.frame

    user_plan = {
        "max_samples": max_samples,
        "sample_selection": sample_selection,
        "missing_data_strategy": missing_data_strategy,
        "fill_constant_value": fill_constant_value,
        "row_coverage_strategy": row_coverage_strategy,
        "sparse_row_min_fraction": sparse_row_min_fraction,
        "row_range_start": row_range_start,
        "row_range_end": row_range_end,
    }
    planner_trace: list[dict[str, Any]] = []
    critic_decision: dict[str, Any] = {}
    normalized_user_hypotheses = _normalize_user_hypotheses(user_hypotheses)
    normalized_feature_hypotheses = _normalize_feature_hypotheses(feature_hypotheses)
    resolved_targets, resolved_predictor_map = _merge_hypotheses_into_analysis_scope(
        target_signals=target_signals,
        predictor_signals_by_target=predictor_signals_by_target,
        user_hypotheses=normalized_user_hypotheses,
    )

    if enable_strategy_search:
        candidate_plans = _planner_generate_strategy_candidates(
            user_plan=user_plan,
            max_candidates=max(1, int(strategy_search_candidates)),
        )
        planner_trace = _planner_evaluate_candidates(
            frame=source_frame,
            candidate_plans=candidate_plans,
            timestamp_column=timestamp_column,
            target_signals=resolved_targets,
            predictor_signals_by_target=resolved_predictor_map,
            max_lag=max_lag,
        )
        chosen = _critic_choose_strategy(planner_trace)
        planner_recommended_id = str(chosen.get("selected_candidate_id", "user_plan"))
        selected_plan = dict(user_plan)
        critic_decision = {
            "selected_candidate_id": "user_plan",
            "planner_recommended_candidate_id": planner_recommended_id,
            "rationale": (
                "User-configured preprocessing is locked and always applied. "
                f"Planner recommendation `{planner_recommended_id}` was advisory only."
            ),
            "margin": chosen.get("margin"),
        }
    else:
        selected_plan = user_plan
        planner_trace = []
        critic_decision = {"selected_candidate_id": "user_plan", "rationale": "strategy_search_disabled"}

    frame, preprocessing = _apply_preprocessing_plan(
        frame=source_frame,
        max_samples=selected_plan.get("max_samples"),
        sample_selection=str(selected_plan.get("sample_selection", "uniform")),
        missing_data_strategy=str(selected_plan.get("missing_data_strategy", "keep")),
        fill_constant_value=selected_plan.get("fill_constant_value"),
        row_coverage_strategy=str(selected_plan.get("row_coverage_strategy", "keep")),
        sparse_row_min_fraction=float(selected_plan.get("sparse_row_min_fraction", 0.8)),
        row_range_start=selected_plan.get("row_range_start"),
        row_range_end=selected_plan.get("row_range_end"),
    )

    quality = run_quality_checks(frame, timestamp_column=timestamp_column)
    diagnostics = run_sensor_diagnostics(frame, timestamp_column=timestamp_column)
    correlations = run_correlation_analysis(
        frame=frame,
        target_signals=resolved_targets,
        predictor_signals_by_target=resolved_predictor_map,
        timestamp_column=timestamp_column,
        max_lag=max_lag,
        include_feature_engineering=include_feature_engineering,
        feature_gain_threshold=feature_gain_threshold,
        top_k_predictors=top_k_predictors,
        feature_scan_predictors=feature_scan_predictors,
        max_feature_opportunities=max_feature_opportunities,
        confidence_top_k=confidence_top_k,
        bootstrap_rounds=bootstrap_rounds,
        stability_windows=stability_windows,
        correlation_hypotheses=normalized_user_hypotheses,
        feature_hypotheses=normalized_feature_hypotheses,
    )
    stationarity_columns = [item.target_signal for item in correlations.target_analyses]
    stationarity = assess_stationarity(frame, signal_columns=stationarity_columns)
    task_profiles = assess_task_profiles(
        frame=frame,
        target_columns=stationarity_columns,
        data_mode=correlations.data_mode,
        task_type_hint=task_type_hint,
    )
    model_strategy = recommend_model_strategies(
        frame=frame,
        correlations=correlations,
        max_lag=max_lag,
    )
    recommendations = recommend_data_trajectories(
        frame=frame,
        correlations=correlations.to_dict(),
        sensor_diagnostics=diagnostics.to_dict(),
    )

    candidates = build_candidate_signals_from_correlations(correlations)
    ranking = rank_surrogate_candidates(
        candidates=candidates,
        physically_available_signals=physically_available_signals,
        non_virtualizable_signals=non_virtualizable_signals,
    )

    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("agent1_%Y%m%d_%H%M%S")
    dataset_slug = _dataset_slug_from_path(data_path)
    forced_directives = _normalize_forced_requests(forced_requests)
    lineage_store = RunStore(base_dir="reports")
    lineage_path = lineage_store.save_lineage(
        dataset_slug=dataset_slug,
        run_id=resolved_run_id,
        data_path=data_path,
        analysis_config={
            "timestamp_column": timestamp_column,
            "target_signals": resolved_targets,
            "predictor_signals_by_target": resolved_predictor_map,
            "user_hypotheses": normalized_user_hypotheses,
            "feature_hypotheses": normalized_feature_hypotheses,
            "max_lag": max_lag,
            "include_feature_engineering": include_feature_engineering,
            "feature_gain_threshold": feature_gain_threshold,
            "top_k_predictors": top_k_predictors,
            "feature_scan_predictors": feature_scan_predictors,
            "max_feature_opportunities": max_feature_opportunities,
            "confidence_top_k": confidence_top_k,
            "bootstrap_rounds": bootstrap_rounds,
            "stability_windows": stability_windows,
            "task_type_hint": task_type_hint,
            "enable_strategy_search": enable_strategy_search,
            "strategy_search_candidates": strategy_search_candidates,
        },
        selected_strategy=selected_plan,
        planner_trace=planner_trace,
        critic_decision=critic_decision,
    )

    report_payload = build_agent1_report_payload(
        data_path=data_path,
        quality=quality,
        stationarity=stationarity,
        correlations=correlations,
        ranking=ranking,
        forced_requests=[asdict(item) for item in forced_directives],
        preprocessing=preprocessing,
        sensor_diagnostics=diagnostics.to_dict(),
        model_strategy_recommendations=model_strategy.to_dict(),
        task_profiles=[item.to_dict() for item in task_profiles],
        experiment_recommendations=recommendations_to_dict(recommendations),
        planner_trace=planner_trace,
        critic_decision=critic_decision,
        lineage_path=lineage_path,
        user_hypotheses={
            "correlation_hypotheses": normalized_user_hypotheses,
            "feature_hypotheses": normalized_feature_hypotheses,
        },
    )

    artifact_paths: dict[str, Any] | None = None
    if save_artifacts:
        artifact_paths = save_agent1_artifacts(
            structured=report_payload["structured"],
            data_path=data_path,
            run_id=resolved_run_id,
        )

    report_payload = build_agent1_report_payload(
        data_path=data_path,
        quality=quality,
        stationarity=stationarity,
        correlations=correlations,
        ranking=ranking,
        forced_requests=[asdict(item) for item in forced_directives],
        preprocessing=preprocessing,
        sensor_diagnostics=diagnostics.to_dict(),
        model_strategy_recommendations=model_strategy.to_dict(),
        task_profiles=[item.to_dict() for item in task_profiles],
        experiment_recommendations=recommendations_to_dict(recommendations),
        planner_trace=planner_trace,
        critic_decision=critic_decision,
        lineage_path=lineage_path,
        artifact_paths=artifact_paths or {},
        user_hypotheses={
            "correlation_hypotheses": normalized_user_hypotheses,
            "feature_hypotheses": normalized_feature_hypotheses,
        },
    )
    report_path: str | None = None
    if save_report:
        report_path = save_agent1_markdown_report(
            markdown=report_payload["markdown"],
            data_path=data_path,
            run_id=resolved_run_id,
        )

    return {
        "status": "ok",
        "data_mode": correlations.data_mode,
        "timestamp_column": correlations.timestamp_column,
        "target_count": len(correlations.target_analyses),
        "candidate_count": len(candidates),
        "ranking": [asdict(item) for item in ranking],
        "forced_requests": [asdict(item) for item in forced_directives],
        "user_hypotheses": normalized_user_hypotheses,
        "feature_hypotheses": normalized_feature_hypotheses,
        "quality": quality.to_dict(),
        "stationarity": stationarity.to_dict(),
        "correlations": correlations.to_dict(),
        "sensor_diagnostics": diagnostics.to_dict(),
        "model_strategy_recommendations": model_strategy.to_dict(),
        "task_profiles": [item.to_dict() for item in task_profiles],
        "experiment_recommendations": recommendations_to_dict(recommendations),
        "planner_trace": planner_trace,
        "critic_decision": critic_decision,
        "lineage_path": lineage_path,
        "artifact_paths": artifact_paths or {},
        "preprocessing": preprocessing,
        "report_path": report_path,
        "report_markdown": report_payload["markdown"],
    }


def _tool_build_modeling_directives(
    *,
    ranked_signals: list[dict[str, Any]],
    forced_requests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ranked = [RankedSignal(**item) for item in ranked_signals]
    forced = [ForcedModelingDirective(**item) for item in (forced_requests or [])]
    directives = build_modeling_directives(ranked_signals=ranked, forced_requests=forced)
    return {"directives": [asdict(item) for item in directives]}


def _tool_train_incremental_linear_surrogate(
    *,
    data_path: str,
    target_column: str,
    feature_columns: list[str],
    sheet_name: str | None = None,
    run_id: str | None = None,
    checkpoint_tag: str | None = None,
) -> dict[str, Any]:
    frame = _load_frame(data_path=data_path, sheet_name=sheet_name)
    model = IncrementalLinearSurrogate(
        feature_columns=feature_columns,
        target_column=target_column,
    )
    rows_used = model.fit_dataframe(frame)
    metrics = model.evaluate_dataframe(frame)

    artifact_store = ArtifactStore()
    run_dir = artifact_store.create_run_dir(run_id=run_id)
    model_state_path = model.save(Path(run_dir) / "model_state.json")
    params_path = artifact_store.save_model_params(
        run_dir=run_dir,
        model_name="incremental_linear_surrogate",
        best_params={"ridge": model.ridge, "training_rows_used": rows_used},
        metrics=metrics,
        feature_columns=feature_columns,
        target_column=target_column,
        split_strategy="full_dataset_baseline",
        extra={"data_path": data_path},
    )

    checkpoint_store = ModelCheckpointStore()
    checkpoint = checkpoint_store.create_checkpoint(
        model_name="incremental_linear_surrogate",
        run_dir=run_dir,
        model_state_path=model_state_path,
        target_column=target_column,
        feature_columns=feature_columns,
        metrics=metrics,
        data_references=[data_path],
        notes=checkpoint_tag or "",
        tags=[checkpoint_tag] if checkpoint_tag else [],
    )
    return {
        "status": "ok",
        "checkpoint_id": checkpoint.checkpoint_id,
        "run_dir": str(run_dir),
        "model_state_path": str(model_state_path),
        "model_params_path": str(params_path),
        "metrics": metrics,
        "rows_used": rows_used,
    }


def _tool_train_surrogate_candidates(
    *,
    data_path: str,
    target_column: str,
    feature_columns: list[str],
    requested_model_family: str,
    sheet_name: str | None = None,
    timestamp_column: str | None = None,
    run_id: str | None = None,
    checkpoint_tag: str | None = None,
    normalize: bool = True,
    missing_data_strategy: str = "fill_median",
    fill_constant_value: float | None = None,
    compare_against_baseline: bool = True,
    lag_horizon_samples: int | None = None,
    threshold_policy: str | None = None,
    decision_threshold: float | None = None,
    task_type_hint: str | None = None,
) -> dict[str, Any]:
    frame = _load_frame(data_path=data_path, sheet_name=sheet_name)
    result = train_surrogate_candidates(
        frame=frame,
        target_column=target_column,
        feature_columns=feature_columns,
        requested_model_family=requested_model_family,
        timestamp_column=timestamp_column,
        normalize=normalize,
        missing_data_strategy=missing_data_strategy,
        fill_constant_value=fill_constant_value,
        compare_against_baseline=compare_against_baseline,
        lag_horizon_samples=lag_horizon_samples,
        threshold_policy=threshold_policy,
        decision_threshold=decision_threshold,
        task_type=task_type_hint,
        run_id=run_id,
        checkpoint_tag=checkpoint_tag,
        data_references=[data_path],
    )
    return result


def _tool_resume_incremental_linear_surrogate(
    *,
    checkpoint_id: str,
    additional_data_path: str,
    sheet_name: str | None = None,
    run_id: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    checkpoint_store = ModelCheckpointStore()
    parent = checkpoint_store.load_checkpoint(checkpoint_id)
    model = IncrementalLinearSurrogate.load(parent.model_state_path)

    frame = _load_frame(data_path=additional_data_path, sheet_name=sheet_name)
    rows_added = model.update_from_dataframe(frame)
    metrics_new_data = model.evaluate_dataframe(frame)

    artifact_store = ArtifactStore()
    run_dir = artifact_store.create_run_dir(run_id=run_id)
    model_state_path = model.save(Path(run_dir) / "model_state.json")
    params_path = artifact_store.save_model_params(
        run_dir=run_dir,
        model_name=parent.model_name,
        best_params={
            "ridge": model.ridge,
            "parent_checkpoint_id": parent.checkpoint_id,
            "rows_added": rows_added,
        },
        metrics=metrics_new_data,
        feature_columns=parent.feature_columns,
        target_column=parent.target_column,
        split_strategy="incremental_retrain",
        extra={"additional_data_path": additional_data_path},
    )

    child = checkpoint_store.create_checkpoint(
        model_name=parent.model_name,
        run_dir=run_dir,
        model_state_path=model_state_path,
        target_column=parent.target_column,
        feature_columns=parent.feature_columns,
        metrics=metrics_new_data,
        parent_checkpoint_id=parent.checkpoint_id,
        data_references=parent.data_references + [additional_data_path],
        notes=note or "",
        tags=["retrain"],
    )

    plan = checkpoint_store.build_retrain_plan(
        checkpoint_id=parent.checkpoint_id,
        additional_data_references=[additional_data_path],
        notes=note or "",
    )
    return {
        "status": "ok",
        "parent_checkpoint_id": parent.checkpoint_id,
        "new_checkpoint_id": child.checkpoint_id,
        "run_dir": str(run_dir),
        "model_state_path": str(model_state_path),
        "model_params_path": str(params_path),
        "rows_added": rows_added,
        "metrics_on_additional_data": metrics_new_data,
        "retrain_plan": plan.to_dict(),
    }


def _tool_list_model_checkpoints(
    *,
    model_name: str | None = None,
    target_column: str | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    checkpoint_store = ModelCheckpointStore()
    items = checkpoint_store.list_checkpoints(
        model_name=model_name,
        target_column=target_column,
        limit=limit,
    )
    return {"checkpoints": [item.to_dict() for item in items]}


def _tool_analyze_model_checkpoint_performance(
    *,
    checkpoint_id: str,
    data_path: str,
    sheet_name: str | None = None,
    top_k_regions: int = 3,
    trajectory_budget: int = 3,
) -> dict[str, Any]:
    checkpoint_store = ModelCheckpointStore()
    checkpoint = checkpoint_store.load_checkpoint(checkpoint_id)
    model = IncrementalLinearSurrogate.load(checkpoint.model_state_path)
    frame = _load_frame(data_path=data_path, sheet_name=sheet_name)
    predictions = model.predict_dataframe(frame)
    feedback = analyze_model_performance(
        y_true=frame[checkpoint.target_column].to_numpy(dtype=float),
        y_pred=predictions,
        feature_frame=frame[checkpoint.feature_columns],
        top_k_regions=top_k_regions,
        trajectory_budget=trajectory_budget,
    )
    return {
        "status": "ok",
        "checkpoint_id": checkpoint.checkpoint_id,
        "feedback": feedback.to_dict(),
    }


def _load_frame(*, data_path: str, sheet_name: str | None) -> Any:
    loaded = load_tabular_data(path=data_path, sheet_name=sheet_name)
    return loaded.frame


def _normalize_forced_requests(
    forced_requests: list[dict[str, Any]] | None,
) -> list[ForcedModelingDirective]:
    directives: list[ForcedModelingDirective] = []
    for item in forced_requests or []:
        target = item.get("target_signal")
        predictors = item.get("predictor_signals")
        reason = item.get("user_reason", "")
        if not isinstance(target, str) or not isinstance(predictors, list):
            continue
        predictor_signals = [str(value) for value in predictors if str(value).strip()]
        if not predictor_signals:
            continue
        directives.append(
            build_forced_directive(
                target_signal=target,
                predictor_signals=predictor_signals,
                user_reason=str(reason),
            )
        )
    return directives


def _normalize_user_hypotheses(
    hypotheses: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in hypotheses or []:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target_signal", "")).strip()
        raw_predictors = item.get("predictor_signals")
        if not isinstance(raw_predictors, list):
            single = item.get("predictor_signal")
            raw_predictors = [single] if isinstance(single, str) else []
        predictors = [str(value).strip() for value in raw_predictors if str(value).strip()]
        if not target or not predictors:
            continue
        normalized.append(
            {
                "target_signal": target,
                "predictor_signals": predictors,
                "user_reason": str(
                    item.get("user_reason", item.get("reason", "user hypothesis"))
                ),
            }
        )
    return normalized


def _normalize_feature_hypotheses(
    hypotheses: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in hypotheses or []:
        if not isinstance(item, dict):
            continue
        base_signal = str(item.get("base_signal", "")).strip()
        transformation = str(item.get("transformation", "")).strip().lower()
        if not base_signal or not transformation:
            continue
        target_signal = str(item.get("target_signal", "")).strip()
        normalized.append(
            {
                "target_signal": target_signal,
                "base_signal": base_signal,
                "transformation": transformation,
                "user_reason": str(
                    item.get("user_reason", item.get("reason", "user hypothesis"))
                ),
            }
        )
    return normalized


def _merge_hypotheses_into_analysis_scope(
    *,
    target_signals: list[str] | None,
    predictor_signals_by_target: dict[str, list[str]] | None,
    user_hypotheses: list[dict[str, Any]],
) -> tuple[list[str] | None, dict[str, list[str]] | None]:
    resolved_targets = list(target_signals) if isinstance(target_signals, list) else None
    resolved_predictor_map: dict[str, list[str]] | None = None
    if isinstance(predictor_signals_by_target, dict):
        resolved_predictor_map = {
            str(key): [str(item) for item in value]
            for key, value in predictor_signals_by_target.items()
            if isinstance(value, list)
        }

    for item in user_hypotheses:
        target = str(item.get("target_signal", "")).strip()
        predictors = [
            str(value).strip() for value in item.get("predictor_signals", []) if str(value).strip()
        ]
        if not target or not predictors:
            continue
        if resolved_targets is not None and target not in resolved_targets:
            resolved_targets.append(target)
        if resolved_predictor_map is None:
            resolved_predictor_map = {}
        current = list(resolved_predictor_map.get(target, []))
        seen = set(current)
        for predictor in predictors:
            if predictor not in seen and predictor != target:
                current.append(predictor)
                seen.add(predictor)
        resolved_predictor_map[target] = current

    return resolved_targets, resolved_predictor_map


def _detect_numeric_signal_columns(frame: Any) -> list[str]:
    numeric_signals: list[str] = []
    for col in frame.columns:
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if int(numeric.notna().sum()) >= 8 and int(numeric.nunique(dropna=True)) > 1:
            numeric_signals.append(str(col))
    return numeric_signals


def _infer_timestamp_column_hint(frame: pd.DataFrame) -> str | None:
    candidates = [
        str(col)
        for col in frame.columns
        if any(token in str(col).lower() for token in ("time", "timestamp", "date"))
    ]
    for col in candidates:
        series = frame[col]
        parsed_dt = pd.to_datetime(series, errors="coerce")
        if float(parsed_dt.notna().mean()) >= 0.8 and bool(parsed_dt.dropna().is_monotonic_increasing):
            return col
        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.dropna()
        if len(valid) < 8:
            continue
        diffs = valid.diff().dropna()
        positive = diffs[diffs > 0]
        if len(positive) >= 5:
            return col
    return None


def _estimate_sample_period_seconds(frame: pd.DataFrame, timestamp_column: str) -> float | None:
    if timestamp_column not in frame.columns:
        return None
    series = frame[timestamp_column]
    parsed_dt = pd.to_datetime(series, errors="coerce")
    if float(parsed_dt.notna().mean()) >= 0.8:
        diffs = parsed_dt.dropna().diff().dt.total_seconds().dropna()
        diffs = diffs[diffs > 0]
        if len(diffs) >= 5:
            return float(diffs.median())

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 8:
        return None
    diffs_num = numeric.diff().dropna()
    diffs_num = diffs_num[diffs_num > 0]
    if len(diffs_num) < 5:
        return None
    median_step = float(diffs_num.median())
    name = timestamp_column.lower()
    if "ms" in name:
        return median_step / 1000.0
    if "us" in name:
        return median_step / 1_000_000.0
    if "ns" in name:
        return median_step / 1_000_000_000.0
    return median_step


def _apply_preprocessing_plan(
    *,
    frame: pd.DataFrame,
    max_samples: int | None,
    sample_selection: str,
    missing_data_strategy: str,
    fill_constant_value: float | None,
    row_coverage_strategy: str,
    sparse_row_min_fraction: float,
    row_range_start: int | None,
    row_range_end: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared = frame.copy()
    initial_rows = int(len(prepared))
    metadata: dict[str, Any] = {
        "initial_rows": initial_rows,
        "sample_plan": {
            "requested_max_samples": int(max_samples) if max_samples is not None else None,
            "selection": sample_selection,
            "applied": False,
            "rows_after": initial_rows,
        },
        "missing_data_plan": {
            "strategy": missing_data_strategy,
            "fill_constant_value": fill_constant_value,
            "applied": False,
            "rows_after": initial_rows,
            "split_leakage_risk": "none",
            "split_leakage_note": "",
            "recommended_split_safe_policy": "",
        },
        "row_coverage_plan": {
            "strategy": row_coverage_strategy,
            "sparse_row_min_fraction": float(sparse_row_min_fraction),
            "row_range_start": row_range_start,
            "row_range_end": row_range_end,
            "applied": False,
            "rows_after": initial_rows,
        },
    }

    if max_samples is not None and int(max_samples) > 0 and int(max_samples) < len(prepared):
        count = int(max_samples)
        method = sample_selection.strip().lower()
        if method == "head":
            prepared = prepared.head(count).copy()
        elif method == "tail":
            prepared = prepared.tail(count).copy()
        else:
            indices = np.linspace(0, len(prepared) - 1, count, dtype=int)
            prepared = prepared.iloc[indices].copy()
            method = "uniform"
        metadata["sample_plan"]["applied"] = True
        metadata["sample_plan"]["selection"] = method
        metadata["sample_plan"]["rows_after"] = int(len(prepared))

    missing_strategy = missing_data_strategy.strip().lower()
    if missing_strategy == "drop_rows":
        prepared = prepared.dropna(axis=0).reset_index(drop=True)
        metadata["missing_data_plan"]["applied"] = True
    elif missing_strategy in {"fill_median", "median"}:
        numeric_cols = [
            col
            for col in prepared.columns
            if pd.to_numeric(prepared[col], errors="coerce").notna().sum() >= 3
        ]
        for col in numeric_cols:
            numeric = pd.to_numeric(prepared[col], errors="coerce")
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            prepared[col] = numeric.fillna(median)
        metadata["missing_data_plan"]["applied"] = True
        metadata["missing_data_plan"]["strategy"] = "fill_median"
    elif missing_strategy in {"fill_constant", "constant"}:
        fill_value = 0.0 if fill_constant_value is None else float(fill_constant_value)
        numeric_cols = [
            col
            for col in prepared.columns
            if pd.to_numeric(prepared[col], errors="coerce").notna().sum() >= 3
        ]
        for col in numeric_cols:
            numeric = pd.to_numeric(prepared[col], errors="coerce")
            prepared[col] = numeric.fillna(fill_value)
        metadata["missing_data_plan"]["applied"] = True
        metadata["missing_data_plan"]["strategy"] = "fill_constant"
        metadata["missing_data_plan"]["fill_constant_value"] = fill_value
    metadata["missing_data_plan"]["rows_after"] = int(len(prepared))
    metadata["missing_data_plan"].update(
        _missing_data_split_leakage_info(
            strategy=str(metadata["missing_data_plan"].get("strategy", "keep")),
            fill_constant_value=metadata["missing_data_plan"].get("fill_constant_value"),
        )
    )

    coverage_strategy = row_coverage_strategy.strip().lower()
    if coverage_strategy in {"drop_sparse_rows", "drop_sparse"}:
        row_cov = prepared.notna().mean(axis=1)
        prepared = prepared.loc[row_cov >= float(sparse_row_min_fraction)].reset_index(drop=True)
        metadata["row_coverage_plan"]["applied"] = True
        metadata["row_coverage_plan"]["strategy"] = "drop_sparse_rows"
    elif coverage_strategy in {"trim_dense_window", "trim"}:
        row_cov = prepared.notna().mean(axis=1)
        keep = np.where(row_cov.to_numpy(dtype=float) >= float(sparse_row_min_fraction))[0]
        if len(keep) > 0:
            prepared = prepared.iloc[int(keep[0]) : int(keep[-1]) + 1].reset_index(drop=True)
            metadata["row_coverage_plan"]["applied"] = True
            metadata["row_coverage_plan"]["strategy"] = "trim_dense_window"
    elif coverage_strategy in {"manual_range", "range"}:
        start = 0 if row_range_start is None else max(0, int(row_range_start))
        end = len(prepared) - 1 if row_range_end is None else min(len(prepared) - 1, int(row_range_end))
        if end < start:
            raise ValueError(
                f"Invalid row range [{start}, {end}] after bounds check; end must be >= start."
            )
        prepared = prepared.iloc[start : end + 1].reset_index(drop=True)
        metadata["row_coverage_plan"]["applied"] = True
        metadata["row_coverage_plan"]["strategy"] = "manual_range"
        metadata["row_coverage_plan"]["row_range_start"] = start
        metadata["row_coverage_plan"]["row_range_end"] = end
    metadata["row_coverage_plan"]["rows_after"] = int(len(prepared))

    if len(prepared) == 0:
        raise ValueError(
            "Preprocessing removed all rows. Relax filtering/range settings and retry."
        )
    metadata["final_rows"] = int(len(prepared))
    return prepared, metadata


def _missing_data_split_leakage_info(
    *,
    strategy: str,
    fill_constant_value: Any,
) -> dict[str, str]:
    normalized = strategy.strip().lower()
    split_policy = (
        "For model training/evaluation: split first, fit missing-data handling on train only, "
        "then apply unchanged transform to validation/test."
    )
    if normalized in {"fill_median", "median"}:
        return {
            "split_leakage_risk": "high",
            "split_leakage_note": (
                "fill_median estimates statistics from the full analyzed dataset in Agent 1. "
                "Reusing this preprocessing before train/validation/test split leaks "
                "validation/test distribution information."
            ),
            "recommended_split_safe_policy": split_policy,
        }
    if normalized in {"fill_constant", "constant"}:
        return {
            "split_leakage_risk": "low",
            "split_leakage_note": (
                "fill_constant is leakage-safe only when the constant is fixed a priori "
                f"(configured value: {fill_constant_value}). "
                "If the value is tuned using full data, it still introduces leakage."
            ),
            "recommended_split_safe_policy": split_policy,
        }
    if normalized == "drop_rows":
        return {
            "split_leakage_risk": "low",
            "split_leakage_note": (
                "drop_rows does not use global statistics, but dropping on the full dataset can "
                "still bias split distributions. Apply row-dropping consistently per split."
            ),
            "recommended_split_safe_policy": split_policy,
        }
    return {
        "split_leakage_risk": "none",
        "split_leakage_note": (
            "No imputation statistics are learned in this step (`keep`)."
        ),
        "recommended_split_safe_policy": split_policy,
    }


def _planner_generate_strategy_candidates(
    *,
    user_plan: dict[str, Any],
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    base = dict(user_plan)
    candidates.append({"candidate_id": "user_plan", "plan": base, "source": "user"})

    # Controlled alternatives for critic scoring.
    variants = [
        {
            "candidate_id": "fill_median_trim",
            "plan": {
                **base,
                "missing_data_strategy": "fill_median",
                "row_coverage_strategy": "trim_dense_window",
                "sparse_row_min_fraction": max(0.7, float(base.get("sparse_row_min_fraction", 0.8))),
            },
            "source": "planner",
        },
        {
            "candidate_id": "drop_rows_drop_sparse",
            "plan": {
                **base,
                "missing_data_strategy": "drop_rows",
                "row_coverage_strategy": "drop_sparse_rows",
                "sparse_row_min_fraction": max(0.75, float(base.get("sparse_row_min_fraction", 0.8))),
            },
            "source": "planner",
        },
        {
            "candidate_id": "fill_constant_keep",
            "plan": {
                **base,
                "missing_data_strategy": "fill_constant",
                "fill_constant_value": 0.0 if base.get("fill_constant_value") is None else base.get("fill_constant_value"),
                "row_coverage_strategy": "keep",
            },
            "source": "planner",
        },
        {
            "candidate_id": "keep_keep",
            "plan": {
                **base,
                "missing_data_strategy": "keep",
                "row_coverage_strategy": "keep",
            },
            "source": "planner",
        },
    ]
    for item in variants:
        if len(candidates) >= max_candidates:
            break
        if not any(c["candidate_id"] == item["candidate_id"] for c in candidates):
            candidates.append(item)
    return candidates[:max_candidates]


def _planner_evaluate_candidates(
    *,
    frame: pd.DataFrame,
    candidate_plans: list[dict[str, Any]],
    timestamp_column: str | None,
    target_signals: list[str] | None,
    predictor_signals_by_target: dict[str, list[str]] | None,
    max_lag: int,
) -> list[dict[str, Any]]:
    trace: list[dict[str, Any]] = []
    for candidate in candidate_plans:
        candidate_id = str(candidate.get("candidate_id", "candidate"))
        plan = dict(candidate.get("plan", {}))
        try:
            prepared, prep_meta = _apply_preprocessing_plan(
                frame=frame,
                max_samples=plan.get("max_samples"),
                sample_selection=str(plan.get("sample_selection", "uniform")),
                missing_data_strategy=str(plan.get("missing_data_strategy", "keep")),
                fill_constant_value=plan.get("fill_constant_value"),
                row_coverage_strategy=str(plan.get("row_coverage_strategy", "keep")),
                sparse_row_min_fraction=float(plan.get("sparse_row_min_fraction", 0.8)),
                row_range_start=plan.get("row_range_start"),
                row_range_end=plan.get("row_range_end"),
            )
            quality = run_quality_checks(prepared, timestamp_column=timestamp_column)
            quick_targets = target_signals if target_signals else _top_numeric_targets(prepared, limit=3)
            quick_corr = run_correlation_analysis(
                frame=prepared,
                target_signals=quick_targets,
                predictor_signals_by_target=predictor_signals_by_target,
                timestamp_column=timestamp_column,
                max_lag=max_lag,
                include_feature_engineering=False,
                top_k_predictors=3,
                confidence_top_k=0,
                bootstrap_rounds=0,
                stability_windows=3,
            )
            top_strength = _avg_top_strength(quick_corr.to_dict())
            warning_penalty = min(0.2, 0.03 * len(quality.warnings))
            rows_ratio = min(1.0, len(prepared) / max(len(frame), 1))
            score = float(
                max(
                    0.0,
                    min(
                        1.0,
                        0.55 * top_strength + 0.35 * quality.completeness_score + 0.10 * rows_ratio - warning_penalty,
                    ),
                )
            )
            trace.append(
                {
                    "candidate_id": candidate_id,
                    "score": score,
                    "rows": int(len(prepared)),
                    "completeness": float(quality.completeness_score),
                    "top_strength": float(top_strength),
                    "warnings": list(quality.warnings),
                    "plan": plan,
                    "preprocessing": prep_meta,
                    "error": "",
                }
            )
        except Exception as exc:
            trace.append(
                {
                    "candidate_id": candidate_id,
                    "score": 0.0,
                    "rows": 0,
                    "completeness": 0.0,
                    "top_strength": 0.0,
                    "warnings": [],
                    "plan": plan,
                    "preprocessing": {},
                    "error": str(exc),
                }
            )
    return trace


def _critic_choose_strategy(trace: list[dict[str, Any]]) -> dict[str, Any]:
    if not trace:
        return {"selected_candidate_id": "user_plan", "rationale": "no_candidates"}
    ranked = sorted(trace, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    margin = (
        float(best.get("score", 0.0)) - float(second.get("score", 0.0))
        if second is not None
        else float(best.get("score", 0.0))
    )
    rationale = (
        f"highest composite score={float(best.get('score', 0.0)):.3f} "
        f"(margin={margin:.3f}, rows={best.get('rows')}, "
        f"completeness={float(best.get('completeness', 0.0)):.3f}, "
        f"top_strength={float(best.get('top_strength', 0.0)):.3f})"
    )
    return {
        "selected_candidate_id": best.get("candidate_id", "user_plan"),
        "rationale": rationale,
        "margin": margin,
    }


def _candidate_plan_by_id(
    candidates: list[dict[str, Any]],
    candidate_id: Any,
) -> dict[str, Any] | None:
    key = str(candidate_id) if candidate_id is not None else ""
    for item in candidates:
        if str(item.get("candidate_id")) == key:
            return dict(item.get("plan", {}))
    return None


def _avg_top_strength(correlations: dict[str, Any]) -> float:
    targets = list(correlations.get("target_analyses", []))
    if not targets:
        return 0.0
    vals: list[float] = []
    for target in targets:
        rows = list(target.get("predictor_results", []))
        if not rows:
            continue
        vals.append(float(rows[0].get("best_abs_score", 0.0)))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def _top_numeric_targets(frame: pd.DataFrame, *, limit: int) -> list[str]:
    cols = _detect_numeric_signal_columns(frame)
    return cols[:limit]


def _dataset_slug_from_path(path: str) -> str:
    stem = Path(path).stem.strip().lower()
    safe = "".join(ch if ch.isalnum() else "_" for ch in stem)
    safe = "_".join(token for token in safe.split("_") if token)
    return safe or "dataset"
