"""Default tool registration for the Corr2Surrogate harness."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from corr2surrogate.analytics import (
    assess_stationarity,
    build_agent1_report_payload,
    build_candidate_signals_from_correlations,
    run_correlation_analysis,
    run_quality_checks,
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
from corr2surrogate.persistence.artifact_store import ArtifactStore

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
            "correlations, feature opportunities, and dependency-aware ranking."
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
) -> dict[str, Any]:
    result = evaluate_training_iteration(
        metrics=metrics,
        acceptance_criteria=acceptance_criteria,
        attempt=attempt,
        max_attempts=max_attempts,
        min_relative_improvement=min_relative_improvement,
        previous_best_score=previous_best_score,
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
    forced_requests: list[dict[str, Any]] | None = None,
    physically_available_signals: list[str] | None = None,
    non_virtualizable_signals: list[str] | None = None,
    max_lag: int = 8,
    include_feature_engineering: bool = True,
    feature_gain_threshold: float = 0.05,
    save_report: bool = True,
    run_id: str | None = None,
) -> dict[str, Any]:
    loaded = load_tabular_data(
        path=data_path,
        sheet_name=sheet_name,
        header_row=header_row,
        data_start_row=data_start_row,
    )
    frame = loaded.frame

    quality = run_quality_checks(frame, timestamp_column=timestamp_column)
    correlations = run_correlation_analysis(
        frame=frame,
        target_signals=target_signals,
        predictor_signals_by_target=predictor_signals_by_target,
        timestamp_column=timestamp_column,
        max_lag=max_lag,
        include_feature_engineering=include_feature_engineering,
        feature_gain_threshold=feature_gain_threshold,
    )
    stationarity_columns = [item.target_signal for item in correlations.target_analyses]
    stationarity = assess_stationarity(frame, signal_columns=stationarity_columns)

    candidates = build_candidate_signals_from_correlations(correlations)
    ranking = rank_surrogate_candidates(
        candidates=candidates,
        physically_available_signals=physically_available_signals,
        non_virtualizable_signals=non_virtualizable_signals,
    )

    forced_directives = _normalize_forced_requests(forced_requests)
    report_payload = build_agent1_report_payload(
        data_path=data_path,
        quality=quality,
        stationarity=stationarity,
        correlations=correlations,
        ranking=ranking,
        forced_requests=[asdict(item) for item in forced_directives],
    )
    report_path: str | None = None
    if save_report:
        report_path = save_agent1_markdown_report(
            markdown=report_payload["markdown"],
            data_path=data_path,
            run_id=run_id,
        )

    return {
        "status": "ok",
        "data_mode": correlations.data_mode,
        "timestamp_column": correlations.timestamp_column,
        "target_count": len(correlations.target_analyses),
        "candidate_count": len(candidates),
        "ranking": [asdict(item) for item in ranking],
        "forced_requests": [asdict(item) for item in forced_directives],
        "quality": quality.to_dict(),
        "stationarity": stationarity.to_dict(),
        "correlations": correlations.to_dict(),
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


def _detect_numeric_signal_columns(frame: Any) -> list[str]:
    numeric_signals: list[str] = []
    for col in frame.columns:
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if int(numeric.notna().sum()) >= 8 and int(numeric.nunique(dropna=True)) > 1:
            numeric_signals.append(str(col))
    return numeric_signals
