"""Agent 1 report assembly helpers."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Any

from corr2surrogate.core.json_utils import write_json
from corr2surrogate.analytics.correlations import CorrelationAnalysisBundle
from corr2surrogate.analytics.quality_checks import QualityCheckResult
from corr2surrogate.analytics.stationarity import StationaritySummary
from corr2surrogate.analytics.ranking import RankedSignal


def build_agent1_report_payload(
    *,
    data_path: str,
    quality: QualityCheckResult,
    stationarity: StationaritySummary,
    correlations: CorrelationAnalysisBundle,
    ranking: list[RankedSignal],
    forced_requests: list[dict[str, Any]] | None = None,
    preprocessing: dict[str, Any] | None = None,
    sensor_diagnostics: dict[str, Any] | None = None,
    model_strategy_recommendations: dict[str, Any] | None = None,
    experiment_recommendations: list[dict[str, Any]] | None = None,
    planner_trace: list[dict[str, Any]] | None = None,
    critic_decision: dict[str, Any] | None = None,
    lineage_path: str | None = None,
    artifact_paths: dict[str, Any] | None = None,
    user_hypotheses: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build structured and markdown report payload for Agent 1 output."""
    timestamp = datetime.now(timezone.utc).isoformat()
    ranking_payload = [asdict(item) for item in ranking]
    structured = {
        "generated_at_utc": timestamp,
        "data_path": data_path,
        "data_mode": correlations.data_mode,
        "timestamp_column": correlations.timestamp_column,
        "quality": quality.to_dict(),
        "stationarity": stationarity.to_dict(),
        "correlations": correlations.to_dict(),
        "ranking": ranking_payload,
        "forced_requests": forced_requests or [],
        "preprocessing": preprocessing or {},
        "sensor_diagnostics": sensor_diagnostics or {},
        "model_strategy_recommendations": model_strategy_recommendations or {},
        "experiment_recommendations": experiment_recommendations or [],
        "planner_trace": planner_trace or [],
        "critic_decision": critic_decision or {},
        "lineage_path": lineage_path,
        "artifact_paths": artifact_paths or {},
        "user_hypotheses": user_hypotheses or {},
    }
    markdown = _build_markdown(structured)
    return {
        "structured": structured,
        "markdown": markdown,
    }


def save_agent1_markdown_report(
    *,
    markdown: str,
    reports_dir: str | Path = "reports",
    data_path: str | None = None,
    run_id: str | None = None,
) -> str:
    """Persist Agent 1 markdown report and return path."""
    base = Path(reports_dir)
    dataset_slug = _dataset_slug_from_data_path(data_path)
    target_dir = base / dataset_slug
    target_dir.mkdir(parents=True, exist_ok=True)
    if run_id is None:
        run_id = datetime.now(timezone.utc).strftime("agent1_%Y%m%d_%H%M%S")
    safe_run_id = _sanitize_token(run_id, fallback="agent1_run")
    path = target_dir / f"{safe_run_id}.md"
    path.write_text(markdown, encoding="utf-8")
    return str(path)


def save_agent1_artifacts(
    *,
    structured: dict[str, Any],
    data_path: str | None = None,
    reports_dir: str | Path = "reports",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Persist machine-readable artifacts (CSV/JSON and optional PNG plots)."""
    base = Path(reports_dir)
    dataset_slug = _dataset_slug_from_data_path(data_path)
    target_dir = base / dataset_slug
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_run = _sanitize_token(run_id or "agent1", fallback="agent1")
    artifact_dir = target_dir / f"{safe_run}_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    top_predictors_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    hypothesis_pair_rows: list[dict[str, Any]] = []
    hypothesis_feature_rows: list[dict[str, Any]] = []
    model_strategy_rows: list[dict[str, Any]] = []
    model_probe_rows: list[dict[str, Any]] = []
    for target in structured.get("correlations", {}).get("target_analyses", []):
        target_signal = str(target.get("target_signal", "unknown"))
        for rank, row in enumerate(list(target.get("predictor_results", []))[:10], start=1):
            merged = dict(row)
            merged["target_signal"] = target_signal
            merged["rank"] = rank
            top_predictors_rows.append(merged)
        for rank, op in enumerate(list(target.get("feature_opportunities", []))[:10], start=1):
            merged = dict(op)
            merged["target_signal"] = target_signal
            merged["rank"] = rank
            feature_rows.append(merged)
        for rank, row in enumerate(list(target.get("hypothesis_pair_checks", []))[:10], start=1):
            merged = dict(row)
            merged["target_signal"] = target_signal
            merged["rank"] = rank
            hypothesis_pair_rows.append(merged)
        for rank, row in enumerate(
            list(target.get("hypothesis_feature_checks", []))[:10], start=1
        ):
            merged = dict(row)
            merged["target_signal"] = target_signal
            merged["rank"] = rank
            hypothesis_feature_rows.append(merged)

    for target in (structured.get("model_strategy_recommendations") or {}).get(
        "target_recommendations", []
    ):
        target_signal = str(target.get("target_signal", "unknown"))
        flattened = {
            key: value
            for key, value in dict(target).items()
            if key != "candidate_models"
        }
        flattened["target_signal"] = target_signal
        flattened["priority_model_families"] = ", ".join(
            str(item) for item in (target.get("priority_model_families") or [])
        )
        model_strategy_rows.append(flattened)
        for rank, item in enumerate(list(target.get("candidate_models", [])), start=1):
            merged = dict(item)
            merged["target_signal"] = target_signal
            merged["rank"] = rank
            model_probe_rows.append(merged)

    csv_paths: dict[str, str] = {}
    csv_paths["top_predictors"] = _write_rows_csv(
        artifact_dir / "top_predictors.csv", top_predictors_rows
    )
    csv_paths["feature_opportunities"] = _write_rows_csv(
        artifact_dir / "feature_opportunities.csv", feature_rows
    )
    csv_paths["experiment_recommendations"] = _write_rows_csv(
        artifact_dir / "experiment_recommendations.csv",
        list(structured.get("experiment_recommendations", [])),
    )
    csv_paths["sensor_diagnostics"] = _write_rows_csv(
        artifact_dir / "sensor_diagnostics.csv",
        list((structured.get("sensor_diagnostics") or {}).get("diagnostics", [])),
    )
    csv_paths["hypothesis_pair_checks"] = _write_rows_csv(
        artifact_dir / "hypothesis_pair_checks.csv",
        hypothesis_pair_rows,
    )
    csv_paths["hypothesis_feature_checks"] = _write_rows_csv(
        artifact_dir / "hypothesis_feature_checks.csv",
        hypothesis_feature_rows,
    )
    csv_paths["planner_trace"] = _write_rows_csv(
        artifact_dir / "planner_trace.csv",
        list(structured.get("planner_trace", [])),
    )
    csv_paths["model_strategy_recommendations"] = _write_rows_csv(
        artifact_dir / "model_strategy_recommendations.csv",
        model_strategy_rows,
    )
    csv_paths["model_strategy_probes"] = _write_rows_csv(
        artifact_dir / "model_strategy_probes.csv",
        model_probe_rows,
    )

    json_path = write_json(artifact_dir / "structured_report.json", structured, indent=2)

    plot_paths: dict[str, str] = {}
    top_plot = _write_top_predictor_plot(
        artifact_dir=artifact_dir,
        rows=top_predictors_rows,
    )
    if top_plot:
        plot_paths["top_predictors"] = top_plot

    payload = {
        "artifact_dir": str(artifact_dir),
        "csv_paths": csv_paths,
        "json_path": str(json_path),
        "plot_paths": plot_paths,
    }
    return payload


def _dataset_slug_from_data_path(data_path: str | None) -> str:
    if not data_path:
        return "dataset"
    stem = Path(data_path).stem
    return _sanitize_token(stem, fallback="dataset")


def _sanitize_token(value: str, *, fallback: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        return fallback
    return normalized


def _build_markdown(structured: dict[str, Any]) -> str:
    quality = structured["quality"]
    stationarity = structured["stationarity"]
    ranking = structured["ranking"]
    correlations = structured["correlations"]["target_analyses"]
    preprocessing = structured.get("preprocessing") or {}
    sensor_diagnostics = structured.get("sensor_diagnostics") or {}
    recommendations = structured.get("experiment_recommendations") or []
    planner_trace = structured.get("planner_trace") or []
    critic = structured.get("critic_decision") or {}
    lineage_path = structured.get("lineage_path")
    artifact_paths = structured.get("artifact_paths") or {}
    user_hypotheses = structured.get("user_hypotheses") or {}
    model_strategy = structured.get("model_strategy_recommendations") or {}

    lines: list[str] = [
        "# Agent 1 Analysis Report",
        "",
        f"- Generated (UTC): {structured['generated_at_utc']}",
        f"- Data path: `{structured['data_path']}`",
        f"- Data mode: `{structured['data_mode']}`",
        f"- Timestamp column: `{structured['timestamp_column']}`",
        "",
        "## Preprocessing Decisions",
    ]
    lines.extend(_render_preprocessing_section(preprocessing))
    lines.extend(["", "## User Hypotheses"])
    lines.extend(_render_user_hypotheses_section(user_hypotheses))
    lines.extend(
        [
            "",
        "## Quality Summary",
        f"- Rows: {quality['rows']}",
        f"- Columns: {quality['columns']}",
        f"- Completeness score: {quality['completeness_score']:.3f}",
        f"- Duplicate rows: {quality['duplicate_rows']}",
    ])
    lines.extend(_render_outlier_section(quality))
    if quality["warnings"]:
        lines.append("- Warnings:")
        for warning in quality["warnings"]:
            lines.append(f"  - {warning}")
    lines.extend(
        [
            "",
            "## Agentic Planning",
        ]
    )
    lines.extend(_render_agentic_section(planner_trace=planner_trace, critic=critic))
    lines.extend(
        [
            "",
            "## Stationarity Summary",
            f"- Analyzed signals: {stationarity['analyzed_signals']}",
            f"- Stationary: {stationarity['stationary_signals']}",
            f"- Non-stationary: {stationarity['non_stationary_signals']}",
            f"- Inconclusive: {stationarity['inconclusive_signals']}",
            "",
            "## Correlation Details (Top 10 Predictors per Target)",
        ]
    )
    for target in correlations:
        target_signal = str(target.get("target_signal", "unknown"))
        predictor_results = list(target.get("predictor_results", []))
        feature_opportunities = list(target.get("feature_opportunities", []))
        lines.extend(
            [
                f"### `{target_signal}`",
                f"- Predictors evaluated: {len(predictor_results)}",
                f"- Feature opportunities found: {len(feature_opportunities)}",
            ]
        )

        if predictor_results:
            lines.extend(
                [
                    "",
                    "| Category | Target | Rank | Predictor | Correlation Type | Strength | Best Method | Best Abs | Pearson | Pearson CI | Pearson p | Spearman | Spearman CI | Spearman p | Kendall | Distance Corr | Lagged Pearson | Stability | Confounder | Partial Pearson | Cond. MI | Best Lag | Samples |",
                    "|---|---|---:|---|---|---|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
                ]
            )
            for idx, row in enumerate(predictor_results[:10], start=1):
                row_dict = dict(row)
                best_method = str(row_dict.get("best_method", "none"))
                best_abs = _safe_float(row_dict.get("best_abs_score"))
                pearson_ci = (
                    f"[{_fmt(_safe_float(row_dict.get('pearson_ci_low')))}, "
                    f"{_fmt(_safe_float(row_dict.get('pearson_ci_high')))}]"
                )
                spearman_ci = (
                    f"[{_fmt(_safe_float(row_dict.get('spearman_ci_low')))}, "
                    f"{_fmt(_safe_float(row_dict.get('spearman_ci_high')))}]"
                )
                lines.append(
                    "| "
                    f"`{_row_category(row_dict)}` | `{target_signal}` | "
                    f"{idx} | `{row_dict.get('predictor_signal', 'n/a')}` | "
                    f"{_correlation_kind_label(row_dict)} | "
                    f"{_strength_label(best_abs)} | "
                    f"`{best_method}` | "
                    f"{_fmt(best_abs)} | "
                    f"{_fmt(_safe_float(row_dict.get('pearson')))} | "
                    f"{pearson_ci} | "
                    f"{_fmt(_safe_float(row_dict.get('pearson_pvalue')))} | "
                    f"{_fmt(_safe_float(row_dict.get('spearman')))} | "
                    f"{spearman_ci} | "
                    f"{_fmt(_safe_float(row_dict.get('spearman_pvalue')))} | "
                    f"{_fmt(_safe_float(row_dict.get('kendall')))} | "
                    f"{_fmt(_safe_float(row_dict.get('distance_corr')))} | "
                    f"{_fmt(_safe_float(row_dict.get('lagged_pearson')))} | "
                    f"{_fmt(_safe_float(row_dict.get('stability_score')))} | "
                    f"`{row_dict.get('confounder_signal', '')}` | "
                    f"{_fmt(_safe_float(row_dict.get('partial_pearson')))} | "
                    f"{_fmt(_safe_float(row_dict.get('conditional_mi')))} | "
                    f"{_fmt_int(row_dict.get('best_lag'))} | "
                    f"{_fmt_int(row_dict.get('sample_count'))} |"
                )
        else:
            lines.append("- No usable predictors after numeric filtering and sample checks.")

        lines.append("")
        lines.append("#### Feature Engineering Opportunities (Top 10)")
        if feature_opportunities:
            lines.extend(
                [
                    "",
                    "| Category | Target | Rank | Expression | Base Signal(s) | Abs Score | Gain vs Raw | Notes |",
                    "|---|---|---:|---|---|---:|---:|---|",
                ]
            )
            for idx, op in enumerate(feature_opportunities[:10], start=1):
                op_dict = dict(op)
                lines.append(
                    "| "
                    f"`{_feature_category(op_dict)}` | `{target_signal}` | "
                    f"{idx} | `{op_dict.get('expression', 'n/a')}` | "
                    f"`{op_dict.get('base_signal', 'n/a')}` | "
                    f"{_fmt(_safe_float(op_dict.get('score_abs')))} | "
                    f"{_fmt(_safe_float(op_dict.get('gain_over_raw')))} | "
                    f"{op_dict.get('notes', '')} |"
                )
        else:
            lines.append("- No feature-engineering opportunity exceeded the gain threshold.")
        lines.append("")
        hypothesis_pairs = list(target.get("hypothesis_pair_checks", []))
        hypothesis_features = list(target.get("hypothesis_feature_checks", []))
        if hypothesis_pairs or hypothesis_features:
            lines.append("#### User Hypothesis Checks")
            if hypothesis_pairs:
                lines.append("- Pair hypotheses:")
                for row in hypothesis_pairs[:10]:
                    row_dict = dict(row)
                    lines.append(
                        "  - "
                        f"target=`{target_signal}` predictor=`{row_dict.get('predictor_signal', 'n/a')}` "
                        f"best_method=`{row_dict.get('best_method', 'n/a')}` "
                        f"best_abs={_fmt(_safe_float(row_dict.get('best_abs_score')))} "
                        f"note={row_dict.get('hypothesis_note', '')}"
                    )
            if hypothesis_features:
                lines.append("- Feature hypotheses:")
                for row in hypothesis_features[:10]:
                    row_dict = dict(row)
                    lines.append(
                        "  - "
                        f"target=`{target_signal}` expr=`{row_dict.get('expression', 'n/a')}` "
                        f"abs_score={_fmt(_safe_float(row_dict.get('score_abs')))} "
                        f"gain={_fmt(_safe_float(row_dict.get('gain_over_raw')))} "
                        f"note={row_dict.get('hypothesis_note', '')}"
                    )
            lines.append("")
    lines.extend(["", "## Model Strategy Recommendations (Agent 2 Planning)"])
    lines.extend(_render_model_strategy_section(model_strategy))
    lines.extend(["", "## Sensor Diagnostics"])
    lines.extend(_render_sensor_diagnostics(sensor_diagnostics))
    lines.extend(["", "## Experiment Recommendations (Pre-Model, Agent 1)"])
    lines.extend(_render_experiment_recommendations(recommendations))
    lines.extend(["", "## Dependency-Aware Ranking"])
    for item in ranking:
        lines.append(
            f"- `{item['target_signal']}`: adjusted_score={item['adjusted_score']:.3f}, "
            f"feasible={item['feasible']} ({item['rationale']})"
        )
    forced = structured.get("forced_requests", [])
    if forced:
        lines.extend(["", "## Forced Modeling Requests"])
        for req in forced:
            lines.append(
                f"- target `{req.get('target_signal')}` with predictors "
                f"{req.get('predictor_signals', [])}"
            )
    if lineage_path:
        lines.extend(["", "## Lineage", f"- Run lineage: `{lineage_path}`"])
    if artifact_paths:
        lines.extend(["", "## Artifacts"])
        lines.extend(_render_artifact_paths(artifact_paths))
    lines.append("")
    return "\n".join(lines)


def _render_preprocessing_section(preprocessing: dict[str, Any]) -> list[str]:
    if not preprocessing:
        return ["- None (raw loaded dataset used)."]
    sample_plan = preprocessing.get("sample_plan") or {}
    missing_plan = preprocessing.get("missing_data_plan") or {}
    coverage_plan = preprocessing.get("row_coverage_plan") or {}
    lines = [
        f"- Initial rows: {preprocessing.get('initial_rows', 'n/a')}",
        f"- Final rows analyzed: {preprocessing.get('final_rows', 'n/a')}",
        "- Sample plan: "
        f"applied={sample_plan.get('applied', False)}, "
        f"requested_max_samples={sample_plan.get('requested_max_samples')}, "
        f"selection={sample_plan.get('selection')}, "
        f"rows_after={sample_plan.get('rows_after')}",
        "- Missing-data plan: "
        f"applied={missing_plan.get('applied', False)}, "
        f"strategy={missing_plan.get('strategy')}, "
        f"fill_constant_value={missing_plan.get('fill_constant_value')}, "
        f"rows_after={missing_plan.get('rows_after')}",
        "- Row-coverage plan: "
        f"applied={coverage_plan.get('applied', False)}, "
        f"strategy={coverage_plan.get('strategy')}, "
        f"threshold={coverage_plan.get('sparse_row_min_fraction')}, "
        f"range=[{coverage_plan.get('row_range_start')}, {coverage_plan.get('row_range_end')}], "
        f"rows_after={coverage_plan.get('rows_after')}",
    ]
    leakage_risk = str(missing_plan.get("split_leakage_risk", "none"))
    leakage_note = str(missing_plan.get("split_leakage_note", "")).strip()
    split_policy = str(missing_plan.get("recommended_split_safe_policy", "")).strip()
    lines.append(f"- Split leakage risk (missing-data plan): `{leakage_risk}`.")
    if leakage_note:
        lines.append(f"- Split leakage note: {leakage_note}")
    if split_policy:
        lines.append(f"- Split-safe policy: {split_policy}")
    return lines


def _render_outlier_section(quality: dict[str, Any]) -> list[str]:
    rows = int(quality.get("rows", 0) or 0)
    threshold = max(5, int(0.02 * rows))
    outlier_counts_raw = quality.get("outlier_count_by_column") or {}
    outlier_counts: dict[str, int] = {}
    if isinstance(outlier_counts_raw, dict):
        for key, value in outlier_counts_raw.items():
            try:
                outlier_counts[str(key)] = int(value)
            except (TypeError, ValueError):
                continue

    extreme = [str(col) for col in quality.get("extreme_outlier_columns", [])]
    lines = [
        "- Outlier detection method: robust-z (|z|>4.0) OR Tukey-IQR fence "
        "(outside [Q1-1.5*IQR, Q3+1.5*IQR]).",
        f"- Extreme outlier rule: outlier_count > max(5, 2% of rows) = {threshold}.",
    ]
    if not extreme:
        lines.append("- Extreme outlier columns: none.")
        return lines

    lines.append(
        f"- Extreme outlier columns: {', '.join(extreme)}."
    )
    lines.append("")
    lines.extend(
        [
            "| Column | Outlier Count | Fraction of Rows |",
            "|---|---:|---:|",
        ]
    )
    ranked = sorted(
        ((col, outlier_counts.get(col, 0)) for col in extreme),
        key=lambda item: item[1],
        reverse=True,
    )
    denom = max(rows, 1)
    for col, count in ranked:
        frac = count / denom
        lines.append(f"| `{col}` | {count} | {frac:.3%} |")
    return lines


def _render_agentic_section(
    *,
    planner_trace: list[dict[str, Any]],
    critic: dict[str, Any],
) -> list[str]:
    if not planner_trace:
        return ["- Planner: single strategy (no strategy search)."]
    lines: list[str] = [f"- Planner evaluated {len(planner_trace)} strategy candidates."]
    top = sorted(
        planner_trace,
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )[:3]
    for row in top:
        lines.append(
            "- Candidate "
            f"`{row.get('candidate_id')}` score={_fmt(_safe_float(row.get('score')))} "
            f"rows={row.get('rows')} completeness={_fmt(_safe_float(row.get('completeness')))} "
            f"top_strength={_fmt(_safe_float(row.get('top_strength')))}."
        )
    if critic:
        lines.append(
            "- Critic selected strategy "
            f"`{critic.get('selected_candidate_id', 'n/a')}` "
            f"because: {critic.get('rationale', 'n/a')}"
        )
    return lines


def _render_user_hypotheses_section(payload: dict[str, Any]) -> list[str]:
    pair_items = list(payload.get("correlation_hypotheses", []))
    feature_items = list(payload.get("feature_hypotheses", []))
    if not pair_items and not feature_items:
        return ["- None provided by user."]
    lines: list[str] = []
    if pair_items:
        lines.append(f"- Correlation hypotheses: {len(pair_items)}")
        for item in pair_items[:10]:
            target = item.get("target_signal", "")
            predictors = item.get("predictor_signals", [])
            reason = item.get("user_reason", "")
            lines.append(
                f"  - `{target}` <- {predictors} ({reason})"
            )
    if feature_items:
        lines.append(f"- Feature hypotheses: {len(feature_items)}")
        for item in feature_items[:10]:
            target = item.get("target_signal", "") or "*"
            base_signal = item.get("base_signal", "")
            transformation = item.get("transformation", "")
            reason = item.get("user_reason", "")
            lines.append(
                f"  - target=`{target}` expression=`{transformation}({base_signal})` ({reason})"
            )
    return lines


def _render_sensor_diagnostics(payload: dict[str, Any]) -> list[str]:
    diagnostics = list(payload.get("diagnostics", []))
    warnings = list(payload.get("warnings", []))
    flagged = list(payload.get("flagged_signals", []))
    lines: list[str] = [
        f"- Signals analyzed: {len(diagnostics)}",
        f"- Flagged signals: {len(flagged)}",
    ]
    if payload.get("timestamp_column"):
        lines.append(
            f"- Timestamp jitter CV: {_fmt(_safe_float(payload.get('timestamp_jitter_cv')))}"
        )
    if warnings:
        lines.append("- Warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")
    if diagnostics:
        lines.extend(
            [
                "",
                "| Signal | Trust Score | Missing | Saturation | Quantization | Drift | Dropout Run | Stuck Run | Flags |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in diagnostics[:10]:
            lines.append(
                "| "
                f"`{row.get('signal', 'n/a')}` | "
                f"{_fmt(_safe_float(row.get('trust_score')))} | "
                f"{_fmt(_safe_float(row.get('missing_fraction')))} | "
                f"{_fmt(_safe_float(row.get('saturation_fraction')))} | "
                f"{_fmt(_safe_float(row.get('quantization_fraction')))} | "
                f"{_fmt(_safe_float(row.get('drift_slope_normalized')))} | "
                f"{_fmt(_safe_float(row.get('dropout_run_fraction')))} | "
                f"{_fmt(_safe_float(row.get('stuck_run_fraction')))} | "
                f"{row.get('flags', [])} |"
            )
    return lines


def _render_model_strategy_section(payload: dict[str, Any]) -> list[str]:
    targets = list(payload.get("target_recommendations", []))
    if not targets:
        return ["- No model-strategy recommendations were generated."]
    lines: list[str] = [
        "- Agent 1 runs lightweight probe models to estimate which Agent 2 families should be tested first.",
        "- Planned Agent 2 order should stay pragmatic: linear/lagged baselines first, then tree ensembles, then sequence models only when justified.",
    ]
    for target in targets:
        target_signal = str(target.get("target_signal", "unknown"))
        priority = ", ".join(str(item) for item in (target.get("priority_model_families") or []))
        lines.extend(
            [
                f"### `{target_signal}`",
                f"- Recommended model family: `{target.get('recommended_model_family', 'n/a')}`",
                f"- Search order: {priority or 'n/a'}",
                f"- Tree model worth testing: {target.get('tree_model_worth_testing', False)}",
                f"- Sequence model worth testing: {target.get('sequence_model_worth_testing', False)}",
                f"- Rationale: {target.get('rationale', '')}",
            ]
        )
        probes = list(target.get("candidate_models", []))
        if probes:
            lines.extend(
                [
                    "",
                    "| Probe Model | MAE | RMSE | R2 | Gain vs Linear | Notes |",
                    "|---|---:|---:|---:|---:|---|",
                ]
            )
            for item in probes:
                lines.append(
                    "| "
                    f"`{item.get('model_family', 'n/a')}` | "
                    f"{_fmt(_safe_float(item.get('mae')))} | "
                    f"{_fmt(_safe_float(item.get('rmse')))} | "
                    f"{_fmt(_safe_float(item.get('r2')))} | "
                    f"{_fmt(_safe_float(item.get('relative_mae_gain_vs_linear')))} | "
                    f"{item.get('notes', '')} |"
                )
        else:
            lines.append(
                "- Probe-model score table omitted because screening did not have enough complete rows for a stable validation split."
            )
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _render_experiment_recommendations(recommendations: list[dict[str, Any]]) -> list[str]:
    if not recommendations:
        return [
            "- These are pre-model data-collection suggestions from Agent 1 "
            "(correlation/stability/diagnostic driven, not post-training error analysis).",
            "- No additional recommendations generated.",
        ]
    lines: list[str] = [
        "- These are pre-model data-collection suggestions from Agent 1 "
        "(correlation/stability/diagnostic driven, not post-training error analysis)."
    ]
    for idx, row in enumerate(recommendations[:10], start=1):
        lines.append(
            f"{idx}. target=`{row.get('target_signal')}` "
            f"type=`{row.get('trajectory_type')}` score={_fmt(_safe_float(row.get('score')))}"
        )
        lines.append(f"   suggestion: {row.get('suggestion')}")
        lines.append(f"   rationale: {row.get('rationale')}")
    return lines


def _render_artifact_paths(payload: dict[str, Any]) -> list[str]:
    lines = []
    artifact_dir = payload.get("artifact_dir")
    if artifact_dir:
        lines.append(f"- Artifact directory: `{artifact_dir}`")
    csv_paths = payload.get("csv_paths") or {}
    for key, path in csv_paths.items():
        lines.append(f"- CSV `{key}`: `{path}`")
    json_path = payload.get("json_path")
    if json_path:
        lines.append(f"- Structured JSON: `{json_path}`")
    plot_paths = payload.get("plot_paths") or {}
    for key, path in plot_paths.items():
        lines.append(f"- Plot `{key}`: `{path}`")
    return lines


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def _fmt_int(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "n/a"


def _strength_label(score_abs: float) -> str:
    if not math.isfinite(score_abs):
        return "n/a"
    if score_abs >= 0.85:
        return "very strong"
    if score_abs >= 0.70:
        return "strong"
    if score_abs >= 0.50:
        return "moderate"
    if score_abs >= 0.30:
        return "weak"
    return "very weak"


def _correlation_kind_label(row: dict[str, Any]) -> str:
    method = str(row.get("best_method", "none"))
    method_value = _safe_float(row.get(method))
    method_family = {
        "pearson": "linear",
        "spearman": "monotonic",
        "kendall": "rank-monotonic",
        "distance_corr": "nonlinear-association",
        "lagged_pearson": "lagged-linear",
    }.get(method, "unspecified")
    if method == "distance_corr":
        return method_family
    if not math.isfinite(method_value):
        return method_family
    direction = "positive" if method_value >= 0 else "negative"
    return f"{direction} {method_family}"


def _row_category(row: dict[str, Any]) -> str:
    if bool(row.get("is_user_hypothesis")):
        return "hypothesis_pair"
    return "predictor_correlation"


def _feature_category(row: dict[str, Any]) -> str:
    if bool(row.get("is_user_hypothesis")):
        return "hypothesis_feature"
    return "feature_engineering"


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> str:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return str(path)
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            skey = str(key)
            if skey not in seen:
                all_keys.append(skey)
                seen.add(skey)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            payload = {str(k): row.get(k) for k in all_keys}
            writer.writerow(payload)
    return str(path)


def _write_top_predictor_plot(*, artifact_dir: Path, rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None
    top = rows[:10]
    labels = [str(row.get("predictor_signal", "n/a")) for row in top]
    scores = [_safe_float(row.get("best_abs_score")) for row in top]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, scores)
    ax.set_ylabel("best_abs_score")
    ax.set_title("Top Predictor Strength")
    ax.set_ylim(0.0, 1.0)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    path = artifact_dir / "top_predictors.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)
