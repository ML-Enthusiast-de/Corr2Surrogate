"""Agent 1 report assembly helpers."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

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
    experiment_recommendations: list[dict[str, Any]] | None = None,
    planner_trace: list[dict[str, Any]] | None = None,
    critic_decision: dict[str, Any] | None = None,
    lineage_path: str | None = None,
    artifact_paths: dict[str, Any] | None = None,
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
        "experiment_recommendations": experiment_recommendations or [],
        "planner_trace": planner_trace or [],
        "critic_decision": critic_decision or {},
        "lineage_path": lineage_path,
        "artifact_paths": artifact_paths or {},
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
    csv_paths["planner_trace"] = _write_rows_csv(
        artifact_dir / "planner_trace.csv",
        list(structured.get("planner_trace", [])),
    )

    json_path = artifact_dir / "structured_report.json"
    json_path.write_text(json.dumps(structured, indent=2), encoding="utf-8")

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
    lines.extend(
        [
            "",
        "## Quality Summary",
        f"- Rows: {quality['rows']}",
        f"- Columns: {quality['columns']}",
        f"- Completeness score: {quality['completeness_score']:.3f}",
        f"- Duplicate rows: {quality['duplicate_rows']}",
        f"- Extreme outlier columns: {', '.join(quality['extreme_outlier_columns']) or 'none'}",
    ])
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
                    f"`predictor_correlation` | `{target_signal}` | "
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
                    f"`feature_engineering` | `{target_signal}` | "
                    f"{idx} | `{op_dict.get('expression', 'n/a')}` | "
                    f"`{op_dict.get('base_signal', 'n/a')}` | "
                    f"{_fmt(_safe_float(op_dict.get('score_abs')))} | "
                    f"{_fmt(_safe_float(op_dict.get('gain_over_raw')))} | "
                    f"{op_dict.get('notes', '')} |"
                )
        else:
            lines.append("- No feature-engineering opportunity exceeded the gain threshold.")
        lines.append("")
    lines.extend(["", "## Sensor Diagnostics"])
    lines.extend(_render_sensor_diagnostics(sensor_diagnostics))
    lines.extend(["", "## Experiment Recommendations"])
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
    return [
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


def _render_experiment_recommendations(recommendations: list[dict[str, Any]]) -> list[str]:
    if not recommendations:
        return ["- No additional recommendations generated."]
    lines: list[str] = []
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
