"""Agent 1 report assembly helpers."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
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

    lines: list[str] = [
        "# Agent 1 Analysis Report",
        "",
        f"- Generated (UTC): {structured['generated_at_utc']}",
        f"- Data path: `{structured['data_path']}`",
        f"- Data mode: `{structured['data_mode']}`",
        f"- Timestamp column: `{structured['timestamp_column']}`",
        "",
        "## Quality Summary",
        f"- Rows: {quality['rows']}",
        f"- Columns: {quality['columns']}",
        f"- Completeness score: {quality['completeness_score']:.3f}",
        f"- Duplicate rows: {quality['duplicate_rows']}",
        f"- Extreme outlier columns: {', '.join(quality['extreme_outlier_columns']) or 'none'}",
    ]
    if quality["warnings"]:
        lines.append("- Warnings:")
        for warning in quality["warnings"]:
            lines.append(f"  - {warning}")
    lines.extend(
        [
            "",
            "## Stationarity Summary",
            f"- Analyzed signals: {stationarity['analyzed_signals']}",
            f"- Stationary: {stationarity['stationary_signals']}",
            f"- Non-stationary: {stationarity['non_stationary_signals']}",
            f"- Inconclusive: {stationarity['inconclusive_signals']}",
            "",
            "## Correlation Highlights",
        ]
    )
    for target in correlations:
        top = target.get("top_predictors", [])[:3]
        opp_count = len(target.get("feature_opportunities", []))
        lines.append(
            f"- `{target['target_signal']}`: top predictors={top or ['none']}, "
            f"feature opportunities={opp_count}"
        )
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
    lines.append("")
    return "\n".join(lines)
