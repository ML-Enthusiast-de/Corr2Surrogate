"""Tabular ingestion utilities for CSV/XLSX with header inference support."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


SUPPORTED_TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class IngestionError(RuntimeError):
    """Base ingestion error."""


class UnsupportedFormatError(IngestionError):
    """Raised when the input file extension is not supported."""


class SheetSelectionRequiredError(IngestionError):
    """Raised when an Excel file has multiple sheets and no sheet was selected."""

    def __init__(self, sheets: list[str]) -> None:
        self.sheets = sheets
        joined = ", ".join(sheets)
        super().__init__(
            f"Multiple sheets found ({joined}). Please select a sheet to continue."
        )


@dataclass(frozen=True)
class HeaderInference:
    """Result of inferred header/data-start detection."""

    header_row: int
    data_start_row: int
    confidence: float
    candidate_rows: list[int]
    needs_user_confirmation: bool
    reason: str


@dataclass(frozen=True)
class IngestionResult:
    """Loaded tabular dataset and metadata required for user interaction."""

    frame: pd.DataFrame
    source_path: Path
    file_type: str
    selected_sheet: Optional[str]
    available_sheets: list[str]
    inferred_header: HeaderInference
    delimiter: Optional[str]


def list_excel_sheets(path: str | Path) -> list[str]:
    """Return sheet names for an Excel file."""
    file_path = Path(path)
    if file_path.suffix.lower() not in {".xlsx", ".xls"}:
        return []
    with pd.ExcelFile(file_path) as xls:
        return list(xls.sheet_names)


def load_tabular_data(
    path: str | Path,
    *,
    sheet_name: str | int | None = None,
    header_row: int | None = None,
    data_start_row: int | None = None,
    delimiter: str | None = None,
    infer_header: bool = True,
    confidence_threshold: float = 0.70,
    preview_rows: int = 40,
) -> IngestionResult:
    """Load CSV/XLSX data and infer header/data start when needed.

    If confidence is low, `needs_user_confirmation` is set to True so Agent 1 can ask.
    """
    file_path = Path(path)
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_TABULAR_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported format '{ext}'. Supported: {sorted(SUPPORTED_TABULAR_EXTENSIONS)}"
        )

    if ext in {".xlsx", ".xls"}:
        sheets = list_excel_sheets(file_path)
        selected_sheet = _resolve_sheet_selection(sheet_name, sheets)
        selected_sheet_name = (
            selected_sheet if isinstance(selected_sheet, str) else sheets[selected_sheet]
        )
        preview = pd.read_excel(
            file_path, sheet_name=selected_sheet, header=None, nrows=preview_rows
        )
        raw = pd.read_excel(file_path, sheet_name=selected_sheet, header=None)
        inferred = _resolve_header_inference(
            preview=preview,
            raw=raw,
            header_row=header_row,
            data_start_row=data_start_row,
            infer_header=infer_header,
            confidence_threshold=confidence_threshold,
        )
        frame = _finalize_raw_table(raw, inferred.header_row, inferred.data_start_row)
        return IngestionResult(
            frame=frame,
            source_path=file_path,
            file_type="xlsx",
            selected_sheet=selected_sheet_name,
            available_sheets=sheets,
            inferred_header=inferred,
            delimiter=None,
        )

    csv_delimiter = delimiter or _detect_csv_delimiter(file_path)
    preview = pd.read_csv(
        file_path,
        sep=csv_delimiter,
        header=None,
        nrows=preview_rows,
        engine="python",
    )
    raw = pd.read_csv(file_path, sep=csv_delimiter, header=None, engine="python")
    inferred = _resolve_header_inference(
        preview=preview,
        raw=raw,
        header_row=header_row,
        data_start_row=data_start_row,
        infer_header=infer_header,
        confidence_threshold=confidence_threshold,
    )
    frame = _finalize_raw_table(raw, inferred.header_row, inferred.data_start_row)
    return IngestionResult(
        frame=frame,
        source_path=file_path,
        file_type="csv",
        selected_sheet=None,
        available_sheets=[],
        inferred_header=inferred,
        delimiter=csv_delimiter,
    )


def _resolve_sheet_selection(
    sheet_name: str | int | None,
    sheets: list[str],
) -> str | int:
    if not sheets:
        raise IngestionError("No sheets were found in the Excel file.")
    if len(sheets) > 1 and sheet_name is None:
        raise SheetSelectionRequiredError(sheets=sheets)
    if sheet_name is None:
        return sheets[0] if sheets else 0
    if isinstance(sheet_name, int):
        if sheet_name < 0 or sheet_name >= len(sheets):
            raise IngestionError(
                f"Invalid sheet index {sheet_name}. Available indices: 0..{len(sheets)-1}"
            )
        return sheet_name
    if sheet_name not in sheets:
        raise IngestionError(
            f"Sheet '{sheet_name}' not found. Available: {', '.join(sheets)}"
        )
    return sheet_name


def _resolve_header_inference(
    *,
    preview: pd.DataFrame,
    raw: pd.DataFrame,
    header_row: int | None,
    data_start_row: int | None,
    infer_header: bool,
    confidence_threshold: float,
) -> HeaderInference:
    if header_row is not None:
        resolved_data_start = (
            data_start_row if data_start_row is not None else header_row + 1
        )
        return HeaderInference(
            header_row=header_row,
            data_start_row=resolved_data_start,
            confidence=1.0,
            candidate_rows=[header_row],
            needs_user_confirmation=False,
            reason="User provided header row.",
        )

    if not infer_header:
        return HeaderInference(
            header_row=0,
            data_start_row=1 if data_start_row is None else data_start_row,
            confidence=0.5,
            candidate_rows=[0],
            needs_user_confirmation=True,
            reason="Header inference disabled; defaulted to row 0.",
        )

    inferred = _infer_header_and_data_start(
        preview, raw, confidence_threshold=confidence_threshold
    )
    if data_start_row is not None:
        return HeaderInference(
            header_row=inferred.header_row,
            data_start_row=data_start_row,
            confidence=inferred.confidence,
            candidate_rows=inferred.candidate_rows,
            needs_user_confirmation=inferred.needs_user_confirmation,
            reason=f"{inferred.reason} Data start overridden by user.",
        )
    return inferred


def _infer_header_and_data_start(
    preview: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    confidence_threshold: float,
) -> HeaderInference:
    max_scan_rows = min(len(preview), 20)
    if max_scan_rows == 0:
        return HeaderInference(
            header_row=0,
            data_start_row=1,
            confidence=0.0,
            candidate_rows=[0],
            needs_user_confirmation=True,
            reason="File appears empty.",
        )

    scored_rows: list[tuple[int, float]] = []
    for row_idx in range(max_scan_rows):
        score = _score_header_candidate(preview.iloc[row_idx])
        scored_rows.append((row_idx, score))

    scored_rows.sort(key=lambda item: item[1], reverse=True)
    best_row, best_score = scored_rows[0]
    candidates = [idx for idx, _ in scored_rows[:3]]
    data_start_row = _infer_data_start_row(raw, start_at=best_row + 1)
    reason = (
        "Header inferred from mostly textual, non-empty, unique row pattern."
        if best_score >= confidence_threshold
        else "Low header confidence. User confirmation recommended."
    )
    return HeaderInference(
        header_row=best_row,
        data_start_row=data_start_row,
        confidence=round(best_score, 3),
        candidate_rows=candidates,
        needs_user_confirmation=best_score < confidence_threshold,
        reason=reason,
    )


def _score_header_candidate(row: pd.Series) -> float:
    non_null = row.dropna()
    if non_null.empty:
        return 0.0
    values = [str(v).strip() for v in non_null.tolist()]
    if not values:
        return 0.0

    numeric_flags = [_looks_numeric(v) for v in values]
    alpha_flags = [_has_alpha(v) for v in values]
    unique_ratio = len(set(values)) / len(values)
    numeric_ratio = sum(numeric_flags) / len(numeric_flags)
    alpha_ratio = sum(alpha_flags) / len(alpha_flags)
    non_empty_ratio = len(values) / max(len(row), 1)

    score = (
        (1.0 - numeric_ratio) * 0.45
        + alpha_ratio * 0.25
        + unique_ratio * 0.20
        + non_empty_ratio * 0.10
    )
    return max(0.0, min(score, 1.0))


def _infer_data_start_row(raw: pd.DataFrame, *, start_at: int) -> int:
    for idx in range(start_at, len(raw)):
        row = raw.iloc[idx]
        non_null = row.dropna()
        if non_null.empty:
            continue
        values = [str(v).strip() for v in non_null.tolist()]
        numeric_ratio = sum(_looks_numeric(v) for v in values) / len(values)
        if numeric_ratio >= 0.5:
            return idx
    return start_at


def _finalize_raw_table(raw: pd.DataFrame, header_row: int, data_start_row: int) -> pd.DataFrame:
    if header_row >= len(raw):
        raise IngestionError(
            f"Header row {header_row} out of range for file with {len(raw)} rows."
        )
    if data_start_row <= header_row:
        raise IngestionError(
            f"Data start row {data_start_row} must be greater than header row {header_row}."
        )

    header_values = raw.iloc[header_row].tolist()
    columns = _make_unique_columns(header_values)
    frame = raw.iloc[data_start_row:].copy()
    frame.columns = columns
    frame = frame.dropna(how="all").reset_index(drop=True)

    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="ignore")
    return frame


def _make_unique_columns(values: list[object]) -> list[str]:
    used: dict[str, int] = {}
    columns: list[str] = []
    for idx, value in enumerate(values):
        base = str(value).strip() if value is not None else ""
        if base == "" or base.lower() == "nan":
            base = f"column_{idx}"
        count = used.get(base, 0)
        used[base] = count + 1
        columns.append(base if count == 0 else f"{base}_{count}")
    return columns


def _detect_csv_delimiter(path: Path) -> str:
    sample = ""
    for encoding in ("utf-8", "latin-1"):
        try:
            sample = path.read_text(encoding=encoding)[:8000]
            break
        except UnicodeDecodeError:
            continue
    if not sample:
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except csv.Error:
        return ","


def _looks_numeric(value: object) -> bool:
    text = str(value).strip().replace(",", ".")
    if text == "":
        return False
    try:
        float(text)
    except ValueError:
        return False
    return True


def _has_alpha(value: object) -> bool:
    return any(ch.isalpha() for ch in str(value))
