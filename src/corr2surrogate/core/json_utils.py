"""JSON helpers that enforce strict JSON compatibility."""

from __future__ import annotations

from datetime import date, datetime
import json
import math
from pathlib import Path
from typing import Any


def to_json_compatible(value: Any) -> Any:
    """Recursively convert values into strict JSON-compatible structures.

    Non-finite floats (NaN, +inf, -inf) are converted to null.
    """
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_compatible(item) for item in value]

    if hasattr(value, "item"):
        try:
            return to_json_compatible(value.item())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return to_json_compatible(value.tolist())
        except Exception:
            pass
    return value


def dumps_json(
    value: Any,
    *,
    indent: int | None = None,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
) -> str:
    """Serialize with strict JSON semantics (allow_nan=False)."""
    return json.dumps(
        to_json_compatible(value),
        indent=indent,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys,
        allow_nan=False,
    )


def write_json(
    path: str | Path,
    value: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = True,
    sort_keys: bool = False,
) -> Path:
    """Write strict JSON to disk and return path."""
    output = Path(path)
    output.write_text(
        dumps_json(
            value,
            indent=indent,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys,
        ),
        encoding="utf-8",
    )
    return output
