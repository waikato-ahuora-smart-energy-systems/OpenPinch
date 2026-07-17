"""Shared JSON-safe workspace view primitives."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Optional

import pandas as pd

from ....contracts.workspace import TableView


def _unit_column(metric: str) -> str:
    return f"{metric} (unit)"


def dataframe_to_table_view(frame: pd.DataFrame) -> TableView:
    """Convert a dataframe to a JSON-safe table view."""
    safe_frame = frame.copy()
    rows = [
        {column: json_safe(value) for column, value in row.items()}
        for row in safe_frame.to_dict(orient="records")
    ]
    return TableView(columns=[str(column) for column in safe_frame.columns], rows=rows)


def numeric_delta(base_value: Any, variant_value: Any) -> Optional[float]:
    """Return a numeric delta when both inputs are plain numbers."""
    if isinstance(base_value, bool) or isinstance(variant_value, bool):
        return None
    if is_number(base_value) and is_number(variant_value):
        return float(variant_value) - float(base_value)
    return None


def is_number(value: Any) -> bool:
    """Return whether a value is a plain numeric scalar."""
    if value is None or isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def maybe_string(value: Any) -> Optional[str]:
    """Return a non-empty string representation when present."""
    if value in (None, ""):
        return None
    return str(value)


def maybe_float(value: Any) -> Optional[float]:
    """Return a finite float when conversion is possible."""
    if value is None:
        return None
    try:
        value = float(value)
    except TypeError, ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def json_safe(value: Any) -> Any:
    """Convert nested values to JSON-safe plain data structures."""
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if hasattr(value, "__fspath__"):
        return str(value)
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="python"))
    if hasattr(value, "to_dict"):
        try:
            return json_safe(value.to_dict())
        except Exception:
            pass
    return str(value)
