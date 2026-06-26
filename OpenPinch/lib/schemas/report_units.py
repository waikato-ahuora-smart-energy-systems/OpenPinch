"""Helpers for attaching and preserving units on report-facing target results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np

from ...classes.value import Value

_ARRAY_PAYLOAD_KEYS = {"kind", "values", "unit"}
_SCALAR_PAYLOAD_KEYS = {"value", "unit", "weights"}
_PERIOD_PAYLOAD_KEYS = {"values", "period_ids", "weights", "unit"}

__all__ = [
    "split_report_value",
]


def _normalise_input_object(value: Any) -> Any:
    if isinstance(value, Value):
        return value
    if hasattr(value, "model_dump") and not isinstance(value, Mapping):
        return value.model_dump(mode="python")
    return value


def _is_bool_like(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(
        value,
        (int, float, np.integer, np.floating),
    ) and not _is_bool_like(value)


def _is_numeric_array_like(value: Any) -> bool:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        return False
    if _is_bool_like(value) or np.isscalar(value):
        return False
    try:
        np.asarray(value, dtype=float).reshape(-1)
    except TypeError, ValueError:
        return False
    return True


def _is_array_value_data(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("kind") == "array"
        and "values" in value
        and set(value).issubset(_ARRAY_PAYLOAD_KEYS)
    )


def _is_scalar_value_data(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and "value" in value
        and set(value).issubset(_SCALAR_PAYLOAD_KEYS)
    )


def _is_period_value_data(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and "values" in value
        and set(value).issubset(_PERIOD_PAYLOAD_KEYS)
    )


def _coerce_array_values(values: Any) -> list[float]:
    resolved = np.asarray(values, dtype=float).reshape(-1)
    if resolved.size == 0:
        raise ValueError("Report value arrays cannot be empty.")
    return [float(item) for item in resolved.tolist()]


def _value_array_to_list(value: Value) -> list[float]:
    return [float(item) for item in value.period_values.tolist()]


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    return None if np.isnan(scalar) else scalar


def _resolve_period_values(
    values: Any, *, period_idx: int | None = None
) -> float | None:
    resolved = np.asarray(values, dtype=float).reshape(-1)
    if resolved.size == 0:
        raise ValueError("Period-valued report values cannot be empty.")
    if resolved.size == 1:
        return _coerce_optional_float(resolved[0])

    resolved_idx = 0 if period_idx is None else int(period_idx)
    if resolved_idx < 0 or resolved_idx >= resolved.size:
        raise ValueError(
            f"period_idx {resolved_idx} is out of range for {resolved.size} period(s)."
        )
    return _coerce_optional_float(resolved[resolved_idx])


def _resolve_scalar_value(value: Any, *, period_idx: int | None = None) -> float | None:
    value = _normalise_input_object(value)
    if value is None:
        return None
    if _is_bool_like(value):
        raise TypeError("Boolean values are not supported.")
    if isinstance(value, Value):
        if value.num_periods == 1:
            return _coerce_optional_float(value.value)
        return _resolve_period_values(value.period_values, period_idx=period_idx)
    if _is_numeric_scalar(value):
        return float(value)
    if _is_scalar_value_data(value):
        return _coerce_optional_float(value.get("value"))
    if _is_period_value_data(value):
        return _resolve_period_values(value.get("values"), period_idx=period_idx)
    if hasattr(value, "value") and hasattr(value, "unit"):
        return _coerce_optional_float(getattr(value, "value"))
    if hasattr(value, "values") and hasattr(value, "unit"):
        return _resolve_period_values(getattr(value, "values"), period_idx=period_idx)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Unsupported scalar report value type: {type(value)!r}.")


def split_report_value(
    value: Any,
    *,
    period_idx: int | None = None,
) -> tuple[Any, Optional[str]]:
    """Return ``(value, unit)`` for scalar, multiperiod, or array report data."""
    value = _normalise_input_object(value)
    if value is None:
        return None, None

    if isinstance(value, float | int):
        return value, None

    if isinstance(value, Value):
        if value.num_periods == 1:
            return _coerce_optional_float(value.value), value.unit
        if period_idx is None:
            return _value_array_to_list(value), value.unit
        return _resolve_period_values(
            value.period_values, period_idx=period_idx
        ), value.unit

    if _is_array_value_data(value):
        return _coerce_array_values(value.get("values")), value.get("unit")

    unit = (
        value.get("unit")
        if isinstance(value, Mapping)
        else getattr(value, "unit", None)
    )
    if _is_numeric_array_like(value):
        return _coerce_array_values(value), unit

    try:
        return _resolve_scalar_value(value, period_idx=period_idx), unit
    except KeyError, TypeError, ValueError:
        return None, unit
