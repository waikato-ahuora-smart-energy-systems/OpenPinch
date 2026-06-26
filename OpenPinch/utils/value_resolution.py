"""Helpers for resolving scalar and multiperiod value payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..classes.value import Value

_SCALAR_PAYLOAD_KEYS = {"value", "unit", "weights"}
_PERIOD_PAYLOAD_KEYS = {"values", "period_ids", "weights", "unit"}

__all__ = [
    "get_scalar_value",
    "get_period_value",
    "resolve_value_array",
]


def _normalise_input_object(value: Any) -> Any:
    if hasattr(value, "model_dump") and not isinstance(value, Mapping):
        return value.model_dump(mode="python")
    return value


def _is_bool_like(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _is_numeric_scalar(value: Any) -> bool:
    return isinstance(
        value, (int, float, np.integer, np.floating)
    ) and not _is_bool_like(value)


def _is_value_with_unit(value: Any) -> bool:
    return hasattr(value, "value") and hasattr(value, "unit")


def _is_period_value_object(value: Any) -> bool:
    return hasattr(value, "values")


def _is_scalar_value_payload(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and "value" in value
        and set(value).issubset(_SCALAR_PAYLOAD_KEYS)
    )


def _is_period_value_payload(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    keys = set(value)
    return keys.issubset(_PERIOD_PAYLOAD_KEYS) and (
        "values" in keys or "period_ids" in keys or "weights" in keys
    )


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    return None if np.isnan(scalar) else scalar


def _resolve_period_values(
    values: Any,
    *,
    period_idx: int | None = None,
    default_allowed: bool = True,
) -> float | None:
    if values is None:
        return None
    if _is_bool_like(values):
        raise TypeError("Boolean values are not supported.")

    resolved = np.asarray(values, dtype=float).reshape(-1)
    if resolved.size == 0:
        raise ValueError("Period values cannot be empty.")
    if resolved.size == 1:
        return _coerce_optional_float(resolved[0])

    if period_idx is None:
        if not default_allowed:
            raise ValueError("period_idx is required for period values.")
        period_idx = 0

    period_idx = int(period_idx)
    if period_idx < 0:
        raise ValueError("period_idx must be a non-negative integer.")
    if period_idx >= resolved.size:
        raise ValueError(
            f"period_idx {period_idx} is out of range for {resolved.size} period(s)."
        )
    return _coerce_optional_float(resolved[period_idx])


def get_period_value(
    value: Any,
    *,
    period_idx: int | None = None,
    default_allowed: bool = True,
) -> float | None:
    """Return one scalar magnitude from a scalar or period value-like object."""
    value = _normalise_input_object(value)
    if value is None:
        return None
    if _is_bool_like(value):
        raise TypeError("Boolean values are not supported.")

    if isinstance(value, Mapping):
        if _is_period_value_payload(value):
            if "values" not in value:
                raise KeyError("values")
            return _resolve_period_values(
                value.get("values"),
                period_idx=period_idx,
                default_allowed=default_allowed,
            )
        if _is_scalar_value_payload(value):
            return get_period_value(
                value.get("value"),
                period_idx=period_idx,
                default_allowed=default_allowed,
            )
        raise TypeError("Unsupported mapping for period value resolution.")

    if isinstance(value, Value):
        if value.num_periods <= 1:
            return _coerce_optional_float(value.value)
        return _resolve_period_values(
            value.period_values,
            period_idx=period_idx,
            default_allowed=default_allowed,
        )

    if _is_period_value_object(value):
        return _resolve_period_values(
            getattr(value, "values"),
            period_idx=period_idx,
            default_allowed=default_allowed,
        )

    if _is_value_with_unit(value):
        return get_period_value(
            getattr(value, "value"),
            period_idx=period_idx,
            default_allowed=default_allowed,
        )

    if _is_numeric_scalar(value):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise TypeError(
                f"Unsupported string value: {value}. "
                "String must be convertible to float."
            ) from exc

    if hasattr(value, "units"):
        return get_period_value(
            Value(value),
            period_idx=period_idx,
            default_allowed=default_allowed,
        )

    raise TypeError(
        "Unsupported type: expected numeric scalar, Value, ValueWithUnit, "
        "or serialized scalar/period value payload."
    )


def get_scalar_value(
    value: Any,
    *,
    period_idx: int | None = None,
    default_allowed: bool = True,
) -> float | None:
    """Resolve one scalar value and reject operator or zone-spec payloads."""
    if value is None:
        return None
    value = _normalise_input_object(value)
    return get_period_value(
        value,
        period_idx=period_idx,
        default_allowed=default_allowed,
    )


def resolve_value_array(value: Any) -> np.ndarray:
    """Resolve scalar or multiperiod value-like payloads into a 1-D float array."""
    value = _normalise_input_object(value)
    if value is None:
        return np.array([], dtype=float)
    if _is_bool_like(value):
        raise TypeError("Boolean values are not supported.")

    if isinstance(value, Mapping):
        if _is_period_value_payload(value):
            if "values" not in value:
                raise KeyError("values")
            return np.asarray(value.get("values"), dtype=float).reshape(-1)
        if _is_scalar_value_payload(value):
            scalar = get_scalar_value(value)
            return (
                np.array([], dtype=float)
                if scalar is None
                else np.asarray([scalar], dtype=float)
            )
        raise TypeError("Unsupported mapping for value array resolution.")

    if isinstance(value, Value):
        return np.asarray(value.period_values, dtype=float).reshape(-1)

    if _is_period_value_object(value):
        return np.asarray(getattr(value, "values"), dtype=float).reshape(-1)

    if _is_value_with_unit(value):
        scalar = get_scalar_value(value)
        return (
            np.array([], dtype=float)
            if scalar is None
            else np.asarray([scalar], dtype=float)
        )

    if _is_numeric_scalar(value):
        return np.asarray([float(value)], dtype=float)

    if isinstance(value, str):
        return np.asarray([get_scalar_value(value)], dtype=float)

    if hasattr(value, "units"):
        return np.asarray(Value(value).period_values, dtype=float).reshape(-1)

    raise TypeError(
        "Unsupported type: expected numeric scalar, Value, ValueWithUnit, "
        "or serialized scalar or multiperiod value payload."
    )
