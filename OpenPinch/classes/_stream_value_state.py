"""Stateless value and operating-period helpers for thermal streams."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from .value import Value


def coerce_to_value(value, *, target_unit: str) -> Value | None:
    """Coerce one supported input to a value expressed in ``target_unit``."""
    if value is None:
        return None

    raw_value = value
    if hasattr(raw_value, "model_dump") and not isinstance(raw_value, Mapping):
        raw_value = raw_value.model_dump(mode="python")

    if isinstance(raw_value, Mapping) and is_period_value_data(raw_value):
        raw_value = {
            "values": raw_value.get("values"),
            "unit": raw_value.get("unit"),
        }

    parsed = Value(raw_value)
    if parsed.unit == target_unit:
        return parsed
    if parsed.unit in {"", "-"}:
        return Value(parsed.value, unit=target_unit)
    return parsed.to(target_unit)


def period_vector_size(values: Iterable[Value | None]) -> int:
    """Return the largest operating-period count represented by ``values``."""
    counts = [value.num_periods for value in values if isinstance(value, Value)]
    return max(counts) if counts else 1


def value_array(
    value: Value | None,
    *,
    size: int,
    stream_name: str,
) -> np.ndarray:
    """Return one value as a scalar-broadcast period vector."""
    if value is None:
        return np.zeros(size, dtype=float)
    if value.num_periods == 1:
        return np.full(size, float(value.value), dtype=float)
    arr = value.period_values.astype(float)
    if arr.size != size:
        raise ValueError(
            f"Period count for stream '{stream_name}' is inconsistent with its values."
        )
    return arr


def build_value(magnitudes, *, unit: str) -> Value:
    """Construct a scalar or multiperiod value from numeric magnitudes."""
    arr = np.asarray(magnitudes, dtype=float).reshape(-1)
    if arr.size == 1:
        return Value(float(arr[0]), unit=unit)
    return Value(arr, unit=unit)


def copy_value(value: Value | None) -> Value | None:
    """Return a defensive value copy while preserving ``None``."""
    if value is None:
        return None
    copied = Value(value)
    reason = getattr(value, "_read_only_reason", None)
    return copied._make_read_only(reason) if reason is not None else copied


def resolve_period_weights(
    period_ids: Mapping[str, int] | Iterable[str],
    weights,
) -> np.ndarray:
    """Return the canonical non-negative weight vector for ``period_ids``.

    Missing trailing values default to one. Excess, non-finite, negative, and
    all-zero vectors are rejected so every consumer sees the same period model.
    """
    expected_len = (
        len(period_ids)
        if hasattr(period_ids, "__len__")
        else len(list(period_ids))
    )
    if expected_len <= 0:
        raise ValueError("At least one operating period is required.")
    if weights is None:
        resolved = np.ones(expected_len, dtype=float)
    else:
        if isinstance(weights, Mapping):
            ordered_period_ids = list(period_ids)
            unknown_ids = set(weights) - set(ordered_period_ids)
            if unknown_ids:
                unknown = ", ".join(sorted(str(item) for item in unknown_ids))
                raise ValueError(f"Unknown period weight id(s): {unknown}.")
            supplied_values = []
            missing_seen = False
            for period_id in ordered_period_ids:
                if period_id not in weights:
                    missing_seen = True
                    continue
                if missing_seen:
                    raise ValueError("Missing period weights must be trailing.")
                supplied_values.append(weights[period_id])
            supplied = np.asarray(supplied_values, dtype=float).reshape(-1)
        else:
            supplied = np.asarray(weights, dtype=float).reshape(-1)
        if supplied.size > expected_len:
            raise ValueError(
                f"Expected at most {expected_len} period weight(s), "
                f"got {supplied.size}."
            )
        resolved = np.ones(expected_len, dtype=float)
        resolved[: supplied.size] = supplied
    if not np.isfinite(resolved).all():
        raise ValueError("Period weights must be finite.")
    if np.any(resolved < 0.0):
        raise ValueError("Period weights must be non-negative.")
    if float(resolved.sum()) <= 0.0:
        raise ValueError("At least one period weight must be positive.")
    return resolved


def is_period_value_data(value: Mapping) -> bool:
    """Return whether a mapping uses the structured period-value contract."""
    keys = set(value)
    return keys.issubset({"values", "period_ids", "weights", "unit"}) and (
        "values" in keys or "period_ids" in keys or "weights" in keys
    )


def normalise_period_ids(
    period_ids: dict[str, int] | list[str] | tuple[str, ...] | None,
) -> dict[str, int] | None:
    """Normalize period identifiers to the canonical identifier-index mapping."""
    if period_ids is None:
        return None
    if isinstance(period_ids, dict):
        return {str(key): int(val) for key, val in period_ids.items()}
    return {str(period_id): idx for idx, period_id in enumerate(period_ids)}


def normalise_weights(
    weights,
    *,
    expected_len: int,
) -> np.ndarray | None:
    """Normalize an operating-period weight vector using existing semantics."""
    if weights is None:
        return None
    arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.size != expected_len:
        raise ValueError("weights length must match the number of periods.")
    total = float(arr.sum())
    if total > 0.0:
        arr = arr / total
    return arr


def validate_num_periods(
    values: Iterable[Value | None],
    *,
    stream_name: str,
) -> int:
    """Return the shared period count or raise the existing mismatch error."""
    counts = np.asarray(
        [value.num_periods for value in values if isinstance(value, Value)],
        dtype=int,
    )
    period_value_counts = counts[counts > 1]
    if period_value_counts.size == 0:
        return 1
    if int(period_value_counts.max()) != int(period_value_counts.min()):
        raise ValueError(f"Stream inputs for {stream_name} have unequal period counts.")
    return int(period_value_counts.max())


__all__: list[str] = []
