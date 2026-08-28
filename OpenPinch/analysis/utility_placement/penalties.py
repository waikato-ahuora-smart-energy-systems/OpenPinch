"""Deterministic aggregation and disjoint feasible/infeasible scalar maps."""

from __future__ import annotations

import math

_DEFAULT_UTILITY_NAMES = frozenset({"hu", "cu"})


def default_utility_penalty(
    *,
    names: tuple[str, ...],
    duties: tuple[float, ...],
    reference_duty: float,
) -> float:
    """Return normalized positive duty assigned to generated default utilities."""
    if len(names) != len(duties):
        raise ValueError("default utility names and duties must align")
    if (
        not math.isfinite(reference_duty)
        or reference_duty < 0.0
        or any(not math.isfinite(duty) or duty < 0.0 for duty in duties)
    ):
        raise ValueError(
            "default utility penalty inputs must be finite and non-negative"
        )
    default_duty = math.fsum(
        duty
        for name, duty in zip(names, duties, strict=True)
        if name.strip().casefold() in _DEFAULT_UTILITY_NAMES
    )
    return default_duty / max(reference_duty, 1.0)


def aggregate_weighted_objective(
    values: tuple[float, ...], weights: tuple[float, ...]
) -> float:
    """Return a raw-weighted canonical sum without normalization."""
    if len(values) != len(weights):
        raise ValueError("objective values and period weights must align")
    if any(not math.isfinite(value) for value in values + weights):
        raise ValueError("objective values and weights must be finite")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("period weights must be non-negative")
    terms = sorted(
        value * weight for value, weight in zip(values, weights, strict=True)
    )
    return math.fsum(terms)


def feasible_objective_scalar(cost: float, *, scale: float) -> float:
    """Map any finite physical cost monotonically into the open interval (0, 1)."""
    if not math.isfinite(cost) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("feasible cost must be finite and scale must be positive")
    mapped = 0.5 + math.atan(cost / scale) / math.pi
    return min(max(mapped, math.nextafter(0.0, 1.0)), math.nextafter(1.0, 0.0))


def infeasible_objective_scalar(normalized_violation: float) -> float:
    """Map a non-negative violation monotonically into [1, 2)."""
    if not math.isfinite(normalized_violation) or normalized_violation < 0.0:
        raise ValueError("normalized violation must be finite and non-negative")
    mapped = 1.0 + normalized_violation / (1.0 + normalized_violation)
    return min(mapped, math.nextafter(2.0, 1.0))


__all__ = [
    "aggregate_weighted_objective",
    "default_utility_penalty",
    "feasible_objective_scalar",
    "infeasible_objective_scalar",
]
