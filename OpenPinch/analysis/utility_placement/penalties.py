"""Deterministic aggregation and disjoint feasible/infeasible scalar maps."""

from __future__ import annotations

import math

from ...domain.enums import PenaltyForm
from ..numerics import g_ineq_penalty


def g_penalty(
    *,
    hot_fallback_duty: float,
    cold_fallback_duty: float,
    required_hot_duty: float,
    required_cold_duty: float,
) -> float:
    """Return the canonical squared default-utility duty penalty."""
    values = (
        hot_fallback_duty,
        cold_fallback_duty,
        required_hot_duty,
        required_cold_duty,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("g_penalty duties must be finite and non-negative")

    def residual(fallback: float, required: float) -> float:
        if required == 0.0:
            if fallback != 0.0:
                raise ValueError("fallback duty requires positive residual duty")
            return 0.0
        return fallback / required

    return float(
        g_ineq_penalty(
            (
                residual(hot_fallback_duty, required_hot_duty),
                residual(cold_fallback_duty, required_cold_duty),
            ),
            form=PenaltyForm.SQUARE,
        )
    )


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


def penalized_feasible_objective_scalar(base: float, penalty: float) -> float:
    """Combine a feasible scalar and penalty while retaining the (0, 1) partition."""
    if (
        not math.isfinite(base)
        or not 0.0 < base < 1.0
        or not math.isfinite(penalty)
        or penalty < 0.0
    ):
        raise ValueError("feasible scalar and penalty must be finite and valid")
    mapped_penalty = penalty / (1.0 + penalty)
    result = base + (1.0 - base) * mapped_penalty
    return min(result, math.nextafter(1.0, 0.0))


def infeasible_objective_scalar(normalized_violation: float) -> float:
    """Map a non-negative violation monotonically into [1, 2)."""
    if not math.isfinite(normalized_violation) or normalized_violation < 0.0:
        raise ValueError("normalized violation must be finite and non-negative")
    mapped = 1.0 + normalized_violation / (1.0 + normalized_violation)
    return min(mapped, math.nextafter(2.0, 1.0))


__all__ = [
    "aggregate_weighted_objective",
    "feasible_objective_scalar",
    "g_penalty",
    "infeasible_objective_scalar",
    "penalized_feasible_objective_scalar",
]
