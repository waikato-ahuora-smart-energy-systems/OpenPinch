"""Inverse process targeting for a global heat-recovery dt_min."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real

import numpy as np

from ...contracts.heat_recovery_dt_min import HeatRecoveryDtMinStatus
from ...domain.stream_collection import StreamCollection
from ...domain.value import Value
from .cascade import (
    _get_precise_process_heat_cascade,
    get_heat_recovery_target_from_pt,
)

DT_MIN_TOLERANCE = 1e-6
_BISECTION_WIDTH = DT_MIN_TOLERANCE / 2.0
RECOVERY_ABSOLUTE_TOLERANCE = 1e-6
RECOVERY_RELATIVE_TOLERANCE = 1e-9
MAXIMUM_ITERATIONS = 100


@dataclass(frozen=True)
class HeatRecoveryDtMinSolution:
    """Canonical numerical result before application-owned unit conversion."""

    dt_min: float
    requested_heat_recovery: float
    achieved_heat_recovery: float
    thermodynamic_limit: float
    heat_recovery_residual: float
    status: HeatRecoveryDtMinStatus
    iterations: int


class HeatRecoveryLimitError(ValueError):
    """Raised when requested recovery exceeds the zero-dt_min limit."""

    def __init__(self, requested: float, limit: float) -> None:
        self.requested = requested
        self.limit = limit
        super().__init__(
            f"Requested heat recovery {requested:g} kW exceeds the "
            f"thermodynamic limit {limit:g} kW."
        )


def _validate_real_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (Real, Decimal, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a finite non-negative scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return 0.0 if result == 0.0 else result


def _validate_requested_recovery(value: object) -> float:
    return _validate_real_scalar(value, name="requested_heat_recovery")


def _validate_period_idx(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError("period_idx must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError("period_idx must be a non-negative integer")
    return result


def _recovery_tolerance(*values: float) -> float:
    scale = max((abs(value) for value in values), default=0.0)
    return max(
        RECOVERY_ABSOLUTE_TOLERANCE,
        RECOVERY_RELATIVE_TOLERANCE * scale,
    )


def _set_global_dt_min(
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
    dt_min: float,
) -> None:
    half_dt_min = Value(dt_min / 2.0, unit="delta_degC")
    for stream in (*hot_streams, *cold_streams):
        stream.delta_t_contribution_multiplier_locked = False
        stream.delta_t_contribution_multiplier = 1.0
        stream.delta_t_contribution = half_dt_min


def _evaluate_detached_process_heat_recovery(
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
    *,
    dt_min: float,
    period_idx: int,
) -> float:
    if not math.isfinite(dt_min) or dt_min < 0.0:
        raise ValueError("dt_min must be finite and non-negative")
    if len(hot_streams) == 0 or len(cold_streams) == 0:
        return 0.0

    _set_global_dt_min(hot_streams, cold_streams, dt_min)
    table = _get_precise_process_heat_cascade(
        hot_streams=hot_streams,
        cold_streams=cold_streams,
        period_idx=period_idx,
    )
    if len(table) == 0:
        return 0.0
    recovery = float(get_heat_recovery_target_from_pt(table))
    if not math.isfinite(recovery):
        raise RuntimeError(
            "Heat-recovery dt_min evaluation produced a non-finite value."
        )
    # Remove only binary summation noise, well below the documented recovery
    # tolerances, so the feasibility predicate remains monotone at coincident
    # shifted temperatures.
    return float(f"{recovery:.15g}")


def evaluate_process_heat_recovery(
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
    *,
    dt_min: float,
    period_idx: int = 0,
) -> float:
    """Evaluate recovery on detached streams at one global dt_min."""
    resolved_dt_min = _validate_real_scalar(dt_min, name="dt_min")
    resolved_period_idx = _validate_period_idx(period_idx)
    detached_hot = hot_streams.copy(deep=True)
    detached_cold = cold_streams.copy(deep=True)
    recovery = _evaluate_detached_process_heat_recovery(
        detached_hot,
        detached_cold,
        dt_min=resolved_dt_min,
        period_idx=resolved_period_idx,
    )
    tolerance = _recovery_tolerance(recovery)
    if recovery < 0.0 and recovery >= -tolerance:
        return 0.0
    if recovery < 0.0:
        raise RuntimeError("Process heat recovery was negative outside tolerance.")
    return recovery


def _no_overlap_upper_bound(
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
    *,
    period_idx: int,
) -> float:
    hot = hot_streams.segment_numeric_view(period_idx)
    cold = cold_streams.segment_numeric_view(period_idx)
    hot_mask = hot.active & np.isfinite(hot.t_max)
    cold_mask = cold.active & np.isfinite(cold.t_min)
    if not np.any(hot_mask) or not np.any(cold_mask):
        return 0.0
    return max(
        0.0,
        float(np.max(hot.t_max[hot_mask]) - np.min(cold.t_min[cold_mask])),
    )


def _solution(
    *,
    dt_min: float,
    requested: float,
    achieved: float,
    limit: float,
    status: HeatRecoveryDtMinStatus,
    iterations: int,
) -> HeatRecoveryDtMinSolution:
    return HeatRecoveryDtMinSolution(
        dt_min=0.0 if dt_min == 0.0 else dt_min,
        requested_heat_recovery=requested,
        achieved_heat_recovery=0.0 if achieved == 0.0 else achieved,
        thermodynamic_limit=0.0 if limit == 0.0 else limit,
        heat_recovery_residual=achieved - requested,
        status=status,
        iterations=iterations,
    )


def solve_heat_recovery_dt_min(
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
    *,
    requested_heat_recovery: float,
    period_idx: int = 0,
) -> HeatRecoveryDtMinSolution:
    """Return the greatest feasible global dt_min for a requested recovery.

    Thermodynamic-limit requests use the same plateau boundary rule, allowing
    threshold problems to return a positive dt_min.
    """
    requested = _validate_requested_recovery(requested_heat_recovery)
    resolved_period_idx = _validate_period_idx(period_idx)
    detached_hot = hot_streams.copy(deep=True)
    detached_cold = cold_streams.copy(deep=True)

    def evaluate(dt_min: float) -> float:
        return _evaluate_detached_process_heat_recovery(
            detached_hot,
            detached_cold,
            dt_min=dt_min,
            period_idx=resolved_period_idx,
        )

    limit = evaluate(0.0)
    limit_tolerance = _recovery_tolerance(limit, requested)
    if limit < 0.0 and limit >= -limit_tolerance:
        limit = 0.0
    if limit < 0.0:
        raise RuntimeError("Thermodynamic heat-recovery limit was negative.")
    if requested > limit:
        raise HeatRecoveryLimitError(requested, limit)
    at_thermodynamic_limit = abs(requested - limit) <= limit_tolerance
    if at_thermodynamic_limit and limit == 0.0:
        return _solution(
            dt_min=0.0,
            requested=requested,
            achieved=limit,
            limit=limit,
            status=HeatRecoveryDtMinStatus.AT_THERMODYNAMIC_LIMIT,
            iterations=0,
        )

    upper = _no_overlap_upper_bound(
        detached_hot,
        detached_cold,
        period_idx=resolved_period_idx,
    )

    def bounded_recovery(dt_min: float) -> float:
        recovery = evaluate(dt_min)
        tolerance = _recovery_tolerance(limit, requested, recovery)
        if recovery < -tolerance or recovery > limit + tolerance:
            raise RuntimeError(
                "Heat-recovery dt_min evaluation left its thermodynamic bounds."
            )
        return min(limit, max(0.0, recovery))

    upper_recovery = bounded_recovery(upper)
    if upper_recovery > limit_tolerance:
        raise RuntimeError(
            "The no-overlap dt_min did not produce a valid zero-recovery bracket."
        )

    low = 0.0
    high = upper
    iterations = 0
    if requested == 0.0:
        while high - low > _BISECTION_WIDTH:
            if iterations >= MAXIMUM_ITERATIONS:
                raise RuntimeError("Heat-recovery dt_min bisection failed to converge.")
            midpoint = (low + high) / 2.0
            midpoint_recovery = bounded_recovery(midpoint)
            iterations += 1
            if midpoint_recovery == 0.0:
                high = midpoint
            else:
                low = midpoint
        low_recovery = bounded_recovery(low)
        high_recovery = bounded_recovery(high)
        if high - low > DT_MIN_TOLERANCE or low_recovery == 0.0 or high_recovery != 0.0:
            raise RuntimeError("Heat-recovery dt_min boundary verification failed.")
        return _solution(
            dt_min=high,
            requested=requested,
            achieved=high_recovery,
            limit=limit,
            status=HeatRecoveryDtMinStatus.ZERO_RECOVERY_BOUNDARY,
            iterations=iterations,
        )

    comparison_target = limit if at_thermodynamic_limit else requested

    def is_feasible(recovery: float) -> bool:
        if at_thermodynamic_limit or requested <= RECOVERY_ABSOLUTE_TOLERANCE:
            return recovery >= comparison_target
        comparison_tolerance = _recovery_tolerance(
            limit,
            comparison_target,
            recovery,
        )
        return recovery + comparison_tolerance >= comparison_target

    while high - low > _BISECTION_WIDTH:
        if iterations >= MAXIMUM_ITERATIONS:
            raise RuntimeError("Heat-recovery dt_min bisection failed to converge.")
        midpoint = (low + high) / 2.0
        midpoint_recovery = bounded_recovery(midpoint)
        iterations += 1
        if is_feasible(midpoint_recovery):
            low = midpoint
        else:
            high = midpoint

    low_recovery = bounded_recovery(low)
    high_recovery = bounded_recovery(high)
    if high - low > DT_MIN_TOLERANCE or not is_feasible(low_recovery):
        raise RuntimeError("Heat-recovery dt_min boundary verification failed.")
    if high < upper and is_feasible(high_recovery):
        raise RuntimeError("Heat-recovery dt_min boundary verification failed.")

    return _solution(
        dt_min=low,
        requested=requested,
        achieved=low_recovery,
        limit=limit,
        status=(
            HeatRecoveryDtMinStatus.AT_THERMODYNAMIC_LIMIT
            if at_thermodynamic_limit
            else HeatRecoveryDtMinStatus.SOLVED
        ),
        iterations=iterations,
    )


__all__ = [
    "HeatRecoveryLimitError",
    "HeatRecoveryDtMinSolution",
    "evaluate_process_heat_recovery",
    "solve_heat_recovery_dt_min",
]
