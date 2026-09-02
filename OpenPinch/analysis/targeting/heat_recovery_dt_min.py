"""Inverse process targeting for a global heat-recovery dt_min."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...contracts.heat_recovery_dt_min import HeatRecoveryDtMinStatus
from ...domain.stream_collection import StreamCollection
from ...domain.value import Value
from .cascade import get_heat_recovery_target_from_pt, get_process_heat_cascade

DT_MIN_TOLERANCE = 1e-6
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


def _validate_requested_recovery(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("requested_heat_recovery must be a finite non-negative scalar")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "requested_heat_recovery must be a finite non-negative scalar"
        ) from exc
    if not math.isfinite(result):
        raise ValueError("requested_heat_recovery must be finite")
    if result < 0.0:
        raise ValueError("requested_heat_recovery must be non-negative")
    return 0.0 if result == 0.0 else result


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
    table = get_process_heat_cascade(
        hot_streams=hot_streams,
        cold_streams=cold_streams,
        is_shifted=True,
        period_idx=period_idx,
        insert_constant_heat_intervals=False,
    )
    if len(table) == 0:
        return 0.0
    recovery = float(get_heat_recovery_target_from_pt(table))
    if not math.isfinite(recovery):
        raise RuntimeError(
            "Heat-recovery dt_min evaluation produced a non-finite value."
        )
    return recovery


def evaluate_process_heat_recovery(
    hot_streams: StreamCollection,
    cold_streams: StreamCollection,
    *,
    dt_min: float,
    period_idx: int = 0,
) -> float:
    """Evaluate recovery on detached streams at one global dt_min."""
    detached_hot = hot_streams.copy(deep=True)
    detached_cold = cold_streams.copy(deep=True)
    recovery = _evaluate_detached_process_heat_recovery(
        detached_hot,
        detached_cold,
        dt_min=float(dt_min),
        period_idx=int(period_idx),
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
    detached_hot = hot_streams.copy(deep=True)
    detached_cold = cold_streams.copy(deep=True)

    def evaluate(dt_min: float) -> float:
        return _evaluate_detached_process_heat_recovery(
            detached_hot,
            detached_cold,
            dt_min=dt_min,
            period_idx=int(period_idx),
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
    if at_thermodynamic_limit and limit <= limit_tolerance:
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
        period_idx=int(period_idx),
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
    if requested <= limit_tolerance:
        while high - low > DT_MIN_TOLERANCE:
            if iterations >= MAXIMUM_ITERATIONS:
                raise RuntimeError("Heat-recovery dt_min bisection failed to converge.")
            midpoint = (low + high) / 2.0
            midpoint_recovery = bounded_recovery(midpoint)
            iterations += 1
            if midpoint_recovery <= limit_tolerance:
                high = midpoint
                upper_recovery = max(0.0, midpoint_recovery)
            else:
                low = midpoint
        return _solution(
            dt_min=high,
            requested=requested,
            achieved=upper_recovery,
            limit=limit,
            status=HeatRecoveryDtMinStatus.ZERO_RECOVERY_BOUNDARY,
            iterations=iterations,
        )

    comparison_target = limit if at_thermodynamic_limit else requested
    feasible_recovery = limit
    while high - low > DT_MIN_TOLERANCE:
        if iterations >= MAXIMUM_ITERATIONS:
            raise RuntimeError("Heat-recovery dt_min bisection failed to converge.")
        midpoint = (low + high) / 2.0
        midpoint_recovery = bounded_recovery(midpoint)
        iterations += 1
        comparison_tolerance = _recovery_tolerance(
            limit,
            comparison_target,
            midpoint_recovery,
        )
        if midpoint_recovery + comparison_tolerance >= comparison_target:
            low = midpoint
            feasible_recovery = min(limit, max(0.0, midpoint_recovery))
        else:
            high = midpoint

    return _solution(
        dt_min=low,
        requested=requested,
        achieved=feasible_recovery,
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
