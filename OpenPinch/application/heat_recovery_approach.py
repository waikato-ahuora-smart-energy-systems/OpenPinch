"""Problem orchestration for inverse heat-recovery approach targeting."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from ..analysis.numerics import get_period_index
from ..analysis.targeting.approach_temperature import (
    HeatRecoveryApproachSolution,
    HeatRecoveryLimitError,
    solve_heat_recovery_approach,
)
from ..contracts.heat_recovery import (
    HeatRecoveryApproachResult,
    HeatRecoveryQuantity,
)
from ..contracts.units import coerce_output_value, standardise_input_value
from ..domain.enums import ZoneType
from ..domain.value import Value

if TYPE_CHECKING:
    from ..domain.zone import Zone
    from .problem import PinchProblem

_SUPPORTED_ZONE_TYPES = frozenset(
    {ZoneType.S.value, ZoneType.P.value, ZoneType.O.value}
)
_SERIALIZED_SCALAR_KEYS = frozenset({"value", "unit"})


def _resolve_zone_and_period(
    problem: "PinchProblem",
    *,
    zone: str | "Zone" | None,
    period_id: str | None,
) -> tuple["Zone", int, str]:
    root = problem._build_execution_master_zone()
    selected = problem._resolve_target_zone(zone, master_zone=root)
    if selected is None:
        raise ValueError(f"Target zone {zone!r} was not found.")
    if selected.type not in _SUPPORTED_ZONE_TYPES:
        raise ValueError(
            "Heat-recovery approach targeting requires a Site, Process Zone, "
            "or Unit Operation scope. Select a direct process-targeting zone "
            "instead of a Community or Region."
        )
    idx, requested_id = get_period_index(
        period_ids=selected.period_ids,
        args={} if period_id is None else {"period_id": period_id},
    )
    canonical_id = requested_id
    if canonical_id is None:
        canonical_id = next(
            (
                candidate
                for candidate, candidate_idx in selected.period_ids.items()
                if candidate_idx == idx
            ),
            str(idx),
        )
    return selected, idx, canonical_id


def _canonical_recovery(heat_recovery: Any, *, config) -> float:
    if isinstance(heat_recovery, bool):
        raise TypeError("heat_recovery must be a finite non-negative scalar")
    value = standardise_input_value(
        heat_recovery,
        field_name="heat_flow",
        config=config,
    )
    if value is None or value.num_periods != 1:
        raise TypeError("heat_recovery must be a finite non-negative scalar")
    return float(value)


def _quantity(
    value: float,
    *,
    source_unit: str,
    metric_name: str,
    config,
) -> HeatRecoveryQuantity:
    converted = coerce_output_value(
        Value(value, unit=source_unit),
        metric_name=metric_name,
        config=config,
    )
    if converted is None:  # pragma: no cover - numeric input always returns a value
        raise RuntimeError(f"Could not convert output metric {metric_name!r}.")
    return HeatRecoveryQuantity(value=float(converted), unit=converted.unit)


def _result_contract(
    solution: HeatRecoveryApproachSolution,
    *,
    zone: "Zone",
    period_id: str,
) -> HeatRecoveryApproachResult:
    heat = {
        "source_unit": "kW",
        "metric_name": "Qr",
        "config": zone.config,
    }
    return HeatRecoveryApproachResult(
        scope=zone.address,
        period_id=period_id,
        approach_temperature=_quantity(
            solution.approach_temperature,
            source_unit="delta_degC",
            metric_name="heat_recovery_approach_temperature",
            config=zone.config,
        ),
        requested_heat_recovery=_quantity(
            solution.requested_heat_recovery,
            **heat,
        ),
        achieved_heat_recovery=_quantity(
            solution.achieved_heat_recovery,
            **heat,
        ),
        thermodynamic_limit=_quantity(solution.thermodynamic_limit, **heat),
        heat_recovery_residual=_quantity(solution.heat_recovery_residual, **heat),
        status=solution.status,
        iterations=solution.iterations,
    )


def calculate_heat_recovery_approach(
    problem: "PinchProblem",
    *,
    heat_recovery,
    zone=None,
    period_id=None,
) -> HeatRecoveryApproachResult:
    """Calculate one selected-period global HRAT without changing the problem."""
    selected, idx, canonical_id = _resolve_zone_and_period(
        problem,
        zone=zone,
        period_id=period_id,
    )
    requested = _canonical_recovery(heat_recovery, config=selected.config)
    try:
        solution = solve_heat_recovery_approach(
            selected.hot_streams,
            selected.cold_streams,
            requested_heat_recovery=requested,
            period_idx=idx,
        )
    except HeatRecoveryLimitError as exc:
        raise ValueError(
            f"Requested heat recovery {exc.requested:g} kW exceeds the "
            f"thermodynamic limit {exc.limit:g} kW for scope "
            f"{selected.address!r}, period {canonical_id!r}."
        ) from exc
    return _result_contract(solution, zone=selected, period_id=canonical_id)


def _is_period_mapping(value: object) -> bool:
    return isinstance(value, Mapping) and not (
        "value" in value and set(value).issubset(_SERIALIZED_SCALAR_KEYS)
    )


def _period_requests(
    heat_recovery,
    *,
    period_ids: tuple[str, ...],
) -> dict[str, Any]:
    if not _is_period_mapping(heat_recovery):
        return {period_id: heat_recovery for period_id in period_ids}
    supplied = tuple(heat_recovery)
    if set(supplied) != set(period_ids) or len(supplied) != len(period_ids):
        raise ValueError(
            "A heat_recovery mapping must contain exactly the canonical period "
            f"IDs: {', '.join(period_ids)}."
        )
    return {period_id: heat_recovery[period_id] for period_id in period_ids}


def calculate_all_period_heat_recovery_approach(
    problem: "PinchProblem",
    *,
    heat_recovery,
    zone=None,
    workers=1,
) -> dict[str, HeatRecoveryApproachResult]:
    """Calculate isolated inverse targets for every canonical period."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    period_ids = tuple(problem.period_ids)
    requests = _period_requests(heat_recovery, period_ids=period_ids)

    def solve(period_id: str) -> HeatRecoveryApproachResult:
        return calculate_heat_recovery_approach(
            problem,
            heat_recovery=requests[period_id],
            zone=zone,
            period_id=period_id,
        )

    if workers == 1:
        return {period_id: solve(period_id) for period_id in period_ids}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        solved = executor.map(solve, period_ids)
        return dict(zip(period_ids, solved, strict=True))


__all__ = [
    "calculate_all_period_heat_recovery_approach",
    "calculate_heat_recovery_approach",
]
