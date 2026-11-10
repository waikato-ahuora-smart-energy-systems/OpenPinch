"""Problem orchestration for heat pump and refrigeration load kinks."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from numbers import Real
from typing import TYPE_CHECKING

from ..analysis.heat_pumps.common.load_kinks import (
    HprLoadKinkProfile,
    find_hpr_load_kinks,
)
from ..analysis.parallelism import supported_workers
from ..contracts.heat_recovery_dt_min import HeatRecoveryQuantity
from ..contracts.hpr_load_kinks import HprLoadKink, HprLoadKinksResult
from ..contracts.units import coerce_output_value
from ..domain.enums import ProblemTableLabel
from ..domain.value import Value
from ._problem.targeting.state import snapshot_problem
from .heat_recovery_dt_min import _resolve_zone_and_period

if TYPE_CHECKING:
    from ..domain.configuration import Configuration
    from ..domain.zone import Zone
    from .problem import PinchProblem


def _quantity(
    value: float, *, source_unit: str, metric_name: str, config: "Configuration"
) -> HeatRecoveryQuantity:
    converted = coerce_output_value(
        Value(value, unit=source_unit), metric_name=metric_name, config=config
    )
    return HeatRecoveryQuantity(value=float(converted), unit=converted.unit)


def _result_contract(
    profile: HprLoadKinkProfile,
    *,
    zone: "Zone",
    period_id: str,
    is_refrigeration: bool,
    steepening: float,
) -> HprLoadKinksResult:
    config = zone.config

    def duty(value: float) -> HeatRecoveryQuantity:
        return _quantity(
            value, source_unit="kW", metric_name="hpr_load_kink_duty", config=config
        )

    def slope(value: float) -> HeatRecoveryQuantity:
        return _quantity(
            value,
            source_unit="delta_degC",
            metric_name="hpr_load_kink_slope",
            config=config,
        )

    return HprLoadKinksResult(
        scope=zone.address,
        period_id=period_id,
        is_refrigeration=is_refrigeration,
        steepening=steepening,
        maximum_duty=duty(profile.maximum_duty),
        vertices=tuple(
            HprLoadKink(
                load_fraction=vertex.load_fraction,
                duty=duty(vertex.duty),
                temperature=_quantity(
                    vertex.temperature,
                    source_unit="degC",
                    metric_name="hpr_load_kink_temperature",
                    config=config,
                ),
                slope_before=slope(vertex.slope_before),
                slope_after=None
                if vertex.slope_after is None
                else slope(vertex.slope_after),
                kind=vertex.kind,
            )
            for vertex in profile.vertices
        ),
        logical_load_fractions=profile.logical_load_fractions,
    )


def calculate_hpr_load_kinks(
    problem: "PinchProblem",
    *,
    zone=None,
    period_id=None,
    is_refrigeration=False,
    steepening=2.0,
) -> HprLoadKinksResult:
    """Classify the net load profile vertices without changing the problem."""
    if not isinstance(is_refrigeration, bool):
        raise TypeError("is_refrigeration must be True or False.")
    if (
        isinstance(steepening, bool)
        or not isinstance(steepening, Real)
        or not math.isfinite(steepening)
        or steepening <= 1.0
    ):
        raise ValueError("steepening must be a finite number greater than 1.")
    selected, _, canonical_id = _resolve_zone_and_period(
        problem,
        zone=zone,
        period_id=period_id,
        requirement="HPR load kinks require",
    )
    direct = snapshot_problem(problem).target.direct_heat_integration(
        zone=selected.address, period_id=canonical_id
    )
    try:
        profile = find_hpr_load_kinks(
            direct.pt[ProblemTableLabel.T],
            direct.pt[ProblemTableLabel.H_NET_COLD],
            direct.pt[ProblemTableLabel.H_NET_HOT],
            is_refrigeration=is_refrigeration,
            steepening=steepening,
        )
    except ValueError as exc:
        raise ValueError(
            f"{exc} Scope {selected.address!r}, period {canonical_id!r}."
        ) from exc
    from ._problem.targeting.provenance import make_provenance

    result = _result_contract(
        profile,
        zone=selected,
        period_id=canonical_id,
        is_refrigeration=is_refrigeration,
        steepening=steepening,
    )
    return result.model_copy(
        update={
            "provenance": make_provenance(
                problem,
                "target.hpr_load_kinks",
                selected,
                period_ids=(canonical_id,),
                settings={
                    **selected.config._values,
                    "is_refrigeration": is_refrigeration,
                    "steepening": float(steepening),
                },
            )
        }
    )


def calculate_all_period_hpr_load_kinks(
    problem: "PinchProblem",
    *,
    zone=None,
    is_refrigeration=False,
    steepening=2.0,
    workers=1,
) -> dict[str, HprLoadKinksResult]:
    """Classify the net load profile vertices of every canonical period."""
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer.")
    period_ids = tuple(problem.period_ids)

    def solve(period_id: str) -> HprLoadKinksResult:
        return calculate_hpr_load_kinks(
            problem,
            zone=zone,
            period_id=period_id,
            is_refrigeration=is_refrigeration,
            steepening=steepening,
        )

    if supported_workers(workers) == 1:
        return {period_id: solve(period_id) for period_id in period_ids}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(zip(period_ids, executor.map(solve, period_ids), strict=True))


__all__ = ["calculate_all_period_hpr_load_kinks", "calculate_hpr_load_kinks"]
