"""Problem orchestration for single-stage refrigerant screening."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..analysis.heat_pumps.refrigerant_screen import (
    RefrigerantScreenThresholds,
    resolve_refrigerants,
    screen_refrigerants,
)
from ..contracts.hpr_refrigerant_screen import (
    HprRefrigerantScreenResult,
    RefrigerantScreenQuantity,
    RefrigerantScreenRow,
)
from ..contracts.units import coerce_output_value, standardise_input_value
from ..domain.configuration import tol
from ..domain.enums import HeatPumpAndRefrigerationCycle
from ..domain.targets import HeatPumpTargetBase
from ..domain.value import Value
from ._problem.arguments import temporary_zone_configuration
from ._problem.targeting.provenance import make_provenance

if TYPE_CHECKING:
    from .problem import PinchProblem

# Source unit and output unit rule of each reported quantity.
_TEMPERATURE = ("degC", "hpr_screen_temperature")
_DIFFERENCE = ("delta_degC", "hpr_screen_temperature_difference")
_PRESSURE = ("Pa", "hpr_screen_pressure")
_VHC = ("kJ/m^3", "hpr_screen_suction_vhc")


_CARNOT_CYCLES = frozenset(
    {
        HeatPumpAndRefrigerationCycle.CascadeCarnot.value,
        HeatPumpAndRefrigerationCycle.ParallelCarnot.value,
    }
)


def _placed_levels(levels, duties) -> list[float]:
    """Distinct levels, in order, of the ports that carry duty."""
    return list(dict.fromkeys(float(t) for t, q in zip(levels, duties) if abs(q) > tol))


def calculate_hpr_refrigerant_screen(
    problem: "PinchProblem",
    *,
    target=None,
    T_evap=None,
    T_cond=None,
    refrigerants: str | Sequence[str] | None = None,
    options: Mapping[str, Any] | None = None,
) -> HprRefrigerantScreenResult:
    """Screen candidate fluids between a Carnot target's or explicit levels."""
    root = problem._build_execution_master_zone()
    zone, period_ids, prerequisites = root, tuple(problem.period_ids), ()
    given = (T_evap is not None, T_cond is not None)
    if any(given) if target is not None else not all(given):
        raise ValueError(
            "Refrigerant screening takes either a Carnot target or both T_evap and "
            "T_cond."
        )
    if target is not None:
        if (
            not isinstance(target, HeatPumpTargetBase)
            or target.hpr_cycle not in _CARNOT_CYCLES
        ):
            raise TypeError(
                "Refrigerant screening needs a Carnot heat pump or refrigeration "
                "target."
            )
        if not target.hpr_success:
            raise ValueError("Refrigerant screening needs a successful Carnot target.")
        zone = problem._resolve_target_zone(
            target.provenance.zone_address, master_zone=root
        )
        period_ids = target.provenance.period_ids
        prerequisites = (target.provenance.identity,)
        details = target.hpr_details
        T_evap = _placed_levels(details.T_evap, details.Q_evap)
        T_cond = _placed_levels(details.T_cond, details.Q_cond)
    with temporary_zone_configuration(root, options):
        config = zone.config

        def quantity(value, kind) -> RefrigerantScreenQuantity | None:
            if value is None:
                return None
            unit, metric = kind
            converted = coerce_output_value(
                Value(value, unit=unit), metric_name=metric, config=config
            )
            return RefrigerantScreenQuantity(
                value=float(converted), unit=converted.unit
            )

        def levels(value) -> tuple[float, ...]:
            items = value if isinstance(value, Sequence) else (value,)
            return tuple(
                float(standardise_input_value(v, field_name="t_supply", config=config))
                for v in items
            )

        thresholds = RefrigerantScreenThresholds.from_configuration(config)
        T_evap_levels, T_cond_levels = levels(T_evap), levels(T_cond)
        fluids = resolve_refrigerants(refrigerants)
        entries = screen_refrigerants(T_evap_levels, T_cond_levels, fluids, thresholds)
        return HprRefrigerantScreenResult(
            provenance=make_provenance(
                problem,
                "target.hpr_refrigerant_screen",
                zone,
                period_ids=period_ids,
                prerequisites=prerequisites,
                settings={
                    **config._values,
                    "T_evap": list(T_evap_levels),
                    "T_cond": list(T_cond_levels),
                    "refrigerants": list(fluids),
                },
            ),
            source="explicit_levels" if target is None else "carnot_target",
            T_evap_levels=tuple(quantity(t, _TEMPERATURE) for t in T_evap_levels),
            T_cond_levels=tuple(quantity(t, _TEMPERATURE) for t in T_cond_levels),
            dt_cont=quantity(thresholds.dt_cont, _DIFFERENCE),
            max_pressure_ratio=thresholds.max_pressure_ratio,
            min_suction_pressure=quantity(thresholds.min_suction_pressure, _PRESSURE),
            critical_margin=quantity(thresholds.critical_margin, _DIFFERENCE),
            rows=tuple(
                RefrigerantScreenRow(
                    fluid=e.fluid,
                    T_evap=quantity(e.T_evap, _TEMPERATURE),
                    T_cond=quantity(e.T_cond, _TEMPERATURE),
                    T_crit=quantity(e.T_crit, _TEMPERATURE),
                    p_evap=quantity(e.p_evap, _PRESSURE),
                    p_cond=quantity(e.p_cond, _PRESSURE),
                    pressure_ratio=e.pressure_ratio,
                    suction_vhc=quantity(e.suction_vhc, _VHC),
                    verdict=e.verdict,
                    rank=e.rank,
                )
                for e in entries
            ),
        )
