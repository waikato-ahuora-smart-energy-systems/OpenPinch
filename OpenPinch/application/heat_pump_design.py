"""Cases for designing a heat exchanger network with a heat pump in it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..analysis.heat_exchanger_networks.heat_pump_cost import heat_pump_record
from ..contracts.hpr import HprLevels
from .case_derivation import _case_with_heat_pump_streams


@dataclass(frozen=True)
class HeatPumpDesignCase:
    """The case a heat pump network is designed on at one dt_cont multiplier.

    ``target`` is the heat pump target solved for that multiplier and
    ``case`` the problem with its streams.
    """

    multiplier: float
    target: Any
    case: Any
    record: Any = None


def heat_pump_design_case(
    problem,
    multiplier: float,
    *,
    cycle: str = "carnot",
    load=None,
    levels=None,
    cycle_options=None,
    search=None,
    options=None,
    machine: str = "HP",
    cost_per_kw: float = 0.0,
) -> HeatPumpDesignCase:
    """The case for a heat pump design at ``multiplier``.

    The heat pump is targeted on a copy of ``problem`` whose root zone
    multiplier is scaled by ``multiplier``, with the heat pump's own
    approach (``levels``' ``minimum_approach_temperature``, else
    ``HPR_DT_CONT``) scaled alike: the contributions the design at that
    multiplier works at. Its streams go into ``problem`` itself at the
    unscaled approach, which the design then scales with every other
    stream's. ``load``, ``levels``, ``cycle_options``, ``search`` and
    ``options`` are those of ``target.heat_pump_and_refrigeration``.
    """
    multiplier = float(multiplier)
    root = problem._require_prepared_root_zone()
    levels = HprLevels.model_validate(levels or {})
    approach = levels.minimum_approach_temperature
    if approach is None:
        approach = float(root.config.hpr.dt_cont)
    zone_multiplier = float(root.dt_cont_multiplier)
    scaled = type(problem)(problem.to_problem_json(), project_name=problem.project_name)
    scaled.set_dt_cont_multiplier(zone_multiplier * multiplier)
    target = scaled.target.heat_pump_and_refrigeration(
        cycle,
        duty="heat_pump",
        placement="process",
        load=load,
        levels=levels.model_copy(
            update={
                "minimum_approach_temperature": approach * zone_multiplier * multiplier
            }
        ),
        cycle_options=cycle_options,
        search=search,
        options=options,
    )
    return HeatPumpDesignCase(
        multiplier=multiplier,
        target=target,
        # The zone multiplier scales the heat pump's streams too.
        case=_case_with_heat_pump_streams(
            problem, target, dt_cont=approach, machine=machine
        ),
        record=heat_pump_record(
            target,
            machine=machine,
            costing=root.config.costing,
            cost_per_kw=cost_per_kw,
        ),
    )


__all__ = ["HeatPumpDesignCase", "heat_pump_design_case"]
