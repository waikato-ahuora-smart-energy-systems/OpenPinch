"""Cases for designing a heat exchanger network with a heat pump in it."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..analysis.heat_exchanger_networks.errors import WorkflowContractError
from ..analysis.heat_exchanger_networks.heat_pump_cost import (
    heat_pump_record,
    network_with_heat_pumps,
)
from ..contracts.hpr import HprLevels, HPRTargetingError
from .case_derivation import (
    _case_with_heat_pump_streams,
    _require_heat_pump_stream_support,
)


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
    costing=None,
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
    _require_heat_pump_stream_support(problem)
    multiplier = float(multiplier)
    root = problem._require_prepared_root_zone()
    levels = HprLevels.model_validate(levels or {})
    # The target's own precedence: the levels' argument, then its options,
    # then the problem's setting.
    approach = levels.minimum_approach_temperature
    if approach is None:
        approach = float((options or {}).get("HPR_DT_CONT", root.config.hpr.dt_cont))
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
            # The costing group of the design this case is for, when that
            # design's options differ from the problem's stored ones.
            costing=root.config.costing if costing is None else costing,
            cost_per_kw=cost_per_kw,
        ),
    )


_REQUEST_KEYS = (
    "cycle",
    "load",
    "levels",
    "cycle_options",
    "search",
    "options",
    "machine",
)


def heat_pump_request(heat_pump) -> dict[str, Any]:
    """The ``heat_pump=`` argument of a design as keyword arguments.

    A cycle name (``"carnot"``), or a mapping with any of ``cycle``
    (``"carnot"`` when left out), ``load``, ``levels``, ``cycle_options``,
    ``search`` and ``options`` as ``target.heat_pump_and_refrigeration``
    takes them, and ``machine``, the name the heat pump's streams and
    record carry.
    """
    if isinstance(heat_pump, str):
        return {"cycle": heat_pump}
    if not isinstance(heat_pump, Mapping):
        raise TypeError(
            "heat_pump must be a cycle name or a mapping with 'cycle' and the "
            "arguments of target.heat_pump_and_refrigeration."
        )
    unknown = sorted(set(heat_pump) - set(_REQUEST_KEYS))
    if unknown:
        raise ValueError(
            f"heat_pump does not take {', '.join(map(repr, unknown))}; it takes "
            + ", ".join(_REQUEST_KEYS)
            + "."
        )
    return {"cycle": "carnot"} | dict(heat_pump)


def design_with_heat_pump(
    problem,
    heat_pump,
    *,
    cost_per_kw: float,
    multipliers,
    arguments: Mapping,
    costing=None,
):
    """Design ``problem`` without a heat pump and, at each of
    ``multipliers``, with one, and rank every network together.

    ``heat_pump`` is a cycle name or a mapping of the keyword arguments of
    ``heat_pump_design_case``; ``arguments`` are those of
    ``design.heat_exchanger_network``.
    """
    request = heat_pump_request(heat_pump)
    # Before any design runs its solvers or replaces the stored design.
    _require_heat_pump_stream_support(problem)
    if not cost_per_kw:
        warnings.warn(
            "heat_pump_cost is 0, so the heat pump designs are ranked with a "
            "free compressor: on electricity and their exchangers alone. Give "
            "heat_pump_cost in $ per kW of condenser capacity.",
            UserWarning,
            stacklevel=2,
        )
    plain = problem.design.heat_exchanger_network(**arguments)
    runs = []
    for multiplier in multipliers:
        try:
            run = heat_pump_design_case(
                problem,
                multiplier,
                cost_per_kw=cost_per_kw,
                costing=costing,
                **request,
            )
            view = run.case.design.heat_exchanger_network(
                **dict(arguments) | {"dt_cont_multipliers": [multiplier]}
            )
        except (HPRTargetingError, WorkflowContractError) as error:
            # No target at this multiplier, or no network around its streams.
            warnings.warn(
                f"No heat pump network at dt_cont multiplier {multiplier:g}: {error}",
                UserWarning,
                stacklevel=2,
            )
            continue
        runs.append((run, view))
    return heat_pump_design_view(problem, plain, runs)


def heat_pump_design_view(problem, plain, runs):
    """One view ranking the designs without the heat pump and with it.

    ``plain`` is the problem's design without a heat pump and ``runs`` pairs
    each heat pump case with the view of its design. Every network of a
    heat pump design takes the case's machine record and its annual cost
    (``network_with_heat_pumps``) and is ranked on that total among the
    rest. The view is ``plain``, its result holding them all and the
    cheapest selected.
    """
    from ..analysis.heat_exchanger_networks.results.selection import (
        ranked_networks,
        select_network,
    )

    result = plain.result
    ranked = list(ranked_networks(result))
    for run, view in runs:
        for outcome in ranked_networks(view.result):
            network = network_with_heat_pumps(outcome.network, (run.record,))
            ranked.append(
                outcome.model_copy(
                    update={
                        "network": network,
                        "objective_value": network.total_annual_cost,
                    }
                )
            )
    result.ranked_networks = tuple(ranked)
    select_network(result)
    return plain


__all__ = ["HeatPumpDesignCase", "heat_pump_design_case", "heat_pump_design_view"]
