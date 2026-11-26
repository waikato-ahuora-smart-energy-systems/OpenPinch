"""Cases for designing a heat exchanger network with a heat pump in it."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..analysis.heat_exchanger_networks.errors import WorkflowContractError
from ..analysis.heat_exchanger_networks.heat_pump_cost import (
    heat_pump_record,
    network_with_heat_pumps,
)
from ..analysis.heat_exchanger_networks.solver.arrays import heat_pump_stream_roles
from ..contracts.hpr import HprLevels, HprLoad, HPRTargetingError
from .case_derivation import (
    _case_with_heat_pump_streams,
    _case_with_utility_heat_pump_zones,
    _require_heat_pump_site_target,
    _require_heat_pump_stream_support,
    heat_pump_stream_records,
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


HEAT_PUMP_SNAP_K = 1e-4
"""How far (K) a heat pump stream may start or end from a process
temperature and be moved onto it."""


def _process_temperatures(problem) -> np.ndarray:
    """Every shifted temperature (degC) a process stream of ``problem``, or
    a segment of one, starts or ends at, in any period."""
    root = problem._require_prepared_root_zone()
    found = []
    for group in (root.hot_streams, root.cold_streams):
        for _key, stream in group.items():
            for part in (stream, *(stream.segments if stream.has_segments else ())):
                for value in (
                    part.shifted_minimum_temperature,
                    part.shifted_maximum_temperature,
                ):
                    found.extend(np.atleast_1d(np.asarray(value.value, dtype=float)))
    return np.unique(np.asarray(found, dtype=float))


def _on_process_temperatures(case, process, *, approach: float):
    """``case`` with its heat pump streams on the process temperatures
    they lie beside.

    A converged target puts a level at a kink of the composite curves, a
    temperature a process stream starts or ends at, to the search's
    tolerance: 1e-5 K under it on the basic pinch case. A condenser that
    far under the temperature a cold stream ends at leaves that stream's
    last 1e-5 K to the hot utility, a second pinch at the condenser, and
    the case has no network. Each temperature of a condenser or evaporator
    stream within ``HEAT_PUMP_SNAP_K`` of a shifted temperature of
    ``process`` (the problem as the design sees it, without the heat pump)
    is moved onto it; ``approach`` (K) is the heat pump's own contribution
    there. The air's streams are left: they sit at the ambient temperature.
    """

    grid = _process_temperatures(process)
    if not grid.size:
        return case
    source = case.to_problem_json()

    def offset(temperature: float, sign: float) -> float:
        """The move that puts ``temperature`` on the process temperature it
        lies beside; 0 when it lies beside none."""
        shifted = temperature + sign * approach
        nearest = float(grid[np.argmin(np.abs(grid - shifted))])
        return nearest - shifted if abs(nearest - shifted) <= HEAT_PUMP_SNAP_K else 0.0

    for record in source["streams"]:
        role = (record.get("heat_pump") or {}).get("role")
        if role not in ("condenser", "evaporator"):
            continue
        # A condenser is a hot stream, shifted down by its contribution; an
        # evaporator a cold one, shifted up.
        sign = -1.0 if role == "condenser" else 1.0
        ends = [
            (part, key)
            for part in (record, *(record.get("segments") or ()))
            for key in ("t_supply", "t_target")
            if part.get(key) is not None
        ]
        if not ends:
            continue
        # The stream moves whole, each period by its own amount: the first
        # of its temperatures that lies beside a process temperature sets
        # the move. Moving one end of a 0.01 K stream alone would change the
        # span the problem table reads its duty over.
        periods = [
            [float(v) for v in part[key].get("values", [part[key].get("value")])]
            for part, key in ends
        ]
        moves = [
            next((m for m in (offset(t, sign) for t in each) if m), 0.0)
            for each in zip(*periods)
        ]
        for (part, key), values in zip(ends, periods):
            placed = [t + move for t, move in zip(values, moves)]
            part[key] = (
                {**part[key], "values": placed}
                if "values" in part[key]
                else {**part[key], "value": placed[0]}
            )
    return type(case)(source, project_name=case.project_name)


def _scaled(problem, multiplier: float):
    """A copy of ``problem`` as a design at ``multiplier`` sees it: its root
    zone multiplier scaled, and so every stream's and utility's contribution."""
    root = problem._require_prepared_root_zone()
    scaled = type(problem)(problem.to_problem_json(), project_name=problem.project_name)
    scaled.set_dt_cont_multiplier(float(root.dt_cont_multiplier) * float(multiplier))
    return scaled


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
    placement: str = "process",
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
    scaled = _scaled(problem, multiplier)
    if len(problem.period_ids) > 1:
        # One machine over every period, as the network is one network:
        # the only target that holds the heat pump's streams in each period.
        options = dict(options or {}) | {"HPR_MULTIPERIOD_OPTIMIZATION_ENABLED": True}
    target = scaled.target.heat_pump_and_refrigeration(
        cycle,
        duty="heat_pump",
        placement=placement,
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
    record = heat_pump_record(
        target,
        machine=machine,
        # The costing group of the design this case is for, when that
        # design's options differ from the problem's stored ones.
        costing=root.config.costing if costing is None else costing,
        cost_per_kw=cost_per_kw,
        # Each period's share of the year, for a machine over several.
        period_weights=dict(
            zip(problem.period_ids, root.config.problem.period_weights)
        ),
    )
    asked = None if load is None else HprLoad.model_validate(load).fraction
    if record.load_fraction is None and asked is not None:
        # A shared design over several periods reports no load summary.
        record = record.model_copy(update={"load_fraction": float(asked)})
    if placement == "utility":
        # The heat pump's zones of the site; their writer takes the approach
        # the root's multiplier scales to and writes it over that multiplier.
        case = _case_with_utility_heat_pump_zones(
            problem, target, dt_cont=approach * zone_multiplier, machine=machine
        )
        # As with_heat_pump_streams does: the case at the design's multiplier
        # must target as the heat pump did.
        _require_heat_pump_site_target(_scaled(case, multiplier), target)
    else:
        # The zone multiplier scales the heat pump's streams too.
        case = _case_with_heat_pump_streams(
            problem, target, dt_cont=approach, machine=machine
        )
        # A level the search left a hair off a process temperature goes on
        # it: the design at this multiplier sees the heat pump's streams
        # shifted by the approach the target stood them off by.
        case = _on_process_temperatures(
            case, scaled, approach=approach * zone_multiplier * multiplier
        )
        # The streams the case added go with the machine, so a network of
        # the case can be solved again from the problem and itself alone.
        record = record.model_copy(
            update={"streams": heat_pump_stream_records(case, machine)}
        )
    return HeatPumpDesignCase(
        multiplier=multiplier, target=target, case=case, record=record
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


# The shares of its full load a utility heat pump is tried at without a
# load, as a retrofit tries its heat pump sizes.
_UTILITY_LOADS = (1.0, 0.75, 0.5)


def _load_fractions(problem) -> list[float]:
    """The heat pump sizes worth trying on ``problem``: the logical load
    fractions of its net load profile, over every period.

    The profile, and so its kinks, differs from period to period; a size
    is tried over all of them, so the kinks of every period are taken.
    """
    fractions = {
        round(float(fraction), 6)
        for result in problem.target.all_periods.hpr_load_kinks().values()
        for fraction in result.logical_load_fractions
    }
    return sorted(fractions)


def design_with_heat_pump(
    problem,
    heat_pump,
    *,
    cost_per_kw: float,
    multipliers,
    arguments: Mapping,
    costing=None,
    max_exchangers: int | None = None,
    placement: str = "process",
):
    """Design ``problem`` without a heat pump and, at each of
    ``multipliers``, with one, and rank every network together.

    ``heat_pump`` is a cycle name or a mapping of the keyword arguments of
    ``heat_pump_design_case``; ``arguments`` are those of
    ``design.heat_exchanger_network``, or with ``placement`` "utility" of
    ``design.total_site_heat_exchanger_network``: the heat pump then heats
    and cools the site's utility levels, and without a load it is tried at
    each of ``_UTILITY_LOADS`` of its full load.
    """
    site = placement == "utility"

    def design(case, multiplier=None):
        if not site:
            at = {} if multiplier is None else {"dt_cont_multipliers": [multiplier]}
            return case.design.heat_exchanger_network(**dict(arguments) | at)
        options = dict(arguments.get("options") or {})
        if multiplier is not None:
            # The site design takes its multipliers from its options.
            options["HENS_DT_CONT_MULTIPLIERS"] = [multiplier]
        return case.design.total_site_heat_exchanger_network(
            **dict(arguments) | {"options": options}
        )

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
    plain = design(problem)
    # A site's case is refused when its levels cannot hold the heat pump as
    # its target assumed: that size is left out, not the site design.
    refused = (ValueError,) if site else ()
    # A heat pump's size is chosen outside the design: each is its own
    # target and its own networks. Without a load, every size worth trying.
    asked = request.pop("load", None)
    levels = HprLevels.model_validate(request.pop("levels", None) or {})
    # One approach: the one given, else HPR_DT_CONT. Only the dt_cont
    # multiplier varies the approach the machine stands off by.
    approaches = [levels.minimum_approach_temperature]
    runs = []
    for multiplier in multipliers:
        # The kinks of the profile as the design at this multiplier sees it:
        # scaling the contributions moves the pinch and the profile's kinks.
        # The kinks are the process profile's, not the site utilities'.
        if asked is not None:
            loads = [asked]
        elif site:
            loads = [{"fraction": fraction} for fraction in _UTILITY_LOADS]
        else:
            loads = [
                {"fraction": fraction}
                for fraction in _load_fractions(_scaled(problem, multiplier))
            ]
        for load in loads:
            for approach in approaches:
                try:
                    run = heat_pump_design_case(
                        problem,
                        multiplier,
                        load=load,
                        levels=levels.model_copy(
                            update={"minimum_approach_temperature": approach}
                        ),
                        cost_per_kw=cost_per_kw,
                        costing=costing,
                        placement=placement,
                        **request,
                    )
                    view = design(run.case, multiplier)
                except (HPRTargetingError, WorkflowContractError, *refused) as error:
                    # No target at this multiplier and size, no network
                    # around its streams, or (on a site) a case that would
                    # not target as the heat pump did.
                    warnings.warn(
                        f"No heat pump network at dt_cont multiplier "
                        f"{multiplier:g} and load {load}: {error}",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                runs.append((run, view))
    return heat_pump_design_view(problem, plain, runs, max_exchangers=max_exchangers)


def capped_heat_pump_streams(case) -> set[str]:
    """The keys of the streams ``HENS_HEAT_PUMP_STREAM_EXCHANGERS`` caps:
    the condensers and evaporators of ``case``, as its networks name them.

    An ambient source is not capped: one air stream can feed several
    evaporators. A machine on the utility system is capped as one on the
    process is.
    """
    return {
        key
        for key, role in heat_pump_stream_roles(case, ("process", "utility")).items()
        if role in ("condenser", "evaporator")
    }


def _most_exchangers(network, keys: set[str]) -> int:
    """The most exchangers with duty that one stream of ``keys`` has.

    ``keys`` are stream keys as the network names them: a stream of the
    same name in another zone is another stream.
    """
    counts = dict.fromkeys(keys, 0)
    for exchanger in network.exchangers:
        if max(state.duty for state in exchanger.period_states) <= 0.0:
            continue
        for key in (exchanger.source_stream, exchanger.sink_stream):
            if key in counts:
                counts[key] += 1
    return max(counts.values(), default=0)


def heat_pump_design_view(problem, plain, runs, *, max_exchangers: int | None = None):
    """One view ranking the designs without the heat pump and with it.

    ``plain`` is the problem's design without a heat pump and ``runs`` pairs
    each heat pump case with the view of its design. Every network of a
    heat pump design takes the case's machine record and its annual cost
    (``network_with_heat_pumps``) and is ranked on that total among the
    rest. The view is ``plain``, its result holding them all and the
    cheapest selected.

    A network that serves a heat pump stream through more than
    ``max_exchangers`` exchangers (``HENS_HEAT_PUMP_STREAM_EXCHANGERS``) is
    left out, and a warning says how many were.
    """
    from ..analysis.heat_exchanger_networks.results.selection import (
        ranked_networks,
        select_network,
    )

    result = plain.result
    ranked = list(ranked_networks(result))
    kept = dropped = 0
    for run, view in runs:
        capped = capped_heat_pump_streams(run.case)
        for outcome in ranked_networks(view.result):
            if (
                max_exchangers is not None
                and _most_exchangers(outcome.network, capped) > max_exchangers
            ):
                dropped += 1
                continue
            kept += 1
            network = network_with_heat_pumps(outcome.network, (run.record,))
            ranked.append(
                outcome.model_copy(
                    update={
                        "network": network,
                        "objective_value": network.total_annual_cost,
                    }
                )
            )
    if dropped:
        warnings.warn(
            f"{dropped} heat pump network(s) put more than {max_exchangers} "
            "exchanger(s) on a heat pump stream "
            f"(HENS_HEAT_PUMP_STREAM_EXCHANGERS) and were left out; {kept} "
            "are ranked.",
            UserWarning,
            stacklevel=2,
        )
    result.ranked_networks = tuple(ranked)
    select_network(result)
    return plain


def heat_pump_network_case(problem, network):
    """The case ``network`` was designed on: ``problem`` with the streams
    its heat pumps carry (``NetworkHeatPump.streams``).

    ``None`` for a network whose machines carry none, and for one whose
    streams ``problem`` already holds (the case itself).
    """
    carried = [stream for machine in network.heat_pumps for stream in machine.streams]
    if not carried:
        return None
    # A zone's address starts at the root zone, named for the project: the
    # streams join the same zones under this problem's root.
    root = problem._require_prepared_root_zone().address
    carried = [
        stream | {"zone": root + "".join(stream["zone"].partition("/")[1:])}
        for stream in carried
    ]
    source = problem.to_problem_json()
    held = {(record["zone"], record["name"]) for record in source["streams"]}
    missing = [s for s in carried if (s["zone"], s["name"]) not in held]
    if not missing:
        return None
    source["streams"].extend(missing)
    return type(problem)(source, project_name=problem.project_name)


def with_heat_pumps(view, heat_pumps):
    """``view`` with ``heat_pumps`` recorded on every network of it and
    their annual cost in its total (``network_with_heat_pumps``)."""
    from ..analysis.heat_exchanger_networks.results.selection import (
        ranked_networks,
        select_network,
    )

    result = view.result
    ranked = []
    for outcome in ranked_networks(result):
        network = network_with_heat_pumps(outcome.network, heat_pumps)
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
    return view


__all__ = [
    "HeatPumpDesignCase",
    "capped_heat_pump_streams",
    "design_with_heat_pump",
    "heat_pump_design_case",
    "heat_pump_design_view",
    "heat_pump_network_case",
    "heat_pump_request",
    "with_heat_pumps",
]
