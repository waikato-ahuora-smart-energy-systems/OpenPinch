"""Heat pump targets for a retrofit, by the HPR service.

The as-piped target serves only the existing network's heaters from only
its coolers, so a heat pump that meets it needs no process-side repiping:
each evaporator goes in series with a cooler, each condenser with a heater.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from ....contracts.synthesis.retrofit import (
    RetrofitHeatPumpTarget,
    RetrofitHeatPumpTargets,
)
from ....domain.enums import (
    HeatExchangerKind,
    HeatPumpAndRefrigerationCycle,
    ProblemTableLabel,
    TargetType,
)
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ....domain.stream import Stream
from ....domain.stream_collection import StreamCollection
from ...heat_pumps.service import compute_profile_heat_pump_target
from ...targeting.cascade import get_process_heat_cascade
from ..solver.arrays import annual_cost_scales, problem_to_solver_arrays
from .search import RetrofitSettings, run_retrofit_search

# Solved network states carry temperatures in K; problem tables in degC.
_KELVIN = 273.15


def retrofit_heat_pumps(
    problem, network: HeatExchangerNetwork, *, args: dict | None = None
) -> RetrofitHeatPumpTargets:
    """The HPR service's heat pump targets for ``problem`` (its master
    zone's direct-integration cascade, already targeted) and for
    ``network``'s solved coolers and heaters (``as_piped_profiles``), with
    the zone's HPR configuration. A ``network`` whose heaters and coolers
    have no solved temperatures is solved first, as the retrofit search's
    existing design with ``design.retrofit``'s defaults (each exchanger held
    to its two streams' contributions). ``args`` takes the HPR search
    budget keys."""

    if not _solved(network):
        network = (
            run_retrofit_search(
                problem, network, RetrofitSettings(max_modifications=0, approach=None)
            )
            .frontier[0]
            .network
        )
    zone = problem.master_zone
    pt = zone.targets[TargetType.DI.value].pt
    profiles = {
        "ideal": (
            np.asarray(pt[ProblemTableLabel.T], dtype=float),
            np.asarray(pt[ProblemTableLabel.H_NET_HOT], dtype=float),
            np.asarray(pt[ProblemTableLabel.H_NET_COLD], dtype=float),
        ),
        "as_piped": as_piped_profiles(problem, network),
    }
    prices = _network_prices(problem, network)
    return RetrofitHeatPumpTargets(
        **{
            name: _summary(
                _profile_target(problem, profile, prices, args),
                profile[1],
                profile[2],
                prices,
            )
            for name, profile in profiles.items()
        }
    )


def as_piped_heat_pump(
    problem,
    network: HeatExchangerNetwork,
    *,
    args=None,
    load_fraction=None,
    cycle: str | None = None,
):
    """The as-piped heat pump target of ``network`` as the heat pump
    service solved it (``HeatPumpTargetOutputs``), with its condenser
    streams (``hpr_hot_streams``), evaporator streams
    (``hpr_cold_streams``) and work; ``None`` when the network's heaters
    carry no load. It is the solve ``retrofit_heat_pumps`` summarises as
    its as-piped target, at the same prices. ``network`` is a solved one:
    its heaters and coolers carry their temperatures. ``cycle``
    ("carnot" or "vapour_compression") solves a cascade of that cycle
    whatever the zone's ``HPR_TYPE``; ``None`` takes ``HPR_TYPE``."""

    return _profile_target(
        problem,
        as_piped_profiles(problem, network),
        _network_prices(problem, network),
        args,
        load_fraction=load_fraction,
        cycle=cycle,
    )


# The cycle model a retrofit heat pump's cycle name solves.
_CYCLE_TYPES = {
    "carnot": HeatPumpAndRefrigerationCycle.CascadeCarnot.value,
    "vapour_compression": HeatPumpAndRefrigerationCycle.CascadeVapourComp.value,
}


def _network_prices(problem, network: HeatExchangerNetwork) -> dict:
    """Prices in $/kW/y as the retrofit's: each side at the utilities the
    network's heaters (or coolers) use, duty-weighted, in the first period;
    and the HPR electricity price."""

    config = problem.master_zone.config
    arrays = problem_to_solver_arrays(problem, 2.0 * config.thermal.dt_cont)
    price_scale, _capital = annual_cost_scales(config.costing)
    return {
        "hot_utility_price": _utility_price(arrays, network, "hot"),
        "cold_utility_price": _utility_price(arrays, network, "cold"),
        "electricity_price": float(config.costing.hpr_ele_price) * price_scale,
    }


def _profile_target(
    problem, profile, prices: dict, args, load_fraction=None, cycle=None
):
    """The heat pump service's result on ``profile`` (shifted temperatures,
    source and sink) at ``prices``, in the first period, whose states the
    as-piped profiles and prices take. ``load_fraction`` caps the load at
    that share of the sink's, and ``cycle`` sets the cycle model, on a copy
    of the configuration."""

    T_vals, H_hot, H_cold = profile
    config = _at_network_prices(problem.master_zone.config, prices)
    if load_fraction is not None and not 0.0 <= float(load_fraction) <= 1.0:
        # update_values does not check the field's bounds; NaN fails here.
        raise ValueError(
            "load_fraction is a share of the heaters' load, between 0 and 1; "
            f"got {load_fraction!r}."
        )
    if load_fraction is not None:
        config = deepcopy(config)
        config.update_values(
            HPR_LOAD_MODE="fraction", HPR_LOAD_FRACTION=float(load_fraction)
        )
    if cycle is not None:
        config = deepcopy(config)
        config.update_values(HPR_TYPE=_CYCLE_TYPES[cycle])
    return compute_profile_heat_pump_target(
        T_vals=T_vals,
        H_hot=H_hot,
        H_cold=H_cold,
        config=config,
        args=args,
        period_id=min(problem.period_ids.items(), key=lambda item: item[1])[0],
        period_idx=0,
    )


def _at_network_prices(config, prices: dict):
    """A copy of ``config`` whose HPR price ratios are ``prices``' own.

    The HPR search prices heat and cooling as ratios of the electricity
    price (``COSTING_HPR_PRICE_RATIO_*``). The targets are reported at the
    network's utility prices, so the search is given those: a machine sized
    for other prices is not the best one at these. Cooling left below the
    cooling water level is refrigeration to the search, priced from a
    default machine's COP; the targets report it at the cold utility price
    too, so that machine is switched off (an efficiency of 0 prices
    refrigeration as cooling water,
    ``default_refrigeration_to_power_ratio``). With no electricity price
    there is no ratio and ``config`` is returned as it is.
    """

    electricity = prices["electricity_price"]
    if electricity <= 0.0:
        return config
    priced = deepcopy(config)
    priced.update_values(
        COSTING_HPR_PRICE_RATIO_HEAT_TO_ELE=prices["hot_utility_price"] / electricity,
        COSTING_HPR_PRICE_RATIO_COOLING_WATER_TO_ELE=(
            prices["cold_utility_price"] / electricity
        ),
        COSTING_HPR_REFRIGERATION_ETA_II=0.0,
    )
    return priced


def _utility_price(arrays, network: HeatExchangerNetwork, side: str) -> float:
    """A side's utility price, $/kW/y: its heaters' (``"hot"``) or coolers'
    utilities weighted by their first-period duties; with none in use, the
    side's cheapest utility."""

    hot = side == "hot"
    axis = arrays.axis_maps["hot_utilities" if hot else "cold_utilities"]
    costs = arrays.arrays["hu_cost_period" if hot else "cu_cost_period"][0]
    kind = HeatExchangerKind.HOT_UTILITY if hot else HeatExchangerKind.COLD_UTILITY
    duty: dict[str, float] = {}
    for exchanger in network.exchangers:
        if exchanger.kind != kind:
            continue
        name = exchanger.source_stream if hot else exchanger.sink_stream
        index = axis.get(name)
        if index is None:
            index = next(
                (u for key, u in axis.items() if key.rsplit(".", 1)[-1] == name), None
            )
        if index is None:
            continue
        duty[index] = duty.get(index, 0.0) + max(
            0.0, float(exchanger.period_states[0].duty)
        )
    total = sum(duty.values())
    if total <= 0.0:
        return float(min(costs))
    return float(sum(float(costs[u]) * q for u, q in duty.items()) / total)


def _solved(network: HeatExchangerNetwork) -> bool:
    """Whether every heater and cooler of ``network`` has solved process-side
    temperatures in its first period."""

    for exchanger in network.exchangers:
        state = exchanger.period_states[0]
        if exchanger.kind == HeatExchangerKind.COLD_UTILITY:
            temperatures = (
                state.source_inlet_temperature,
                state.source_outlet_temperature,
            )
        elif exchanger.kind == HeatExchangerKind.HOT_UTILITY:
            temperatures = (state.sink_inlet_temperature, state.sink_outlet_temperature)
        else:
            continue
        if None in temperatures:
            return False
    return True


def _summary(result, H_hot, H_cold, prices: dict) -> RetrofitHeatPumpTarget:
    """``result`` against the utility its profiles carry, at ``prices``;
    with no result (no load), a heat pump that delivers nothing."""

    if result is None:
        return RetrofitHeatPumpTarget(
            hot_utility_before=float(np.abs(H_cold).max()),
            cold_utility_before=float(np.abs(H_hot).max()),
            heat_delivered=0.0,
            process_heat_absorbed=0.0,
            ambient_heat=0.0,
            work=0.0,
            cop=0.0,
            condensing_temperatures=(),
            condenser_duties=(),
            evaporating_temperatures=(),
            evaporator_duties=(),
            **prices,
        )
    condenser = np.atleast_1d(np.asarray(result.Q_cond, dtype=float))
    evaporator = np.atleast_1d(np.asarray(result.Q_evap, dtype=float))
    # Air heat the evaporators take (the service's ambient source); the
    # ambient sink is source heat left to air, not taken by the heat pump.
    ambient = float(result.Q_amb_hot)
    return RetrofitHeatPumpTarget(
        hot_utility_before=float(np.abs(H_cold).max()),
        cold_utility_before=float(np.abs(H_hot).max()),
        heat_delivered=float(condenser.sum()),
        process_heat_absorbed=float(evaporator.sum()) - ambient,
        ambient_heat=ambient,
        work=float(result.w_net),
        cop=float(result.cop_h),
        condensing_temperatures=tuple(float(t) for t in np.atleast_1d(result.T_cond)),
        condenser_duties=tuple(float(q) for q in condenser),
        evaporating_temperatures=tuple(float(t) for t in np.atleast_1d(result.T_evap)),
        evaporator_duties=tuple(float(q) for q in evaporator),
        total_annualized_cost=result.hpr_total_annualized_cost,
        **prices,
    )


def as_piped_profiles(
    problem,
    network: HeatExchangerNetwork,
    *,
    period: int = 0,
    grid_periods: tuple[int, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The shifted temperatures (degC, hottest first) and the source and
    sink profiles of ``network``'s coolers and heaters, in the period of
    index ``period`` (the first by default).

    Each cooler is a hot segment and each heater a cold one, between the
    process stream's solved inlet and outlet temperatures (K) at the
    exchanger's duty, shifted by that stream's temperature contribution
    and its multiplier.
    The sources and sinks are cascaded apart, so none recovers heat from
    another. The temperatures of the periods of index ``grid_periods``
    join the grid too, carrying no duty.
    """

    sources, sinks = _as_piped_segments(problem, network, period)
    every = sources + sinks
    for other in grid_periods:
        # Fresh segments: a collection renames a stream it adds whose name
        # it already holds.
        more_sources, more_sinks = _as_piped_segments(problem, network, other)
        every = every + more_sources + more_sinks
    source = get_process_heat_cascade(
        hot_streams=sources,
        cold_streams=StreamCollection(),
        all_streams=every,
        is_shifted=True,
        is_full_analysis=True,
    )
    sink = get_process_heat_cascade(
        hot_streams=StreamCollection(),
        cold_streams=sinks,
        all_streams=every,
        is_shifted=True,
        is_full_analysis=True,
    )
    return (
        np.asarray(source[ProblemTableLabel.T], dtype=float),
        np.asarray(source[ProblemTableLabel.H_NET_HOT], dtype=float),
        np.asarray(sink[ProblemTableLabel.H_NET_COLD], dtype=float),
    )


def _as_piped_segments(
    problem, network: HeatExchangerNetwork, period: int
) -> tuple[StreamCollection, StreamCollection]:
    """``network``'s coolers as hot segments and heaters as cold ones, in
    the period of index ``period``; see ``as_piped_profiles``."""

    zone = problem.master_zone
    sources = StreamCollection()
    sinks = StreamCollection()
    for exchanger in network.exchangers:
        if exchanger.kind == HeatExchangerKind.COLD_UTILITY:
            name, side = exchanger.source_stream, "source"
            segments, process = sources, zone.hot_streams
        elif exchanger.kind == HeatExchangerKind.HOT_UTILITY:
            name, side = exchanger.sink_stream, "sink"
            segments, process = sinks, zone.cold_streams
        else:
            continue
        state = exchanger.period_states[period]
        stream = _process_stream(process, name)
        segments.add(
            Stream(
                str(exchanger.exchanger_id or name),
                getattr(state, f"{side}_inlet_temperature") - _KELVIN,
                getattr(state, f"{side}_outlet_temperature") - _KELVIN,
                heat_flow=state.duty,
                delta_t_contribution=stream.delta_t_contribution,
                delta_t_contribution_multiplier=(
                    stream.delta_t_contribution_multiplier
                ),
            )
        )
    return sources, sinks


def as_piped_period_profiles(
    problem, network: HeatExchangerNetwork
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``as_piped_profiles`` in each of the problem's periods, by period id,
    all on one temperature grid, as a shared heat pump design needs."""

    indices = tuple(problem.period_ids.values())
    return {
        period: as_piped_profiles(problem, network, period=index, grid_periods=indices)
        for period, index in problem.period_ids.items()
    }


def _process_stream(streams: StreamCollection, name: str) -> Stream:
    """The process stream ``name`` names, as is or without its zone prefix
    (``H1`` for ``Process A.H1``); a bare name several zones share raises
    ``ValueError``."""

    exact = [stream for key, stream in streams.items() if key == name]
    if exact:
        return exact[0]
    matches = [
        (str(key), stream)
        for key, stream in streams.items()
        if str(key).endswith("." + name)
    ]
    if not matches:
        raise ValueError(f"the problem has no process stream {name!r}.")
    if len(matches) > 1:
        raise ValueError(
            f"stream {name!r} could be "
            + " or ".join(sorted(key for key, _ in matches))
            + "; name it with its zone."
        )
    return matches[0][1]
