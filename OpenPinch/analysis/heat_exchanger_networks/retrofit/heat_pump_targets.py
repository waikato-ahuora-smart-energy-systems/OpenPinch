"""Heat pump targets for a retrofit, by the HPR service.

The as-piped target serves only the existing network's heaters from only
its coolers, so a heat pump that meets it needs no process-side repiping:
each evaporator goes in series with a cooler, each condenser with a heater.
"""

from __future__ import annotations

import numpy as np

from ....contracts.synthesis.retrofit import (
    RetrofitHeatPumpTarget,
    RetrofitHeatPumpTargets,
)
from ....domain.enums import HeatExchangerKind, ProblemTableLabel, TargetType
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
    existing design. ``args`` takes the HPR search budget keys."""

    if not _solved(network):
        network = (
            run_retrofit_search(problem, network, RetrofitSettings(max_modifications=0))
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
    # Prices in $/kW/y as the retrofit's: the first hot and cold utility,
    # and the HPR electricity price.
    config = zone.config
    data = problem_to_solver_arrays(problem, 2.0 * config.thermal.dt_cont).arrays
    price_scale, _capital = annual_cost_scales(config.costing)
    prices = {
        "hot_utility_price": float(data["hu_cost_period"][0][0]),
        "cold_utility_price": float(data["cu_cost_period"][0][0]),
        "electricity_price": float(config.costing.hpr_ele_price) * price_scale,
    }
    # The first period, whose states the as-piped profiles and prices take.
    period_id = min(problem.period_ids.items(), key=lambda item: item[1])[0]
    targets = {}
    for name, (T_vals, H_hot, H_cold) in profiles.items():
        result = compute_profile_heat_pump_target(
            T_vals=T_vals,
            H_hot=H_hot,
            H_cold=H_cold,
            config=config,
            args=args,
            period_id=period_id,
            period_idx=0,
        )
        targets[name] = _summary(result, H_hot, H_cold, prices)
    return RetrofitHeatPumpTargets(**targets)


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
    problem, network: HeatExchangerNetwork
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The shifted temperatures (degC, hottest first) and the source and
    sink profiles of ``network``'s coolers and heaters, in its first period.

    Each cooler is a hot segment and each heater a cold one, between the
    process stream's solved inlet and outlet temperatures (K) at the
    exchanger's duty, shifted by that stream's temperature contribution
    and its multiplier.
    The sources and sinks are cascaded apart, so none recovers heat from
    another.
    """

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
        state = exchanger.period_states[0]
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
    every = sources + sinks
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
