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

# Solved network states carry temperatures in K; problem tables in degC.
_KELVIN = 273.15


def retrofit_heat_pump_targets(
    problem, network: HeatExchangerNetwork, *, args: dict | None = None
) -> RetrofitHeatPumpTargets:
    """The HPR service's heat pump targets for ``problem`` (its master
    zone's direct-integration cascade, already targeted) and for
    ``network``'s solved coolers and heaters (``as_piped_profiles``), with
    the zone's HPR configuration. ``args`` takes the HPR search budget
    keys."""

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
    targets = {}
    for name, (T_vals, H_hot, H_cold) in profiles.items():
        result = compute_profile_heat_pump_target(
            T_vals=T_vals, H_hot=H_hot, H_cold=H_cold, config=zone.config, args=args
        )
        targets[name] = _summary(result, H_hot, H_cold)
    return RetrofitHeatPumpTargets(**targets)


def _summary(result, H_hot, H_cold) -> RetrofitHeatPumpTarget:
    """``result`` against the utility its profiles carry."""

    condenser = np.atleast_1d(np.asarray(result.Q_cond, dtype=float))
    evaporator = np.atleast_1d(np.asarray(result.Q_evap, dtype=float))
    ambient = float(result.Q_amb_cold)
    return RetrofitHeatPumpTarget(
        hot_utility_before=float(np.abs(H_cold).max()),
        cold_utility_before=float(np.abs(H_hot).max()),
        heat_delivered=float(condenser.sum()),
        process_heat_absorbed=float(evaporator.sum()) - ambient,
        ambient_heat=ambient,
        work=float(result.w_net),
        cop=float(result.cop_h),
        condensing_temperatures=tuple(
            float(t) for t in np.atleast_1d(result.T_cond)
        ),
        condenser_duties=tuple(float(q) for q in condenser),
        evaporating_temperatures=tuple(
            float(t) for t in np.atleast_1d(result.T_evap)
        ),
        evaporator_duties=tuple(float(q) for q in evaporator),
        total_annualized_cost=result.hpr_total_annualized_cost,
    )


def as_piped_profiles(
    problem, network: HeatExchangerNetwork
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The shifted temperatures (degC, hottest first) and the source and
    sink profiles of ``network``'s coolers and heaters, in its first period.

    Each cooler is a hot segment and each heater a cold one, between the
    process stream's solved inlet and outlet temperatures (K) at the
    exchanger's duty, shifted by that stream's temperature contribution.
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
        segments.add(
            Stream(
                str(exchanger.exchanger_id or name),
                getattr(state, f"{side}_inlet_temperature") - _KELVIN,
                getattr(state, f"{side}_outlet_temperature") - _KELVIN,
                heat_flow=state.duty,
                delta_t_contribution=_process_stream(
                    process, name
                ).delta_t_contribution,
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
    (``H1`` for ``Process A.H1``)."""

    matches = [
        stream
        for key, stream in streams.items()
        if key == name or str(key).endswith("." + name)
    ]
    if not matches:
        raise ValueError(f"the problem has no process stream {name!r}.")
    return matches[0]
