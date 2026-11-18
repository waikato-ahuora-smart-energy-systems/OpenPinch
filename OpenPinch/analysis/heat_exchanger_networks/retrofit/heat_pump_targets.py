"""Heat pump targets for a retrofit, by the HPR service.

The as-piped target serves only the existing network's heaters from only
its coolers, so a heat pump that meets it needs no process-side repiping:
each evaporator goes in series with a cooler, each condenser with a heater.
"""

from __future__ import annotations

import numpy as np

from ....domain.enums import HeatExchangerKind, ProblemTableLabel
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ....domain.stream import Stream
from ....domain.stream_collection import StreamCollection
from ...targeting.cascade import get_process_heat_cascade

# Solved network states carry temperatures in K; problem tables in degC.
_KELVIN = 273.15


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
