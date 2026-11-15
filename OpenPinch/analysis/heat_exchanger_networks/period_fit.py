"""Fitting a network designed on one basis to every operating period.

A network synthesised on one period (or a synthetic one) may leave another
period's stream with load no exchanger can take. Before the fixed-structure
fit over all periods, each stream without a utility at its end gets one, the
cheapest that can serve it there in every period; the fit's zero-duty pruning
removes those no period uses.
"""

from __future__ import annotations

import numpy as np

from ...domain.enums import HeatExchangerKind
from ...domain.heat_exchanger_network import HeatExchangerNetwork
from .solver.arrays import problem_to_solver_arrays

# The arrays are read for temperatures and prices only, which no approach
# temperature changes.
_ARRAYS_APPROACH = 1.0


def with_stream_end_utilities(
    problem, network: HeatExchangerNetwork, *, approach: float
) -> HeatExchangerNetwork:
    """``network`` plus a heater or cooler at each stream end that has none.

    Each cold stream gets the cheapest hot utility (and each hot stream the
    cheapest cold utility) that keeps ``approach`` at both ends of an
    exchanger at the stream end, in every period the stream runs. Price is
    the period-weighted utility price. A stream no utility can serve that way
    gets none.
    """

    arrays = problem_to_solver_arrays(problem, _ARRAYS_APPROACH)
    values, axes = arrays.arrays, arrays.axis_maps
    weights = np.asarray(values["period_weights"], dtype=float)
    recovery, heaters, coolers = [], [], []
    served = set()
    for exchanger in network.exchangers:
        if exchanger.kind == HeatExchangerKind.RECOVERY:
            recovery.append(
                (exchanger.source_stream, exchanger.sink_stream, exchanger.stage)
            )
            continue
        heater = exchanger.kind == HeatExchangerKind.HOT_UTILITY
        stream, utility = (
            (exchanger.sink_stream, exchanger.source_stream)
            if heater
            else (exchanger.source_stream, exchanger.sink_stream)
        )
        placement = (
            (stream, utility)
            if exchanger.stage is None
            else (stream, utility, exchanger.stage)
        )
        (heaters if heater else coolers).append(placement)
        if exchanger.stage is None:
            served.add(stream)

    def cheapest(stream_index, side):
        if side == "hot":  # a heater on a cold stream
            stream_in, stream_out = values["T_c_in_period"], values["T_c_out_period"]
            util_in, util_out = values["T_hu_in_period"], values["T_hu_out_period"]
            running = values["f_c_period"][:, stream_index] > 0.0
            prices, names = values["hu_cost_period"], axes["hot_utilities"]
        else:  # a cooler on a hot stream
            stream_in, stream_out = values["T_h_in_period"], values["T_h_out_period"]
            util_in, util_out = values["T_cu_in_period"], values["T_cu_out_period"]
            running = values["f_h_period"][:, stream_index] > 0.0
            prices, names = values["cu_cost_period"], axes["cold_utilities"]
        best = None
        for name, u in names.items():
            # Hot side: the utility's outlet against the stream's inlet, its
            # inlet against the stream's target; cold side mirrored.
            sign = 1.0 if side == "hot" else -1.0
            gaps = np.minimum(
                sign * (util_out[:, u] - stream_in[:, stream_index]),
                sign * (util_in[:, u] - stream_out[:, stream_index]),
            )
            if np.any(gaps[running] < approach):
                continue
            price = float(np.dot(weights, prices[:, u]))
            if best is None or price < best[0]:
                best = (price, name)
        return None if best is None else best[1]

    for side, streams, placements in (
        ("hot", axes["cold_process_streams"], heaters),
        ("cold", axes["hot_process_streams"], coolers),
    ):
        for stream, index in streams.items():
            if stream in served:
                continue
            utility = cheapest(index, side)
            if utility is not None:
                placements.append((stream, utility))

    return HeatExchangerNetwork.from_structure(
        recovery=recovery,
        heaters=heaters,
        coolers=coolers,
        stage_count=network.stage_count,
    )


__all__ = ["with_stream_end_utilities"]
