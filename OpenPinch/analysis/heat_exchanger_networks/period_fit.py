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
from .solver.arrays import nominal_approach_temperature, problem_to_solver_arrays
from .solver.utility_slices import (
    slice_label,
    slice_labels,
    slice_parent,
    with_named_slices,
)


def with_stream_end_utilities(
    problem,
    network: HeatExchangerNetwork,
    *,
    approach: float | None = None,
    dt_cont_multiplier: float = 1.0,
    levels: frozenset[str] = frozenset(),
) -> HeatExchangerNetwork:
    """``network`` with a heater or cooler at each stream end every period can use.

    The approach a stream and a utility must keep is ``approach`` when given
    (a floor for every exchanger), else the pair's own temperature
    contributions scaled by ``dt_cont_multiplier``, the approach a fit at
    that multiplier holds. An end utility of ``network`` that cannot keep it
    at both ends in every period its stream runs is dropped. Each stream then
    without one gets the cheapest that can: a hot utility for a cold stream,
    a cold utility for a hot stream, priced by the period-weighted utility
    price. A stream no utility can serve that way gets none. Every utility
    named in ``levels`` (levels held at their targets) is also placed at each
    stream end it can serve, so a level has exchangers to carry its target;
    the fit removes those it leaves unused.
    """

    # A sliced network keeps its slices: each heater or cooler named by its
    # slice runs on that slice's range, not the whole utility's.
    root = problem._require_prepared_root_zone()
    arrays = with_named_slices(
        problem_to_solver_arrays(
            problem, nominal_approach_temperature(root.config, dt_cont_multiplier)
        ),
        slice_labels(network.exchangers),
    )
    values, axes = arrays.arrays, arrays.axis_maps
    weights = np.asarray(values["period_weights"], dtype=float)

    def contributions(prefix: str) -> np.ndarray:
        """Each period's largest segment contribution of every stream on a
        side, as the fit's profiles hold it."""
        masked = np.where(
            values[f"{prefix}_segment_mask_period"],
            values[f"{prefix}_segment_dt_cont_period"],
            -np.inf,
        )
        return masked.max(axis=2)

    def terminal_contributions(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        """Each period's inlet and outlet segment contributions of every
        utility on a side, the terminals the fit's approach equations use."""
        segments = values[f"{prefix}_segment_dt_cont_period"]
        last = np.maximum(values[f"{prefix}_segment_count"] - 1, 0)
        outlet = np.take_along_axis(
            segments,
            np.broadcast_to(last[None, :, None], (*segments.shape[:2], 1)),
            axis=2,
        )[..., 0]
        return segments[..., 0], outlet

    def running_by_period(prefix: str) -> np.ndarray:
        """Whether each stream on a side runs in each period: a heat capacity
        flowrate, or (a segmented stream, which has none) segment duty."""
        segment_duty = np.where(
            values[f"{prefix}_segment_mask_period"],
            values[f"{prefix}_segment_duty_period"],
            0.0,
        ).sum(axis=2)
        return (values[f"f_{prefix[0]}_period"] > 0.0) | (segment_duty > 0.0)

    runs = {"hot": running_by_period("cold"), "cold": running_by_period("hot")}

    held = {
        "hot": (contributions("cold"), terminal_contributions("hot_utility")),
        "cold": (contributions("hot"), terminal_contributions("cold_utility")),
    }

    def reaches(side: str, stream_index: int, utility_index: int) -> bool:
        """Whether a utility at the stream's end keeps the pair's approach in
        every period the stream runs: a heater (``side="hot"``) on a cold
        stream, or a cooler on a hot stream."""
        if side == "hot":
            stream_in, stream_out = values["T_c_in_period"], values["T_c_out_period"]
            util_in, util_out = values["T_hu_in_period"], values["T_hu_out_period"]
            sign = 1.0
        else:
            stream_in, stream_out = values["T_h_in_period"], values["T_h_out_period"]
            util_in, util_out = values["T_cu_in_period"], values["T_cu_out_period"]
            sign = -1.0
        running = runs[side][:, stream_index]
        if approach is None:
            stream_cont, (inlet_cont, outlet_cont) = held[side]
            needed_out = stream_cont[:, stream_index] + outlet_cont[:, utility_index]
            needed_in = stream_cont[:, stream_index] + inlet_cont[:, utility_index]
        else:
            needed_out = needed_in = np.full(len(weights), float(approach))
        # Hot side: the utility's outlet against the stream's inlet, its inlet
        # against the stream's target; cold side mirrored. Each gap keeps the
        # contribution of the utility segment at that end.
        outlet_short = (
            sign * (util_out[:, utility_index] - stream_in[:, stream_index])
            < needed_out
        )
        inlet_short = (
            sign * (util_in[:, utility_index] - stream_out[:, stream_index]) < needed_in
        )
        return not np.any((outlet_short | inlet_short)[running])

    def cheapest(stream_index: int, side: str):
        names = axes["hot_utilities" if side == "hot" else "cold_utilities"]
        prices = values["hu_cost_period" if side == "hot" else "cu_cost_period"]
        best = None
        for name, u in names.items():
            if not reaches(side, stream_index, u):
                continue
            price = float(np.dot(weights, prices[:, u]))
            if best is None or price < best[0]:
                best = (price, name)
        return None if best is None else best[1]

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
        utility = slice_label(exchanger) or utility
        if exchanger.stage is not None:
            (heaters if heater else coolers).append((stream, utility, exchanger.stage))
            continue
        # An end utility the basis period could use may not reach the stream
        # in another period: it is replaced by one that does.
        side = "hot" if heater else "cold"
        streams = axes["cold_process_streams" if heater else "hot_process_streams"]
        utilities = axes["hot_utilities" if heater else "cold_utilities"]
        if not reaches(side, streams[stream], utilities[utility]):
            continue
        (heaters if heater else coolers).append((stream, utility))
        served.add(stream)

    for side, streams, placements in (
        ("hot", axes["cold_process_streams"], heaters),
        ("cold", axes["hot_process_streams"], coolers),
    ):
        names = axes["hot_utilities" if side == "hot" else "cold_utilities"]
        for stream, index in streams.items():
            if stream not in served:
                utility = cheapest(index, side)
                if utility is not None:
                    placements.append((stream, utility))
            for name, u in names.items():
                if (
                    slice_parent(name, levels) in levels
                    and (stream, name) not in placements
                    and reaches(side, index, u)
                ):
                    placements.append((stream, name))

    return HeatExchangerNetwork.from_structure(
        recovery=recovery,
        heaters=heaters,
        coolers=coolers,
        stage_count=network.stage_count,
    )


__all__ = ["with_stream_end_utilities"]
