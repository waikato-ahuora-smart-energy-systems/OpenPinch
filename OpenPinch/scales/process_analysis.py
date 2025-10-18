from operator import attrgetter
from typing import List, Tuple

from ..analysis import (
    get_pinch_temperatures,
    get_process_heat_cascade,
    get_utility_targets,
)
from ..classes import *
from ..lib import *
from ..utils import timing_decorator

__all__ = ["get_process_targets"]

#######################################################################################################
# Public API
#######################################################################################################


def get_process_targets(zone: Zone):
    """Populate a ``Zone`` with detailed process-level pinch targets.

    The function aggregates problem-table calculations, multi-utility targeting,
    pinch temperature detection, and graph preparation.  Results are cached on
    the provided ``zone`` via :meth:`Zone.add_target_from_results` and used later
    by site and regional aggregation routines.
    """
    (
        config,
        hot_streams,
        cold_streams,
        all_streams,
        hot_utilities,
        cold_utilities,
        zone_identifier,
        subzones,
    ) = attrgetter(
        "config",
        "hot_streams",
        "cold_streams",
        "all_streams",
        "hot_utilities",
        "cold_utilities",
        "identifier",
        "subzones",
    )(zone)

    pt, pt_real, target_values = get_process_heat_cascade(
        hot_streams, cold_streams, all_streams, config
    )
    pt, pt_real, hot_utilities, cold_utilities = get_utility_targets(
        pt, pt_real, hot_utilities, cold_utilities
    )
    net_hot_streams, net_cold_streams = _get_net_hot_and_cold_segments(
        zone_identifier, pt, hot_utilities, cold_utilities
    )
    _get_balanced_cc(pt)
    _get_balanced_cc(pt_real)
    hot_pinch, cold_pinch = get_pinch_temperatures(pt)

    # Target heat transfer area and number of exchanger units based on Balanced CC
    config: Configuration
    if config.DO_AREA_TARGETING and 0:
        # area = get_area_targets(pt_real, config)
        # num_units = get_min_number_hx(pt)
        # capital_cost = compute_capital_cost(area, num_units, config)
        # annual_capital_cost = compute_annual_capital_cost(area, num_units, config)
        # zone.add_target_from_results(TargetType.DI.value, {
        #     "capital_cost": capital_cost,
        #     "annual_capital_cost": annual_capital_cost,
        #     "num_units": num_units,
        #     "area": area,
        # })
        pass

    graph_data = _save_graph_data(pt, pt_real)
    zone.add_target_from_results(
        TargetType.DI.value,
        {
            "pt": pt,
            "pt_real": pt_real,
            "target_values": target_values,
            "graphs": graph_data,
            "hot_utilities": hot_utilities,
            "cold_utilities": cold_utilities,
            "net_hot_streams": net_hot_streams,
            "net_cold_streams": net_cold_streams,
            "hot_pinch": hot_pinch,
            "cold_pinch": cold_pinch,
        },
    )

    if len(subzones) > 0:
        for z in subzones.values():
            z = get_process_targets(z)

    return zone


#######################################################################################################
# Helper functions
#######################################################################################################

@timing_decorator
def _get_net_hot_and_cold_segments(
    zone_identifier: str,
    pt: ProblemTable,
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
) -> Tuple[StreamCollection, StreamCollection]:
    """Derive net process streams that represent overall utility demand for site aggregation."""
    total_utility_demand = sum([u.heat_flow for u in hot_utilities]) + sum(
        [u.heat_flow for u in cold_utilities]
    )
    if zone_identifier == ZoneType.P.value and total_utility_demand > tol:
        net_hot_streams, net_cold_streams = _save_data_for_site_analysis(
            pt, PT.T.value, PT.H_NET_A.value, hot_utilities, cold_utilities
        )
    else:
        net_hot_streams, net_cold_streams = StreamCollection(), StreamCollection()
    return net_hot_streams, net_cold_streams


def _save_data_for_site_analysis(
    pt: ProblemTable,
    col_T: str,
    col_H: str,
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
) -> Tuple[StreamCollection, StreamCollection]:
    """Constructs net stream segments that require utility input across temperature intervals."""
    net_hot_streams = StreamCollection()
    net_cold_streams = StreamCollection()

    if pt.delta_col(col_T)[1:].min() < tol:
        raise ValueError("Infeasible temperature interval detected in _store_TSP_data")

    T_vals = pt.col[col_T]
    dh_vals = pt.delta_col(col_H)[1:]

    hot_utilities_seq = list(hot_utilities)
    cold_utilities_seq = list(cold_utilities)
    hot_remaining = [u.heat_flow for u in hot_utilities_seq]
    cold_remaining = [u.heat_flow for u in cold_utilities_seq]

    hu_idx = _initialise_utility_index(hot_utilities_seq, hot_remaining)
    cu_idx = _initialise_utility_index(cold_utilities_seq, cold_remaining)

    k = 1
    for i, dh in enumerate(dh_vals):
        if dh > tol and hu_idx >= 0:
            hu_idx, k = _add_net_segment_stateful(
                T_vals[i],
                T_vals[i + 1],
                hu_idx,
                dh,
                hot_utilities_seq,
                hot_remaining,
                net_cold_streams,
                k,
            )
        elif -dh > tol and cu_idx >= 0:
            cu_idx, k = _add_net_segment_stateful(
                T_vals[i],
                T_vals[i + 1],
                cu_idx,
                abs(dh),
                cold_utilities_seq,
                cold_remaining,
                net_hot_streams,
                k,
            )

    return net_hot_streams, net_cold_streams


def _add_net_segment_stateful(
    T_ub: float,
    T_lb: float,
    curr_idx: int,
    dh_req: float,
    utilities: List[Stream],
    remaining: List[float],
    net_streams: StreamCollection,
    k: int,
    j: int = 0,
):
    """Adds a net utility segment and recursively handles segmentation if needed."""
    if curr_idx < 0 or not utilities or dh_req <= tol:
        return curr_idx, k

    curr_u = utilities[curr_idx]
    available = remaining[curr_idx] if curr_idx < len(remaining) else 0.0

    if available <= tol:
        next_idx = _find_next_available_utility(curr_idx + 1, utilities, remaining)
        if next_idx == curr_idx:
            return curr_idx, k
        return _add_net_segment_stateful(
            T_ub,
            T_lb,
            next_idx,
            dh_req,
            utilities,
            remaining,
            net_streams,
            k,
            j,
        )

    dh_curr = min(dh_req, available)
    remaining[curr_idx] = max(available - dh_curr, 0.0)
    dh_next = dh_req - dh_curr

    T_span = T_ub - T_lb
    T_i = T_ub - (dh_curr / dh_req) * T_span if T_span and dh_req > tol else T_lb

    net_streams.add(
        Stream(
            name=f"Segment {k}" if j == 0 else f"Segment {k}-{j}",
            t_supply=T_i if curr_u.type == StreamType.Hot.value else T_ub,
            t_target=T_ub if curr_u.type == StreamType.Hot.value else T_i,
            heat_flow=dh_curr,
            dt_cont=curr_u.dt_cont,
            htc=1.0,
            is_process_stream=True,
        )
    )

    if dh_next > tol:
        next_idx = _find_next_available_utility(curr_idx + 1, utilities, remaining)
        if next_idx == curr_idx:
            return curr_idx, k + 1
        return _add_net_segment_stateful(
            T_i,
            T_lb,
            next_idx,
            dh_next,
            utilities,
            remaining,
            net_streams,
            k,
            j + 1,
        )

    next_idx = (
        curr_idx
        if remaining[curr_idx] > tol
        else _find_next_available_utility(curr_idx + 1, utilities, remaining)
    )
    return next_idx, k + 1


def _initialise_utility_index(
    utilities: List[Stream], remaining: List[float]
) -> int:
    """Returns the index of the first available utility with remaining capacity."""
    for idx, residual in enumerate(remaining):
        if residual > tol:
            return idx
    return len(utilities) - 1 if utilities else -1


def _find_next_available_utility(
    start: int, utilities: List[Stream], remaining: List[float]
) -> int:
    """Return the index of the next utility that still has remaining duty."""
    if not utilities:
        return -1
    for idx in range(max(start, 0), len(utilities)):
        if remaining[idx] > tol:
            return idx
    return len(utilities) - 1


@timing_decorator
def _save_graph_data(pt: ProblemTable, pt_real: ProblemTable) -> Zone:
    """Assemble the problem-table slices required for composite/comparison plots."""
    pt.round(decimals=4)
    pt_real.round(decimals=4)
    return {
        GT.CC.value: pt_real[[PT.T.value, PT.H_HOT.value, PT.H_COLD.value]],
        GT.SCC.value: pt[[PT.T.value, PT.H_HOT.value, PT.H_COLD.value]],
        GT.BCC.value: pt_real[[PT.T.value, PT.H_HOT_BAL.value, PT.H_COLD_BAL.value]],
        GT.GCC.value: pt[[PT.T.value, PT.H_NET.value, PT.H_NET_NP.value, PT.H_NET_V.value, PT.H_NET_A.value, PT.H_UT_NET.value]],
        GT.GCC_R.value: pt_real[[PT.T.value, PT.H_NET.value, PT.H_UT_NET.value]],
        GT.NLC.value: pt[[PT.T.value, PT.H_HOT_NET.value, PT.H_COLD_NET.value, PT.H_HOT_UT.value, PT.H_COLD_UT.value]],
    }


def _get_balanced_cc(pt: ProblemTable) -> ProblemTable:
    """Returns the balanced composite curves of both process and utility streams"""
    pt.col[PT.H_HOT_BAL.value] = pt.col[PT.H_HOT.value] + pt.col[PT.H_HOT_UT.value] 
    pt.col[PT.H_COLD_BAL.value] = pt.col[PT.H_COLD.value] + pt.col[PT.H_COLD_UT.value]
    return pt
