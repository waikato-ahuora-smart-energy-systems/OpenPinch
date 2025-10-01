from typing import Tuple
from copy import deepcopy
from ..lib import *
from ..classes import Zone, StreamCollection, ProblemTable, Stream
from .problem_table_analysis import problem_table_algorithm
from .utility_targeting import get_zonal_utility_targets
from .support_methods import get_pinch_temperatures


__all__ = ["get_process_pinch_targets"]

#######################################################################################################
# Public API
#######################################################################################################

def get_process_pinch_targets(zone: Zone):
    """Populate a Zone`` with detailed process-level pinch targets.

    The function aggregates problem-table calculations, multi-utility targeting,
    pinch temperature detection, and graph preparation.  Results are cached on
    the provided ``zone`` via :meth:`Zone.add_target_from_results` and used later
    by site and regional aggregation routines.
    """
    config = zone.config
    hot_streams, cold_streams, all_streams = zone.hot_streams, zone.cold_streams, zone.all_streams
    hot_utilities, cold_utilities = zone.hot_utilities, zone.cold_utilities

    pt, pt_real, target_values = problem_table_algorithm(hot_streams, cold_streams, all_streams, config)
    pt, pt_real, hot_utilities, cold_utilities = get_zonal_utility_targets(pt, pt_real, hot_utilities, cold_utilities)
    net_hot_streams, net_cold_streams = _get_net_hot_and_cold_segments(zone.identifier, pt, hot_utilities, cold_utilities)
    hot_pinch, cold_pinch = get_pinch_temperatures(pt)

    # z = get_additional_zonal_pinch_analysis(z)
    graphs = _save_graph_data(pt, pt_real)
    zone.add_target_from_results(TargetType.DI.value, {
        "pt": pt,
        "pt_real": pt_real,
        "target_values": target_values,
        "graphs": graphs,
        "hot_utilities": hot_utilities,
        "cold_utilities": cold_utilities,
        "net_hot_streams": net_hot_streams,
        "net_cold_streams": net_cold_streams,
        "hot_pinch": hot_pinch,
        "cold_pinch": cold_pinch,
    })

    if len(zone.subzones) > 0:
        for z in zone.subzones.values():
            z = get_process_pinch_targets(z)

    return zone

#######################################################################################################
# Helper functions
#######################################################################################################

def _get_net_hot_and_cold_segments(zone_identifier: str, pt: ProblemTable, hot_utilities: StreamCollection, cold_utilities: StreamCollection) -> Tuple[StreamCollection, StreamCollection]:
    """Derive net process streams that represent overall utility demand for site aggregation."""
    total_utility_demand = sum([u.heat_flow for u in hot_utilities]) + sum([u.heat_flow for u in cold_utilities])
    if zone_identifier == ZoneType.P.value and total_utility_demand > ZERO:
        net_hot_streams, net_cold_streams = _save_data_for_total_site_analysis(pt, PT.T.value, PT.H_NET_A.value, hot_utilities, cold_utilities)        
    else:
        net_hot_streams, net_cold_streams = StreamCollection(), StreamCollection()
    return net_hot_streams, net_cold_streams


def _save_data_for_total_site_analysis(pt: ProblemTable, col_T: str, col_H: str, hot_utilities: StreamCollection, cold_utilities: StreamCollection) -> Tuple[StreamCollection, StreamCollection]:
    """Constructs net stream segments that require utility input across temperature intervals."""
    net_hot_streams = StreamCollection()
    net_cold_streams = StreamCollection()
    #Stephen changed the value from the ZERO constant to a very small number to deal with some janky behaviour
    if pt.delta_col(col_T)[1:].min() < 0.00000000000000000000000000000000000001:
        raise ValueError("Infeasible temperature interval detected in _store_TSP_data") 

    T_vals = pt.col[col_T]
    dh_vals = pt.delta_col(col_H)[1:]

    hot_utilities = deepcopy(hot_utilities)
    cold_utilities = deepcopy(cold_utilities)

    hu: Stream = _initialise_utility_selected(hot_utilities)
    cu: Stream = _initialise_utility_selected(cold_utilities)

    k = 1
    for i, dh in enumerate(dh_vals):
        if dh > ZERO:
            hu, hot_utilities, net_cold_streams, k = _add_net_segment(T_vals[i], T_vals[i+1], hu, dh, hot_utilities, net_cold_streams, k)
        elif -dh > ZERO:
            cu, cold_utilities, net_hot_streams, k = _add_net_segment(T_vals[i], T_vals[i+1], cu, dh, cold_utilities, net_hot_streams, k)

    return net_hot_streams, net_cold_streams


def _add_net_segment(T_ub: float, T_lb: float, curr_u: Stream, dh_req: float, utilities: StreamCollection, net_streams: StreamCollection, k: int, j: int = 0):
    """Adds a net utility segment and recursively handles segmentation if needed."""
    next_u = _advance_utility_if_needed(dh_req, curr_u, utilities)
    
    dh_next = max(-curr_u.heat_flow, 0.0)
    curr_u.heat_flow = max(curr_u.heat_flow, 0.0)
    dh_curr = abs(dh_req) - dh_next
    T_i = T_ub - (dh_curr / abs(dh_req)) * (T_ub - T_lb)

    net_streams.add(
        Stream(
            name=f"Segment {k}" if j == 0 else f"Segment {k}-{j}",
            t_supply=T_i if curr_u.type == StreamType.Hot.value else T_ub,
            t_target=T_ub if curr_u.type == StreamType.Hot.value else T_i,
            heat_flow=dh_curr,
            dt_cont=curr_u.dt_cont,
            htc=1.0, # Might be a way to calculate this.
            is_process_stream=True
        )
    )
    if dh_next > ZERO:
        return _add_net_segment(T_i, T_lb, next_u, dh_next, utilities, net_streams, k, j+1)
    else:
        return next_u, utilities, net_streams, k+1


def _initialise_utility_selected(utilities: StreamCollection = None):
    """Returns the first available utility with remaining capacity."""
    for j in range(len(utilities)):
        if utilities[j].heat_flow > ZERO:
            return utilities[j]
    return utilities[-1]


def _advance_utility_if_needed(dh_used: float, current_u: Stream = None, utilities: StreamCollection = None) -> Stream:
    """Advances to the next utility if the current one is exhausted."""
    current_u.heat_flow -= abs(dh_used)
    if current_u.heat_flow > ZERO:
        return current_u

    k = utilities.get_index(current_u) + 1
    for j in range(k, len(utilities)):
        if utilities[j].heat_flow > ZERO:
            return utilities[j]
        
    return utilities[-1]


def _save_graph_data(pt: ProblemTable, pt_real: ProblemTable) -> Zone:
    """Assemble the problem-table slices required for composite/comparison plots."""
    pt.round(decimals=4)
    pt_real.round(decimals=4)
    return {
        GT.CC.value: pt_real[[PT.T.value, PT.H_HOT.value, PT.H_COLD.value]],
        GT.SCC.value: pt[[PT.T.value, PT.H_HOT.value, PT.H_COLD.value]],
        GT.GCC.value: pt[[PT.T.value, PT.H_NET.value]],
        GT.GCC_NP.value: pt[[PT.T.value, PT.H_NET_NP.value]],
        GT.GCC_Ex.value: pt[[PT.T.value, PT.H_NET_V.value]],
        GT.GCC_Act.value: pt[[PT.T.value, PT.H_NET_A.value, PT.H_UT_NET.value]],
        GT.SHL.value: pt[[PT.T.value, PT.H_HOT_NET.value, PT.H_COLD_NET.value]],   
        GT.GCC_Ut.value: pt_real[[PT.T.value, PT.H_UT_NET.value]],
        GT.GCC_Ut_star.value: pt[[PT.T.value, PT.H_UT_NET.value]],
    }
