from operator import attrgetter
from typing import List, Tuple

from ..classes import *
from ..lib import *
from ..utils import *
from . import (
    get_process_heat_cascade,
    get_utility_targets,
    get_area_targets,
    get_min_number_hx,
    get_capital_cost_targets,
    get_balanced_CC,
    get_additional_GCCs,
    get_heat_pump_targets,
    calc_heat_pump_cascade,
    plot_multi_hp_profiles_from_results,
)

__all__ = ["compute_direct_integration_targets"]

#######################################################################################################
# Public API
#######################################################################################################


def compute_direct_integration_targets(zone: Zone):
    """Populate a ``Zone`` with detailed direct heat integration pinch targets.

    The function aggregates problem-table calculations, multi-utility targeting,
    pinch temperature detection, and graph preparation.  Results are cached on
    the provided ``zone`` and used later by site and regional aggregation routines.
    """
    zone_config: Configuration = zone.config
    hot_streams: StreamCollection = zone.hot_streams
    cold_streams: StreamCollection = zone.cold_streams
    all_streams: StreamCollection = zone.all_streams
    hot_utilities: StreamCollection = zone.hot_utilities
    cold_utilities: StreamCollection = zone.cold_utilities
    res: dict = {}

    pt, pt_real, target_values = get_process_heat_cascade(
        hot_streams, cold_streams, all_streams, zone_config
    )
    hot_pinch, cold_pinch = pt.pinch_temperatures()
    pt = get_additional_GCCs(
        pt, 
        is_direct_integration=True, 
        do_vert_cc_calc=zone_config.DO_VERTICAL_GCC,
        do_assisted_ht_calc=zone_config.DO_ASSITED_HT,
    )
    
    if zone.identifier in [Z.P.value]:
        # T_vals=pt.col[PT.T.value]
        # H_hot=pt.col[PT.H_HOT_UT.value]
        # H_cold=pt.col[PT.H_COLD_UT.value]
        if _validate_heat_pump_targeting_required(pt, True, zone_config):
            hp_res = get_heat_pump_targets(
                T_vals=pt.col[PT.T.value],
                H_hot=pt.col[PT.H_NET_HOT.value],
                H_cold=pt.col[PT.H_NET_COLD.value],
                zone_config=zone_config, 
                is_direct_integration=True,
                is_heat_pumping=True,
            )
            res.update(
                hp_res
            )
            calc_heat_pump_cascade(
                pt=pt,
                res=hp_res,
                is_T_vals_shifted=True,
                is_direct_integration=True,
            )
            if 0:
                plot_multi_hp_profiles_from_results(
                    T_hot=pt.col[PT.T.value],
                    H_hot=pt.col[PT.H_NET_HOT.value],
                    T_cold=pt.col[PT.T.value],                    
                    H_cold=pt.col[PT.H_NET_COLD.value],
                    hp_hot_streams=hp_res.cond_streams,
                    hp_cold_streams=hp_res.evap_streams,
                )
                plot_multi_hp_profiles_from_results(
                    T_hot=pt.col[PT.T.value],
                    H_hot=pt.col[PT.H_NET_W_AIR.value],
                )
                pass
    
    get_utility_targets(
        pt, pt_real, hot_utilities, cold_utilities, is_direct_integration=True
    )
    net_hot_streams, net_cold_streams = _create_net_hot_and_cold_stream_collections_for_site_analysis(
        pt.col[PT.T.value], pt.col[PT.H_NET_A.value], hot_utilities, cold_utilities
    )
    if zone_config.DO_BALANCED_CC or zone_config.DO_AREA_TARGETING:
        pt.update(
            get_balanced_CC(
                pt.col[PT.H_HOT.value],
                pt.col[PT.H_COLD.value],
                pt.col[PT.H_HOT_UT.value],
                pt.col[PT.H_COLD_UT.value],
            )
        )  
        pt_real.update(
            get_balanced_CC(
                pt_real.col[PT.H_HOT.value],
                pt_real.col[PT.H_COLD.value],
                pt_real.col[PT.H_HOT_UT.value],
                pt_real.col[PT.H_COLD_UT.value],
                pt_real.col[PT.DELTA_T.value],
                pt_real.col[PT.RCP_HOT.value],
                pt_real.col[PT.RCP_COLD.value],
                pt_real.col[PT.RCP_HOT_UT.value],
                pt_real.col[PT.RCP_COLD_UT.value],
            )
        )
        # Target capital cost and heat transfer area and number of exchanger units based on Balanced CC
        if zone_config.DO_AREA_TARGETING:
            num_units = get_min_number_hx(
                pt.col[PT.T.value],
                pt.col[PT.H_HOT_BAL.value],
                pt.col[PT.H_COLD_BAL.value],
                hot_streams,
                cold_streams,
                hot_utilities,
                cold_utilities,               
            )
            area = get_area_targets(
                pt_real.col[PT.T.value],
                pt_real.col[PT.H_HOT_BAL.value],
                pt_real.col[PT.H_COLD_BAL.value],
                pt_real.col[PT.R_HOT_BAL.value],
                pt_real.col[PT.R_COLD_BAL.value], 
            )
            capital_cost, annual_capital_cost = get_capital_cost_targets(
                area,
                num_units,
                zone_config,
            )
            res.update(
                {
                    "Area target": area,
                    "Units target": num_units,
                    "Capital cost target": capital_cost,
                    "Annualised capital cost target": annual_capital_cost,
                }
            )

    res.update(
        {
            "pt": pt,
            "pt_real": pt_real,
            "target_values": target_values,
            "graphs": _save_graph_data(pt, pt_real),
            "hot_utilities": hot_utilities,
            "cold_utilities": cold_utilities,
            "net_hot_streams": net_hot_streams,
            "net_cold_streams": net_cold_streams,
            "hot_pinch": hot_pinch,
            "cold_pinch": cold_pinch,
        }
    )
    zone.add_target_from_results(TargetType.DI.value, res)
    return zone


#######################################################################################################
# Helper functions
#######################################################################################################


def _create_net_hot_and_cold_stream_collections_for_site_analysis(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
) -> Tuple[StreamCollection, StreamCollection]:
    """Constructs net stream segments that require utility input across temperature intervals."""
    net_hot_streams = StreamCollection()
    net_cold_streams = StreamCollection()

    if (sum([u.heat_flow for u in hot_utilities]) + 
        sum([u.heat_flow for u in cold_utilities])) < tol:
        # If no utility is needed, there is no net streams for indirect integration. 
        return net_hot_streams, net_cold_streams
    
    if delta_vals(T_vals).min() < tol:
        raise ValueError("Infeasible temperature interval detected in _store_TSP_data")

    T_vals = T_vals
    dh_vals = delta_vals(H_vals)

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


def _save_graph_data(pt: ProblemTable, pt_real: ProblemTable) -> Zone:
    """Assemble the problem-table slices required for composite/comparison plots."""
    pt.round(decimals=4)
    pt_real.round(decimals=4)
    return {
        GT.CC.value: pt_real[[PT.T.value, PT.H_HOT.value, PT.H_COLD.value]],
        GT.SCC.value: pt[[PT.T.value, PT.H_HOT.value, PT.H_COLD.value]],
        GT.BCC.value: pt_real[[PT.T.value, PT.H_HOT_BAL.value, PT.H_COLD_BAL.value]],
        GT.GCC.value: pt[[PT.T.value, PT.H_NET.value, PT.H_NET_NP.value, PT.H_NET_V.value, PT.H_NET_A.value, PT.H_NET_UT.value]],
        GT.GCC_R.value: pt_real[[PT.T.value, PT.H_NET.value, PT.H_NET_UT.value]],
        GT.NLC.value: pt[[PT.T.value, PT.H_NET_HOT.value, PT.H_NET_COLD.value, PT.H_HOT_UT.value, PT.H_COLD_UT.value]],
        GT.GCC_HP.value: pt[[PT.T.value, PT.H_NET_W_AIR.value, PT.H_NET_HP_PRO.value]],
    }


def _validate_heat_pump_targeting_required(
    pt: ProblemTable,
    is_heat_pumping: bool,
    zone_config: Configuration,
) -> bool:
    return False if (
        (zone_config.DO_PROCESS_HP_TARGETING == False)
        or 
        (np.abs(pt.col[PT.H_NET_COLD.value]).max() < tol and is_heat_pumping == True)
        or
        (np.abs(pt.col[PT.H_NET_HOT.value]).max() < tol and is_heat_pumping == False)
        or 
        (zone_config.HP_LOAD_FRACTION < tol)
    ) else True
