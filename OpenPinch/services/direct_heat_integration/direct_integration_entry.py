"""Direct heat integration entry point for process and unit-level targeting."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ...classes.problem_table import ProblemTable
from ...classes.stream import Stream
from ...classes.stream_collection import StreamCollection
from ...classes.zone import Zone
from ...lib.config import tol
from ...lib.enums import GT, PT, ST, TT
from ...lib.schemas.targets import DirectIntegrationTarget
from ..common.capital_cost_and_area_targeting import (
    get_area_targets,
    get_balanced_CC,
    get_capital_cost_targets,
    get_min_number_hx,
)
from ..common.gcc_manipulation import get_additional_GCCs
from ..common.miscellaneous import delta_vals, get_state_index
from ..common.problem_table_analysis import (
    get_heat_recovery_target_from_pt,
    get_process_heat_cascade,
    set_zonal_targets,
)
from ..common.utility_targeting import get_utility_targets

__all__ = ["compute_direct_integration_targets"]

################################################################################
# Public API
################################################################################


def compute_direct_integration_targets(
    zone: Zone,
    args: dict | None = None,
) -> DirectIntegrationTarget:
    """Populate a ``Zone`` with detailed direct heat integration pinch targets.

    The function aggregates Problem Table calculations, multi-utility targeting,
    pinch temperature detection, and graph preparation.  Results are cached on
    the provided ``zone`` and used later by site and regional aggregation routines.
    """
    idx, sid = get_state_index(state_ids=zone.state_ids, args=args)
    all_streams = zone.all_streams
    pt = get_process_heat_cascade(
        hot_streams=zone.hot_streams,
        cold_streams=zone.cold_streams,
        all_streams=all_streams,
        is_shifted=True,
        idx=idx,
    )
    pt_real = get_process_heat_cascade(
        hot_streams=zone.hot_streams,
        cold_streams=zone.cold_streams,
        all_streams=all_streams,
        is_shifted=False,
        known_heat_recovery=get_heat_recovery_target_from_pt(pt),
        idx=idx,
    )
    hot_pinch, cold_pinch = pt.pinch_temperatures()
    pt = get_additional_GCCs(
        pt,
        do_vert_cc_calc=zone.config.DO_VERTICAL_GCC,
        do_assisted_ht_calc=zone.config.DO_ASSITED_HT,
        assisted_ht_dt_cut=zone.config.DT_ASSISTED_HT,
    )

    get_utility_targets(
        pt=pt,
        pt_real=pt_real,
        hot_utilities=zone.hot_utilities,
        cold_utilities=zone.cold_utilities,
        is_direct_integration=True,
        idx=idx,
    )
    zone.net_hot_streams, zone.net_cold_streams = (
        _create_net_hot_and_cold_stream_collections_for_site_analysis(
            T_vals=pt[PT.T],
            H_vals=pt[PT.H_NET_A],
            hot_utilities=zone.hot_utilities,
            cold_utilities=zone.cold_utilities,
            idx=idx,
        )
    )
    if zone.config.DO_BALANCED_CC or zone.config.DO_AREA_TARGETING:
        pt.update(
            **get_balanced_CC(
                T_col=pt[PT.T],
                H_hot=pt[PT.H_HOT],
                H_cold=pt[PT.H_COLD],
                H_hot_ut=pt[PT.H_HOT_UT],
                H_cold_ut=pt[PT.H_COLD_UT],
            )
        )
        pt_real.update(
            **get_balanced_CC(
                T_col=pt_real[PT.T],
                H_hot=pt_real[PT.H_HOT],
                H_cold=pt_real[PT.H_COLD],
                H_hot_ut=pt_real[PT.H_HOT_UT],
                H_cold_ut=pt_real[PT.H_COLD_UT],
                dT_vals=pt_real[PT.DELTA_T],
                RCP_hot=pt_real[PT.RCP_HOT],
                RCP_cold=pt_real[PT.RCP_COLD],
                RCP_hot_ut=pt_real[PT.RCP_HOT_UT],
                RCP_cold_ut=pt_real[PT.RCP_COLD_UT],
            )
        )
        # Target capital cost, area, and exchanger count from the balanced CC.
        if zone.config.DO_AREA_TARGETING:
            num_units = get_min_number_hx(
                T_vals=pt[PT.T],
                H_hot_bal=pt[PT.H_HOT_BAL],
                H_cold_bal=pt[PT.H_COLD_BAL],
                hot_streams=zone.hot_streams,
                cold_streams=zone.cold_streams,
                hot_utilities=zone.hot_utilities,
                cold_utilities=zone.cold_utilities,
                idx=idx,
            )
            area = get_area_targets(
                T_vals=pt_real[PT.T],
                H_hot_bal=pt_real[PT.H_HOT_BAL],
                H_cold_bal=pt_real[PT.H_COLD_BAL],
                R_hot_bal=pt_real[PT.R_HOT_BAL],
                R_cold_bal=pt_real[PT.R_COLD_BAL],
            )
            capital_cost, annual_capital_cost = get_capital_cost_targets(
                area=area,
                num_units=num_units,
                zone_config=zone.config,
            )
            area_payload = {
                "area": area,
                "num_units": num_units,
                "capital_cost": capital_cost,
                "total_cost": annual_capital_cost,
            }
        else:
            area_payload = {}
    else:
        area_payload = {}

    payload = (
        set_zonal_targets(
            pt=pt,
            pt_real=pt_real,
        )
        | {
            "zone_name": zone.name,
            "type": TT.DI.value,
            "parent_zone": zone.parent_zone,
            "config": zone.config,
            "pt": pt,
            "pt_real": pt_real,
            "graphs": _save_graph_data(pt, pt_real),
            "hot_utilities": zone.hot_utilities,
            "cold_utilities": zone.cold_utilities,
            "hot_pinch": hot_pinch,
            "cold_pinch": cold_pinch,
            "state_id": sid,
            "state_idx": idx,
        }
        | area_payload
    )
    return DirectIntegrationTarget.model_validate(payload)


################################################################################
# Helper functions
################################################################################


def _create_net_hot_and_cold_stream_collections_for_site_analysis(
    T_vals: np.ndarray,
    H_vals: np.ndarray,
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
    idx: int | None = None,
) -> Tuple[StreamCollection, StreamCollection]:
    """Construct net stream segments requiring utility input by interval."""
    net_hot_streams = StreamCollection()
    net_cold_streams = StreamCollection()

    hot_utilities_seq = list(hot_utilities)
    cold_utilities_seq = list(cold_utilities)
    hot_remaining = [
        StreamCollection._value_at_idx(utility._heat_flow, idx)
        for utility in hot_utilities_seq
    ]
    cold_remaining = [
        StreamCollection._value_at_idx(utility._heat_flow, idx)
        for utility in cold_utilities_seq
    ]

    if np.nansum(hot_remaining) + np.nansum(cold_remaining) < tol:
        # If no utility is needed, there is no net streams for indirect integration.
        return net_hot_streams, net_cold_streams

    if delta_vals(T_vals).min() < tol:
        raise ValueError("Infeasible temperature interval detected in _store_TSP_data")

    T_vals = T_vals
    dh_vals = delta_vals(H_vals)

    hu_idx = _initialise_utility_index(hot_utilities_seq, hot_remaining)
    cu_idx = _initialise_utility_index(cold_utilities_seq, cold_remaining)

    k = 1
    for i, dh in enumerate(dh_vals):
        if dh > tol and hu_idx >= 0:
            hu_idx, k = _add_net_segment_stateful(
                T_ub=T_vals[i],
                T_lb=T_vals[i + 1],
                curr_idx=hu_idx,
                dh_req=dh,
                utilities=hot_utilities_seq,
                remaining=hot_remaining,
                net_streams=net_cold_streams,
                k=k,
                idx=idx,
            )
        elif -dh > tol and cu_idx >= 0:
            cu_idx, k = _add_net_segment_stateful(
                T_ub=T_vals[i],
                T_lb=T_vals[i + 1],
                curr_idx=cu_idx,
                dh_req=abs(dh),
                utilities=cold_utilities_seq,
                remaining=cold_remaining,
                net_streams=net_hot_streams,
                k=k,
                idx=idx,
            )

    return net_hot_streams, net_cold_streams


def _add_net_segment_stateful(
    T_ub: float,
    T_lb: float,
    curr_idx: int,
    dh_req: float,
    utilities: StreamCollection,
    remaining: List[float],
    net_streams: StreamCollection,
    k: int,
    j: int = 0,
    idx: int | None = None,
) -> Tuple[int, int]:
    """Add net utility segments, splitting iteratively across utilities."""
    if curr_idx < 0 or not utilities or dh_req <= tol:
        return curr_idx, k

    segment_upper = float(T_ub)
    segment_lower = float(T_lb)
    remaining_dh = float(dh_req)
    split_idx = int(j)

    while remaining_dh > tol and curr_idx >= 0:
        available = float(remaining[curr_idx]) if curr_idx < len(remaining) else 0.0
        if available <= tol:
            next_idx = _find_next_available_utility(curr_idx + 1, utilities, remaining)
            if next_idx == curr_idx:
                break
            curr_idx = next_idx
            continue

        curr_u = utilities[curr_idx]
        dh_curr = float(min(remaining_dh, available))
        remaining[curr_idx] = max(available - dh_curr, 0.0)

        span = segment_upper - segment_lower
        split_temp = (
            segment_upper - (dh_curr / remaining_dh) * span
            if span and remaining_dh > tol
            else segment_lower
        )

        net_streams.add(
            Stream(
                name=f"Segment {k}" if split_idx == 0 else f"Segment {k}-{split_idx}",
                t_supply=split_temp if curr_u.type == ST.Hot.value else segment_upper,
                t_target=segment_upper if curr_u.type == ST.Hot.value else split_temp,
                heat_flow=dh_curr,
                dt_cont=StreamCollection._value_at_idx(curr_u._dt_cont, idx),
                dt_cont_multiplier=curr_u.dt_cont_multiplier,
                htc=1.0,
                is_process_stream=True,
            )
        )

        remaining_dh -= dh_curr
        if remaining_dh <= tol:
            break

        segment_upper = split_temp
        curr_idx = _find_next_available_utility(curr_idx + 1, utilities, remaining)
        split_idx += 1

    next_idx = (
        curr_idx
        if 0 <= curr_idx < len(remaining) and remaining[curr_idx] > tol
        else _find_next_available_utility(curr_idx + 1, utilities, remaining)
    )
    return next_idx, k + 1


def _initialise_utility_index(
    utilities: StreamCollection, remaining: List[float]
) -> int:
    """Returns the index of the first available utility with remaining capacity."""
    for idx, residual in enumerate(remaining):
        if residual > tol:
            return idx
    return len(utilities) - 1 if utilities else -1


def _find_next_available_utility(
    start: int, utilities: StreamCollection, remaining: List[float]
) -> int:
    """Return the index of the next utility that still has remaining duty."""
    if not utilities:
        return -1
    for idx in range(max(start, 0), len(utilities)):
        if remaining[idx] > tol:
            return idx
    return len(utilities) - 1


def _save_graph_data(pt: ProblemTable, pt_real: ProblemTable) -> dict:
    """Assemble the Problem Table slices required for composite/comparison plots."""
    pt.round(decimals=4)
    pt_real.round(decimals=4)
    return {
        GT.CC.value: pt_real.slice([PT.T, PT.H_HOT, PT.H_COLD]),
        GT.SCC.value: pt.slice([PT.T, PT.H_HOT, PT.H_COLD]),
        GT.BCC.value: pt_real.slice([PT.T, PT.H_HOT_BAL, PT.H_COLD_BAL]),
        GT.GCC.value: pt.slice(
            [PT.T, PT.H_NET, PT.H_NET_NP, PT.H_NET_V, PT.H_NET_A, PT.H_NET_UT]
        ),
        GT.GCC_R.value: pt_real.slice([PT.T, PT.H_NET, PT.H_NET_UT]),
        GT.NLP.value: pt.slice(
            [
                PT.T,
                PT.H_NET_HOT,
                PT.H_NET_COLD,
                PT.H_HOT_UT,
                PT.H_COLD_UT,
                PT.H_HOT_HP,
                PT.H_COLD_HP,
            ]
        ),
        GT.GCC_HP.value: pt.slice([PT.T, PT.H_NET_W_AIR, PT.H_NET_HP]),
    }
