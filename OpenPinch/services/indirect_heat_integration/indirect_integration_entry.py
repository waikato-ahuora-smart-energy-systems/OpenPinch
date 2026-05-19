"""Indirect heat-integration entry point for total site style targeting.

The routines in this module aggregate process-level direct-integration outputs
from subzones, construct site process/utility cascades, and calculate net
utility balances after feasible inter-zone heat recovery.
"""

from copy import deepcopy
from typing import Tuple

import numpy as np

from ...classes.problem_table import ProblemTable
from ...classes.stream_collection import StreamCollection
from ...classes.zone import Zone
from ...lib.config import tol
from ...lib.enums import GT, PT, TT
from ...lib.problem_table_types import ProblemTableUpdateKwargs
from ...lib.schemas.targets import TotalProcessTarget, TotalSiteTarget
from ..common.problem_table_analysis import (
    get_heat_recovery_target_from_pt,
    get_process_heat_cascade,
    problem_table_algorithm,
)

__all__ = [
    "compute_total_subzone_utility_targets",
    "compute_indirect_integration_targets",
]


#######################################################################################################
# Public API
#######################################################################################################


def compute_total_subzone_utility_targets(zone: Zone) -> TotalProcessTarget:
    """Sums and records zonal targets."""
    # Sum targets from subzones
    hot_utility_target = cold_utility_target = heat_recovery_target = 0.0
    utility_cost = num_units = area = capital_cost = total_cost = 0.0

    hot_utilities = deepcopy(zone.hot_utilities).reset_heat_flows()
    cold_utilities = deepcopy(zone.cold_utilities).reset_heat_flows()
    for subzone in zone.subzones.values():
        t = subzone.targets[TT.DI.value]
        hot_utility_target += t.hot_utility_target
        cold_utility_target += t.cold_utility_target
        heat_recovery_target += t.heat_recovery_target
        utility_cost += t.utility_cost

        for j in range(len(hot_utilities)):
            hot_utilities[j].set_heat_flow(
                hot_utilities[j].heat_flow + t.hot_utilities[j].heat_flow
            )

        for j in range(len(cold_utilities)):
            cold_utilities[j].set_heat_flow(
                cold_utilities[j].heat_flow + t.cold_utilities[j].heat_flow
            )

        if area > tol:
            num_units += t.num_units
            area += t.area
            # capital_cost = t.capital_cost

    heat_recovery_limit = zone.targets[TT.DI.value].heat_recovery_limit
    payload = {
        "zone_name": zone.name,
        "type": TT.TZ.value,
        "parent_zone": zone.parent_zone,
        "config": zone.config,
        "hot_utilities": hot_utilities,
        "cold_utilities": cold_utilities,
        "hot_utility_target": hot_utility_target,
        "cold_utility_target": cold_utility_target,
        "heat_recovery_target": heat_recovery_target,
        "heat_recovery_limit": heat_recovery_limit,
        "degree_of_int": (
            (heat_recovery_target / heat_recovery_limit)
            if heat_recovery_limit > 0
            else 1.0
        ),
        "utility_cost": utility_cost,
    }
    return TotalProcessTarget.model_validate(payload)


def compute_indirect_integration_targets(zone: Zone) -> TotalSiteTarget:
    """Compute indirect integration targets for an aggregated zone.

    The routine assumes the relevant child zones have already been solved for
    direct integration. It then sums subzone targets, builds site-level net
    stream cascades, performs utility-to-utility balancing, and records the
    resulting total-site style target on ``zone`` before returning it.
    """
    s_tzt = zone.targets[TT.TZ.value]

    # Total site profiles - process side
    pt = get_process_heat_cascade(
        hot_streams=zone.net_hot_streams,
        cold_streams=zone.net_cold_streams,
        is_shifted=True, # Second shift of process streams to align with the real utility scale
    )
    pt.update(
        **_shift_site_process_profiles(
            T_col=pt.col[PT.T.value],
            H_hot=pt.col[PT.H_HOT.value],
            H_cold=pt.col[PT.H_COLD.value],
        )
    )
    # Apply the problem table algorithm to subzone utility use
    pt.update(
        **_build_site_utility_profile(
            hot_utilities=s_tzt.hot_utilities,
            cold_utilities=s_tzt.cold_utilities,
            is_shifted=False,
        )
    )

    # Extract overall heat integration targets
    hot_utility_target = pt.loc[0, PT.H_NET_UT.value]
    cold_utility_target = pt.loc[-1, PT.H_NET_UT.value]
    heat_recovery_target = s_tzt.heat_recovery_target + (
        s_tzt.hot_utility_target - hot_utility_target
    )
    hot_pinch, cold_pinch = pt.pinch_temperatures(col_H=PT.H_NET_UT.value)

    # Apply the utility targeting method to determine the net utility use and generation
    hot_utilities, cold_utilities = _match_utility_gen_and_use_at_same_level(
        hot_utilities=deepcopy(s_tzt.hot_utilities),
        cold_utilities=deepcopy(s_tzt.cold_utilities),
    )

    payload = {
        "zone_name": zone.name,
        "type": TT.TS.value,
        "parent_zone": zone.parent_zone,
        "config": zone.config,
        "pt": pt,
        "graphs": _save_graph_data(pt),
        "hot_pinch": hot_pinch,
        "cold_pinch": cold_pinch,
        "hot_utilities": hot_utilities,
        "cold_utilities": cold_utilities,
        "hot_utility_target": hot_utility_target,
        "cold_utility_target": cold_utility_target,
        "heat_recovery_target": heat_recovery_target,
        "heat_recovery_limit": s_tzt.heat_recovery_limit,
        "degree_of_int": (
            (heat_recovery_target / s_tzt.heat_recovery_limit)
            if s_tzt.heat_recovery_limit > 0
            else 1.0
        ),
        "utility_cost": _compute_utility_cost(hot_utilities, cold_utilities),
    }
    return TotalSiteTarget.model_validate(payload)


#######################################################################################################
# Helper Functions
#######################################################################################################


def _match_utility_gen_and_use_at_same_level(
    hot_utilities: StreamCollection, cold_utilities: StreamCollection
) -> Tuple[StreamCollection, StreamCollection]:
    for u_h in hot_utilities:
        for u_c in cold_utilities:
            if (
                abs(u_h.t_supply - u_c.t_target) < 1
                and abs(u_h.t_target - u_c.t_supply) < 1
            ):
                Q = min(u_h.heat_flow, u_c.heat_flow)
                u_h.set_heat_flow(u_h.heat_flow - Q)
                u_c.set_heat_flow(u_c.heat_flow - Q)
    return hot_utilities, cold_utilities


def _compute_utility_cost(
    hot_utilities: StreamCollection, cold_utilities: StreamCollection
) -> float:
    utility_cost = 0.0
    for u in hot_utilities + cold_utilities:
        utility_cost += u.ut_cost
    return utility_cost


def _shift_site_process_profiles(
    T_col: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
) -> ProblemTableUpdateKwargs:
    return {
        "T_col": T_col,
        "updates": {
            PT.H_HOT.value: H_hot - H_hot[0],
            PT.H_COLD.value: H_cold - H_cold[-1],
        },
    }


def _build_site_utility_profile(
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
    is_shifted: bool = False,
) -> ProblemTableUpdateKwargs:
    pt_ut = get_process_heat_cascade(
        hot_streams=hot_utilities,
        cold_streams=cold_utilities,
        is_shifted=is_shifted,
    )
    h_net_ut = pt_ut.col[PT.H_HOT.value] - pt_ut.col[PT.H_COLD.value]
    return {
        "T_col": pt_ut.col[PT.T.value],
        "updates": {
            PT.H_NET_UT.value: h_net_ut - h_net_ut.min(),
            PT.H_HOT_UT.value: pt_ut.col[PT.H_HOT.value],
            PT.H_COLD_UT.value: pt_ut.col[PT.H_COLD.value]
            - pt_ut.col[PT.H_COLD.value].max(),
        },
    }


def _save_graph_data(
    pt: ProblemTable,
) -> Zone:
    """Prepare graph-ready tables capturing site-level utility composite curves."""
    pt.round(decimals=4)
    return {
        GT.TSP.value: pt.slice([PT.T, PT.H_HOT, PT.H_COLD, PT.H_HOT_UT, PT.H_COLD_UT]),
        GT.SUGCC.value: pt.slice([PT.T, PT.H_NET_UT, PT.H_NET_HP]),
    }
