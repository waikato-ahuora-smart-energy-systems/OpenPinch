"""Indirect heat-integration entry point for total site style targeting.

The routines in this module aggregate process-level direct-integration outputs
from subzones, construct site process/utility cascades, and calculate net
utility balances after feasible inter-zone heat recovery.
"""

from copy import deepcopy
from typing import Dict, Tuple

import numpy as np

from ...classes.energy_target import EnergyTarget
from ...classes.problem_table import ProblemTable
from ...classes.stream import Stream
from ...classes.stream_collection import StreamCollection
from ...classes.zone import Zone
from ...lib.config import tol
from ...lib.enums import GT, PT, TargetType, Z
from ..heat_pump_integration.heat_pump_and_refrigeration_entry import (
    get_indirect_heat_pump_and_refrigeration_target,
)
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

def compute_total_subzone_utility_targets(zone: Zone) -> EnergyTarget:
    """Sums and records zonal targets."""
    target = EnergyTarget(
        zone_name=zone.name,
        identifier=TargetType.TZ.value,
        parent_zone=zone.parent_zone,
        zone_config=zone.config,
    )
    # Sum targets from subzones
    hot_utility_target = cold_utility_target = heat_recovery_target = 0.0
    utility_cost = num_units = area = capital_cost = total_cost = 0.0

    hot_utilities = deepcopy(zone.hot_utilities)
    cold_utilities = deepcopy(zone.cold_utilities)
    hot_utilities, cold_utilities = _reset_utility_heat_flows(
        hot_utilities, cold_utilities
    )
    for z in zone.subzones.values():
        t = z.targets[TargetType.DI.value]
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

    heat_recovery_limit = zone.targets[TargetType.DI.value].heat_recovery_limit
    target.update(
        {
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
        }        
    )
    return target


def compute_indirect_integration_targets(zone: Zone) -> EnergyTarget:
    """Compute indirect integration targets for an aggregated zone.

    The routine assumes the relevant child zones have already been solved for
    direct integration. It then sums subzone targets, builds site-level net
    stream cascades, performs utility-to-utility balancing, and records the
    resulting total-site style target on ``zone`` before returning it.
    """ 
    target = EnergyTarget(
        zone_name=zone.name,
        identifier=TargetType.TS.value,
        parent_zone=zone.parent_zone,
        zone_config=zone.config,
    )
    # Total site profiles - process side
    pt = get_process_heat_cascade(
        hot_streams=zone.net_hot_streams,
        cold_streams=zone.net_cold_streams,
        all_streams=zone.all_net_streams,
        zone_config=zone.config,
        is_shifted=True,
    )
    pt_real = get_process_heat_cascade(
        hot_streams=zone.net_hot_streams,
        cold_streams=zone.net_cold_streams,
        all_streams=zone.all_net_streams,
        zone_config=zone.config,
        is_shifted=False,
        known_heat_recovery=get_heat_recovery_target_from_pt(pt),
    )
    pt.update(
        _get_site_process_heat_load_profiles(
            pt.col[PT.H_HOT.value], pt.col[PT.H_COLD.value]
        )
    )
    pt_real.update(
        _get_site_process_heat_load_profiles(
            pt_real.col[PT.H_HOT.value], pt_real.col[PT.H_COLD.value]
        )
    )

    # Get utility duties based on the summation of subzones
    s_tzt = zone.targets[TargetType.TZ.value]
    hot_utilities = deepcopy(s_tzt.hot_utilities)
    cold_utilities = deepcopy(s_tzt.cold_utilities)

    # Apply the problem table algorithm to the simple sum of subzone utility use
    pt.update(
        _get_site_utility_heat_cascade(
            pt.col[PT.T.value],
            hot_utilities,
            cold_utilities,
            is_shifted=True,
        )
    )
    pt_real.update(
        _get_site_utility_heat_cascade(
            pt_real.col[PT.T.value],
            hot_utilities,
            cold_utilities,
            is_shifted=False,
        )
    )

    # Apply the utility targeting method to determine the net utility use and generation
    _match_utility_gen_and_use_at_same_level(hot_utilities, cold_utilities)

    # Determine if heat pump or refrigeration targeting is warranted based on the utility cascade profiles and user settings
    if zone.identifier == Z.S.value and (
        zone.config.DO_UTILITY_HP_TARGETING or zone.config.DO_UTILITY_RFRG_TARGETING
    ):
        target.update(
            get_indirect_heat_pump_and_refrigeration_target(
                pt=pt,
                hot_utilities=hot_utilities,
                cold_utilities=cold_utilities,
                zone_name=zone.name,
                zone_config=zone.config,
            )
        )

    # Extract overall heat integration targets
    hot_utility_target = pt.loc[0, PT.H_NET_UT.value]
    cold_utility_target = pt.loc[-1, PT.H_NET_UT.value]
    heat_recovery_target = s_tzt.heat_recovery_target + (
        s_tzt.hot_utility_target - hot_utility_target
    )
    hot_pinch, cold_pinch = pt.pinch_temperatures(col_H=PT.H_NET_UT.value)

    target.update(
        {
            "pt": pt,
            "pt_real": pt_real,
            "graphs": _save_graph_data(pt, pt_real),          
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
    )
    return target


#######################################################################################################
# Helper Functions
#######################################################################################################


def _reset_utility_heat_flows(
    hot_utilities: StreamCollection, cold_utilities: StreamCollection
) -> Tuple[StreamCollection, StreamCollection]:
    """Zero out utility heat flows prior to accumulating site-level demands."""
    hu: Stream
    for hu in hot_utilities:
        hu.heat_flow = 0.0
    cu: Stream
    for cu in cold_utilities:
        cu.heat_flow = 0.0
    return hot_utilities, cold_utilities


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


def _get_site_process_heat_load_profiles(
    H_hot: np.ndarray,
    H_cold: np.ndarray,
) -> dict:
    return {
        PT.H_NET_HOT.value: H_hot - H_hot[0],
        PT.H_NET_COLD.value: H_cold - H_cold[-1],
    }


def _get_site_utility_heat_cascade(
    T_int_vals: np.ndarray,
    hot_utilities: StreamCollection = None,
    cold_utilities: StreamCollection = None,
    is_shifted: bool = True,
) -> Dict[str, np.ndarray]:
    """Prepare and calculate the utility heat cascade a given set of hot and cold utilities."""
    pt_ut_hot = ProblemTable({PT.T.value: T_int_vals})
    problem_table_algorithm(pt_ut_hot, hot_streams=hot_utilities, is_shifted=is_shifted)

    pt_ut_cld = ProblemTable({PT.T.value: T_int_vals})
    problem_table_algorithm(
        pt_ut_cld, cold_streams=cold_utilities, is_shifted=is_shifted
    )

    pt_ut = ProblemTable({PT.T.value: T_int_vals})
    problem_table_algorithm(pt_ut, hot_utilities, cold_utilities, is_shifted=is_shifted)

    h_net_values = pt_ut.col[PT.H_NET.value]
    H_NET_UT = h_net_values.max() - h_net_values

    h_ut_cc = pt_ut_hot.col[PT.H_HOT.value]
    c_ut_cc = pt_ut_cld.col[PT.H_COLD.value] - pt_ut_cld.col[PT.H_COLD.value].max()

    return {
        PT.H_NET_UT.value: H_NET_UT,
        PT.H_HOT_UT.value: h_ut_cc,
        PT.H_COLD_UT.value: c_ut_cc,
    }


def _save_graph_data(
    pt: ProblemTable,
    pt_real: ProblemTable,
) -> Zone:
    """Prepare graph-ready tables capturing site-level utility composite curves."""
    pt.round(decimals=4)
    pt_real.round(decimals=4)
    return {
        GT.TSP.value: pt[
            [
                PT.T.value,
                PT.H_NET_HOT.value,
                PT.H_NET_COLD.value,
                PT.H_HOT_UT.value,
                PT.H_COLD_UT.value,
            ]
        ],
        GT.SUGCC.value: pt_real[[PT.T.value, PT.H_NET_UT.value, PT.H_NET_HP.value]],
    }
