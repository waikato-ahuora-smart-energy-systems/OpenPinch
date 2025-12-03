from copy import deepcopy
from typing import Tuple

from ..classes import *
from ..lib import *
from ..utils import *
from . import (
    get_process_heat_cascade,
    get_utility_heat_cascade,
    get_utility_targets,
    get_additional_GCCs,
    get_heat_pump_targets,
    calc_heat_pump_cascade,
    plot_multi_hp_profiles_from_results
)

__all__ = ["compute_indirect_integration_targets"]


#######################################################################################################
# Public API
#######################################################################################################


def compute_indirect_integration_targets(zone: Zone) -> Zone:
    """Targets indirect heat integration, such as for Total Site, 
    after computing direct heat integration in subzones.
    """
    zone_config: Configuration = zone.config
    res: dict = {}

    # Sum targets from subzones 
    _sum_subzone_targets(zone)

    # Total site profiles - process side
    zone.import_hot_and_cold_streams_from_sub_zones(get_net_streams=True)
    pt, pt_real, _ = get_process_heat_cascade(
        zone.net_hot_streams, zone.net_cold_streams, zone.all_net_streams, zone.config
    )
    pt.update(
        _shift_composite_curves(pt.col[PT.H_HOT.value], pt.col[PT.H_COLD.value])
    )
    pt_real.update(
        _shift_composite_curves(pt_real.col[PT.H_HOT.value], pt_real.col[PT.H_COLD.value])
    )    

    # Get utility duties based on the summation of subzones
    s_tzt: EnergyTarget = zone.targets[key_name(zone.name, TargetType.TZ.value)]
    hot_utilities = deepcopy(s_tzt.hot_utilities)
    cold_utilities = deepcopy(s_tzt.cold_utilities)
    
    # Apply the problem table algorithm to the simple sum of subzone utility use 
    pt.update(
        get_utility_heat_cascade(
            pt.col[PT.T.value], 
            hot_utilities, 
            cold_utilities, 
            is_shifted=True,
        )
    )
    pt_real.update(
        get_utility_heat_cascade(
            pt_real.col[PT.T.value], 
            hot_utilities, 
            cold_utilities, 
            is_shifted=False,
        )
    )

    # Apply the utility targeting method to determine the net utility use and generation 
    _match_utility_gen_and_use_at_same_level(
        hot_utilities, cold_utilities
    )
    pt = get_additional_GCCs(
        pt, 
        is_direct_integration=False
    )
    get_utility_targets(
        pt, pt_real, 
        hot_utilities, cold_utilities,
        is_direct_integration=False
    )

    # Extract overall heat integration targets
    hot_utility_target = pt.loc[0, PT.H_NET_UT.value]
    cold_utility_target = pt.loc[-1, PT.H_NET_UT.value]
    heat_recovery_target = s_tzt.heat_recovery_target + (
        s_tzt.hot_utility_target - hot_utility_target
    )
    hot_pinch, cold_pinch = pt.pinch_temperatures(col_H=PT.H_NET_UT.value)

    if zone.identifier in [Z.S.value]:
        if _validate_heat_pump_targeting_required(pt, True, zone_config):
            hp_res = get_heat_pump_targets(
                T_vals=pt.col[PT.T.value],
                H_hot=pt.col[PT.H_HOT_UT.value],
                H_cold=pt.col[PT.H_COLD_UT.value],
                zone_config=zone_config, 
                is_direct_integration=False,
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
                    hp_hot_streams=hp_res.hp_hot_streams,
                    hp_cold_streams=hp_res.hp_cold_streams,
                )
                plot_multi_hp_profiles_from_results(
                    T_hot=pt.col[PT.T.value],
                    H_hot=pt.col[PT.H_NET_W_AIR.value],
                )
                pass

    # if zone_config.DO_TURBINE_WORK:
    #     work_target = 0.0
    #     if zone_config.ABOVE_PINCH_CHECKBOX:
    #         pass
    #         # s_tsi = get_power_cogeneration_above_pinch(s_tsi)
    #     utility_cost = utility_cost - work_target / 1000 * zone_config.ELECTRICITY_PRICE * zone_config.ANNUAL_OP_TIME

    graphs = _save_graph_data(pt, pt_real)

    target_values = _set_sites_targets(
        hot_utility_target,
        cold_utility_target,
        heat_recovery_target,
        s_tzt.heat_recovery_limit,
    )
    res.update(
        {
            "pt": pt,
            "pt_real": pt_real,
            "target_values": target_values,
            "graphs": graphs,
            "hot_utilities": hot_utilities,
            "cold_utilities": cold_utilities,
            "hot_pinch": hot_pinch,
            "cold_pinch": cold_pinch,
            "utility_cost": _compute_utility_cost(hot_utilities, cold_utilities),
        }            
    )
    zone.add_target_from_results(TargetType.TS.value, res)
    return zone


#######################################################################################################
# Helper Functions
#######################################################################################################


def _sum_subzone_targets(zone: Zone) -> Zone:
    """Sums and records zonal targets."""
    hot_utility_target = cold_utility_target = heat_recovery_target = 0.0
    utility_cost = num_units = area = capital_cost = total_cost = 0.0

    hot_utilities = deepcopy(zone.hot_utilities)
    cold_utilities = deepcopy(zone.cold_utilities)
    hot_utilities, cold_utilities = _reset_utility_heat_flows(
        hot_utilities, cold_utilities
    )

    for z in zone.subzones.values():
        z: Zone
        t: EnergyTarget
        t = z.targets[f"{z.name}/{TargetType.DI.value}"]
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

    heat_recovery_limit = zone.targets[
        f"{zone.name}/{TargetType.DI.value}"
    ].heat_recovery_limit

    # Target co-generation of heat and power
    # if zone_config.DO_TURBINE_WORK:
    #     st = get_power_cogeneration_above_pinch(st)
    #     utility_cost = utility_cost - work_target / 1000 * zone_config.ELECTRICITY_PRICE * zone_config.ANNUAL_OP_TIME

    target_values = _set_sites_targets(
        hot_utility_target,
        cold_utility_target,
        heat_recovery_target,
        heat_recovery_limit,
    )
    zone.add_target_from_results(
        TargetType.TZ.value,
        {
            "target_values": target_values,
            "hot_utilities": hot_utilities,
            "cold_utilities": cold_utilities,
        },
    )
    return zone


def _reset_utility_heat_flows(
    hot_utilities: StreamCollection, 
    cold_utilities: StreamCollection
) -> Tuple[StreamCollection, StreamCollection]:
    """Zero out utility heat flows prior to accumulating site-level demands."""
    hu: Stream
    for hu in hot_utilities:
        hu.heat_flow = 0.0
    cu: Stream
    for cu in cold_utilities:
        cu.heat_flow = 0.0
    return hot_utilities, cold_utilities


def _set_sites_targets(
    hot_utility_target, cold_utility_target, heat_recovery_target, heat_recovery_limit
) -> dict:
    """Assign thermal targets and integration degree to the zone based on site analysis methods."""
    return {
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


def _match_utility_gen_and_use_at_same_level(
    hot_utilities: StreamCollection, 
    cold_utilities: StreamCollection
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


def _compute_utility_cost(hot_utilities: StreamCollection, cold_utilities: StreamCollection) -> float:
    utility_cost = 0.0
    for u in hot_utilities + cold_utilities:
        utility_cost += u.ut_cost
    return utility_cost


def _shift_composite_curves(H_hot: np.ndarray, H_cold: np.ndarray) -> dict:
    return {
        PT.H_HOT.value: H_hot - H_hot[0],
        PT.H_COLD.value: H_cold - H_cold[-1],
    }


def _save_graph_data(pt: ProblemTable, pt_real: ProblemTable) -> Zone:
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
        GT.SUGCC.value: pt[[PT.T.value, PT.H_NET_UT.value]],
    }


def _validate_heat_pump_targeting_required(
    pt: ProblemTable,
    is_heat_pumping: bool,
    zone_config: Configuration,
) -> bool:
    return False if (
        (zone_config.DO_UTILITY_HP_TARGETING == False)
        or 
        (pt.col[PT.H_NET_UT.value][0] < tol and is_heat_pumping == True)
        or
        (pt.col[PT.H_NET_UT.value][-1] < tol and is_heat_pumping == False)
        or 
        (zone_config.HP_LOAD_FRACTION < tol)
    ) else True
