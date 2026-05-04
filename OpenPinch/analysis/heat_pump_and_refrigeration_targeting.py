"""Public entrypoint for heat-pump and refrigeration targeting."""

from ast import literal_eval

import numpy as np

from ..classes.problem_table import ProblemTable
from ..classes.zone import Zone
from ..classes.stream_collection import StreamCollection
from ..lib.config import Configuration, tol
from ..lib.enums import HPRcycle, PT
from ..lib.schema import HPRTargetOutputs
from ..utils.miscellaneous import get_value
from .gcc_manipulation import get_additional_GCCs
from .heat_pump_and_refrigeration_placement.brayton import (
    optimise_brayton_heat_pump_placement,
)
from .heat_pump_and_refrigeration_placement.cascade_vapour_compression import (
    optimise_cascade_heat_pump_placement,
)
from .heat_pump_and_refrigeration_placement.multi_simple_carnot import (
    optimise_multi_simple_carnot_heat_pump_placement,
)
from .heat_pump_and_refrigeration_placement.multi_simple_vapour_compression import (
    optimise_multi_simple_heat_pump_placement,
)
from .heat_pump_and_refrigeration_placement.multi_temperature_carnot import (
    optimise_multi_temperature_carnot_heat_pump_placement,
)
from .heat_pump_and_refrigeration_placement.preprocessing import (
    construct_HPRTargetInputs,
)
from .heat_pump_and_refrigeration_placement.shared import (
    get_process_heat_cascade,
    get_utility_heat_cascade,
    plot_multi_hp_profiles_from_results,
)
from .problem_table_analysis import create_problem_table_with_t_int

__all__ = [
    "get_indirect_heat_pump_and_refrigeration_target",
    "_validate_hpr_required",
    "_get_hpr_targets",
    "_calc_hpr_cascade",
    "plot_multi_hp_profiles_from_results",
]


#######################################################################################################
# Public API
#######################################################################################################


def get_direct_heat_pump_and_refrigeration_target(
    pt: ProblemTable,
    zone_name: str,
    zone_config: Configuration,
) -> dict:
    res = {}
    hp_target_load = _validate_hpr_required(
        pt,
        is_heat_pumping=zone_config.DO_PROCESS_HP_TARGETING,
        zone_name=zone_name,
        zone_config=zone_config,
    )
    if hp_target_load > tol:
        res["Heat pump"] = _get_hpr_targets(
            Q_hpr_target=hp_target_load,
            T_vals=pt.col[PT.T.value],
            H_hot=pt.col[PT.H_NET_HOT.value],
            H_cold=pt.col[PT.H_NET_COLD.value],
            zone_config=zone_config,
            is_heat_pumping=True,
        )
        _calc_hpr_cascade(
            pt=pt,
            res=res["Heat pump"],
            is_T_vals_shifted=True,
            is_heat_pumping=True,
        )

    r_target_load = _validate_hpr_required(
        pt,
        is_refrigeration=zone_config.DO_PROCESS_RFRG_TARGETING,
        zone_name=zone_name,
        zone_config=zone_config,
    )
    if r_target_load > tol:
        res["Refrigeration"] = _get_hpr_targets(
            Q_hpr_target=r_target_load,
            T_vals=pt.col[PT.T.value],
            H_hot=pt.col[PT.H_NET_HOT.value],
            H_cold=pt.col[PT.H_NET_COLD.value],
            zone_config=zone_config,
            is_heat_pumping=False,
        )
        _calc_hpr_cascade(
            pt=pt,
            res=res["Refrigeration"],
            is_T_vals_shifted=True,
            is_heat_pumping=False,
        )
    return res


def get_indirect_heat_pump_and_refrigeration_target(
    pt: ProblemTable,
    hot_utilities: StreamCollection,
    cold_utilities: StreamCollection,
    zone_name: str,
    zone_config: Configuration,
) -> dict:
    res = {}
    # Create problem table based on inverted utility streams
    pt_ut_gen = get_process_heat_cascade(
        hot_streams=cold_utilities.get_hot_streams(invert_utility=True),
        cold_streams=hot_utilities.get_cold_streams(invert_utility=True),
        zone_config=zone_config,
        is_shifted=True,
    )
    get_additional_GCCs(pt_ut_gen)
    # Perform heat pump and/or refrigeration targeting on the correct cascades
    hp_target_load = _validate_hpr_required(
        pt,
        is_heat_pumping=zone_config.DO_UTILITY_HP_TARGETING,
        zone_name=zone_name,
        zone_config=zone_config,
    )
    if hp_target_load > tol:
        res["Heat pump"] = _get_hpr_targets(
            Q_hpr_target=hp_target_load,
            T_vals=pt_ut_gen.col[PT.T.value],
            H_hot=pt_ut_gen.col[PT.H_NET_HOT.value],
            H_cold=pt_ut_gen.col[PT.H_NET_COLD.value],
            zone_config=zone_config,
            is_heat_pumping=True,
        )
        _calc_hpr_cascade(
            pt=pt,
            res=res["Heat pump"],
            is_T_vals_shifted=True,
            is_heat_pumping=True,
        )

    r_target_load = _validate_hpr_required(
        pt,
        is_refrigeration=zone_config.DO_UTILITY_RFRG_TARGETING,
        zone_name=zone_name,
        zone_config=zone_config,
    )
    if r_target_load > tol:
        res["Refrigeration"] = _get_hpr_targets(
            Q_hpr_target=r_target_load,
            T_vals=pt_ut_gen.col[PT.T.value],
            H_hot=pt_ut_gen.col[PT.H_NET_HOT.value],
            H_cold=pt_ut_gen.col[PT.H_NET_COLD.value],
            zone_config=zone_config,
            is_heat_pumping=False,
        )
        _calc_hpr_cascade(
            pt=pt,
            res=res["Refrigeration"],
            is_T_vals_shifted=True,
            is_heat_pumping=False,
        )
    return res


#######################################################################################################
# Helper functions API
#######################################################################################################


def _validate_hpr_required(
    pt: ProblemTable,
    is_heat_pumping: bool = False,
    is_refrigeration: bool = False,
    zone_name: str = None,
    zone_config: Configuration = Configuration(),
    r: dict | float | int = None,
) -> float:
    if is_heat_pumping:
        Q_max = np.abs(pt.col[PT.H_NET_COLD.value]).max()
    elif is_refrigeration:
        Q_max = np.abs(pt.col[PT.H_NET_HOT.value]).max()
    else:
        return 0

    Q = Q_max
    hpr_load = zone_config.HPR_LOAD_VALUE if r is None else r
    if isinstance(hpr_load, float | int):
        Q = Q_max * hpr_load
    elif isinstance(hpr_load, dict):
        Q = min(get_value(hpr_load, val2=Q_max, zone_name=zone_name), Q_max)
    elif isinstance(hpr_load, str):
        hpr_load = literal_eval(hpr_load.strip())
        if isinstance(hpr_load, float | int | dict):
            Q = _validate_hpr_required(
                pt=pt,
                is_heat_pumping=is_heat_pumping,
                is_refrigeration=is_refrigeration,
                zone_name=zone_name,
                zone_config=zone_config,
                r=hpr_load,
            )
    return Q


def _get_hpr_targets(
    Q_hpr_target: float,
    T_vals: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    zone_config: Configuration,
    is_heat_pumping: bool,
) -> HPRTargetOutputs:
    # zone_config.HPR_TYPE = HPRcycle.MultiTempCarnot.value
    # zone_config.N_COND = 2
    # zone_config.N_EVAP = 2
    args = construct_HPRTargetInputs(
        Q_hpr_target=Q_hpr_target,
        T_vals=T_vals,
        H_hot=np.abs(H_hot) * -1,
        H_cold=np.abs(H_cold),
        is_heat_pumping=is_heat_pumping,
        zone_config=zone_config,
        debug=False, #True,
    )
    handler = _HP_PLACEMENT_HANDLERS.get(zone_config.HPR_TYPE)
    if handler is None:
        raise ValueError("No valid heat pump targeting type selected.")
    return HPRTargetOutputs.model_validate(handler(args))


def _calc_hpr_cascade(
    pt: ProblemTable,
    res: HPRTargetOutputs,
    is_T_vals_shifted: bool,
    is_heat_pumping: bool,
) -> ProblemTable:
    pt_hp = create_problem_table_with_t_int(
        streams=res.hpr_hot_streams + res.hpr_cold_streams,
        is_shifted=is_T_vals_shifted,
    )
    pt.insert_temperature_interval(pt_hp[PT.T.value].to_list())
    pt_hpr = get_utility_heat_cascade(
        T_int_vals=pt.col[PT.T.value],
        hot_utilities=res.hpr_hot_streams,
        cold_utilities=res.hpr_cold_streams,
        is_shifted=is_T_vals_shifted,
    )
    if is_heat_pumping:
        pt.update(
            {
                PT.H_NET_HP.value: pt_hpr[PT.H_NET_UT.value],
                PT.H_HOT_HP.value: pt_hpr[PT.H_HOT_UT.value],
                PT.H_COLD_HP.value: pt_hpr[PT.H_COLD_UT.value],
            }
        )
    else:
        pt.update(
            {
                PT.H_NET_RFRG.value: pt_hpr[PT.H_NET_UT.value],
                PT.H_HOT_RFRG.value: pt_hpr[PT.H_HOT_UT.value],
                PT.H_COLD_RFRG.value: pt_hpr[PT.H_COLD_UT.value],
            }
        )

    if len(res.amb_streams) > 0:
        pt_air = get_process_heat_cascade(
            hot_streams=res.amb_streams.get_hot_streams(),
            cold_streams=res.amb_streams.get_cold_streams(),
            is_shifted=True,
        )
        T_ls = pt[PT.T.value].to_list() + pt_air[PT.T.value].to_list()
        pt_air.insert_temperature_interval(T_ls)
        pt.insert_temperature_interval(T_ls)
        pt.col[PT.H_NET_W_AIR.value] = (
            pt.col[PT.H_NET_A.value] + pt_air.col[PT.H_NET.value]
        )

        if res.Q_amb_hot > tol:
            pt.col[PT.H_NET_HOT.value] -= pt_air.col[PT.H_NET.value]
        elif res.Q_amb_cold > tol:
            pt.col[PT.H_NET_COLD.value] += pt_air.col[PT.H_NET.value]
    else:
        pt.col[PT.H_NET_W_AIR.value] = pt.col[PT.H_NET_A.value]

    return pt


_HP_PLACEMENT_HANDLERS = {
    HPRcycle.Brayton.value: optimise_brayton_heat_pump_placement,
    HPRcycle.MultiTempCarnot.value: optimise_multi_temperature_carnot_heat_pump_placement,
    HPRcycle.MultiSimpleVapourComp.value: optimise_multi_simple_heat_pump_placement,
    HPRcycle.CascadeVapourComp.value: optimise_cascade_heat_pump_placement,
    HPRcycle.MultiSimpleCarnot.value: optimise_multi_simple_carnot_heat_pump_placement,
}
