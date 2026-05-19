"""Public entrypoint for heat pump and refrigeration targeting."""

from ast import literal_eval
from copy import deepcopy

import numpy as np

from ...classes.problem_table import ProblemTable
from ...classes.zone import Zone
from ...lib.config import Configuration, tol
from ...lib.enums import GT, PT, TT, HPRcycle
from ...lib.schemas.hpr import HeatPumpTargetOutputs
from ...lib.schemas.targets import (
    DirectHeatPumpTarget,
    DirectRefrigerationTarget,
    IndirectHeatPumpTarget,
    IndirectRefrigerationTarget,
)
from ...utils.miscellaneous import get_value
from ..common.problem_table_analysis import create_problem_table_with_t_int
from .common.preprocessing import (
    construct_HPRTargetInputs,
)
from .common.shared import (
    get_process_heat_cascade,
    get_utility_heat_cascade,
)
from .cycles.brayton import (
    optimise_brayton_heat_pump_placement,
)
from .cycles.cascade_vapour_compression import (
    optimise_cascade_heat_pump_placement,
)
from .cycles.multi_simple_carnot import (
    optimise_multi_simple_carnot_heat_pump_placement,
)
from .cycles.multi_simple_vapour_compression import (
    optimise_multi_simple_heat_pump_placement,
)
from .cycles.multi_temperature_carnot import (
    optimise_multi_temperature_carnot_heat_pump_placement,
)

__all__ = [
    "compute_direct_heat_pump_or_refrigeration_target",
    "compute_indirect_heat_pump_or_refrigeration_target",
]


#######################################################################################################
# Public API
#######################################################################################################


def compute_direct_heat_pump_or_refrigeration_target(
    zone: Zone,
    is_heat_pumping: bool,
) -> DirectHeatPumpTarget | DirectRefrigerationTarget | None:
    """Solve an explicit direct heat-pump or refrigeration target for one zone."""
    is_refrigeration = not (is_heat_pumping)
    pt = deepcopy(zone.targets[TT.DI.value].pt)
    target_load = _validate_hpr_required(
        pt,
        is_heat_pumping=is_heat_pumping,
        is_refrigeration=is_refrigeration,
        zone_name=zone.name,
        zone_config=zone.config,
    )
    if target_load < tol:
        return None

    res = _get_hpr_targets(
        Q_hpr_target=target_load,
        T_vals=pt.col[PT.T.value],
        H_hot=pt.col[PT.H_NET_HOT.value],
        H_cold=pt.col[PT.H_NET_COLD.value],
        zone_config=zone.config,
        is_heat_pumping=is_heat_pumping,
    )
    pt = _calc_hpr_cascade(
        pt=pt,
        res=res,
        is_T_vals_shifted=True,
        is_heat_pumping=is_heat_pumping,
    )
    payload = {
        "zone_name": zone.name,
        "type": TT.DHP.value if is_heat_pumping else TT.DR.value,
        "parent_zone": zone.parent_zone,
        "config": zone.config,
        "pt": pt,
        "graphs": _get_hpr_graphs(
            pt=pt,
            is_direct=True,
            is_heat_pumping=is_heat_pumping,
        ),
    } | _get_hpr_target_summary(res, zone)
    model_cls = DirectHeatPumpTarget if is_heat_pumping else DirectRefrigerationTarget
    return model_cls.model_validate(payload)


def compute_indirect_heat_pump_or_refrigeration_target(
    zone: Zone,
    is_heat_pumping: bool,
) -> IndirectHeatPumpTarget | IndirectRefrigerationTarget | None:
    """Solve an indirect / utility-system heat-pump or refrigeration target."""
    is_refrigeration = not (is_heat_pumping)
    pt = deepcopy(zone.targets[TT.TS.value].pt)
    # Create problem table based on inverted utility streams
    pt_ut_gen = get_process_heat_cascade(
        hot_streams=zone.cold_utilities.get_hot_streams(invert_utility=True),
        cold_streams=zone.hot_utilities.get_cold_streams(invert_utility=True),
        is_shifted=True,
        is_full_analysis=True,
    )
    # Perform heat pump and/or refrigeration targeting on the correct cascades
    target_load = _validate_hpr_required(
        pt,
        is_heat_pumping=is_heat_pumping,
        is_refrigeration=is_refrigeration,
        zone_name=zone.name,
        zone_config=zone.config,
    )
    if target_load < tol:
        return None

    res = _get_hpr_targets(
        Q_hpr_target=target_load,
        T_vals=pt_ut_gen.col[PT.T.value],
        H_hot=pt_ut_gen.col[PT.H_NET_HOT.value],
        H_cold=pt_ut_gen.col[PT.H_NET_COLD.value],
        zone_config=zone.config,
        is_heat_pumping=is_heat_pumping,
    )
    pt = _calc_hpr_cascade(
        pt=pt,
        res=res,
        is_T_vals_shifted=True,
        is_heat_pumping=is_heat_pumping,
    )
    payload = {
        "zone_name": zone.name,
        "type": TT.IHP.value if is_heat_pumping else TT.IR.value,
        "parent_zone": zone.parent_zone,
        "config": zone.config,
        "pt": pt,
        "graphs": _get_hpr_graphs(
            pt=pt,
            is_direct=False,
            is_heat_pumping=is_heat_pumping,
        ),
    } | _get_hpr_target_summary(res, zone)
    model_cls = (
        IndirectHeatPumpTarget if is_heat_pumping else IndirectRefrigerationTarget
    )
    return model_cls.model_validate(payload)


#######################################################################################################
# Helper functions API
#######################################################################################################


def _validate_hpr_required(
    pt: ProblemTable,
    is_heat_pumping: bool = False,
    is_refrigeration: bool = False,
    zone_name: str = None,
    zone_config: Configuration | None = None,
    r: dict | float | int = None,
) -> float:
    if zone_config is None:
        raise ValueError("zone_config must be provided for HPR targeting.")

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
) -> HeatPumpTargetOutputs:
    args = construct_HPRTargetInputs(
        Q_hpr_target=Q_hpr_target,
        T_vals=T_vals,
        H_hot=np.abs(H_hot) * -1,
        H_cold=np.abs(H_cold),
        is_heat_pumping=is_heat_pumping,
        zone_config=zone_config,
        debug=False,  # True,
    )
    handler = _HP_PLACEMENT_HANDLERS.get(zone_config.HPR_TYPE)
    if handler is None:
        raise ValueError("No valid heat pump targeting type selected.")
    result = handler(args)
    return HeatPumpTargetOutputs.model_validate(result.to_output_payload())


def _get_hpr_target_summary(
    res: HeatPumpTargetOutputs,
    zone: Zone,
) -> dict:
    return {
        "hpr_cycle": str(zone.config.HPR_TYPE),
        "hpr_utility_total": res.utility_tot,
        "hpr_work": res.w_net,
        "hpr_external_utility": res.Q_ext,
        "hpr_ambient_hot": res.Q_amb_hot,
        "hpr_ambient_cold": res.Q_amb_cold,
        "hpr_cop": res.cop_h,
        "hpr_eta_he": res.eta_he,
        "hpr_success": res.success,
        "hpr_hot_streams": res.hpr_hot_streams,
        "hpr_cold_streams": res.hpr_cold_streams,
        "hpr_details": res,
    }


def _get_hpr_graphs(
    pt: ProblemTable,
    *,
    is_direct: bool,
    is_heat_pumping: bool,
) -> dict:
    if not is_heat_pumping:
        return {}

    if is_direct:
        return {
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

    return {GT.SUGCC.value: pt.slice([PT.T, PT.H_NET_UT, PT.H_NET_HP])}


def _calc_hpr_cascade(
    pt: ProblemTable,
    res: HeatPumpTargetOutputs,
    is_T_vals_shifted: bool = True,
    is_heat_pumping: bool = True,
) -> ProblemTable:
    # Add new temperature intervals to the process heat cascade
    pt_hpr_grid = create_problem_table_with_t_int(
        streams=(
            res.hpr_hot_streams
            + res.hpr_cold_streams
            + res.amb_streams.get_hot_streams()
            + res.amb_streams.get_cold_streams()
        ),
        is_shifted=is_T_vals_shifted,
    )
    pt.share_temperature_intervals(pt_hpr_grid)

    # Ambient air addition to the process stream set
    pt.col[PT.H_NET_W_AIR.value] = pt.col[PT.H_NET_A.value]
    if len(res.amb_streams) > 0:
        pt_air = get_process_heat_cascade(
            hot_streams=res.amb_streams.get_hot_streams(),
            cold_streams=res.amb_streams.get_cold_streams(),
            is_shifted=is_T_vals_shifted,
            is_full_analysis=True,
        )
        pt.share_temperature_intervals(pt_air)
        pt.col[PT.H_NET_W_AIR.value] += pt_air.col[PT.H_NET.value]
        pt.col[PT.H_NET_HOT.value] += pt_air.col[PT.H_NET_HOT.value]
        pt.col[PT.H_NET_COLD.value] += pt_air.col[PT.H_NET_COLD.value]

    # Heat pump or refrigeration cascade
    hpr_profile = get_utility_heat_cascade(
        T_int_vals=pt.col[PT.T.value],
        hot_utilities=res.hpr_hot_streams,
        cold_utilities=res.hpr_cold_streams,
        is_shifted=is_T_vals_shifted,
    )
    hpr_updates = hpr_profile["updates"]
    if is_heat_pumping:
        pt.update(
            T_col=hpr_profile["T_col"],
            updates={
                PT.H_NET_HP.value: hpr_updates[PT.H_NET_UT.value],
                PT.H_HOT_HP.value: hpr_updates[PT.H_HOT_UT.value],
                PT.H_COLD_HP.value: hpr_updates[PT.H_COLD_UT.value],
            },
        )
    else:
        pt.update(
            T_col=hpr_profile["T_col"],
            updates={
                PT.H_NET_RFRG.value: hpr_updates[PT.H_NET_UT.value],
                PT.H_HOT_RFRG.value: hpr_updates[PT.H_HOT_UT.value],
                PT.H_COLD_RFRG.value: hpr_updates[PT.H_COLD_UT.value],
            },
        )

    return pt


_HP_PLACEMENT_HANDLERS = {
    HPRcycle.Brayton.value: optimise_brayton_heat_pump_placement,
    HPRcycle.MultiTempCarnot.value: (
        optimise_multi_temperature_carnot_heat_pump_placement
    ),
    HPRcycle.MultiSimpleVapourComp.value: optimise_multi_simple_heat_pump_placement,
    HPRcycle.CascadeVapourComp.value: optimise_cascade_heat_pump_placement,
    HPRcycle.MultiSimpleCarnot.value: optimise_multi_simple_carnot_heat_pump_placement,
}
