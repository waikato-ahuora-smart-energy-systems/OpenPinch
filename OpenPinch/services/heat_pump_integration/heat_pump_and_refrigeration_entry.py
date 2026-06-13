"""Public entrypoint for heat pump and refrigeration targeting."""

from __future__ import annotations

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
from ...utils.value_resolution import evaluate_value_spec
from ..common.miscellaneous import get_state_index
from ..common.problem_table_analysis import create_problem_table_with_t_int
from .common.postprocessing import _get_hpr_residual_utility_summary
from .common.preprocessing import (
    construct_HPRTargetInputs,
)
from .common.shared import (
    get_process_heat_cascade,
    get_utility_heat_cascade,
)
from .targeting_services.brayton import (
    optimise_brayton_heat_pump_placement,
)
from .targeting_services.cascade_vapour_compression import (
    optimise_cascade_heat_pump_placement,
)
from .targeting_services.multi_simple_carnot import (
    optimise_multi_simple_carnot_heat_pump_placement,
)
from .targeting_services.multi_simple_vapour_compression import (
    optimise_multi_simple_heat_pump_placement,
)
from .targeting_services.multi_temperature_carnot import (
    optimise_multi_temperature_carnot_heat_pump_placement,
)
from .targeting_services.vapour_compression_mvr import (
    optimise_vapour_compression_mvr_heat_pump_placement,
)

__all__ = [
    "compute_direct_heat_pump_or_refrigeration_target",
    "compute_indirect_heat_pump_or_refrigeration_target",
]


################################################################################
# Public API
################################################################################


def compute_direct_heat_pump_or_refrigeration_target(
    zone: Zone,
    is_heat_pumping: bool,
    args: dict | None = None,
) -> DirectHeatPumpTarget | DirectRefrigerationTarget | None:
    """Solve an explicit direct Heat Pump or refrigeration target for one zone."""
    idx, state_id = get_state_index(state_ids=zone.state_ids, args=args)
    is_refrigeration = not (is_heat_pumping)
    base_target = zone.targets[TT.DI.value]
    pt = deepcopy(base_target.pt)
    target_load = _validate_hpr_required(
        H_net_cold=pt[PT.H_NET_COLD],
        H_net_hot=pt[PT.H_NET_HOT],
        is_heat_pumping=is_heat_pumping,
        is_refrigeration=is_refrigeration,
        zone_name=zone.name,
        zone_config=zone.config,
        idx=idx,
    )
    if target_load < tol:
        return None

    res = _get_hpr_targets(
        Q_hpr_target=target_load,
        T_vals=pt[PT.T],
        H_hot=pt[PT.H_NET_HOT],
        H_cold=pt[PT.H_NET_COLD],
        zone_config=zone.config,
        is_heat_pumping=is_heat_pumping,
        idx=idx,
    )
    pt = _calc_hpr_cascade(
        pt=pt,
        res=res,
        is_T_vals_shifted=True,
        is_heat_pumping=is_heat_pumping,
        idx=idx,
    )
    general_results = {
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
        "state_id": state_id,
        "state_idx": idx,
    }
    hpr_results = _get_hpr_target_summary(res, zone)
    util_results = _get_hpr_residual_utility_summary(
        pt=pt,
        base_target=base_target,
        idx=idx,
        is_direct=True,
        is_heat_pumping=is_heat_pumping,
    )
    model_cls = DirectHeatPumpTarget if is_heat_pumping else DirectRefrigerationTarget
    return model_cls.model_validate(general_results | hpr_results | util_results)


def compute_indirect_heat_pump_or_refrigeration_target(
    zone: Zone,
    is_heat_pumping: bool,
    args: dict | None = None,
) -> IndirectHeatPumpTarget | IndirectRefrigerationTarget | None:
    """Solve an indirect / utility system Heat Pump or refrigeration target."""
    idx, state_id = get_state_index(state_ids=zone.state_ids, args=args)
    is_refrigeration = not (is_heat_pumping)
    base_target = zone.targets[TT.TS.value]
    pt = deepcopy(base_target.pt)
    # Create problem table based on inverted utility streams
    pt_ut_gen = get_process_heat_cascade(
        hot_streams=zone.cold_utilities.get_hot_streams(invert_utility=True),
        cold_streams=zone.hot_utilities.get_cold_streams(invert_utility=True),
        is_shifted=True,
        is_full_analysis=True,
        idx=idx,
    )
    # Perform heat pump and/or refrigeration targeting on the correct cascades
    target_load = _validate_hpr_required(
        H_net_cold=pt_ut_gen[PT.H_NET_COLD],
        H_net_hot=pt_ut_gen[PT.H_NET_HOT],
        is_heat_pumping=is_heat_pumping,
        is_refrigeration=is_refrigeration,
        zone_name=zone.name,
        zone_config=zone.config,
        idx=idx,
    )
    if target_load < tol:
        return None

    res = _get_hpr_targets(
        Q_hpr_target=target_load,
        T_vals=pt_ut_gen[PT.T],
        H_hot=pt_ut_gen[PT.H_NET_HOT],
        H_cold=pt_ut_gen[PT.H_NET_COLD],
        zone_config=zone.config,
        is_heat_pumping=is_heat_pumping,
        idx=idx,
    )
    pt = _calc_hpr_cascade(
        pt=pt,
        res=res,
        is_T_vals_shifted=True,
        is_heat_pumping=is_heat_pumping,
        idx=idx,
    )
    general_results = {
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
        "state_id": state_id,
        "state_idx": idx,
    }
    hpr_results = _get_hpr_target_summary(res, zone)
    util_results = _get_hpr_residual_utility_summary(
        pt=pt,
        base_target=base_target,
        idx=idx,
        is_direct=False,
        is_heat_pumping=is_heat_pumping,
    )
    model_cls = (
        IndirectHeatPumpTarget if is_heat_pumping else IndirectRefrigerationTarget
    )
    return model_cls.model_validate(general_results | hpr_results | util_results)


################################################################################
# Helper functions API
################################################################################


def _validate_hpr_required(
    H_net_cold: np.ndarray,
    H_net_hot: np.ndarray,
    is_heat_pumping: bool = False,
    is_refrigeration: bool = False,
    zone_name: str = None,
    zone_config: Configuration | None = None,
    r: dict | float | int = None,
    idx: int | None = None,
) -> float:
    if zone_config is None:
        raise ValueError("zone_config must be provided for HPR targeting.")

    if is_heat_pumping:
        load_values = H_net_cold
    elif is_refrigeration:
        load_values = H_net_hot
    else:
        return 0
    Q_max = float(np.nanmax(np.abs(load_values), initial=0.0))

    Q = Q_max
    hpr_load = zone_config.HPR_LOAD_VALUE if r is None else r
    if isinstance(hpr_load, float | int):
        Q = Q_max * hpr_load
    elif isinstance(hpr_load, dict):
        Q = min(
            evaluate_value_spec(
                hpr_load,
                default_value=Q_max,
                zone_name=zone_name,
                idx=idx,
            ),
            Q_max,
        )
    elif isinstance(hpr_load, str):
        hpr_load = literal_eval(hpr_load.strip())
        if isinstance(hpr_load, float | int | dict):
            Q = _validate_hpr_required(
                H_net_cold=H_net_cold,
                H_net_hot=H_net_hot,
                is_heat_pumping=is_heat_pumping,
                is_refrigeration=is_refrigeration,
                zone_name=zone_name,
                zone_config=zone_config,
                r=hpr_load,
                idx=idx,
            )
    return Q


def _get_hpr_targets(
    Q_hpr_target: float,
    T_vals: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    zone_config: Configuration,
    is_heat_pumping: bool,
    idx: int = 0,
) -> HeatPumpTargetOutputs:
    args = construct_HPRTargetInputs(
        Q_hpr_target=Q_hpr_target,
        T_vals=T_vals,
        H_hot=np.abs(H_hot) * -1,
        H_cold=np.abs(H_cold),
        is_heat_pumping=is_heat_pumping,
        zone_config=zone_config,
        idx=idx,
        debug=False,
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
            GT.NLP_HP.value: pt.slice(
                [
                    PT.T,
                    PT.H_NET_HOT,
                    PT.H_NET_COLD,
                    PT.H_HOT_HP,
                    PT.H_COLD_HP,
                ]
            ),
            GT.GCC_HP.value: pt.slice(
                [
                    PT.T,
                    PT.H_NET_W_AIR,
                    PT.H_NET_HP,
                ]
            ),
        }

    return {
        GT.SUGCC.value: pt.slice(
            [
                PT.T,
                PT.H_NET_UT,
                PT.H_NET_HP,
            ]
        )
    }


def _calc_hpr_cascade(
    pt: ProblemTable,
    res: HeatPumpTargetOutputs,
    is_T_vals_shifted: bool = True,
    is_heat_pumping: bool = True,
    idx: int | None = None,
) -> ProblemTable:
    # Add new temperature intervals to the process heat cascade
    pt_grid_kwargs = {
        "streams": (
            res.hpr_hot_streams
            + res.hpr_cold_streams
            + res.amb_streams.get_hot_streams()
            + res.amb_streams.get_cold_streams()
        ),
        "is_shifted": is_T_vals_shifted,
    }
    if idx is not None:
        pt_grid_kwargs["idx"] = idx
    try:
        pt_hpr_grid = create_problem_table_with_t_int(**pt_grid_kwargs)
    except TypeError:
        pt_grid_kwargs.pop("idx", None)
        pt_hpr_grid = create_problem_table_with_t_int(**pt_grid_kwargs)
    pt.share_temperature_intervals(pt_hpr_grid)

    # Ambient air addition to the process stream set
    pt[PT.H_NET_W_AIR] = pt[PT.H_NET_A]
    if len(res.amb_streams) > 0:
        pt_air_kwargs = {
            "hot_streams": res.amb_streams.get_hot_streams(),
            "cold_streams": res.amb_streams.get_cold_streams(),
            "is_shifted": is_T_vals_shifted,
            "is_full_analysis": True,
        }
        if idx is not None:
            pt_air_kwargs["idx"] = idx
        try:
            pt_air = get_process_heat_cascade(**pt_air_kwargs)
        except TypeError:
            pt_air_kwargs.pop("idx", None)
            pt_air = get_process_heat_cascade(**pt_air_kwargs)
        pt.share_temperature_intervals(pt_air)
        pt[PT.H_NET_W_AIR] += pt_air[PT.H_NET]
        pt[PT.H_NET_HOT] += pt_air[PT.H_NET_HOT]
        pt[PT.H_NET_COLD] += pt_air[PT.H_NET_COLD]

    # Heat pump or refrigeration cascade
    hpr_cascade_kwargs = {
        "T_int_vals": pt[PT.T],
        "hot_utilities": res.hpr_hot_streams,
        "cold_utilities": res.hpr_cold_streams,
        "is_shifted": is_T_vals_shifted,
    }
    if idx is not None:
        hpr_cascade_kwargs["idx"] = idx
    try:
        hpr_profile = get_utility_heat_cascade(**hpr_cascade_kwargs)
    except TypeError:
        hpr_cascade_kwargs.pop("idx", None)
        hpr_profile = get_utility_heat_cascade(**hpr_cascade_kwargs)
    hpr_updates = hpr_profile["updates"]
    if is_heat_pumping:
        pt.update(
            T_col=hpr_profile["T_col"],
            updates={
                PT.H_NET_HP: hpr_updates[PT.H_NET_UT],
                PT.H_HOT_HP: hpr_updates[PT.H_HOT_UT],
                PT.H_COLD_HP: hpr_updates[PT.H_COLD_UT],
            },
        )
    else:
        pt.update(
            T_col=hpr_profile["T_col"],
            updates={
                PT.H_NET_RFRG: hpr_updates[PT.H_NET_UT],
                PT.H_HOT_RFRG: hpr_updates[PT.H_HOT_UT],
                PT.H_COLD_RFRG: hpr_updates[PT.H_COLD_UT],
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
    HPRcycle.VapourCompMVR.value: (optimise_vapour_compression_mvr_heat_pump_placement),
    HPRcycle.MultiSimpleCarnot.value: optimise_multi_simple_carnot_heat_pump_placement,
}
