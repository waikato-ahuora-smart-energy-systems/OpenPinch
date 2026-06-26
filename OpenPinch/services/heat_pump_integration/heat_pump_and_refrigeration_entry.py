"""Public entrypoint for heat pump and refrigeration targeting."""

from __future__ import annotations

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
from ..common.miscellaneous import get_period_index
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
from .targeting_services.cascade_carnot import (
    optimise_cascade_carnot_heat_pump_placement,
)
from .targeting_services.cascade_vapour_compression import (
    optimise_cascade_heat_pump_placement,
)
from .targeting_services.parallel_carnot import (
    optimise_parallel_carnot_heat_pump_placement,
)
from .targeting_services.parallel_vapour_compression import (
    optimise_parallel_heat_pump_placement,
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
    idx, period_id = get_period_index(period_ids=zone.period_ids, args=args)
    is_refrigeration = not (is_heat_pumping)
    base_target = zone.targets[TT.DI.value]
    pt = deepcopy(base_target.pt)
    target_load = _validate_hpr_required(
        H_net_cold=pt[PT.H_NET_COLD],
        H_net_hot=pt[PT.H_NET_HOT],
        is_heat_pumping=is_heat_pumping,
        is_refrigeration=is_refrigeration,
        zone_name=zone.name,
        config=zone.config,
        period_idx=idx,
    )
    if target_load < tol:
        return None

    res = _get_hpr_targets(
        Q_hpr_target=target_load,
        T_vals=pt[PT.T],
        H_hot=pt[PT.H_NET_HOT],
        H_cold=pt[PT.H_NET_COLD],
        config=zone.config,
        is_heat_pumping=is_heat_pumping,
        period_idx=idx,
    )
    pt = _calc_hpr_cascade(
        pt=pt,
        res=res,
        is_T_vals_shifted=True,
        is_heat_pumping=is_heat_pumping,
        period_idx=idx,
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
        "period_id": period_id,
        "period_idx": idx,
    }
    hpr_results = _get_hpr_target_summary(res, zone)
    util_results = _get_hpr_residual_utility_summary(
        pt=pt,
        base_target=base_target,
        period_idx=idx,
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
    idx, period_id = get_period_index(period_ids=zone.period_ids, args=args)
    is_refrigeration = not (is_heat_pumping)
    base_target = zone.targets[TT.TS.value]
    pt = deepcopy(base_target.pt)
    # Create problem table based on inverted utility streams
    pt_ut_gen = get_process_heat_cascade(
        hot_streams=zone.cold_utilities.get_hot_streams(invert_utility=True),
        cold_streams=zone.hot_utilities.get_cold_streams(invert_utility=True),
        is_shifted=True,
        is_full_analysis=True,
        period_idx=idx,
    )
    # Perform heat pump and/or refrigeration targeting on the correct cascades
    target_load = _validate_hpr_required(
        H_net_cold=pt_ut_gen[PT.H_NET_COLD],
        H_net_hot=pt_ut_gen[PT.H_NET_HOT],
        is_heat_pumping=is_heat_pumping,
        is_refrigeration=is_refrigeration,
        zone_name=zone.name,
        config=zone.config,
        period_idx=idx,
    )
    if target_load < tol:
        return None

    res = _get_hpr_targets(
        Q_hpr_target=target_load,
        T_vals=pt_ut_gen[PT.T],
        H_hot=pt_ut_gen[PT.H_NET_HOT],
        H_cold=pt_ut_gen[PT.H_NET_COLD],
        config=zone.config,
        is_heat_pumping=is_heat_pumping,
        period_idx=idx,
    )
    pt = _calc_hpr_cascade(
        pt=pt,
        res=res,
        is_T_vals_shifted=True,
        is_heat_pumping=is_heat_pumping,
        period_idx=idx,
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
        "period_id": period_id,
        "period_idx": idx,
    }
    hpr_results = _get_hpr_target_summary(res, zone)
    util_results = _get_hpr_residual_utility_summary(
        pt=pt,
        base_target=base_target,
        period_idx=idx,
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
    config: Configuration | None = None,
    r: dict | float | int = None,
    period_idx: int | None = None,
) -> float:
    if config is None:
        raise ValueError("config must be provided for HPR targeting.")

    if is_heat_pumping:
        load_values = H_net_cold
    elif is_refrigeration:
        load_values = H_net_hot
    else:
        return 0
    Q_max = float(np.nanmax(np.abs(load_values), initial=0.0))

    if r is not None:
        return _resolve_legacy_hpr_load_override(
            r,
            Q_max=Q_max,
            zone_name=zone_name,
            period_idx=period_idx,
        )

    hpr = config.hpr
    if hpr.load_mode == "fraction":
        return Q_max * float(hpr.load_fraction)
    if hpr.load_mode == "duty":
        return min(float(hpr.load_duty), Q_max)
    if hpr.load_mode == "period_values":
        return min(
            evaluate_value_spec(
                hpr.load_period_values,
                default_value=Q_max,
                zone_name=zone_name,
                period_idx=period_idx,
            ),
            Q_max,
        )
    raise ValueError(f"Unsupported HPR_LOAD_MODE {hpr.load_mode!r}.")


def _resolve_legacy_hpr_load_override(
    hpr_load: dict | float | int,
    *,
    Q_max: float,
    zone_name: str | None,
    period_idx: int | None,
) -> float:
    """Resolve direct test/helper load overrides without accepting flat config keys."""
    if isinstance(hpr_load, float | int):
        return Q_max * float(hpr_load)
    if isinstance(hpr_load, dict):
        return min(
            evaluate_value_spec(
                hpr_load,
                default_value=Q_max,
                zone_name=zone_name,
                period_idx=period_idx,
            ),
            Q_max,
        )
    raise ValueError("HPR load override must be a number or period-value mapping.")


def _get_hpr_targets(
    Q_hpr_target: float,
    T_vals: np.ndarray,
    H_hot: np.ndarray,
    H_cold: np.ndarray,
    config: Configuration,
    is_heat_pumping: bool,
    period_idx: int = 0,
) -> HeatPumpTargetOutputs:
    args = construct_HPRTargetInputs(
        Q_hpr_target=Q_hpr_target,
        T_vals=T_vals,
        H_hot=np.abs(H_hot) * -1,
        H_cold=np.abs(H_cold),
        is_heat_pumping=is_heat_pumping,
        config=config,
        period_idx=period_idx,
        debug=False,
    )
    handler = _HP_PLACEMENT_HANDLERS.get(args.hpr_type)
    if handler is None:
        raise ValueError("No valid heat pump targeting type selected.")
    result = handler(args)
    return HeatPumpTargetOutputs.model_validate(result.to_output_fields())


def _get_hpr_target_summary(
    res: HeatPumpTargetOutputs,
    zone: Zone,
) -> dict:
    return {
        "hpr_cycle": str(zone.config.hpr.type),
        "hpr_utility_total": res.utility_tot,
        "hpr_work": res.w_net,
        "hpr_external_utility": res.Q_ext,
        "hpr_ambient_hot": res.Q_amb_hot,
        "hpr_ambient_cold": res.Q_amb_cold,
        "hpr_cop": res.cop_h,
        "hpr_eta_he": res.eta_he,
        "hpr_operating_cost": res.hpr_operating_cost,
        "hpr_capital_cost": res.hpr_capital_cost,
        "hpr_annualized_capital_cost": res.hpr_annualized_capital_cost,
        "hpr_total_annualized_cost": res.hpr_total_annualized_cost,
        "hpr_compressor_capital_cost": res.hpr_compressor_capital_cost,
        "hpr_heat_exchanger_capital_cost": res.hpr_heat_exchanger_capital_cost,
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
    period_idx: int | None = None,
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
    if period_idx is not None:
        pt_grid_kwargs["period_idx"] = period_idx
    try:
        pt_hpr_grid = create_problem_table_with_t_int(**pt_grid_kwargs)
    except TypeError:
        pt_grid_kwargs.pop("period_idx", None)
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
        if period_idx is not None:
            pt_air_kwargs["period_idx"] = period_idx
        try:
            pt_air = get_process_heat_cascade(**pt_air_kwargs)
        except TypeError:
            pt_air_kwargs.pop("period_idx", None)
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
    if period_idx is not None:
        hpr_cascade_kwargs["period_idx"] = period_idx
    try:
        hpr_profile = get_utility_heat_cascade(**hpr_cascade_kwargs)
    except TypeError:
        hpr_cascade_kwargs.pop("period_idx", None)
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
    HPRcycle.CascadeCarnot.value: (optimise_cascade_carnot_heat_pump_placement),
    HPRcycle.ParallelVapourComp.value: optimise_parallel_heat_pump_placement,
    HPRcycle.CascadeVapourComp.value: optimise_cascade_heat_pump_placement,
    HPRcycle.VapourCompMVR.value: (optimise_vapour_compression_mvr_heat_pump_placement),
    HPRcycle.ParallelCarnot.value: optimise_parallel_carnot_heat_pump_placement,
}
