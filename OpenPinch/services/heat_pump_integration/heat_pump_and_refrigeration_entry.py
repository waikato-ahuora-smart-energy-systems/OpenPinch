"""Public entrypoint for heat pump and refrigeration targeting."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np

from ...classes.problem_table import ProblemTable
from ...classes.zone import Zone
from ...lib.config import Configuration, tol
from ...lib.enums import GT, PT, TT, HPRcycle
from ...lib.schemas.hpr import (
    HeatPumpTargetOutputs,
    HPRPeriodCase,
    MultiPeriodHPRTargetInputs,
)
from ...lib.schemas.targets import (
    DirectHeatPumpTarget,
    DirectRefrigerationTarget,
    IndirectHeatPumpTarget,
    IndirectRefrigerationTarget,
)
from ..common.miscellaneous import get_period_index
from ..common.problem_table_analysis import create_problem_table_with_t_int
from ..direct_heat_integration.direct_integration_entry import (
    compute_direct_integration_targets,
)
from ..indirect_heat_integration.indirect_integration_entry import (
    compute_indirect_integration_targets,
    compute_total_subzone_utility_targets,
)
from .common.postprocessing import _get_hpr_residual_utility_summary
from .common.preprocessing import (
    construct_HPRTargetInputs,
)
from .common.shared import (
    get_process_heat_cascade,
    get_utility_heat_cascade,
    solve_hpr_multiperiod_placement,
    validate_vapour_hp_refrigerant_ls,
)
from .targeting_services.brayton import (
    optimise_brayton_heat_pump_placement,
)
from .targeting_services.cascade_carnot import (
    _compute_cascade_carnot_cycle_obj,
    _get_cascade_carnot_hp_opt_setup,
    optimise_cascade_carnot_heat_pump_placement,
)
from .targeting_services.cascade_vapour_compression import (
    _compute_cascade_hp_system_obj,
    _get_cascade_hp_opt_setup,
    optimise_cascade_heat_pump_placement,
)
from .targeting_services.parallel_carnot import (
    _compute_parallel_carnot_hp_opt_obj,
    _get_parallel_carnot_hp_opt_setup,
    optimise_parallel_carnot_heat_pump_placement,
)
from .targeting_services.parallel_vapour_compression import (
    _compute_parallel_hp_system_obj,
    _get_parallel_hp_opt_setup,
    optimise_parallel_heat_pump_placement,
)
from .targeting_services.vapour_compression_mvr import (
    _compute_vc_mvr_system_obj,
    _get_vc_mvr_opt_setup,
    _normalise_fluid_list,
    _num_vc_stages,
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
    if _use_multiperiod_hpr_optimization(zone):
        return _compute_multiperiod_heat_pump_or_refrigeration_target(
            zone=zone,
            is_heat_pumping=is_heat_pumping,
            is_direct=True,
            args=args,
        )

    idx, period_id = get_period_index(period_ids=zone.period_ids, args=args)
    is_refrigeration = not (is_heat_pumping)
    base_target = zone.targets[TT.DI.value]
    pt = deepcopy(base_target.pt)
    target_load = _validate_hpr_required(
        H_net_cold=pt[PT.H_NET_COLD],
        H_net_hot=pt[PT.H_NET_HOT],
        is_heat_pumping=is_heat_pumping,
        is_refrigeration=is_refrigeration,
        config=zone.config,
        period_id=period_id,
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
    if _use_multiperiod_hpr_optimization(zone):
        return _compute_multiperiod_heat_pump_or_refrigeration_target(
            zone=zone,
            is_heat_pumping=is_heat_pumping,
            is_direct=False,
            args=args,
        )

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
        config=zone.config,
        period_id=period_id,
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


def _use_multiperiod_hpr_optimization(zone: Zone) -> bool:
    return bool(getattr(zone.config.hpr, "multiperiod_optimization_enabled", False))


def _compute_multiperiod_heat_pump_or_refrigeration_target(
    *,
    zone: Zone,
    is_heat_pumping: bool,
    is_direct: bool,
    args: dict | None = None,
) -> (
    DirectHeatPumpTarget
    | DirectRefrigerationTarget
    | IndirectHeatPumpTarget
    | IndirectRefrigerationTarget
):
    idx, period_id = get_period_index(period_ids=zone.period_ids, args=args)
    selected_period_id = _period_id_for_index(zone, idx)
    cases = _build_multiperiod_hpr_cases(
        zone=zone,
        is_heat_pumping=is_heat_pumping,
        is_direct=is_direct,
        selected_period_id=selected_period_id,
        args=args,
    )
    res = _get_multiperiod_hpr_targets(
        period_cases=cases,
        selected_period_id=selected_period_id,
        selected_period_idx=idx,
    )
    selected_case = _period_case_by_id(cases, selected_period_id)
    pt = _calc_hpr_cascade(
        pt=deepcopy(selected_case.base_target.pt),
        res=res,
        is_T_vals_shifted=True,
        is_heat_pumping=is_heat_pumping,
        period_idx=idx,
    )
    general_results = {
        "zone_name": zone.name,
        "type": _hpr_target_type(is_direct=is_direct, is_heat_pumping=is_heat_pumping),
        "parent_zone": zone.parent_zone,
        "config": zone.config,
        "pt": pt,
        "graphs": _get_hpr_graphs(
            pt=pt,
            is_direct=is_direct,
            is_heat_pumping=is_heat_pumping,
        ),
        "period_id": period_id,
        "period_idx": idx,
    }
    hpr_results = _get_hpr_target_summary(res, zone)
    util_results = _get_hpr_residual_utility_summary(
        pt=pt,
        base_target=selected_case.base_target,
        period_idx=idx,
        is_direct=is_direct,
        is_heat_pumping=is_heat_pumping,
    )
    return _hpr_target_model_cls(
        is_direct=is_direct,
        is_heat_pumping=is_heat_pumping,
    ).model_validate(general_results | hpr_results | util_results)


def _build_multiperiod_hpr_cases(
    *,
    zone: Zone,
    is_heat_pumping: bool,
    is_direct: bool,
    selected_period_id: str,
    args: dict | None = None,
) -> list[HPRPeriodCase]:
    raw_cases = []
    period_items = _canonical_period_items(zone)
    weights = _canonical_period_weights(zone)
    for period_id, period_idx in period_items:
        period_args = _period_args(args, period_id=period_id, period_idx=period_idx)
        base_target = _compute_hpr_base_target_for_period(
            zone=zone,
            period_args=period_args,
            is_direct=is_direct,
        )
        if base_target is None:
            raise ValueError(
                "Multi-period HPR optimisation requires a valid base target for "
                f"period {period_id!r}."
            )
        optimizer_pt = _optimizer_problem_table_for_hpr(
            zone=zone,
            base_target=base_target,
            is_direct=is_direct,
            period_idx=period_idx,
        )
        raw_cases.append(
            {
                "period_id": period_id,
                "period_idx": period_idx,
                "weight": float(weights[period_idx]),
                "base_target": base_target,
                "optimizer_pt": optimizer_pt,
            }
        )

    _align_hpr_problem_tables([case["optimizer_pt"] for case in raw_cases])

    period_cases = []
    for case in raw_cases:
        pt = case["optimizer_pt"]
        period_id = case["period_id"]
        period_idx = case["period_idx"]
        target_load = _validate_hpr_required(
            H_net_cold=pt[PT.H_NET_COLD],
            H_net_hot=pt[PT.H_NET_HOT],
            is_heat_pumping=is_heat_pumping,
            is_refrigeration=not is_heat_pumping,
            config=zone.config,
            period_id=period_id,
            period_idx=period_idx,
        )
        if target_load < tol:
            raise ValueError(
                "Multi-period HPR optimisation requires a non-zero HPR load for "
                f"period {period_id!r}."
            )
        period_cases.append(
            HPRPeriodCase(
                period_id=period_id,
                period_idx=period_idx,
                weight=case["weight"],
                args=construct_HPRTargetInputs(
                    Q_hpr_target=target_load,
                    T_vals=pt[PT.T],
                    H_hot=np.abs(pt[PT.H_NET_HOT]) * -1,
                    H_cold=np.abs(pt[PT.H_NET_COLD]),
                    is_heat_pumping=is_heat_pumping,
                    config=zone.config,
                    period_idx=period_idx,
                    debug=False,
                ),
                base_target=case["base_target"],
                optimizer_pt=pt,
            )
        )

    _period_case_by_id(period_cases, selected_period_id)
    return period_cases


def _compute_hpr_base_target_for_period(
    *,
    zone: Zone,
    period_args: dict,
    is_direct: bool,
):
    if is_direct:
        return compute_direct_integration_targets(zone, period_args)
    _refresh_direct_targets_for_subtree(zone, period_args)
    zone.import_hot_and_cold_streams_from_sub_zones(
        get_net_streams=True,
        is_n_zone_depth=False,
        is_new_stream_collection=True,
    )
    zone.add_target(compute_total_subzone_utility_targets(zone, period_args))
    return compute_indirect_integration_targets(zone, period_args)


def _refresh_direct_targets_for_subtree(zone: Zone, period_args: dict) -> None:
    for subzone in zone.subzones.values():
        _refresh_direct_targets_for_subtree(subzone, period_args)
    zone.add_target(compute_direct_integration_targets(zone, period_args))


def _optimizer_problem_table_for_hpr(
    *,
    zone: Zone,
    base_target,
    is_direct: bool,
    period_idx: int,
) -> ProblemTable:
    if is_direct:
        return deepcopy(base_target.pt)
    return get_process_heat_cascade(
        hot_streams=zone.cold_utilities.get_hot_streams(invert_utility=True),
        cold_streams=zone.hot_utilities.get_cold_streams(invert_utility=True),
        is_shifted=True,
        is_full_analysis=True,
        period_idx=period_idx,
    )


def _align_hpr_problem_tables(tables: list[ProblemTable]) -> None:
    if not tables:
        raise ValueError("At least one HPR problem table is required.")
    for i, table in enumerate(tables):
        for other in tables[i + 1 :]:
            table.share_temperature_intervals(other)

    reference = tables[0][PT.T]
    for table in tables[1:]:
        if len(table[PT.T]) != len(reference) or not np.allclose(
            table[PT.T],
            reference,
            rtol=0.0,
            atol=tol,
        ):
            raise ValueError(
                "Multi-period HPR optimisation requires aligned PT temperature grids."
            )


def _get_multiperiod_hpr_targets(
    *,
    period_cases: list[HPRPeriodCase],
    selected_period_id: str,
    selected_period_idx: int,
) -> HeatPumpTargetOutputs:
    selected_case = _period_case_by_id(period_cases, selected_period_id)
    x0_ls, bnds, objective = _get_multiperiod_hpr_opt_setup(
        period_cases,
        selected_case=selected_case,
    )
    mp_args = MultiPeriodHPRTargetInputs(
        period_cases=period_cases,
        selected_period_id=selected_period_id,
        selected_period_idx=selected_period_idx,
        hpr_type=selected_case.args.hpr_type,
        max_multi_start=selected_case.args.max_multi_start,
        bb_minimiser=selected_case.args.bb_minimiser,
        debug=selected_case.args.debug,
    )
    result = solve_hpr_multiperiod_placement(
        f_obj=objective,
        x0_ls=x0_ls,
        bnds=bnds,
        args=mp_args,
    )
    return HeatPumpTargetOutputs.model_validate(result.to_output_fields())


def _get_multiperiod_hpr_opt_setup(
    period_cases: list[HPRPeriodCase],
    *,
    selected_case: HPRPeriodCase,
) -> tuple[np.ndarray | None, list, Callable]:
    hpr_type = selected_case.args.hpr_type
    if hpr_type == HPRcycle.CascadeCarnot.value:
        x0_ls, bnds = _get_cascade_carnot_hp_opt_setup(selected_case.args)
        return x0_ls, bnds, _compute_cascade_carnot_cycle_obj

    if hpr_type == HPRcycle.ParallelCarnot.value:
        n_stages = max(selected_case.args.n_cond, selected_case.args.n_evap)
        for case in period_cases:
            case.args.n_cond = case.args.n_evap = n_stages
        x0_ls, bnds = _get_parallel_carnot_hp_opt_setup(selected_case.args)
        return x0_ls, bnds, _compute_parallel_carnot_hp_opt_obj

    if hpr_type == HPRcycle.CascadeVapourComp.value:
        num_stages = int(selected_case.args.n_cond + selected_case.args.n_evap - 1)
        init_res = (
            optimise_cascade_carnot_heat_pump_placement(selected_case.args)
            if selected_case.args.initialise_simulated_cycle
            else None
        )
        for case in period_cases:
            case.args.refrigerant_ls = validate_vapour_hp_refrigerant_ls(
                num_stages,
                case.args,
            )
        x0_ls, bnds = _get_cascade_hp_opt_setup(init_res, selected_case.args)
        return x0_ls, bnds, _compute_cascade_hp_system_obj

    if hpr_type == HPRcycle.ParallelVapourComp.value:
        num_stages = max(selected_case.args.n_cond, selected_case.args.n_evap)
        for case in period_cases:
            case.args.n_cond = case.args.n_evap = int(num_stages)
        init_res = (
            optimise_parallel_carnot_heat_pump_placement(selected_case.args)
            if selected_case.args.initialise_simulated_cycle
            else None
        )
        for case in period_cases:
            case.args.refrigerant_ls = validate_vapour_hp_refrigerant_ls(
                int(num_stages),
                case.args,
            )
        x0_ls, bnds = _get_parallel_hp_opt_setup(init_res, selected_case.args)
        return x0_ls, bnds, _compute_parallel_hp_system_obj

    if hpr_type == HPRcycle.VapourCompMVR.value:
        if not selected_case.args.is_heat_pumping:
            raise ValueError(
                "Vapour compression with MVR cascade is heat-pump-only; "
                "refrigeration targets are not supported."
            )
        n_vc = _num_vc_stages(selected_case.args)
        init_res = (
            optimise_cascade_carnot_heat_pump_placement(selected_case.args)
            if selected_case.args.initialise_simulated_cycle
            else None
        )
        for case in period_cases:
            case.args.refrigerant_ls = validate_vapour_hp_refrigerant_ls(
                n_vc,
                case.args,
            )
            case.args.mvr_fluid_ls = _normalise_fluid_list(case.args.mvr_fluid_ls)
        x0_ls, bnds = _get_vc_mvr_opt_setup(init_res, selected_case.args)
        return x0_ls, bnds, _compute_vc_mvr_system_obj

    if hpr_type == HPRcycle.Brayton.value:
        raise NotImplementedError(
            "Brayton HPR targeting is not supported for multi-period optimisation."
        )
    raise ValueError("No valid heat pump targeting type selected.")


def _canonical_period_items(zone: Zone) -> list[tuple[str, int]]:
    period_ids = zone.period_ids or {"0": 0}
    return [(str(period_id), int(idx)) for period_id, idx in period_ids.items()]


def _canonical_period_weights(zone: Zone) -> np.ndarray:
    items = _canonical_period_items(zone)
    weights = getattr(zone, "weights", None)
    if weights is None or len(weights) != len(items):
        return np.ones(len(items), dtype=float)
    return np.asarray(weights, dtype=float).reshape(-1)


def _period_args(
    args: dict | None,
    *,
    period_id: str,
    period_idx: int,
) -> dict:
    period_args = dict(args or {})
    period_args["period_id"] = period_id
    period_args["period_idx"] = period_idx
    return period_args


def _period_id_for_index(zone: Zone, period_idx: int) -> str:
    for period_id, idx in (zone.period_ids or {"0": 0}).items():
        if int(idx) == int(period_idx):
            return str(period_id)
    raise ValueError(f"period_idx {period_idx!r} was not found on this zone.")


def _period_case_by_id(
    period_cases: list[HPRPeriodCase],
    period_id: str,
) -> HPRPeriodCase:
    for case in period_cases:
        if str(case.period_id) == str(period_id):
            return case
    raise ValueError(f"period_id {period_id!r} was not prepared for HPR targeting.")


def _hpr_target_type(*, is_direct: bool, is_heat_pumping: bool) -> str:
    if is_direct:
        return TT.DHP.value if is_heat_pumping else TT.DR.value
    return TT.IHP.value if is_heat_pumping else TT.IR.value


def _hpr_target_model_cls(*, is_direct: bool, is_heat_pumping: bool):
    if is_direct:
        return DirectHeatPumpTarget if is_heat_pumping else DirectRefrigerationTarget
    return IndirectHeatPumpTarget if is_heat_pumping else IndirectRefrigerationTarget


def _validate_hpr_required(
    H_net_cold: np.ndarray,
    H_net_hot: np.ndarray,
    is_heat_pumping: bool = False,
    is_refrigeration: bool = False,
    config: Configuration | None = None,
    period_id: str | None = None,
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

    hpr = config.hpr
    if hpr.load_mode == "fraction":
        return Q_max * float(hpr.load_fraction)
    if hpr.load_mode == "duty":
        return min(float(hpr.load_duty), Q_max)
    if hpr.load_mode == "period_values":
        return min(
            _resolve_hpr_period_load(
                hpr.load_period_values,
                period_id=period_id,
                period_idx=period_idx,
            ),
            Q_max,
        )
    raise ValueError(f"Unsupported HPR_LOAD_MODE {hpr.load_mode!r}.")


def _resolve_hpr_period_load(
    load_by_period: dict[str, float],
    *,
    period_id: str | None,
    period_idx: int | None,
) -> float:
    if period_id is not None and str(period_id) in load_by_period:
        return float(load_by_period[str(period_id)])
    if period_idx is not None and str(period_idx) in load_by_period:
        return float(load_by_period[str(period_idx)])
    available = ", ".join(sorted(str(key) for key in load_by_period)) or "<none>"
    raise ValueError(
        "HPR_LOAD_PERIOD_VALUES does not define a load for the selected period. "
        f"Expected period_id {period_id!r} or period index {period_idx!r}; "
        f"available keys: {available}."
    )


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
