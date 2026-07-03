"""Multi-period shared-design HPR targeting orchestration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ....classes.problem_table import ProblemTable
from ....classes.zone import Zone
from ....lib.config import tol
from ....lib.enums import PT, HPRcycle
from ....lib.schemas.hpr import (
    HeatPumpTargetOutputs,
    HPRPeriodCase,
    MultiPeriodHPRTargetInputs,
)
from ...direct_heat_integration.direct_integration_entry import (
    compute_direct_integration_targets,
)
from ...indirect_heat_integration.indirect_integration_entry import (
    compute_indirect_integration_targets,
    compute_total_subzone_utility_targets,
)
from ..common.load_selection import resolve_hpr_target_load
from ..common.preprocessing import construct_HPRTargetInputs
from ..common.shared import (
    get_process_heat_cascade,
    solve_hpr_multiperiod_placement,
    validate_vapour_hp_refrigerant_ls,
)
from .cascade_carnot import (
    _compute_cascade_carnot_cycle_obj,
    _get_cascade_carnot_hp_opt_setup,
    optimise_cascade_carnot_heat_pump_placement,
)
from .cascade_vapour_compression import (
    _compute_cascade_hp_system_obj,
    _get_cascade_hp_opt_setup,
)
from .parallel_carnot import (
    _compute_parallel_carnot_hp_opt_obj,
    _get_parallel_carnot_hp_opt_setup,
    optimise_parallel_carnot_heat_pump_placement,
)
from .parallel_vapour_compression import (
    _compute_parallel_hp_system_obj,
    _get_parallel_hp_opt_setup,
)
from .vapour_compression_mvr import (
    _compute_vc_mvr_system_obj,
    _get_vc_mvr_opt_setup,
    _normalise_fluid_list,
    _num_vc_stages,
)

__all__ = [
    "PreparedHPRPeriodCase",
    "build_multiperiod_hpr_cases",
    "get_multiperiod_hpr_targets",
    "period_case_by_id",
    "period_id_for_index",
]


@dataclass
class PreparedHPRPeriodCase:
    """Runtime data for one period in a shared HPR design solve."""

    period_id: str
    period_idx: int
    weight: float
    solver_case: HPRPeriodCase
    base_target: Any
    optimizer_pt: ProblemTable


def build_multiperiod_hpr_cases(
    *,
    zone: Zone,
    is_heat_pumping: bool,
    is_direct: bool,
    args: dict | None = None,
) -> list[PreparedHPRPeriodCase]:
    """Prepare aligned single-period HPR inputs for one shared design vector."""
    raw_cases = []
    weights = _canonical_period_weights(zone)
    for period_id, period_idx in _canonical_period_items(zone):
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
                "weight": weights[period_id],
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
        target_load = resolve_hpr_target_load(
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
            PreparedHPRPeriodCase(
                period_id=period_id,
                period_idx=period_idx,
                weight=case["weight"],
                solver_case=HPRPeriodCase(
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
                ),
                base_target=case["base_target"],
                optimizer_pt=pt,
            )
        )
    return period_cases


def get_multiperiod_hpr_targets(
    *,
    period_cases: list[PreparedHPRPeriodCase],
    selected_period_id: str,
    selected_period_idx: int,
) -> HeatPumpTargetOutputs:
    """Solve a shared HPR design and project it into the public output schema."""
    solver_cases = [case.solver_case for case in period_cases]
    selected_case = period_case_by_id(period_cases, selected_period_id).solver_case
    x0_ls, bnds, objective = _get_multiperiod_hpr_opt_setup(
        solver_cases,
        selected_case=selected_case,
    )
    mp_args = MultiPeriodHPRTargetInputs(
        period_cases=solver_cases,
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


def period_id_for_index(zone: Zone, period_idx: int) -> str:
    for period_id, idx in (zone.period_ids or {"0": 0}).items():
        if int(idx) == int(period_idx):
            return str(period_id)
    raise ValueError(f"period_idx {period_idx!r} was not found on this zone.")


def period_case_by_id(
    period_cases: list[PreparedHPRPeriodCase],
    period_id: str,
) -> PreparedHPRPeriodCase:
    for case in period_cases:
        if str(case.period_id) == str(period_id):
            return case
    raise ValueError(f"period_id {period_id!r} was not prepared for HPR targeting.")


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


def _canonical_period_weights(zone: Zone) -> dict[str, float]:
    items = _canonical_period_items(zone)
    weights = getattr(zone, "weights", None)
    if weights is None or len(weights) != len(items):
        return {period_id: 1.0 for period_id, _idx in items}
    flat_weights = np.asarray(weights, dtype=float).reshape(-1)
    return {
        period_id: float(flat_weights[period_idx]) for period_id, period_idx in items
    }


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
