"""Build cycle-specific optimisation inputs for a shared HPR design."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ....contracts.hpr import HPRPeriodCase
from ....domain.enums import HeatPumpAndRefrigerationCycle
from ..common.shared import validate_vapour_hp_refrigerant_ls
from ..optimisation_adapter import initialise_hpr_seed
from ..performance_maps.coolprop_preflight import (
    preflight_coolprop_fluid_names,
    preflight_coolprop_hpr_targeting,
    representative_hpr_point,
)
from ..targeting.cascade_carnot import (
    _compute_cascade_carnot_cycle_obj,
    _get_cascade_carnot_hp_opt_setup,
    optimise_cascade_carnot_heat_pump_placement,
)
from ..targeting.cascade_vapour_compression import (
    _cascade_stage_preflight_state,
    _compute_cascade_hp_system_obj,
    _get_cascade_hp_opt_setup,
    _parse_cascade_hp_state_variables,
)
from ..targeting.parallel_carnot import (
    _compute_parallel_carnot_hp_opt_obj,
    _get_parallel_carnot_hp_opt_setup,
    optimise_parallel_carnot_heat_pump_placement,
)
from ..targeting.parallel_vapour_compression import (
    _compute_parallel_hp_system_obj,
    _get_parallel_hp_opt_setup,
    _parse_parallel_hp_state_temperatures,
)
from ..targeting.vapour_compression_mvr import (
    _compute_vc_mvr_system_obj,
    _get_vc_mvr_opt_setup,
    _normalise_fluid_list,
    _num_vc_stages,
    _parse_vc_mvr_state_variables,
)


def get_multiperiod_hpr_optimisation_setup(
    period_cases: list[HPRPeriodCase],
    *,
    selected_case: HPRPeriodCase,
) -> tuple[np.ndarray | None, list, Callable]:
    """Return starts, bounds, and the period objective for one HPR cycle."""
    hpr_type = selected_case.args.hpr_type
    topology = {
        HeatPumpAndRefrigerationCycle.CascadeVapourComp.value: (
            "cascade_vapour_compression"
        ),
        HeatPumpAndRefrigerationCycle.ParallelVapourComp.value: (
            "parallel_vapour_compression"
        ),
        HeatPumpAndRefrigerationCycle.VapourCompMVR.value: "vapour_compression_mvr",
    }.get(hpr_type)
    if topology is not None:
        for case in period_cases:
            if case.args.simulation_backend == "coolprop":
                preflight_coolprop_fluid_names(case.args, topology)
    if hpr_type == HeatPumpAndRefrigerationCycle.CascadeCarnot.value:
        x0_ls, bounds = _get_cascade_carnot_hp_opt_setup(selected_case.args)
        return x0_ls, bounds, _compute_cascade_carnot_cycle_obj

    if hpr_type == HeatPumpAndRefrigerationCycle.ParallelCarnot.value:
        n_stages = max(selected_case.args.n_cond, selected_case.args.n_evap)
        for case in period_cases:
            case.args.n_cond = case.args.n_evap = n_stages
        x0_ls, bounds = _get_parallel_carnot_hp_opt_setup(selected_case.args)
        return x0_ls, bounds, _compute_parallel_carnot_hp_opt_obj

    if hpr_type == HeatPumpAndRefrigerationCycle.CascadeVapourComp.value:
        num_stages = int(selected_case.args.n_cond + selected_case.args.n_evap - 1)
        initial_result = initialise_hpr_seed(
            optimise_cascade_carnot_heat_pump_placement, selected_case.args
        )
        for case in period_cases:
            case.args.refrigerant_ls = validate_vapour_hp_refrigerant_ls(
                num_stages,
                case.args,
            )
        x0_ls, bounds = _get_cascade_hp_opt_setup(
            initial_result,
            selected_case.args,
        )
        if selected_case.args.simulation_backend == "coolprop":
            point = representative_hpr_point(x0_ls, bounds)
            for case in period_cases:
                state = _parse_cascade_hp_state_variables(point, case.args)
                preflight_coolprop_hpr_targeting(
                    args=case.args,
                    state=_cascade_stage_preflight_state(state, case.args),
                    topology_id="cascade_vapour_compression",
                )
        return x0_ls, bounds, _compute_cascade_hp_system_obj

    if hpr_type == HeatPumpAndRefrigerationCycle.ParallelVapourComp.value:
        num_stages = max(selected_case.args.n_cond, selected_case.args.n_evap)
        for case in period_cases:
            case.args.n_cond = case.args.n_evap = int(num_stages)
        initial_result = initialise_hpr_seed(
            optimise_parallel_carnot_heat_pump_placement, selected_case.args
        )
        for case in period_cases:
            case.args.refrigerant_ls = validate_vapour_hp_refrigerant_ls(
                int(num_stages),
                case.args,
            )
        x0_ls, bounds = _get_parallel_hp_opt_setup(
            initial_result,
            selected_case.args,
        )
        if selected_case.args.simulation_backend == "coolprop":
            point = representative_hpr_point(x0_ls, bounds)
            for case in period_cases:
                preflight_coolprop_hpr_targeting(
                    args=case.args,
                    state=_parse_parallel_hp_state_temperatures(point, case.args),
                    topology_id="parallel_vapour_compression",
                )
        return x0_ls, bounds, _compute_parallel_hp_system_obj

    if hpr_type == HeatPumpAndRefrigerationCycle.VapourCompMVR.value:
        if not selected_case.args.is_heat_pumping:
            raise ValueError(
                "Vapour compression with MVR cascade is heat-pump-only; "
                "refrigeration targets are not supported."
            )
        n_vc = _num_vc_stages(selected_case.args)
        initial_result = initialise_hpr_seed(
            optimise_cascade_carnot_heat_pump_placement, selected_case.args
        )
        for case in period_cases:
            case.args.refrigerant_ls = validate_vapour_hp_refrigerant_ls(
                n_vc,
                case.args,
            )
            case.args.mvr_fluid_ls = _normalise_fluid_list(case.args.mvr_fluid_ls)
        x0_ls, bounds = _get_vc_mvr_opt_setup(initial_result, selected_case.args)
        if selected_case.args.simulation_backend == "coolprop":
            point = representative_hpr_point(x0_ls, bounds)
            for case in period_cases:
                preflight_coolprop_hpr_targeting(
                    args=case.args,
                    state=_parse_vc_mvr_state_variables(point, case.args),
                    topology_id="vapour_compression_mvr",
                )
        return x0_ls, bounds, _compute_vc_mvr_system_obj

    if hpr_type == HeatPumpAndRefrigerationCycle.Brayton.value:
        raise NotImplementedError(
            "Brayton HPR targeting is not supported for multi-period optimisation."
        )
    raise ValueError("No valid heat pump targeting type selected.")


__all__ = ["get_multiperiod_hpr_optimisation_setup"]
