"""Build cycle-specific optimisation inputs for a shared HPR design."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ....contracts.hpr import HPRPeriodCase
from ....domain.enums import HeatPumpAndRefrigerationCycle
from ..common.shared import validate_vapour_hp_refrigerant_ls
from ..cycles.registry import family_handlers, resolve_handler
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
    _num_vc_stages,
    _parse_vc_mvr_state_variables,
    _vc_mvr_idle_point,
    use_mvr_cascade_fluids,
)


def get_multiperiod_hpr_optimisation_setup(
    period_cases: list[HPRPeriodCase],
    *,
    selected_case: HPRPeriodCase,
) -> tuple[np.ndarray | None, list, Callable, bool]:
    """Return starts, bounds, the period objective and whether the shared
    design is one installed machine, designed and operated per period
    (vapour compression cycles), rather than one shared vector (Carnot).

    The selected case's model family picks the setup from the cycle registry.
    """
    hpr_type = selected_case.args.hpr_type
    try:
        handlers = family_handlers(hpr_type)
    except ValueError:
        raise ValueError("No valid heat pump targeting type selected.") from None
    if handlers.multiperiod_setup is None:
        raise NotImplementedError(
            f"{hpr_type} HPR targeting is not supported for multi-period optimisation."
        )
    return resolve_handler(handlers.multiperiod_setup)(period_cases, selected_case)


def multiperiod_idle_point(hpr_type: str) -> Callable | None:
    """The cycle's machine off (``x, args -> x``) where its lower bounds are
    not; ``None`` where they are."""
    if hpr_type == HeatPumpAndRefrigerationCycle.VapourCompMVR.value:
        return _vc_mvr_idle_point
    return None


def _preflight_fluid_names(period_cases, topology_id: str) -> None:
    for case in period_cases:
        if case.args.simulation_backend == "coolprop":
            preflight_coolprop_fluid_names(case.args, topology_id)


def _cascade_carnot_setup(period_cases, selected_case):
    x0_ls, bounds = _get_cascade_carnot_hp_opt_setup(selected_case.args)
    return x0_ls, bounds, _compute_cascade_carnot_cycle_obj, False


def _parallel_carnot_setup(period_cases, selected_case):
    n_stages = max(selected_case.args.n_cond, selected_case.args.n_evap)
    for case in period_cases:
        case.args.n_cond = case.args.n_evap = n_stages
    x0_ls, bounds = _get_parallel_carnot_hp_opt_setup(selected_case.args)
    return x0_ls, bounds, _compute_parallel_carnot_hp_opt_obj, False


def _cascade_vapour_compression_setup(period_cases, selected_case):
    _preflight_fluid_names(period_cases, "cascade_vapour_compression")
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
    return x0_ls, bounds, _compute_cascade_hp_system_obj, True


def _parallel_vapour_compression_setup(period_cases, selected_case):
    _preflight_fluid_names(period_cases, "parallel_vapour_compression")
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
    return x0_ls, bounds, _compute_parallel_hp_system_obj, True


def _vapour_compression_mvr_setup(period_cases, selected_case):
    for case in period_cases:
        use_mvr_cascade_fluids(case.args)
    _preflight_fluid_names(period_cases, "vapour_compression_mvr")
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
    x0_ls, bounds = _get_vc_mvr_opt_setup(initial_result, selected_case.args)
    if selected_case.args.simulation_backend == "coolprop":
        point = representative_hpr_point(x0_ls, bounds)
        for case in period_cases:
            preflight_coolprop_hpr_targeting(
                args=case.args,
                state=_parse_vc_mvr_state_variables(point, case.args),
                topology_id="vapour_compression_mvr",
            )
    return x0_ls, bounds, _compute_vc_mvr_system_obj, True


__all__ = ["get_multiperiod_hpr_optimisation_setup"]
