"""Parallel vapour-compression HP targeting."""

from __future__ import annotations

import numpy as np

from ....contracts.hpr import (
    HeatPumpTargetInputs,
    HPRBackendResult,
    HPREvaluationMode,
    HPRParsedState,
    HPRTopologyIdentifier,
)
from ..common._shared.streams import get_Q_vals_at_T_hpr_from_bckgrd_profile
from ..common.encoding import (
    DutyAllocationRequest,
    decode_available_fractions,
    encode_available_fractions,
    limit_available_duty,
    map_T_arr_to_x_arr,
    map_x_arr_to_DT_arr,
    map_x_arr_to_T_arr,
)
from ..common.layout import HPRoptVectorLayout
from ..common.multi_vc_objective import _evaluate_multi_vc_objective
from ..common.shared import (
    condensing_temperature_search_range,
    evaluate_vapour_hpr_result,
    validate_vapour_hp_refrigerant_ls,
)
from ..cycles.parallel_vapour_compression_cycles import (
    ParallelVapourCompressionCycles,
)
from ..optimisation_adapter import initialise_hpr_seed, solve_hpr_placement
from ..performance_maps.coolprop_preflight import (
    preflight_coolprop_fluid_names,
    preflight_coolprop_hpr_targeting,
    representative_hpr_point,
)
from ..performance_maps.target_records import build_coolprop_target_simulation_record
from .parallel_carnot import optimise_parallel_carnot_heat_pump_placement

__all__ = [
    "optimise_parallel_heat_pump_placement",
]


################################################################################
# Public API
################################################################################


def optimise_parallel_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    """Optimise multiple parallel vapour-compression stages for the HPR case."""
    num_stages = args.n_cond = args.n_evap = int(max(args.n_cond, args.n_evap))
    preflight_coolprop_fluid_names(
        args, HPRTopologyIdentifier.PARALLEL_VAPOUR_COMPRESSION
    )
    init_res = initialise_hpr_seed(optimise_parallel_carnot_heat_pump_placement, args)
    args.refrigerant_ls = validate_vapour_hp_refrigerant_ls(num_stages, args)
    x0_ls, bnds = _get_parallel_hp_opt_setup(init_res, args)
    preflight_coolprop_hpr_targeting(
        args=args,
        state=_parse_parallel_hp_state_temperatures(
            representative_hpr_point(x0_ls, bnds), args
        ),
        topology_id=HPRTopologyIdentifier.PARALLEL_VAPOUR_COMPRESSION,
    )
    return solve_hpr_placement(
        f_obj=_compute_parallel_hp_system_obj,
        x0_ls=x0_ls,
        bnds=bnds,
        args=args,
    )


def _parallel_layout(args: HeatPumpTargetInputs) -> HPRoptVectorLayout:
    """Vector layout: one availability fraction per unit on the primary side."""
    is_heat_pumping = getattr(args, "is_heat_pumping", True)
    n_units = int(args.n_cond)
    return HPRoptVectorLayout(
        n_cond=n_units,
        n_evap=int(args.n_evap),
        n_subcool=n_units,
        n_heat_split=n_units if is_heat_pumping else 0,
        n_cool_split=0 if is_heat_pumping else n_units,
        n_ihx=n_units,
    )


def _get_parallel_hp_opt_setup(
    init_res: HPRBackendResult | None,
    args: HeatPumpTargetInputs,
) -> tuple[np.ndarray | None, list]:
    is_heat_pumping = getattr(args, "is_heat_pumping", True)
    layout = _parallel_layout(args)
    bnds = layout.build_bounds(
        x_cond=(0.0, 1.0),
        x_evap=(0.0, 1.0),
        x_subcool=(0.0, 1.0),
        x_heat_split=(0.0, 1.0),
        x_cool_split=(0.0, 1.0),
        x_ihx=(0.0, 1.0),
    )
    if init_res is None:
        return None, bnds

    # The Carnot warm start knows nothing about critical points, so clip it
    # into the refrigerant's subcritical condensing range.
    x_cond = np.clip(
        map_T_arr_to_x_arr(init_res.T_cond, *condensing_temperature_search_range(args)),
        0.0,
        1.0,
    ).tolist()
    x_evap = map_T_arr_to_x_arr(
        init_res.T_evap[::-1], args.T_hot[-1], args.T_hot[0]
    ).tolist()
    x_subcool = [0.0] * int(args.n_cond)
    init_primary_duty = init_res.Q_cond if is_heat_pumping else init_res.Q_evap
    x_ihx = [0.0] * int(args.n_cond)
    pack_kwargs = {
        "x_cond": x_cond,
        "x_evap": x_evap,
        "x_subcool": x_subcool,
        "x_ihx": x_ihx,
    }
    split_key = "x_heat_split" if is_heat_pumping else "x_cool_split"
    pack_kwargs[split_key] = np.ones(int(args.n_cond), dtype=float)
    # Encode the seed duties against the availability at the seed's own
    # temperatures, exactly as the search will decode them.
    state = _parse_parallel_hp_state_temperatures(layout.pack(**pack_kwargs), args)
    Q_available = state.Q_heat_available if is_heat_pumping else state.Q_cool_available
    pack_kwargs[split_key] = encode_available_fractions(
        np.asarray(init_primary_duty, dtype=float), Q_available
    )
    return layout.pack(**pack_kwargs), bnds


################################################################################
# Helper Functions
################################################################################


def _parse_parallel_hp_state_temperatures(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> HPRParsedState:
    is_heat_pumping = getattr(args, "is_heat_pumping", True)
    parts = _parallel_layout(args).unpack(x)
    x_cond = parts["x_cond"]
    x_evap = parts["x_evap"]
    x_subcool = parts["x_subcool"]
    x_heat_split = parts["x_heat_split"]
    x_cool_split = parts["x_cool_split"]
    x_ihx = parts["x_ihx"]

    T_cond = map_x_arr_to_T_arr(x_cond, *condensing_temperature_search_range(args))
    T_evap = map_x_arr_to_T_arr(x_evap, args.T_hot[-1], args.T_hot[0])
    dT_subcool = map_x_arr_to_DT_arr(x_subcool, T_cond, T_evap)
    Q_heat_available = (
        limit_available_duty(
            get_Q_vals_at_T_hpr_from_bckgrd_profile(
                T_cond,
                args.T_cold,
                args.H_cold,
                is_cond=True,
            ),
            args.Q_heat_max,
        )
        if is_heat_pumping
        else None
    )
    Q_cool_available = (
        None
        if is_heat_pumping
        else limit_available_duty(
            get_Q_vals_at_T_hpr_from_bckgrd_profile(
                T_evap,
                args.T_hot,
                args.H_hot,
                is_cond=False,
            ),
            args.Q_cool_max,
        )
    )
    Q_heat_base = (
        float(decode_available_fractions(x_heat_split, Q_heat_available).sum())
        if is_heat_pumping
        else None
    )
    Q_cool_base = (
        None
        if is_heat_pumping
        else float(decode_available_fractions(x_cool_split, Q_cool_available).sum())
    )
    dT_ihx_gas_side = map_x_arr_to_DT_arr(x_ihx, T_cond, T_evap)
    return HPRParsedState(
        T_cond=T_cond,
        dT_subcool=dT_subcool,
        T_evap=T_evap,
        dT_ihx_gas_side=dT_ihx_gas_side,
        Q_heat_base=Q_heat_base,
        Q_cool_base=Q_cool_base,
        x_heat_split=x_heat_split if is_heat_pumping else None,
        x_cool_split=None if is_heat_pumping else x_cool_split,
        Q_heat_available=Q_heat_available,
        Q_cool_available=Q_cool_available,
    )


def _compute_parallel_hp_system_obj(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    *,
    debug: bool = False,
    artifact_mode: HPREvaluationMode = HPREvaluationMode.FINAL,
) -> HPRBackendResult:
    return _evaluate_multi_vc_objective(
        x,
        args,
        parse_state=_parse_parallel_hp_state_temperatures,
        build_and_solve=_build_and_solve_parallel_cycles,
        topology_id=_parallel_topology_id,
        label="Parallel",
        evaluate_result=evaluate_vapour_hpr_result,
        build_record=build_coolprop_target_simulation_record,
        check_temperature_lift=True,
        result_state_fields=("dT_superheat",),
        debug=debug,
        artifact_mode=artifact_mode,
    )


def _parallel_topology_id(_args: HeatPumpTargetInputs) -> HPRTopologyIdentifier:
    return HPRTopologyIdentifier.PARALLEL_VAPOUR_COMPRESSION


def _build_and_solve_parallel_cycles(
    state_vars: HPRParsedState,
    args: HeatPumpTargetInputs,
    is_heat_pumping: bool,
) -> ParallelVapourCompressionCycles:
    hp = ParallelVapourCompressionCycles()
    hp.solve(
        T_evap=state_vars.T_evap,
        T_cond=state_vars.T_cond,
        dtcont=args.dtcont_hp,
        dT_subcool=state_vars.dT_subcool,
        eta_comp=args.eta_comp,
        refrigerant=args.refrigerant_ls,
        dT_ihx_gas_side=state_vars.dT_ihx_gas_side,
        duty_allocation=DutyAllocationRequest.from_state(state_vars),
        is_heat_pump=is_heat_pumping,
    )
    return hp
