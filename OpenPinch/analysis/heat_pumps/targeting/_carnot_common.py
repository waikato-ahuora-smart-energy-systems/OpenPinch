"""Shared Carnot HPR targeting for the cascade and parallel topologies.

Both Carnot screening topologies use the same optimisation vector, state
parsing and objective; they differ only in the cycle model, whether the
condenser and evaporator stage counts are equalised, and which extra
cycle attributes are forwarded to the result builder.  The topology-specific
modules (``cascade_carnot`` and ``parallel_carnot``) keep their public and
private entry points as thin wrappers around the functions defined here, and
pass the cycle class, state parser and result evaluator at call time so that
module-level names remain patchable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ....contracts.hpr import (
    HeatPumpTargetInputs,
    HPRBackendResult,
    HPREvaluationMode,
    HPRParsedState,
)
from ..common._shared.streams import get_Q_vals_at_T_hpr_from_bckgrd_profile
from ..common.encoding import (
    AMBIENT_X_BOUNDS,
    DutyAllocationRequest,
    decode_available_fractions,
    limit_available_duty,
    map_x_arr_to_T_arr,
    map_x_to_Q_amb,
)
from ..common.layout import HPRoptVectorLayout

__all__ = [
    "CarnotTopology",
    "carnot_opt_setup",
    "compute_carnot_objective",
    "normalise_carnot_stage_counts",
    "parse_carnot_state_variables",
]


@dataclass(frozen=True)
class CarnotTopology:
    """Static description of one Carnot HPR topology."""

    equal_stage_counts: bool
    extra_result_kwargs: Callable[[Any], dict[str, Any]]


def normalise_carnot_stage_counts(
    topology: CarnotTopology,
    args: HeatPumpTargetInputs,
) -> None:
    """Apply the topology's stage-count rule to ``args`` in place.

    Parallel stages pair one condenser with one evaporator, so both counts are
    raised to the larger of the two.  Callers rely on the normalised counts
    remaining on ``args`` after the solve.
    """
    if topology.equal_stage_counts:
        args.n_cond = args.n_evap = max(args.n_cond, args.n_evap)


def carnot_opt_setup(
    args: HeatPumpTargetInputs,
) -> tuple[np.ndarray, list]:
    """Return the initial point and bounds of the Carnot optimisation vector."""
    n_stages = int(args.n_cond)
    layout = _carnot_layout(args)
    # Start every condenser stage at all the duty it can deliver.
    return layout.pack(
        x_amb=0.0,
        x_cond=[0.0] * layout.n_cond,
        x_evap=[0.0] * layout.n_evap,
        x_heat_split=[1.0] * n_stages,
    ), layout.build_bounds(
        x_amb=AMBIENT_X_BOUNDS,
        x_cond=(0.0, 1.0),
        x_evap=(0.0, 1.0),
        x_heat_split=(0.0, 1.0),
    )


def _carnot_layout(args: HeatPumpTargetInputs) -> HPRoptVectorLayout:
    return HPRoptVectorLayout(
        n_cond=int(args.n_cond),
        n_evap=int(args.n_evap),
        n_heat_split=int(args.n_cond),
    )


def parse_carnot_state_variables(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> HPRParsedState:
    """Decode a Carnot optimisation vector into temperatures and duties."""
    parts = _carnot_layout(args).unpack(x)
    T_cond = map_x_arr_to_T_arr(parts["x_cond"], args.T_cold[0], args.T_cold[-1])
    T_evap = map_x_arr_to_T_arr(parts["x_evap"], args.T_hot[-1], args.T_hot[0])
    Q_amb_hot, Q_amb_cold = map_x_to_Q_amb(
        parts["x_amb"], max(args.Q_heat_max, args.Q_cool_max)
    )
    H_cold_with_amb = args.H_cold + args.z_amb_cold * Q_amb_cold
    H_hot_with_amb = args.H_hot + args.z_amb_hot * Q_amb_hot
    Q_heat_available = limit_available_duty(
        get_Q_vals_at_T_hpr_from_bckgrd_profile(
            T_cond,
            args.T_cold,
            H_cold_with_amb,
            is_cond=True,
        ),
        args.Q_heat_max + (0.0 if args.is_heat_pumping else Q_amb_cold),
    )
    Q_heat_base = float(
        decode_available_fractions(parts["x_heat_split"], Q_heat_available).sum()
    )
    Q_cool_available = get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_evap,
        args.T_hot,
        H_hot_with_amb,
        is_cond=False,
    )
    return HPRParsedState(
        T_cond=T_cond,
        T_evap=T_evap,
        Q_amb_hot=Q_amb_hot,
        Q_amb_cold=Q_amb_cold,
        Q_heat_base=Q_heat_base,
        x_heat_split=parts["x_heat_split"],
        Q_heat_available=Q_heat_available,
        Q_cool_available=Q_cool_available,
    )


def compute_carnot_objective(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    *,
    topology: CarnotTopology,
    parse_state: Callable[[np.ndarray, HeatPumpTargetInputs], Any],
    cycle_cls: Callable[[], Any],
    evaluate: Callable[..., HPRBackendResult],
    debug: bool = False,
    artifact_mode: HPREvaluationMode = HPREvaluationMode.FINAL,
) -> HPRBackendResult:
    """Solve the topology's Carnot cycle at ``x`` and build the backend result."""
    state_vars = parse_state(x, args)
    if not isinstance(state_vars, HPRParsedState):
        state_vars = HPRParsedState.model_validate(state_vars)

    cycle = cycle_cls()
    cycle.solve(
        T_cond=state_vars.T_cond,
        T_evap=state_vars.T_evap,
        duty_allocation=DutyAllocationRequest.from_state(state_vars),
        eta_ii_hpr_carnot=args.eta_ii_hpr_carnot,
        eta_ii_he_carnot=args.eta_ii_he_carnot,
        args=args,
    )

    return evaluate(
        args=args,
        state=state_vars,
        w_net=cycle.work,
        w_hpr=cycle.w_hpr,
        w_he=cycle.w_he,
        heat_recovery=cycle.heat_recovery,
        cop_h=cycle.COP_h,
        Q_cond_total=cycle.Q_cond,
        Q_evap_total=cycle.Q_evap,
        penalty_terms=cycle.penalty,
        debug=debug,
        artifact_mode=artifact_mode,
        **topology.extra_result_kwargs(cycle),
    )
