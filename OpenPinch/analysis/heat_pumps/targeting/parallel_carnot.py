"""Parallel Carnot HP targeting."""

from __future__ import annotations

import numpy as np

from ....contracts.hpr import (
    HeatPumpTargetInputs,
    HPRBackendResult,
    HPREvaluationMode,
    HPRParsedState,
)
from ..common.shared import (
    evaluate_carnot_hpr_result,
)
from ..cycles.carnot_cycles import ParallelCarnotCycles
from ..optimisation_adapter import solve_hpr_placement
from ._carnot_common import (
    CarnotTopology,
    carnot_opt_setup,
    compute_carnot_objective,
    normalise_carnot_stage_counts,
    parse_carnot_state_variables,
)

__all__ = [
    "optimise_parallel_carnot_heat_pump_placement",
]

_PARALLEL_CARNOT = CarnotTopology(
    equal_stage_counts=True,
    extra_result_kwargs=lambda cycle: {"eta_he": cycle.eta_he},
)


################################################################################
# Public API
################################################################################


def optimise_parallel_carnot_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    """Optimise parallel simple Carnot stages for a screening-level HPR solve."""
    normalise_carnot_stage_counts(_PARALLEL_CARNOT, args)
    x0_ls, bnds = _get_parallel_carnot_hp_opt_setup(args)
    return solve_hpr_placement(
        f_obj=_compute_parallel_carnot_hp_opt_obj,
        x0_ls=x0_ls,
        bnds=bnds,
        args=args,
    )


################################################################################
# Helper Functions
################################################################################


def _get_parallel_carnot_hp_opt_setup(
    args: HeatPumpTargetInputs,
) -> tuple[np.ndarray, list]:
    return carnot_opt_setup(args)


def _parse_parallel_carnot_hp_state_variables(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> HPRParsedState:
    return parse_carnot_state_variables(x, args)


def _compute_parallel_carnot_hp_opt_obj(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    *,
    debug: bool = False,
    artifact_mode: HPREvaluationMode = HPREvaluationMode.FINAL,
) -> HPRBackendResult:
    return compute_carnot_objective(
        x,
        args,
        topology=_PARALLEL_CARNOT,
        parse_state=_parse_parallel_carnot_hp_state_variables,
        cycle_cls=ParallelCarnotCycles,
        evaluate=evaluate_carnot_hpr_result,
        debug=debug,
        artifact_mode=artifact_mode,
    )
