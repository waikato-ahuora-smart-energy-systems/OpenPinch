"""Cascade Carnot HP targeting."""

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
from ..cycles.carnot_cycles import CascadeCarnotCycle
from ..optimisation_adapter import solve_hpr_placement
from ._carnot_common import (
    CarnotTopology,
    carnot_opt_setup,
    compute_carnot_objective,
    normalise_carnot_stage_counts,
    parse_carnot_state_variables,
)

__all__ = [
    "optimise_cascade_carnot_heat_pump_placement",
]

_CASCADE_CARNOT = CarnotTopology(
    equal_stage_counts=False,
    extra_result_kwargs=lambda cycle: {
        "Q_cond": cycle.Q_cond - cycle.Q_cond_he,
        "Q_evap": cycle.Q_evap - cycle.Q_evap_he,
        "Q_cond_he": cycle.Q_cond_he,
        "Q_evap_he": cycle.Q_evap_he,
    },
)


################################################################################
# Public API
################################################################################


def optimise_cascade_carnot_heat_pump_placement(
    args: HeatPumpTargetInputs,
) -> HPRBackendResult:
    """Optimise cascade Carnot stages for the prepared HPR case."""
    normalise_carnot_stage_counts(_CASCADE_CARNOT, args)
    x0_ls, bnds = _get_cascade_carnot_hp_opt_setup(args)
    return solve_hpr_placement(
        f_obj=_compute_cascade_carnot_cycle_obj,
        x0_ls=x0_ls,
        bnds=bnds,
        args=args,
    )


################################################################################
# Helper Functions
################################################################################


def _get_cascade_carnot_hp_opt_setup(
    args: HeatPumpTargetInputs,
) -> tuple[np.ndarray, list]:
    return carnot_opt_setup(args)


def _parse_cascade_carnot_cycle_state_variables(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
) -> HPRParsedState:
    return parse_carnot_state_variables(x, args)


def _compute_cascade_carnot_cycle_obj(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    *,
    debug: bool = False,
    artifact_mode: HPREvaluationMode = HPREvaluationMode.FINAL,
) -> HPRBackendResult:
    return compute_carnot_objective(
        x,
        args,
        topology=_CASCADE_CARNOT,
        parse_state=_parse_cascade_carnot_cycle_state_variables,
        cycle_cls=CascadeCarnotCycle,
        evaluate=evaluate_carnot_hpr_result,
        debug=debug,
        artifact_mode=artifact_mode,
    )
