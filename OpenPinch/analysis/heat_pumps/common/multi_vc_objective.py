"""Shared CoolProp objective for multi-stage vapour-compression HPR topologies.

The cascade and parallel vapour-compression targeting modules evaluate a
candidate with the same sequence: parse the optimisation vector, solve the
topology's cycle, build its stream collection, account for work and duties,
and attach a CoolProp simulation record for final evaluations.  The topology
modules keep their objective functions as thin wrappers that supply the
topology-specific pieces (state parser, cycle builder/solver, topology id)
and the module-level collaborators they import, so those names stay
patchable where they are defined.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ....contracts.hpr import (
    ZERO_USEFUL_DUTY_REASON,
    HeatPumpTargetInputs,
    HPRBackendResult,
    HPREvaluationMode,
    HPRParsedState,
    HPRTopologyIdentifier,
)

__all__ = ["_evaluate_multi_vc_objective"]


def _evaluate_multi_vc_objective(
    x: np.ndarray,
    args: HeatPumpTargetInputs,
    *,
    parse_state: Callable[[np.ndarray, HeatPumpTargetInputs], Any],
    build_and_solve: Callable[[HPRParsedState, HeatPumpTargetInputs, bool], Any],
    topology_id: Callable[[HeatPumpTargetInputs], HPRTopologyIdentifier],
    label: str,
    evaluate_result: Callable[..., HPRBackendResult],
    build_record: Callable[..., Any],
    check_temperature_lift: bool = False,
    result_state_fields: Sequence[str] = (),
    debug: bool = False,
    artifact_mode: HPREvaluationMode = HPREvaluationMode.FINAL,
) -> HPRBackendResult:
    """Evaluate one multi-stage vapour-compression candidate.

    ``build_and_solve(state, args, is_heat_pumping)`` constructs the cycle
    model, solves it and returns it.  ``label`` names the topology in the
    unsolved-cycle failure reason.  ``result_state_fields`` lists extra parsed
    state attributes forwarded to ``evaluate_result`` by name.
    ``check_temperature_lift`` rejects candidates with a negative lift across
    any stage before the cycle is solved.
    """
    is_heat_pumping = getattr(args, "is_heat_pumping", True)
    state_vars = parse_state(x, args)
    if not isinstance(state_vars, HPRParsedState):
        state_vars = HPRParsedState.model_validate(state_vars)

    if check_temperature_lift:
        T_diff = (
            state_vars.T_cond
            - state_vars.dT_subcool
            - state_vars.T_evap
            - args.dtcont_hp
        )
        if T_diff.min() < 0:
            return HPRBackendResult.failure(
                reason=(
                    "Invalid simulated vapour candidate with negative temperature lift."
                ),
                Q_amb_hot=state_vars.Q_amb_hot,
                Q_amb_cold=state_vars.Q_amb_cold,
            )

    cycle_evaluated = False
    try:
        hp = build_and_solve(state_vars, args, is_heat_pumping)
        if not hp.solved:
            return HPRBackendResult.failure(
                reason=f"{label} vapour-compression cycle failed to solve.",
                Q_amb_hot=state_vars.Q_amb_hot,
                Q_amb_cold=state_vars.Q_amb_cold,
            )

        hpr_streams = hp.build_stream_collection(
            include_cond=True,
            include_evap=True,
            is_process_stream=False,
            dtcont=args.dtcont_hp,
        )
        cycle_evaluated = True
        w_hpr = hp.work
        primary_duty = hp.Q_heat_arr.sum() if is_heat_pumping else hp.Q_cool_arr.sum()
        if not np.isfinite(primary_duty) or primary_duty < 0.0:
            return HPRBackendResult.failure(reason="Cycle delivers negative duty.")
        if primary_duty == 0.0 and artifact_mode is HPREvaluationMode.FINAL:
            return HPRBackendResult.failure(reason=ZERO_USEFUL_DUTY_REASON)
        cop = primary_duty / w_hpr if w_hpr > 0 else 1.0
        result = evaluate_result(
            args=args,
            state=state_vars,
            work=w_hpr,
            work_arr=hp.work_arr,
            Q_heat=hp.Q_heat_arr,
            Q_cool=hp.Q_cool_arr,
            cop_h=cop,
            hpr_streams=hpr_streams,
            model=hp,
            penalty_terms=[hp.penalty],
            dT_subcool=state_vars.dT_subcool,
            debug=debug,
            artifact_mode=artifact_mode,
            **{name: getattr(state_vars, name) for name in result_state_fields},
        )
        record = (
            build_record(
                args=args,
                state=state_vars,
                cycle=hp,
                topology_id=topology_id(args),
            )
            if artifact_mode is HPREvaluationMode.FINAL
            else None
        )
        return result.with_updates(
            simulation_backend=(
                "coolprop" if artifact_mode is HPREvaluationMode.FINAL else None
            ),
            target_simulation_record=record,
        )
    except ValueError as exc:
        if cycle_evaluated:
            raise
        return HPRBackendResult.failure(
            reason=str(exc),
            Q_amb_hot=state_vars.Q_amb_hot,
            Q_amb_cold=state_vars.Q_amb_cold,
        )
