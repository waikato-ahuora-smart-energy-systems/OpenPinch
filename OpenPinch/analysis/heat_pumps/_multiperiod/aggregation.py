"""Evaluate and aggregate one shared HPR design across prepared periods."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ....contracts.hpr import (
    ZERO_USEFUL_DUTY_REASON,
    HPRBackendResult,
    HPREvaluationMode,
    HPRPeriodCase,
    MultiPeriodHPRTargetInputs,
)
from ..optimisation_adapter import (
    aggregate_hpr_period_results,
    evaluate_hpr_candidate,
)


def evaluate_multiperiod_candidate(
    point: Sequence[float] | np.ndarray,
    args: MultiPeriodHPRTargetInputs,
    *,
    period_objective: Callable,
    debug: bool = False,
    artifact_mode: HPREvaluationMode = HPREvaluationMode.FINAL,
) -> HPRBackendResult:
    """Evaluate one design vector in every period and apply shared policies."""
    if not args.period_cases:
        return HPRBackendResult.failure(reason="No period cases were prepared.")

    period_outputs: dict[str, HPRBackendResult] = {}
    zero_duty_failure: HPRBackendResult | None = None
    for case in args.period_cases:
        result = evaluate_hpr_candidate(
            objective=period_objective,
            point=point,
            args=case.args,
            debug=debug,
            artifact_mode=artifact_mode,
        )
        if result.failure_reason == ZERO_USEFUL_DUTY_REASON:
            # Keep evaluating: the shared design is "no heat pump" only when it
            # delivers zero useful duty in every period.
            zero_duty_failure = zero_duty_failure or result
            continue
        if not result.success or not np.isfinite(float(result.obj)):
            reason = result.failure_reason or "candidate failed"
            return HPRBackendResult.failure(
                reason=f"HPR period {case.period_id!r} failed: {reason}",
                Q_amb_hot=result.Q_amb_hot,
                Q_amb_cold=result.Q_amb_cold,
            )
        period_outputs[str(case.period_id)] = result

    if zero_duty_failure is not None:
        if not period_outputs:
            # Preserve the sentinel unwrapped so the solver can stop and
            # report the typed "no beneficial heat pump" outcome.
            return HPRBackendResult.failure(
                reason=ZERO_USEFUL_DUTY_REASON,
                Q_amb_hot=zero_duty_failure.Q_amb_hot,
                Q_amb_cold=zero_duty_failure.Q_amb_cold,
            )
        idle_periods = [
            str(case.period_id)
            for case in args.period_cases
            if str(case.period_id) not in period_outputs
        ]
        return HPRBackendResult.failure(
            reason=(f"HPR periods {idle_periods!r} failed: {ZERO_USEFUL_DUTY_REASON}"),
            Q_amb_hot=zero_duty_failure.Q_amb_hot,
            Q_amb_cold=zero_duty_failure.Q_amb_cold,
        )

    weights = np.asarray([case.weight for case in args.period_cases], dtype=float)
    weighted, shared_objective = aggregate_hpr_period_results(
        period_outputs,
        weights,
    )
    selected = period_outputs[str(selected_period_case(args).period_id)]
    return selected.with_updates(
        obj=shared_objective,
        period_outputs=period_outputs,
        weighted_output=weighted,
        design_vector=np.asarray(point, dtype=float),
        period_ids=[str(case.period_id) for case in args.period_cases],
        period_weights=[float(case.weight) for case in args.period_cases],
    )


def selected_period_case(args: MultiPeriodHPRTargetInputs) -> HPRPeriodCase:
    """Resolve the selected solver case by stable id, then by numeric index."""
    for case in args.period_cases:
        if str(case.period_id) == str(args.selected_period_id):
            return case
    for case in args.period_cases:
        if int(case.period_idx) == int(args.selected_period_idx):
            return case
    raise ValueError(
        "Selected period is not present in the prepared multi-period HPR cases."
    )


__all__ = ["evaluate_multiperiod_candidate", "selected_period_case"]
