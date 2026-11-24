"""Stage-packing constraints for HEN integer model variants."""

from __future__ import annotations

from typing import Any

_STAGE_DUTY_RELATIVE_TOLERANCE = 1e-5


def add_recovery_stage_packing_constraints(model: Any) -> None:
    """Force active recovery stages to be contiguous in integer models.

    Contiguous among the stages with an allowed match: an empty stage is
    skipped.
    """

    if not getattr(model, "integers", False) or getattr(model, "S", 0) <= 1:
        return

    stage_capacities = [_stage_capacity(model, k) for k in range(model.S)]
    duty_scale = max(stage_capacities)
    if duty_scale <= 0.0:
        return

    threshold = max(
        float(getattr(model, "tol", 0.0)),
        duty_scale * _STAGE_DUTY_RELATIVE_TOLERANCE,
    )
    _add_active_match_duty_constraints(model)
    stage_active = []
    stage_duties = []
    for k, capacity in enumerate(stage_capacities):
        if capacity <= 0.0:
            stage_active.append(model.m.Param(value=0))
            stage_duties.append(model.m.Param(value=0.0))
            continue

        active = model.m.Var(
            value=1,
            ub=1,
            lb=0,
            integer=True,
        )
        duty = model.m.Intermediate(
            model.m.sum(
                [
                    duty
                    for i in range(model.I)
                    for j in range(model.J)
                    if model.z_allowed[i][j][k] > 0
                    for duty in _period_duties(model, i, j, k)
                ]
            ),
        )
        model.m.Equation(duty <= capacity * active)
        model.m.Equation(duty >= threshold * active)
        stage_active.append(active)
        stage_duties.append(duty)

    # Chain only the stages that can carry duty: a stage with no allowed
    # match is skipped, not a barrier that empties every stage after it.
    open_stages = [k for k, capacity in enumerate(stage_capacities) if capacity > 0.0]
    for k, later in zip(open_stages, open_stages[1:]):
        model.m.Equation(stage_active[k] >= stage_active[later])

    model.recovery_stage_active = stage_active
    model.recovery_stage_duty = stage_duties


def _period_duties(model: Any, i: int, j: int, k: int) -> list:
    """A match's duty in every period (period 0 alone without per-period
    duties): a match may carry load only in some periods."""
    by_period = getattr(model, "Q_r_by_period", None)
    if by_period is None:
        return [model.Q_r[i][j][k]]
    return [by_period[n][i][j][k] for n in range(len(by_period))]


def _period_capacities(model: Any, i: int, j: int) -> list[float]:
    """A match's duty bound in every period (``Q_max`` without them)."""
    by_period = getattr(model, "Q_max_period", None)
    if by_period is None or getattr(model, "Q_r_by_period", None) is None:
        return [max(float(model.Q_max[i][j]), 0.0)]
    return [max(float(by_period[n][i][j]), 0.0) for n in range(len(by_period))]


def _stage_capacity(model: Any, stage: int) -> float:
    return sum(
        capacity
        for i in range(model.I)
        for j in range(model.J)
        if model.z_allowed[i][j][stage] > 0
        for capacity in _period_capacities(model, i, j)
    )


def _add_active_match_duty_constraints(model: Any) -> None:
    for k in range(model.S):
        for j in range(model.J):
            for i in range(model.I):
                if model.z_allowed[i][j][k] <= 0:
                    continue
                capacity = max(float(model.Q_max[i][j]), 0.0)
                if capacity <= 0.0:
                    continue
                # An active match carries duty in some period, not
                # necessarily the first.
                duties = _period_duties(model, i, j, k)
                model.m.Equation(
                    (duties[0] if len(duties) == 1 else model.m.sum(duties))
                    >= capacity * _STAGE_DUTY_RELATIVE_TOLERANCE * model.z[i][j][k]
                )


__all__ = ["add_recovery_stage_packing_constraints"]
