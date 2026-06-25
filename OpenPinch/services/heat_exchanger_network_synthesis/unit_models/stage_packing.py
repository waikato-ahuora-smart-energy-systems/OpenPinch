"""Stage-packing constraints for HEN integer model variants."""

from __future__ import annotations

from typing import Any

_STAGE_DUTY_RELATIVE_TOLERANCE = 1e-5


def add_recovery_stage_packing_constraints(model: Any) -> None:
    """Force active recovery stages to be contiguous in integer models."""

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
            stage_active.append(
                model.m.Param(value=0, name=f"recovery_stage_active_{k}")
            )
            stage_duties.append(
                model.m.Param(value=0.0, name=f"recovery_stage_duty_{k}")
            )
            continue

        active = model.m.Var(
            value=1,
            ub=1,
            lb=0,
            integer=True,
            name=f"recovery_stage_active_{k}",
        )
        duty = model.m.Intermediate(
            model.m.sum(
                [
                    model.Q_r[i][j][k]
                    for i in range(model.I)
                    for j in range(model.J)
                    if model.z_allowed[i][j][k] > 0
                ]
            ),
            name=f"recovery_stage_duty_{k}",
        )
        model.m.Equation(duty <= capacity * active)
        model.m.Equation(duty >= threshold * active)
        stage_active.append(active)
        stage_duties.append(duty)

    for k in range(model.S - 1):
        model.m.Equation(stage_active[k] >= stage_active[k + 1])

    model.recovery_stage_active = stage_active
    model.recovery_stage_duty = stage_duties


def _stage_capacity(model: Any, stage: int) -> float:
    return sum(
        max(float(model.Q_max[i][j]), 0.0)
        for i in range(model.I)
        for j in range(model.J)
        if model.z_allowed[i][j][stage] > 0
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
                model.m.Equation(
                    model.Q_r[i][j][k]
                    >= capacity * _STAGE_DUTY_RELATIVE_TOLERANCE * model.z[i][j][k]
                )


__all__ = ["add_recovery_stage_packing_constraints"]
