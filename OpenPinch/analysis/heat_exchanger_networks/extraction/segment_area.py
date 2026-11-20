"""Extract segment-resolved exchanger area contributions."""

from __future__ import annotations

from typing import Any

from ..solver.arrays import PreparedSolverArrays
from ..solver.piecewise import (
    branch_fractions,
    branch_profile,
    duty_aligned_area_contributions,
    profile_from_solver_arrays,
)
from .metadata import _index, _optional_float


def _recovery_segment_contributions(
    solved_model: Any,
    solver_arrays: PreparedSolverArrays,
    *,
    hot_index: int,
    cold_index: int,
    stage_index: int,
    tolerance: float,
):
    model_contributions = _model_recovery_segment_contributions(
        solved_model,
        hot_index=hot_index,
        cold_index=cold_index,
        stage_index=stage_index,
    )
    if model_contributions:
        return model_contributions
    arrays = solver_arrays.arrays
    required = {"hot_segment_count", "cold_segment_count"}
    if not required.issubset(arrays):
        return ()
    if (
        int(arrays["hot_segment_count"][hot_index]) == 1
        and int(arrays["cold_segment_count"][cold_index]) == 1
    ):
        return ()

    period_ids = [str(value) for value in arrays.get("period_ids", ["0"])]
    q_by_period = getattr(solved_model, "Q_r_by_period", None)
    hot_temperatures = getattr(solved_model, "T_h_by_period", None)
    cold_temperatures = getattr(solved_model, "T_c_by_period", None)
    contributions = []
    for period_index, period_id in enumerate(period_ids):
        duty_value = (
            _index(q_by_period, period_index, hot_index, cold_index, stage_index)
            if q_by_period is not None
            else _index(
                getattr(solved_model, "Q_r", None),
                hot_index,
                cold_index,
                stage_index,
            )
        )
        duty = _optional_float(duty_value) or 0.0
        if duty <= tolerance:
            continue
        hot_inlet = _optional_float(
            _index(hot_temperatures, period_index, hot_index, stage_index)
            if hot_temperatures is not None
            else _index(getattr(solved_model, "T_h", None), hot_index, stage_index)
        )
        cold_inlet = _optional_float(
            _index(cold_temperatures, period_index, cold_index, stage_index + 1)
            if cold_temperatures is not None
            else _index(getattr(solved_model, "T_c", None), cold_index, stage_index + 1)
        )
        if hot_inlet is None or cold_inlet is None:
            raise ValueError(
                "Segment-aware HEN extraction requires stage boundary temperatures."
            )
        hot_share, cold_share = _solved_splits(
            solved_model, period_index, hot_index, cold_index, stage_index
        ) or branch_fractions(
            lambda a, b, c, n=period_index: _match_duty(solved_model, n, a, b, c),
            len(arrays["hot_segment_count"]),
            len(arrays["cold_segment_count"]),
            hot_index,
            cold_index,
            stage_index,
        )
        contributions.extend(
            duty_aligned_area_contributions(
                branch_profile(
                    profile_from_solver_arrays(
                        solver_arrays,
                        side="hot",
                        parent_index=hot_index,
                        period_index=period_index,
                    ),
                    hot_share,
                ),
                branch_profile(
                    profile_from_solver_arrays(
                        solver_arrays,
                        side="cold",
                        parent_index=cold_index,
                        period_index=period_index,
                    ),
                    cold_share,
                ),
                duty=duty,
                hot_inlet_temperature=hot_inlet,
                cold_inlet_temperature=cold_inlet,
                period=period_id,
                tolerance=tolerance,
            )
        )
    return tuple(contributions)


def _solved_splits(solved_model: Any, n: int, i: int, j: int, k: int):
    """A non-isothermal model's solved hot and cold branch splits, or None."""
    if not getattr(solved_model, "non_isothermal_model", False):
        return None
    hot = _optional_float(
        _index(getattr(solved_model, "X_by_period", None), n, i, j, k)
    )
    cold = _optional_float(
        _index(getattr(solved_model, "Y_by_period", None), n, j, i, k)
    )
    if hot is None or cold is None:
        return None
    return (hot if hot > 0.0 else 1.0, cold if cold > 0.0 else 1.0)


def _match_duty(solved_model: Any, period_index: int, i: int, j: int, k: int) -> float:
    """A recovery match's solved duty in one period, 0 when unset."""
    by_period = getattr(solved_model, "Q_r_by_period", None)
    value = (
        _index(by_period, period_index, i, j, k)
        if by_period is not None
        else _index(getattr(solved_model, "Q_r", None), i, j, k)
    )
    return _optional_float(value) or 0.0


def _model_recovery_segment_contributions(
    solved_model: Any,
    *,
    hot_index: int,
    cold_index: int,
    stage_index: int,
):
    grid = getattr(solved_model, "segment_area_contributions_by_period", None)
    if grid is None:
        return ()
    contributions = []
    for period_index in range(len(grid)):
        values = _index(
            grid,
            period_index,
            hot_index,
            cold_index,
            stage_index,
        )
        if values:
            contributions.extend(values)
    return tuple(contributions)


def _model_utility_segment_contributions(
    solved_model: Any,
    attribute_name: str,
    parent_index: int,
    utility_index: int,
):
    grid = getattr(solved_model, attribute_name, None)
    if grid is None:
        return ()
    contributions = []
    for period_values in grid:
        values = _index(period_values, parent_index, utility_index)
        if values:
            contributions.extend(values)
    return tuple(contributions)
