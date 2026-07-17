"""Extract segment-resolved exchanger area contributions."""

from __future__ import annotations

from typing import Any

from ..solver.arrays import PreparedSolverArrays
from ..solver.piecewise import (
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
        contributions.extend(
            duty_aligned_area_contributions(
                profile_from_solver_arrays(
                    solver_arrays,
                    side="hot",
                    parent_index=hot_index,
                    period_index=period_index,
                ),
                profile_from_solver_arrays(
                    solver_arrays,
                    side="cold",
                    parent_index=cold_index,
                    period_index=period_index,
                ),
                duty=duty,
                hot_inlet_temperature=hot_inlet,
                cold_inlet_temperature=cold_inlet,
                period=period_id,
                tolerance=tolerance,
            )
        )
    return tuple(contributions)


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
):
    grid = getattr(solved_model, attribute_name, None)
    if grid is None:
        return ()
    contributions = []
    for period_values in grid:
        values = _index(period_values, parent_index)
        if values:
            contributions.extend(values)
    return tuple(contributions)
