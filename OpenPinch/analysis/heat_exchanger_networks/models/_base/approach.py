"""Approach-temperature and match-restriction equations for base HEN models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _set_minimum_approach_temperatures(model) -> None:
    """Derive pair-specific approach limits from stream contributions."""

    model.dT_r_period = np.array(
        [
            [
                [
                    model.T_h_cont_period[n][i] + model.T_c_cont_period[n][j]
                    for j in range(len(model.T_c_cont_period[n]))
                ]
                for i in range(len(model.T_h_cont_period[n]))
            ]
            for n in range(model.N_periods)
        ],
        dtype=float,
    )
    model.dT_hu_period = np.array(
        [
            [
                (
                    model.T_hu_cont_period[n][0]
                    if len(model.T_hu_cont_period[n])
                    else model.dTmin / 2.0
                )
                + model.T_c_cont_period[n][j]
                for j in range(len(model.T_c_cont_period[n]))
            ]
            for n in range(model.N_periods)
        ],
        dtype=float,
    )
    model.dT_cu_period = np.array(
        [
            [
                model.T_h_cont_period[n][i]
                + (
                    model.T_cu_cont_period[n][0]
                    if len(model.T_cu_cont_period[n])
                    else model.dTmin / 2.0
                )
                for i in range(len(model.T_h_cont_period[n]))
            ]
            for n in range(model.N_periods)
        ],
        dtype=float,
    )
    model.dT_r = model.dT_r_period[0].copy()
    model.dT_hu = model.dT_hu_period[0].copy()
    model.dT_cu = model.dT_cu_period[0].copy()


def _recovery_approach_temperature(
    model,
    i: int,
    j: int,
    period_idx: int = 0,
) -> float:
    if not hasattr(model, "dT_r"):
        return float(model.dTmin)
    if hasattr(model, "dT_r_period"):
        return float(model.dT_r_period[period_idx][i][j])
    return float(model.dT_r[i][j])


def _hot_utility_inlet_approach_temperature(
    model,
    j: int,
    period_idx: int = 0,
) -> float:
    contribution = (
        model.T_hu_in_cont_by_period[period_idx]
        if hasattr(model, "T_hu_in_cont_by_period")
        else model.T_hu_cont_period[period_idx][0]
    )
    return float(contribution + model.T_c_cont_period[period_idx][j])


def _hot_utility_outlet_approach_temperature(
    model,
    j: int,
    period_idx: int = 0,
    heat_duty: float | None = None,
):
    contribution = model._utility_outlet_temperature_contribution(
        "hot",
        period_idx,
        match_index=j,
        heat_duty=heat_duty,
    )
    return contribution + model.T_c_cont_period[period_idx][j]


def _cold_utility_inlet_approach_temperature(
    model,
    i: int,
    period_idx: int = 0,
) -> float:
    contribution = (
        model.T_cu_in_cont_by_period[period_idx]
        if hasattr(model, "T_cu_in_cont_by_period")
        else model.T_cu_cont_period[period_idx][0]
    )
    return float(model.T_h_cont_period[period_idx][i] + contribution)


def _cold_utility_outlet_approach_temperature(
    model,
    i: int,
    period_idx: int = 0,
    heat_duty: float | None = None,
):
    contribution = model._utility_outlet_temperature_contribution(
        "cold",
        period_idx,
        match_index=i,
        heat_duty=heat_duty,
    )
    return model.T_h_cont_period[period_idx][i] + contribution


def _utility_outlet_temperature_contribution(
    model,
    side: str,
    period_idx: int,
    match_index: int,
    heat_duty: float | None = None,
):
    if heat_duty is None and hasattr(
        model,
        f"T_{side[0]}u_out_cont_by_period",
    ):
        return getattr(model, f"T_{side[0]}u_out_cont_by_period")[period_idx][
            match_index
        ]
    scalar = (
        model.T_hu_cont_period[period_idx][0]
        if side == "hot"
        else model.T_cu_cont_period[period_idx][0]
    )
    if heat_duty is None or not model._utility_is_segmented(side):
        return float(scalar)

    from ...solver.piecewise import profile_from_solver_arrays

    profile = profile_from_solver_arrays(
        model.solver_arrays,
        side=f"{side}_utility",
        parent_index=0,
        period_index=period_idx,
    )
    return profile.temperature_contribution_at_heat(heat_duty)


def _utility_solved_outlet_temperature(
    model,
    side: str,
    period_idx: int,
    match_index: int,
    heat_duty: float | None = None,
):
    if heat_duty is None and hasattr(
        model,
        f"T_{side[0]}u_solved_out_by_period",
    ):
        return getattr(model, f"T_{side[0]}u_solved_out_by_period")[period_idx][
            match_index
        ]
    if heat_duty is None or not model._utility_is_segmented(side):
        return (
            model.T_hu_out_period[period_idx][0]
            if side == "hot"
            else model.T_cu_out_period[period_idx][0]
        )

    from ...solver.piecewise import profile_from_solver_arrays

    return profile_from_solver_arrays(
        model.solver_arrays,
        side=f"{side}_utility",
        parent_index=0,
        period_index=period_idx,
    ).temperature_at_heat(heat_duty)


def _utility_max_temperature_contribution(
    model,
    side: str,
    period_idx: int,
) -> float:
    scalar = (
        model.T_hu_cont_period[period_idx][0]
        if side == "hot"
        else model.T_cu_cont_period[period_idx][0]
    )
    if not model._utility_is_segmented(side):
        return float(scalar)
    values = model.solver_arrays.arrays[f"{side}_utility_segment_dt_cont_period"][
        period_idx, 0
    ]
    count = int(model.solver_arrays.arrays[f"{side}_utility_segment_count"][0])
    return float(np.max(values[:count]))


def _set_multiperiod_utility_approach_equations(model) -> None:
    """Constrain both utility terminals with local segment contributions."""
    for n in range(model.N_periods):
        for j in range(model.J):
            if model.z_hu_allowed[j] <= 0 or not model._utility_is_segmented("hot"):
                continue
            inlet_approach = model._hot_utility_inlet_approach_temperature(j, n)
            outlet_approach = model._hot_utility_outlet_approach_temperature(j, n)
            maximum_approach = (
                model._utility_max_temperature_contribution("hot", n)
                + model.T_c_cont_period[n][j]
            )
            big_m = max(
                abs(model.T_hu_in_period[n][0] - model.T_c_out_period[n][j]),
                abs(model.T_hu_in_period[n][0] - model.T_c_in_period[n][j]),
                abs(model.T_hu_out_period[n][0] - model.T_c_out_period[n][j]),
                abs(model.T_hu_out_period[n][0] - model.T_c_in_period[n][j]),
            ) + max(maximum_approach, float(model.dTmin))
            inlet_delta = model.T_hu_in_period[n][0] - model.T_c_out_period[n][j]
            if type(model.z_hu[j]).__name__ == "GKParameter":
                if (
                    model._solver_value(model.z_hu[j].VALUE.value) > model.tol
                    and inlet_delta + model.tol < inlet_approach
                ):
                    raise ValueError(
                        f"Hot utility match {j} violates its inlet approach "
                        f"temperature in period {model.period_ids[n]!r}."
                    )
            else:
                model.m.Equation(
                    inlet_delta >= inlet_approach - big_m * (1 - model.z_hu[j])
                )
            model.m.Equation(
                model._utility_solved_outlet_temperature("hot", n, j)
                - model.T_c_by_period[n][j][0]
                >= outlet_approach - big_m * (1 - model.z_hu[j])
            )

        for i in range(model.I):
            if model.z_cu_allowed[i] <= 0 or not model._utility_is_segmented("cold"):
                continue
            inlet_approach = model._cold_utility_inlet_approach_temperature(i, n)
            outlet_approach = model._cold_utility_outlet_approach_temperature(i, n)
            maximum_approach = model.T_h_cont_period[n][
                i
            ] + model._utility_max_temperature_contribution("cold", n)
            big_m = max(
                abs(model.T_h_in_period[n][i] - model.T_cu_out_period[n][0]),
                abs(model.T_h_in_period[n][i] - model.T_cu_in_period[n][0]),
                abs(model.T_h_out_period[n][i] - model.T_cu_out_period[n][0]),
                abs(model.T_h_out_period[n][i] - model.T_cu_in_period[n][0]),
            ) + max(maximum_approach, float(model.dTmin))
            model.m.Equation(
                model.T_h_by_period[n][i][model.S]
                - model._utility_solved_outlet_temperature("cold", n, i)
                >= outlet_approach - big_m * (1 - model.z_cu[i])
            )
            inlet_delta = model.T_h_out_period[n][i] - model.T_cu_in_period[n][0]
            if type(model.z_cu[i]).__name__ == "GKParameter":
                if (
                    model._solver_value(model.z_cu[i].VALUE.value) > model.tol
                    and inlet_delta + model.tol < inlet_approach
                ):
                    raise ValueError(
                        f"Cold utility match {i} violates its inlet approach "
                        f"temperature in period {model.period_ids[n]!r}."
                    )
            else:
                model.m.Equation(
                    inlet_delta >= inlet_approach - big_m * (1 - model.z_cu[i])
                )


def _weighted_state_average(model, values: Sequence[Any]) -> Any:
    """Return ``sum_s(w_s * value_s) / sum_s(w_s)`` for GEKKO expressions."""

    return (
        sum(float(model.period_weights[n]) * values[n] for n in range(model.N_periods))
        / model.period_weight_sum
    )


def set_match_restrictions(model, restrictions) -> None:
    """Apply inherited topology restrictions in the source array shape."""

    if restrictions is None:
        restrictions = [None, None, None]
    z_restriction, zhu_restriction, zcu_restriction = (
        restrictions[0],
        restrictions[1],
        restrictions[2],
    )

    if z_restriction is not None:
        if isinstance(z_restriction[0][0][0], int):
            model.z_allowed = [
                [
                    [
                        1 if z_restriction[i][j][k] > model.tol else 0
                        for k in range(model.S)
                    ]
                    for j in range(model.J)
                ]
                for i in range(model.I)
            ]
        elif isinstance(z_restriction[0][0][0], list):
            model.z_allowed = [
                [
                    [
                        1 if z_restriction[i][j][k][0] > model.tol else 0
                        for k in range(model.S)
                    ]
                    for j in range(model.J)
                ]
                for i in range(model.I)
            ]
        elif type(z_restriction[0][0][0]).__name__ in {
            "GKVariable",
            "GKParameter",
        }:
            model.z_allowed = [
                [
                    [
                        1 if z_restriction[i][j][k][0] > model.tol else 0
                        for k in range(model.S)
                    ]
                    for j in range(model.J)
                ]
                for i in range(model.I)
            ]
        else:
            raise ValueError("Invalid restriction type")
    else:
        model.z_allowed = model.z_feasible

    if zhu_restriction is not None:
        if isinstance(zhu_restriction[0], int):
            model.z_hu_allowed = [
                1 if zhu_restriction[j] > model.tol else 0 for j in range(model.J)
            ]
        else:
            model.z_hu_allowed = [
                1 if zhu_restriction[j][0] > model.tol else 0 for j in range(model.J)
            ]
    else:
        model.z_hu_allowed = model.z_hu_feasible

    if zcu_restriction is not None:
        if isinstance(zcu_restriction[0], int):
            model.z_cu_allowed = [
                1 if zcu_restriction[i] > model.tol else 0 for i in range(model.I)
            ]
        else:
            model.z_cu_allowed = [
                1 if zcu_restriction[i][0] > model.tol else 0 for i in range(model.I)
            ]
    else:
        model.z_cu_allowed = model.z_cu_feasible
