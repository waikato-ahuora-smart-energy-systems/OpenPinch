"""Approach-temperature and match-restriction equations for base HEN models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from . import execution as _execution
from . import piecewise as _piecewise


def _set_minimum_approach_temperatures(model) -> None:
    """Derive pair-specific approach limits from stream contributions.

    Heaters are ``dT_hu_period[period][utility][cold]`` and coolers
    ``dT_cu_period[period][hot][utility]``, the orientation of the recovery
    pairs. A model with no utility on a side gets one at ``dTmin / 2``.
    """

    def contributions(values):
        return [float(value) for value in values] or [model.dTmin / 2.0]

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
                [utility + cold for cold in model.T_c_cont_period[n]]
                for utility in contributions(model.T_hu_cont_period[n])
            ]
            for n in range(model.N_periods)
        ],
        dtype=float,
    )
    model.dT_cu_period = np.array(
        [
            [
                [hot + utility for utility in contributions(model.T_cu_cont_period[n])]
                for hot in model.T_h_cont_period[n]
            ]
            for n in range(model.N_periods)
        ],
        dtype=float,
    )
    model.dT_hu_min_period = utility_minimum_approach_by_period(model, "hot")
    model.dT_cu_min_period = utility_minimum_approach_by_period(model, "cold")
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
    u: int,
    period_idx: int,
) -> float:
    contribution = (
        model.T_hu_in_cont_by_period[period_idx][u]
        if hasattr(model, "T_hu_in_cont_by_period")
        else model.T_hu_cont_period[period_idx][u]
    )
    return float(contribution + model.T_c_cont_period[period_idx][j])


def _hot_utility_outlet_approach_temperature(
    model,
    j: int,
    u: int,
    period_idx: int,
    heat_duty: float | None = None,
):
    contribution = _utility_outlet_temperature_contribution(
        model,
        "hot",
        period_idx,
        match_index=j,
        utility=u,
        heat_duty=heat_duty,
    )
    return contribution + model.T_c_cont_period[period_idx][j]


def _cold_utility_inlet_approach_temperature(
    model,
    i: int,
    u: int,
    period_idx: int,
) -> float:
    contribution = (
        model.T_cu_in_cont_by_period[period_idx][u]
        if hasattr(model, "T_cu_in_cont_by_period")
        else model.T_cu_cont_period[period_idx][u]
    )
    return float(model.T_h_cont_period[period_idx][i] + contribution)


def _cold_utility_outlet_approach_temperature(
    model,
    i: int,
    u: int,
    period_idx: int,
    heat_duty: float | None = None,
):
    contribution = _utility_outlet_temperature_contribution(
        model,
        "cold",
        period_idx,
        match_index=i,
        utility=u,
        heat_duty=heat_duty,
    )
    return model.T_h_cont_period[period_idx][i] + contribution


def _utility_outlet_temperature_contribution(
    model,
    side: str,
    period_idx: int,
    match_index: int,
    utility: int,
    heat_duty: float | None = None,
):
    if heat_duty is None and hasattr(
        model,
        f"T_{side[0]}u_out_cont_by_period",
    ):
        return getattr(model, f"T_{side[0]}u_out_cont_by_period")[period_idx][utility][
            match_index
        ]
    scalar = (
        model.T_hu_cont_period[period_idx][utility]
        if side == "hot"
        else model.T_cu_cont_period[period_idx][utility]
    )
    if heat_duty is None or not _piecewise._utility_is_segmented(model, side, utility):
        return float(scalar)

    from ...solver.piecewise import profile_from_solver_arrays

    profile = profile_from_solver_arrays(
        model.solver_arrays,
        side=f"{side}_utility",
        parent_index=utility,
        period_index=period_idx,
    )
    return profile.temperature_contribution_at_heat(heat_duty)


def _utility_solved_outlet_temperature(
    model,
    side: str,
    period_idx: int,
    match_index: int,
    utility: int,
    heat_duty: float | None = None,
):
    if heat_duty is None and hasattr(
        model,
        f"T_{side[0]}u_solved_out_by_period",
    ):
        return getattr(model, f"T_{side[0]}u_solved_out_by_period")[period_idx][
            utility
        ][match_index]
    if heat_duty is None or not _piecewise._utility_is_segmented(model, side, utility):
        return (
            model.T_hu_out_period[period_idx][utility]
            if side == "hot"
            else model.T_cu_out_period[period_idx][utility]
        )

    from ...solver.piecewise import profile_from_solver_arrays

    return profile_from_solver_arrays(
        model.solver_arrays,
        side=f"{side}_utility",
        parent_index=utility,
        period_index=period_idx,
    ).temperature_at_heat(heat_duty)


def _utility_segment_contributions(model, side: str, period_idx: int, u: int):
    values = model.solver_arrays.arrays[f"{side}_utility_segment_dt_cont_period"][
        period_idx, u
    ]
    count = int(model.solver_arrays.arrays[f"{side}_utility_segment_count"][u])
    return values[:count]


def _utility_max_temperature_contribution(
    model,
    side: str,
    period_idx: int,
    u: int,
) -> float:
    if not _piecewise._utility_is_segmented(model, side, u):
        return float(
            model.T_hu_cont_period[period_idx][u]
            if side == "hot"
            else model.T_cu_cont_period[period_idx][u]
        )
    return float(np.max(_utility_segment_contributions(model, side, period_idx, u)))


def utility_minimum_approach_by_period(model, side: str) -> list:
    """Return the least approach any heater or cooler may have, by period.

    Hot utilities give ``[period][utility][cold stream]`` and cold utilities
    ``[period][hot stream][utility]``, the recovery pairs' orientation. A
    segmented utility's approach varies along it, so its least segment
    contribution bounds every terminal.
    """
    by_period = []
    for n in range(model.N_periods):
        if side == "hot":
            rows = [[float(value) for value in row] for row in model.dT_hu_period[n]]
            for u in range(len(rows)):
                if _piecewise._utility_is_segmented(model, "hot", u):
                    least = float(
                        np.min(_utility_segment_contributions(model, "hot", n, u))
                    )
                    rows[u] = [least + float(c) for c in model.T_c_cont_period[n]]
        else:
            rows = [[float(value) for value in row] for row in model.dT_cu_period[n]]
            for u in range(len(rows[0]) if rows else 0):
                if _piecewise._utility_is_segmented(model, "cold", u):
                    least = float(
                        np.min(_utility_segment_contributions(model, "cold", n, u))
                    )
                    for i, hot in enumerate(model.T_h_cont_period[n]):
                        rows[i][u] = float(hot) + least
        by_period.append(rows)
    return by_period


def utility_match_feasibility(
    owner,
    active_periods=None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Flag each (stream, utility) pair whose fixed end can meet its approach.

    A heater's utility inlet faces the cold stream's target, and a cooler's
    utility inlet the hot stream's target. Both temperatures are fixed, so a
    pair below its least approach there in any period the stream runs can
    never be used. ``active_periods(side, index)`` names those periods.
    """
    periods = active_periods or (lambda _side, _index: range(owner.N_periods))
    hot = [
        [
            int(
                all(
                    owner.T_hu_in_period[n][u] - owner.T_c_out_period[n][j]
                    >= owner.dT_hu_min_period[n][u][j] - owner.tol
                    for n in periods("cold", j)
                )
            )
            for u in range(owner.NHU)
        ]
        for j in range(owner.J)
    ]
    cold = [
        [
            int(
                all(
                    owner.T_h_out_period[n][i] - owner.T_cu_in_period[n][u]
                    >= owner.dT_cu_min_period[n][i][u] - owner.tol
                    for n in periods("hot", i)
                )
            )
            for u in range(owner.NCU)
        ]
        for i in range(owner.I)
    ]
    return hot, cold


def _stream_active(model, side: str, period_idx: int, index: int) -> bool:
    """Return whether a stream takes part in period ``period_idx``.

    Pinch-decomposition models clip a stream absent from their side of the
    pinch in a period to 0 K; other models have every stream in every period.
    """
    flags = getattr(model, f"z_{'i' if side == 'hot' else 'j'}_active_period", None)
    return flags is None or flags[period_idx][index] > 0


def _set_multiperiod_utility_approach_equations(model) -> None:
    """Constrain both terminals of every heater and cooler.

    Each (stream, utility) pair is its own unit. Segmented utilities use the
    local segment contribution at each terminal; flat utilities use their
    single contribution.
    """
    for n in range(model.N_periods):
        for j in range(model.J):
            if not _stream_active(model, "cold", n, j):
                continue
            for u in range(model.NHU):
                if model.z_hu_allowed[j][u] <= 0:
                    continue
                _set_hot_utility_approach_equations(model, n, j, u)
        for i in range(model.I):
            if not _stream_active(model, "hot", n, i):
                continue
            for u in range(model.NCU):
                if model.z_cu_allowed[i][u] <= 0:
                    continue
                _set_cold_utility_approach_equations(model, n, i, u)


def _set_hot_utility_approach_equations(model, n: int, j: int, u: int) -> None:
    inlet_approach = _hot_utility_inlet_approach_temperature(model, j, u, n)
    outlet_approach = _hot_utility_outlet_approach_temperature(model, j, u, n)
    maximum_approach = (
        _utility_max_temperature_contribution(model, "hot", n, u)
        + model.T_c_cont_period[n][j]
    )
    big_m = max(
        abs(model.T_hu_in_period[n][u] - model.T_c_out_period[n][j]),
        abs(model.T_hu_in_period[n][u] - model.T_c_in_period[n][j]),
        abs(model.T_hu_out_period[n][u] - model.T_c_out_period[n][j]),
        abs(model.T_hu_out_period[n][u] - model.T_c_in_period[n][j]),
    ) + max(maximum_approach, float(model.dTmin))
    z = model.z_hu[j][u]
    inlet_delta = model.T_hu_in_period[n][u] - model.T_c_out_period[n][j]
    if type(z).__name__ == "GKParameter":
        if (
            _execution._solver_value(model, z.VALUE.value) > model.tol
            and inlet_delta + model.tol < inlet_approach
        ):
            raise ValueError(
                f"Hot utility {u} match {j} violates its inlet approach "
                f"temperature in period {model.period_ids[n]!r}."
            )
    else:
        model.m.Equation(inlet_delta >= inlet_approach - big_m * (1 - z))
    model.m.Equation(
        model._utility_solved_outlet_temperature("hot", n, j, u)
        - model.T_c_by_period[n][j][0]
        >= outlet_approach - big_m * (1 - z)
    )


def _set_cold_utility_approach_equations(model, n: int, i: int, u: int) -> None:
    inlet_approach = _cold_utility_inlet_approach_temperature(model, i, u, n)
    outlet_approach = _cold_utility_outlet_approach_temperature(model, i, u, n)
    maximum_approach = model.T_h_cont_period[n][
        i
    ] + _utility_max_temperature_contribution(model, "cold", n, u)
    big_m = max(
        abs(model.T_h_in_period[n][i] - model.T_cu_out_period[n][u]),
        abs(model.T_h_in_period[n][i] - model.T_cu_in_period[n][u]),
        abs(model.T_h_out_period[n][i] - model.T_cu_out_period[n][u]),
        abs(model.T_h_out_period[n][i] - model.T_cu_in_period[n][u]),
    ) + max(maximum_approach, float(model.dTmin))
    z = model.z_cu[i][u]
    model.m.Equation(
        model.T_h_by_period[n][i][model.S]
        - model._utility_solved_outlet_temperature("cold", n, i, u)
        >= outlet_approach - big_m * (1 - z)
    )
    inlet_delta = model.T_h_out_period[n][i] - model.T_cu_in_period[n][u]
    if type(z).__name__ == "GKParameter":
        if (
            _execution._solver_value(model, z.VALUE.value) > model.tol
            and inlet_delta + model.tol < inlet_approach
        ):
            raise ValueError(
                f"Cold utility {u} match {i} violates its inlet approach "
                f"temperature in period {model.period_ids[n]!r}."
            )
    else:
        model.m.Equation(inlet_delta >= inlet_approach - big_m * (1 - z))


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

    model.z_hu_allowed = (
        _utility_restriction(model, zhu_restriction, model.J, model.NHU)
        if zhu_restriction is not None
        else model.z_hu_feasible
    )
    model.z_cu_allowed = (
        _utility_restriction(model, zcu_restriction, model.I, model.NCU)
        if zcu_restriction is not None
        else model.z_cu_feasible
    )


def _utility_restriction(model, restriction, streams: int, utilities: int):
    """Return ``[stream][utility]`` 0/1 flags from solved or given values."""

    def flag(value) -> int:
        number = value if isinstance(value, int | float) else value[0]
        return 1 if number > model.tol else 0

    return [
        [flag(restriction[index][u]) for u in range(utilities)]
        for index in range(streams)
    ]
