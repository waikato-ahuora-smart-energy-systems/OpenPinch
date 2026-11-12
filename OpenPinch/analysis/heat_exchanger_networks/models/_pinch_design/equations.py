"""Pinch-decomposition superstructure and objective equations."""

from __future__ import annotations

from ...indexing import build_index_grid
from .._base import approach as _approach
from .._base import piecewise as _piecewise
from .._base.superstructure import create_match_binaries, create_utility_duty_grids


def set_stage_wise_superstructure(owner) -> None:
    """Create PDM variables, constraints, and binaries."""

    owner._set_multiperiod_stage_wise_superstructure()


def _set_multiperiod_stage_wise_superstructure(owner) -> None:
    owner.Q_r_by_period = build_index_grid(
        lambda n, i, j, k: (
            owner.m.Var(
                value=owner.Q_max_period[n][i][j] / 3,
                ub=owner.Q_max_period[n][i][j],
                lb=0.0,
                name=f"Q_H{i}_to_C{j}_at_S{k}_period{n}",
            )
            if owner.z_allowed[i][j][k] > 0
            else owner.m.Param(
                value=0.0,
                name=f"Q_H{i}_to_C{j}_at_S{k}_period{n}",
            )
        ),
        (owner.N_periods, owner.I, owner.J, owner.S),
    )
    create_utility_duty_grids(owner)
    owner.T_h_by_period = build_index_grid(
        lambda n, i, k: (
            owner.m.Var(
                value=owner.T_h_in_period[n][i],
                ub=owner.T_h_in_period[n][i],
                lb=owner.T_h_out_period[n][i],
                name=f"T_H{i}_at_B{k}_period{n}",
            )
            if k > 0 and owner.z_i_active[i] > 0
            else owner.m.Param(
                value=owner.T_h_in_period[n][i],
                name=f"T_H{i}_at_B{k}_period{n}",
            )
        ),
        (owner.N_periods, owner.I, owner.K),
    )
    owner.T_c_by_period = build_index_grid(
        lambda n, j, k: (
            owner.m.Var(
                value=owner.T_c_in_period[n][j],
                ub=owner.T_c_out_period[n][j],
                lb=owner.T_c_in_period[n][j],
                name=f"T_C{j}_at_B{k}_period{n}",
            )
            if k < owner.S and owner.z_j_active[j] > 0
            else owner.m.Param(
                value=owner.T_c_in_period[n][j],
                name=f"T_C{j}_at_B{k}_period{n}",
            )
        ),
        (owner.N_periods, owner.J, owner.K),
    )
    owner.Q_r = owner.Q_r_by_period[0]
    owner.Q_c = owner.Q_c_by_period[0]
    owner.Q_h = owner.Q_h_by_period[0]
    owner.T_h = owner.T_h_by_period[0]
    owner.T_c = owner.T_c_by_period[0]

    _piecewise._set_piecewise_stage_heat_coordinates(owner)

    for n in range(owner.N_periods):
        _ = [
            (
                owner.m.Equation(
                    owner.Qtot_sh_period[n][i]
                    - sum(
                        owner.Q_r_by_period[n][i][j][k]
                        for k in range(owner.S)
                        for j in range(owner.J)
                    )
                    - sum(owner.Q_c_by_period[n][i])
                    == 0.0
                )
                if owner.z_i_active_period[n][i] > 0
                else None
            )
            for i in range(owner.I)
        ]
        _ = [
            (
                owner.m.Equation(
                    owner.Qtot_sc_period[n][j]
                    - sum(
                        owner.Q_r_by_period[n][i][j][k]
                        for k in range(owner.S)
                        for i in range(owner.I)
                    )
                    - sum(owner.Q_h_by_period[n][j])
                    == 0.0
                )
                if owner.z_j_active_period[n][j] > 0
                else None
            )
            for j in range(owner.J)
        ]
        _ = [
            (
                owner.m.Equation(
                    (owner.T_h_by_period[n][i][k + 1] - owner.T_h_by_period[n][i][k])
                    * owner.f_h_period[n][i]
                    + sum(owner.Q_r_by_period[n][i][j][k] for j in range(owner.J))
                    == 0
                )
                if owner.z_i_active_period[n][i] > 0
                and not _piecewise._hot_parent_segmented(owner, i)
                else None
            )
            for k in range(owner.S)
            for i in range(owner.I)
        ]
        _ = [
            (
                owner.m.Equation(
                    (owner.T_c_by_period[n][j][k + 1] - owner.T_c_by_period[n][j][k])
                    * owner.f_c_period[n][j]
                    + sum(owner.Q_r_by_period[n][i][j][k] for i in range(owner.I))
                    == 0
                )
                if owner.z_j_active_period[n][j] > 0
                and not _piecewise._cold_parent_segmented(owner, j)
                else None
            )
            for k in range(owner.S)
            for j in range(owner.J)
        ]

    create_match_binaries(owner)
    if owner.integers:
        for n in range(owner.N_periods):
            _ = [
                (
                    owner.m.Equation(
                        owner.Q_r_by_period[n][i][j][k] * (1 - owner.z[i][j][k]) == 0.0
                    )
                    if owner.z_allowed[i][j][k] > 0
                    else None
                )
                for k in range(owner.S)
                for j in range(owner.J)
                for i in range(owner.I)
            ]
            _ = [
                (
                    owner.m.Equation(
                        owner.Q_c_by_period[n][i][u] * (1 - owner.z_cu[i][u]) == 0.0
                    )
                    if owner.z_cu_allowed[i][u] > 0
                    else None
                )
                for i in range(owner.I)
                for u in range(owner.NCU)
            ]
            _ = [
                (
                    owner.m.Equation(
                        owner.Q_h_by_period[n][j][u] * (1 - owner.z_hu[j][u]) == 0.0
                    )
                    if owner.z_hu_allowed[j][u] > 0
                    else None
                )
                for j in range(owner.J)
                for u in range(owner.NHU)
            ]

    _approach._set_multiperiod_utility_approach_equations(owner)

    M_ij_period = [
        [
            [
                max(
                    abs(owner.T_h_in_period[n][i] - owner.T_c_in_period[n][j]),
                    abs(owner.T_h_in_period[n][i] - owner.T_c_out_period[n][j]),
                    abs(owner.T_h_out_period[n][i] - owner.T_c_in_period[n][j]),
                    abs(owner.T_h_out_period[n][i] - owner.T_c_out_period[n][j]),
                )
                + _approach._recovery_approach_temperature(owner, i, j, n)
                for j in range(owner.J)
            ]
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]
    # A stream absent from this side of the pinch in period n is clipped to
    # 0 K there, so its matches carry no approach constraint in that period.
    for n in range(owner.N_periods):
        _ = [
            (
                owner.m.Equation(
                    (owner.T_h_by_period[n][i][k] - owner.T_c_by_period[n][j][k])
                    >= _approach._recovery_approach_temperature(owner, i, j, n)
                    - M_ij_period[n][i][j] * (1 - owner.z[i][j][k])
                )
                if owner.z_allowed[i][j][k] > 0
                and owner.z_i_active_period[n][i] > 0
                and owner.z_j_active_period[n][j] > 0
                else None
            )
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        ]
        _ = [
            (
                owner.m.Equation(
                    (
                        owner.T_h_by_period[n][i][k + 1]
                        - owner.T_c_by_period[n][j][k + 1]
                    )
                    >= _approach._recovery_approach_temperature(owner, i, j, n)
                    - M_ij_period[n][i][j] * (1 - owner.z[i][j][k])
                )
                if owner.z_allowed[i][j][k] > 0
                and owner.z_i_active_period[n][i] > 0
                and owner.z_j_active_period[n][j] > 0
                else None
            )
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        ]

    owner.dqda = []
    owner.alpha = []


UTILITY_GOALS = frozenset({"hot utility", "cold utility", "total utility"})
PDM_GOALS = UTILITY_GOALS | {"heat recovery", "min units"}


def _total_duty(grid):
    """Return the summed duty of a ``[stream][utility]`` grid."""
    return sum(load for row in grid for load in row)


def _has_utility_choice(owner) -> bool:
    return owner.NHU > 1 or owner.NCU > 1


def _utility_cost(owner):
    """Return the weighted utility cost of both sides.

    The utility targets fix how much utility each side uses, but not which
    utility: with more than one, their cost picks the cheaper feasible ones.
    """
    return _approach._weighted_state_average(
        owner,
        [
            sum(
                _piecewise._utility_cost_expression(
                    owner,
                    side,
                    n,
                    sum(row[u] for row in grid),
                    utility=u,
                    name=f"pdm_{side}_utility{u}_period_{n}",
                )
                for side, grid, count in (
                    ("hot", owner.Q_h_by_period[n], owner.NHU),
                    ("cold", owner.Q_c_by_period[n], owner.NCU),
                )
                for u in range(count)
            )
            for n in range(owner.N_periods)
        ],
    )


def _heat_recovery(owner, n: int):
    return owner.m.sum(
        [
            owner.Q_r_by_period[n][i][j][k]
            for i in range(owner.I)
            for j in range(owner.J)
            for k in range(owner.S)
        ]
    )


def _units(owner):
    return owner.m.sum(
        [
            owner.z[i][j][k]
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        ]
    )


def _set_utility_target(owner, duty_grids, targets) -> None:
    for n in range(owner.N_periods):
        owner.m.Equation(_total_duty(duty_grids[n]) - targets[n] >= 0.0)


def _goal_duty(owner, n: int):
    """Return the period-``n`` utility duty a utility goal minimises."""
    if owner.minimisation_goal == "hot utility":
        return _total_duty(owner.Q_h_by_period[n])
    if owner.minimisation_goal == "cold utility":
        return _total_duty(owner.Q_c_by_period[n])
    return _total_duty(owner.Q_h_by_period[n]) + _total_duty(owner.Q_c_by_period[n])


def set_obj(owner) -> None:
    """Attach the one PDM objective.

    With one hot and one cold utility the objective is the goal itself. With
    a choice of utilities on either side, the utility goals minimise utility
    cost under their targets (the targets already fix the duty), and heat
    recovery and min units add utility cost to the goal in one minimised
    objective. Heat recovery is then expressed as unrecovered hot-stream
    heat, which keeps the objective non-negative.
    """

    goal = owner.minimisation_goal
    if goal not in PDM_GOALS:
        raise ValueError(
            f"Unsupported pinch-decomposition minimisation goal: {goal!r}."
        )
    periods = range(owner.N_periods)
    choice = _has_utility_choice(owner)

    if goal == "hot utility":
        _set_utility_target(owner, owner.Q_h_by_period, owner.HU_target_by_period)
    elif goal == "cold utility":
        _set_utility_target(owner, owner.Q_c_by_period, owner.CU_target_by_period)

    if goal in UTILITY_GOALS:
        owner.m.Minimize(
            _utility_cost(owner)
            if choice
            else _approach._weighted_state_average(
                owner, [_goal_duty(owner, n) for n in periods]
            )
        )
    elif goal == "heat recovery" and not choice:
        owner.m.Maximize(
            _approach._weighted_state_average(
                owner, [_heat_recovery(owner, n) for n in periods]
            )
        )
    elif goal == "heat recovery":
        owner.m.Minimize(
            _utility_cost(owner)
            + _approach._weighted_state_average(
                owner,
                [
                    sum(owner.Qtot_sh_period[n]) - _heat_recovery(owner, n)
                    for n in periods
                ],
            )
        )
    else:
        owner.m.Minimize(
            _utility_cost(owner) + _units(owner) if choice else _units(owner)
        )
