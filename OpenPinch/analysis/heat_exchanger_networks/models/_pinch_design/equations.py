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


def _total_duty(grid):
    """Return the summed duty of a ``[stream][utility]`` grid."""
    return sum(load for row in grid for load in row)


def _set_utility_choice_objective(owner) -> None:
    """Price the utilities when a side has a choice of them.

    The utility targets fix how much utility each side uses, but not which
    utility: with more than one, their cost picks the cheaper feasible ones.
    """
    if owner.NHU <= 1 and owner.NCU <= 1:
        return
    owner.m.Minimize(
        _approach._weighted_state_average(
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
    )


def set_obj(owner) -> None:
    """Attach PDM objective expressions."""

    _set_utility_choice_objective(owner)

    if owner.minimisation_goal == "hot utility":
        for n in range(owner.N_periods):
            owner.m.Equation(
                _total_duty(owner.Q_h_by_period[n]) - owner.HU_target_by_period[n]
                >= 0.0
            )
        owner.m.Minimize(
            _approach._weighted_state_average(
                owner,
                [_total_duty(owner.Q_h_by_period[n]) for n in range(owner.N_periods)],
            )
        )
    elif owner.minimisation_goal == "cold utility":
        for n in range(owner.N_periods):
            owner.m.Equation(
                _total_duty(owner.Q_c_by_period[n]) - owner.CU_target_by_period[n]
                >= 0.0
            )
        owner.m.Minimize(
            _approach._weighted_state_average(
                owner,
                [_total_duty(owner.Q_c_by_period[n]) for n in range(owner.N_periods)],
            )
        )
    elif owner.minimisation_goal == "total utility":
        owner.m.Minimize(
            _approach._weighted_state_average(
                owner,
                [
                    _total_duty(owner.Q_h_by_period[n])
                    + _total_duty(owner.Q_c_by_period[n])
                    for n in range(owner.N_periods)
                ],
            )
        )
    elif owner.minimisation_goal == "heat recovery":
        owner.m.Maximize(
            _approach._weighted_state_average(
                owner,
                [
                    owner.m.sum(
                        [
                            owner.Q_r_by_period[n][i][j][k]
                            for i in range(owner.I)
                            for j in range(owner.J)
                            for k in range(owner.S)
                        ]
                    )
                    for n in range(owner.N_periods)
                ],
            )
        )
    elif owner.minimisation_goal == "min units":
        owner.m.Minimize(
            owner.m.sum(
                [
                    owner.z[i][j][k]
                    for k in range(owner.S)
                    for j in range(owner.J)
                    for i in range(owner.I)
                ]
            )
        )
    else:
        raise ValueError(
            "Unsupported pinch-decomposition minimisation goal: "
            f"{owner.minimisation_goal!r}."
        )
