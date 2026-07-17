"""Pinch-decomposition superstructure and objective equations."""

from __future__ import annotations

from ...indexing import build_index_grid


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
    owner.Q_c_by_period = build_index_grid(
        lambda n, i: (
            owner.m.Var(
                value=0,
                ub=owner.Qtot_sh_period[n][i],
                lb=0.0,
                name=f"Q_H{i}_to_CU_period{n}",
            )
            if owner.z_cu_allowed[i] > 0
            else owner.m.Param(value=0, name=f"Q_H{i}_to_CU_period{n}")
        ),
        (owner.N_periods, owner.I),
    )
    owner.Q_h_by_period = build_index_grid(
        lambda n, j: (
            owner.m.Var(
                value=0,
                ub=owner.Qtot_sc_period[n][j],
                lb=0.0,
                name=f"Q_HU_to_C{j}_period{n}",
            )
            if owner.z_hu_allowed[j] > 0
            else owner.m.Param(value=0, name=f"Q_HU_to_C{j}_period{n}")
        ),
        (owner.N_periods, owner.J),
    )
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

    owner._set_piecewise_stage_heat_coordinates()

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
                    - owner.Q_c_by_period[n][i]
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
                    - owner.Q_h_by_period[n][j]
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
                and not owner._hot_parent_segmented(i)
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
                and not owner._cold_parent_segmented(j)
                else None
            )
            for k in range(owner.S)
            for j in range(owner.J)
        ]

    if owner.integers:
        owner.z = [
            [
                [
                    (
                        owner.m.Var(
                            value=1,
                            ub=1,
                            lb=0,
                            integer=True,
                            name=f"z_H{i}_to_C{j}_at_S{k}",
                        )
                        if owner.z_allowed[i][j][k] > 0
                        else owner.m.Param(value=0, name=f"z_H{i}_to_C{j}_at_S{k}")
                    )
                    for k in range(owner.S)
                ]
                for j in range(owner.J)
            ]
            for i in range(owner.I)
        ]
        owner.z_cu = [
            (
                owner.m.Var(
                    value=1,
                    ub=1,
                    lb=0,
                    integer=True,
                    name=f"z_H{i}_to_CU",
                )
                if owner.z_cu_allowed[i] > 0
                else owner.m.Param(value=0, name=f"z_H{i}_to_CU")
            )
            for i in range(owner.I)
        ]
        owner.z_hu = [
            (
                owner.m.Var(
                    value=1,
                    ub=1,
                    lb=0,
                    integer=True,
                    name=f"z_HU_to_C{j}",
                )
                if owner.z_hu_allowed[j] > 0
                else owner.m.Param(value=0, name=f"z_HU_to_C{j}")
            )
            for j in range(owner.J)
        ]
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
                        owner.Q_c_by_period[n][i] * (1 - owner.z_cu[i]) == 0.0
                    )
                    if owner.z_cu_allowed[i] > 0
                    else None
                )
                for i in range(owner.I)
            ]
            _ = [
                (
                    owner.m.Equation(
                        owner.Q_h_by_period[n][j] * (1 - owner.z_hu[j]) == 0.0
                    )
                    if owner.z_hu_allowed[j] > 0
                    else None
                )
                for j in range(owner.J)
            ]
    else:
        owner.z = [
            [
                [
                    (
                        owner.m.Param(value=1, name=f"z_H{i}_to_C{j}_at_S{k}")
                        if owner.z_allowed[i][j][k] > 0
                        else owner.m.Param(value=0, name=f"z_H{i}_to_C{j}_at_S{k}")
                    )
                    for k in range(owner.S)
                ]
                for j in range(owner.J)
            ]
            for i in range(owner.I)
        ]
        owner.z_hu = [
            (
                owner.m.Param(value=1, name=f"z_HU_to_C{j}")
                if owner.z_hu_allowed[j] > 0
                else owner.m.Param(value=0, name=f"z_HU_to_C{j}")
            )
            for j in range(owner.J)
        ]
        owner.z_cu = [
            (
                owner.m.Param(value=1, name=f"z_H{i}_to_CU")
                if owner.z_cu_allowed[i] > 0
                else owner.m.Param(value=0, name=f"z_H{i}_to_CU")
            )
            for i in range(owner.I)
        ]

    owner._set_multiperiod_utility_approach_equations()

    M_ij_period = [
        [
            [
                max(
                    abs(owner.T_h_in_period[n][i] - owner.T_c_in_period[n][j]),
                    abs(owner.T_h_in_period[n][i] - owner.T_c_out_period[n][j]),
                    abs(owner.T_h_out_period[n][i] - owner.T_c_in_period[n][j]),
                    abs(owner.T_h_out_period[n][i] - owner.T_c_out_period[n][j]),
                )
                + owner._recovery_approach_temperature(i, j, n)
                for j in range(owner.J)
            ]
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]
    for n in range(owner.N_periods):
        _ = [
            (
                owner.m.Equation(
                    (owner.T_h_by_period[n][i][k] - owner.T_c_by_period[n][j][k])
                    >= owner._recovery_approach_temperature(i, j, n)
                    - M_ij_period[n][i][j] * (1 - owner.z[i][j][k])
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
                    (
                        owner.T_h_by_period[n][i][k + 1]
                        - owner.T_c_by_period[n][j][k + 1]
                    )
                    >= owner._recovery_approach_temperature(i, j, n)
                    - M_ij_period[n][i][j] * (1 - owner.z[i][j][k])
                )
                if owner.z_allowed[i][j][k] > 0
                else None
            )
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        ]

    owner.dqda = []
    owner.alpha = []


def set_obj(owner) -> None:
    """Attach PDM objective expressions."""

    if owner.minimisation_goal == "hot utility":
        for n in range(owner.N_periods):
            owner.m.Equation(
                sum(owner.Q_h_by_period[n]) - owner.HU_target_by_period[n] >= 0.0
            )
        owner.m.Minimize(
            owner._weighted_state_average(
                [sum(owner.Q_h_by_period[n]) for n in range(owner.N_periods)]
            )
        )
    elif owner.minimisation_goal == "cold utility":
        for n in range(owner.N_periods):
            owner.m.Equation(
                sum(owner.Q_c_by_period[n]) - owner.CU_target_by_period[n] >= 0.0
            )
        owner.m.Minimize(
            owner._weighted_state_average(
                [sum(owner.Q_c_by_period[n]) for n in range(owner.N_periods)]
            )
        )
    elif owner.minimisation_goal == "total utility":
        owner.m.Minimize(
            owner._weighted_state_average(
                [
                    sum(owner.Q_h_by_period[n]) + sum(owner.Q_c_by_period[n])
                    for n in range(owner.N_periods)
                ]
            )
        )
    elif owner.minimisation_goal == "heat recovery":
        owner.m.Maximize(
            owner._weighted_state_average(
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
                ]
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
