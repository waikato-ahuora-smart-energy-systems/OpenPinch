"""StageWise superstructure and dQ/dA equation construction."""

from __future__ import annotations

from ...indexing import build_index_grid


def set_stage_wise_superstructure(owner) -> None:
    """Create StageWise variables, constraints, and binaries."""

    owner._set_multiperiod_stage_wise_superstructure()


def _set_multiperiod_stage_wise_superstructure(owner) -> None:
    """Create shared topology with state-indexed operating variables."""

    owner.Q_r_by_period = [
        [
            [
                [
                    (
                        owner.m.Var(
                            value=owner.Q_max_period[n][i][j] / 3,
                            ub=owner.Q_max_period[n][i][j],
                            lb=0.0,
                            name=(f"Q_H{i}_to_C{j}_at_S{k}_period{n}"),
                        )
                        if owner.z_allowed[i][j][k] > 0
                        else owner.m.Param(
                            value=0.0,
                            name=f"Q_H{i}_to_C{j}_at_S{k}_period{n}",
                        )
                    )
                    for k in range(owner.S)
                ]
                for j in range(owner.J)
            ]
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]
    owner.Q_c_by_period = [
        [
            (
                owner.m.Var(
                    value=0,
                    ub=owner.Qtot_sh_period[n][i],
                    lb=0.0,
                    name=f"Q_H{i}_to_CU_period{n}",
                )
                if owner.z_cu_allowed[i] > 0
                else owner.m.Param(value=0, name=f"Q_H{i}_to_CU_period{n}")
            )
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]
    owner.Q_h_by_period = [
        [
            (
                owner.m.Var(
                    value=0,
                    ub=owner.Qtot_sc_period[n][j],
                    lb=0.0,
                    name=f"Q_HU_to_C{j}_period{n}",
                )
                if owner.z_hu_allowed[j] > 0
                else owner.m.Param(value=0, name=f"Q_HU_to_C{j}_period{n}")
            )
            for j in range(owner.J)
        ]
        for n in range(owner.N_periods)
    ]
    owner.T_h_by_period = [
        [
            [
                (
                    owner.m.Var(
                        value=owner.T_h_in_period[n][i],
                        ub=owner.T_h_in_period[n][i],
                        lb=owner.T_h_out_period[n][i],
                        name=f"T_H{i}_at_B{k}_period{n}",
                    )
                    if k > 0
                    else owner.m.Param(
                        value=owner.T_h_in_period[n][i],
                        name=f"T_H{i}_at_B{k}_period{n}",
                    )
                )
                for k in range(owner.K)
            ]
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]
    owner.T_c_by_period = [
        [
            [
                (
                    owner.m.Var(
                        value=owner.T_c_in_period[n][j],
                        ub=owner.T_c_out_period[n][j],
                        lb=owner.T_c_in_period[n][j],
                        name=f"T_C{j}_at_B{k}_period{n}",
                    )
                    if k < owner.S
                    else owner.m.Param(
                        value=owner.T_c_in_period[n][j],
                        name=f"T_C{j}_at_B{k}_period{n}",
                    )
                )
                for k in range(owner.K)
            ]
            for j in range(owner.J)
        ]
        for n in range(owner.N_periods)
    ]
    owner.theta_1_by_period = [
        [
            [
                [
                    (
                        owner.m.Var(
                            value=owner._recovery_approach_temperature(
                                i,
                                j,
                                n,
                            ),
                            ub=abs(
                                owner.T_h_in_period[n][i] - owner.T_c_in_period[n][j]
                            ),
                            lb=owner._recovery_approach_temperature(
                                i,
                                j,
                                n,
                            ),
                            name=(f"approach_T1_H{i}_to_C{j}_at_S{k}_period{n}"),
                        )
                        if owner.z_allowed[i][j][k] > 0
                        else owner.m.Param(
                            value=owner._recovery_approach_temperature(
                                i,
                                j,
                                n,
                            ),
                            name=(f"approach_T1_H{i}_to_C{j}_at_S{k}_period{n}"),
                        )
                    )
                    for k in range(owner.S)
                ]
                for j in range(owner.J)
            ]
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]
    owner.theta_2_by_period = [
        [
            [
                [
                    (
                        owner.m.Var(
                            value=owner._recovery_approach_temperature(
                                i,
                                j,
                                n,
                            ),
                            ub=abs(
                                owner.T_h_in_period[n][i] - owner.T_c_in_period[n][j]
                            ),
                            lb=owner._recovery_approach_temperature(
                                i,
                                j,
                                n,
                            ),
                            name=(f"approach_T2_H{i}_to_C{j}_at_S{k}_period{n}"),
                        )
                        if owner.z_allowed[i][j][k] > 0
                        else owner.m.Param(
                            value=owner._recovery_approach_temperature(
                                i,
                                j,
                                n,
                            ),
                            name=(f"approach_T2_H{i}_to_C{j}_at_S{k}_period{n}"),
                        )
                    )
                    for k in range(owner.S)
                ]
                for j in range(owner.J)
            ]
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]

    owner.Q_r = owner.Q_r_by_period[0]
    owner.Q_c = owner.Q_c_by_period[0]
    owner.Q_h = owner.Q_h_by_period[0]
    owner.T_h = owner.T_h_by_period[0]
    owner.T_c = owner.T_c_by_period[0]
    owner.theta_1 = owner.theta_1_by_period[0]
    owner.theta_2 = owner.theta_2_by_period[0]

    owner._set_piecewise_stage_heat_coordinates()

    for n in range(owner.N_periods):
        owner.m.Equations(
            [
                owner.Qtot_sh_period[n][i]
                - owner.m.sum(
                    [
                        owner.Q_r_by_period[n][i][j][k]
                        for k in range(owner.S)
                        for j in range(owner.J)
                    ]
                )
                - owner.Q_c_by_period[n][i]
                == 0.0
                for i in range(owner.I)
            ]
        )
        owner.m.Equations(
            [
                owner.Qtot_sc_period[n][j]
                - owner.m.sum(
                    [
                        owner.Q_r_by_period[n][i][j][k]
                        for k in range(owner.S)
                        for i in range(owner.I)
                    ]
                )
                - owner.Q_h_by_period[n][j]
                == 0.0
                for j in range(owner.J)
            ]
        )
        owner.m.Equations(
            [
                (owner.T_h_by_period[n][i][k + 1] - owner.T_h_by_period[n][i][k])
                * owner.f_h_period[n][i]
                + sum(owner.Q_r_by_period[n][i][j][k] for j in range(owner.J))
                == 0
                for k in range(owner.S)
                for i in range(owner.I)
                if not owner._hot_parent_segmented(i)
            ]
        )
        owner.m.Equations(
            [
                (owner.T_c_by_period[n][j][k + 1] - owner.T_c_by_period[n][j][k])
                * owner.f_c_period[n][j]
                + sum(owner.Q_r_by_period[n][i][j][k] for i in range(owner.I))
                == 0
                for k in range(owner.S)
                for j in range(owner.J)
                if not owner._cold_parent_segmented(j)
            ]
        )

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
                        owner.Q_r_by_period[n][i][j][k]
                        <= owner.Q_max_period[n][i][j] * owner.z[i][j][k]
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
                        owner.Q_c_by_period[n][i]
                        <= owner.Qtot_sh_period[n][i] * owner.z_cu[i]
                    )
                    if owner.z_cu_allowed[i] > 0
                    else None
                )
                for i in range(owner.I)
            ]
            _ = [
                (
                    owner.m.Equation(
                        owner.Q_h_by_period[n][j]
                        <= owner.Qtot_sc_period[n][j] * owner.z_hu[j]
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

    if owner.non_isothermal_model:
        owner._set_multiperiod_non_isothermal_equations()
    else:
        owner._set_multiperiod_isothermal_approach_equations()

    owner.dqda = []
    owner.alpha = []


def _set_multiperiod_isothermal_approach_equations(owner) -> None:
    for n in range(owner.N_periods):
        _ = [
            (
                owner.m.Equation(
                    owner.theta_1_by_period[n][i][j][k]
                    <= (owner.T_h_by_period[n][i][k] - owner.T_c_by_period[n][j][k])
                    + owner.M_ij_period[n][i][j] * (1 - owner.z[i][j][k])
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
                    owner.theta_1_by_period[n][i][j][k]
                    >= (owner.T_h_by_period[n][i][k] - owner.T_c_by_period[n][j][k])
                    - owner.M_ij_period[n][i][j] * (1 - owner.z[i][j][k])
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
                    owner.theta_2_by_period[n][i][j][k]
                    <= (
                        owner.T_h_by_period[n][i][k + 1]
                        - owner.T_c_by_period[n][j][k + 1]
                    )
                    + owner.M_ij_period[n][i][j] * (1 - owner.z[i][j][k])
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
                    owner.theta_2_by_period[n][i][j][k]
                    >= (
                        owner.T_h_by_period[n][i][k + 1]
                        - owner.T_c_by_period[n][j][k + 1]
                    )
                    - owner.M_ij_period[n][i][j] * (1 - owner.z[i][j][k])
                )
                if owner.z_allowed[i][j][k] > 0
                else None
            )
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        ]


def _set_multiperiod_non_isothermal_equations(owner) -> None:
    owner.X_by_period = [
        [
            [
                [
                    (
                        owner.m.Var(
                            value=0.5,
                            ub=1.0,
                            lb=0.0,
                            name=f"X_H{i}_to_C{j}_at_S{k}_period{n}",
                        )
                        if owner.z_allowed[i][j][k] > 0
                        else owner.m.Param(
                            value=0.0,
                            name=f"X_H{i}_to_C{j}_at_S{k}_period{n}",
                        )
                    )
                    for k in range(owner.S)
                ]
                for j in range(owner.J)
            ]
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]
    owner.Y_by_period = [
        [
            [
                [
                    (
                        owner.m.Var(
                            value=0.5,
                            ub=1.0,
                            lb=0.0,
                            name=f"Y_C{j}_to_H{i}_at_S{k}_period{n}",
                        )
                        if owner.z_allowed[i][j][k] > 0
                        else owner.m.Param(
                            value=0.0,
                            name=f"Y_C{j}_to_H{i}_at_S{k}_period{n}",
                        )
                    )
                    for k in range(owner.S)
                ]
                for i in range(owner.I)
            ]
            for j in range(owner.J)
        ]
        for n in range(owner.N_periods)
    ]
    owner.T_h_out_x_by_period = [
        [
            [
                [
                    (
                        owner.m.Var(
                            value=owner.T_h_in_period[n][i],
                            ub=owner.T_h_in_period[n][i],
                            lb=owner.T_h_out_period[n][i],
                            name=f"Thout_x_H{i}_to_C{j}_at_S{k}_period{n}",
                        )
                        if owner.z_allowed[i][j][k] > 0
                        else owner.m.Param(
                            value=0,
                            name=f"Tx_H{i}_to_C{j}_at_S{k}_period{n}",
                        )
                    )
                    for k in range(owner.S)
                ]
                for j in range(owner.J)
            ]
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]
    owner.T_c_out_y_by_period = [
        [
            [
                [
                    (
                        owner.m.Var(
                            value=owner.T_c_in_period[n][j],
                            ub=owner.T_c_out_period[n][j],
                            lb=owner.T_c_in_period[n][j],
                            name=f"Tcout_y_C{j}_to_H{i}_at_S{k}_period{n}",
                        )
                        if owner.z_allowed[i][j][k] > 0
                        else owner.m.Param(
                            value=0,
                            name=f"Ty_C{j}_to_H{i}_at_S{k}_period{n}",
                        )
                    )
                    for k in range(owner.S)
                ]
                for i in range(owner.I)
            ]
            for j in range(owner.J)
        ]
        for n in range(owner.N_periods)
    ]
    owner.X = owner.X_by_period[0]
    owner.Y = owner.Y_by_period[0]
    owner.T_h_out_x = owner.T_h_out_x_by_period[0]
    owner.T_c_out_y = owner.T_c_out_y_by_period[0]

    owner._set_piecewise_match_outlet_equations()

    for n in range(owner.N_periods):
        _ = [
            (
                owner.m.Equation(
                    owner.theta_1_by_period[n][i][j][k]
                    <= (
                        owner.T_h_by_period[n][i][k]
                        - owner.T_c_out_y_by_period[n][j][i][k]
                    )
                    + owner.M_ij_period[n][i][j] * (1 - owner.z[i][j][k])
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
                    owner.theta_2_by_period[n][i][j][k]
                    <= (
                        owner.T_h_out_x_by_period[n][i][j][k]
                        - owner.T_c_by_period[n][j][k + 1]
                    )
                    + owner.M_ij_period[n][i][j] * (1 - owner.z[i][j][k])
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
                    owner.Q_r_by_period[n][i][j][k]
                    - owner.X_by_period[n][i][j][k]
                    * owner.f_h_period[n][i]
                    * (
                        owner.T_h_by_period[n][i][k]
                        - owner.T_h_out_x_by_period[n][i][j][k]
                    )
                    == 0.0
                )
                if owner.z_allowed[i][j][k] > 0 and not owner._hot_parent_segmented(i)
                else None
            )
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        ]
        _ = [
            (
                owner.m.Equation(
                    owner.Q_r_by_period[n][i][j][k]
                    - owner.Y_by_period[n][j][i][k]
                    * owner.f_c_period[n][j]
                    * (
                        owner.T_c_out_y_by_period[n][j][i][k]
                        - owner.T_c_by_period[n][j][k + 1]
                    )
                    == 0.0
                )
                if owner.z_allowed[i][j][k] > 0 and not owner._cold_parent_segmented(j)
                else None
            )
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        ]
        _ = [
            (
                owner.m.Equation(
                    owner.m.sum([owner.X_by_period[n][i][j][k] for j in range(owner.J)])
                    == 1.0
                )
                if sum(owner.z_allowed[i][j][k] for j in range(owner.J)) > 0
                else None
            )
            for k in range(owner.S)
            for i in range(owner.I)
        ]
        _ = [
            (
                owner.m.Equation(
                    owner.m.sum([owner.Y_by_period[n][j][i][k] for i in range(owner.I)])
                    == 1.0
                )
                if sum(owner.z_allowed[i][j][k] for i in range(owner.I)) > 0
                else None
            )
            for k in range(owner.S)
            for j in range(owner.J)
        ]


def set_dqda_equations(owner) -> None:
    """Apply the source TDM minimum dQ/dA restriction."""

    if getattr(owner, "N_periods", 1) > 1:
        owner.min_dqda_int = build_index_grid(
            lambda n, i, j, k: (
                owner.m.Intermediate(
                    owner.min_dqda
                    * (owner.T_h_by_period[n][i][k] - owner.T_c_by_period[n][j][k + 1])
                    - owner.theta_1_by_period[n][i][j][k]
                    * owner.theta_2_by_period[n][i][j][k]
                    * owner.U_r_period[n][i][j],
                    name=f"dqda_calc_H{i}_to_C{j}_at_S{k}_period{n}",
                )
                if owner.z_allowed[i][j][k] > 0
                else None
            ),
            (owner.N_periods, owner.I, owner.J, owner.S),
        )
        owner.min_dQ_dA_eqn = [
            (
                owner.m.Equation(
                    owner.min_dqda_int[n][i][j][k]
                    <= owner.min_dqda
                    * owner.M_ij_period[n][i][j]
                    * (1 - owner.z[i][j][k])
                )
                if owner.z_allowed[i][j][k] > 0
                else None
            )
            for n in range(owner.N_periods)
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        ]
        return

    owner.min_dqda_int = build_index_grid(
        lambda i, j, k: (
            owner.m.Intermediate(
                owner.min_dqda * (owner.T_h[i][k] - owner.T_c[j][k + 1])
                - owner.theta_1[i][j][k] * owner.theta_2[i][j][k] * owner.U_r[i][j],
                name=f"dqda_calc_H{i}_to_C{j}_at_S{k}",
            )
            if owner.z_allowed[i][j][k] > 0
            else None
        ),
        (owner.I, owner.J, owner.S),
    )
    owner.min_dQ_dA_eqn = [
        (
            owner.m.Equation(
                owner.min_dqda_int[i][j][k]
                <= owner.min_dqda * owner.M_ij[i][j] * (1 - owner.z[i][j][k])
            )
            if owner.z_allowed[i][j][k] > 0
            else None
        )
        for k in range(owner.S)
        for j in range(owner.J)
        for i in range(owner.I)
    ]
