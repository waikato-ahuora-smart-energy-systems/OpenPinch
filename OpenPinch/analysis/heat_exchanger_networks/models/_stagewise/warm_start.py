"""StageWise warm-start value transfer."""

from __future__ import annotations


def set_initial_values_for_variables(
    owner,
    init_solution,
    *,
    brackets: bool = False,
) -> None:
    """Warm-start this model from a solved parent model."""

    if getattr(owner, "N_periods", 1) > 1 and hasattr(owner, "Q_r_by_period"):
        owner._set_multiperiod_initial_values(init_solution, brackets=brackets)
        return

    for k in range(owner.S):
        for j in range(owner.J):
            for i in range(owner.I):
                if owner.z_allowed[i][j][k] > 0:
                    owner._set_value(
                        owner.Q_r[i][j][k],
                        init_solution.Q_r[i][j][k].VALUE[0],
                        brackets=brackets,
                    )
                    owner._set_value(
                        owner.z[i][j][k],
                        init_solution.z[i][j][k][0],
                        brackets=brackets,
                    )
                    owner._set_value(
                        owner.theta_1[i][j][k],
                        init_solution.theta_1[i][j][k].VALUE[0],
                        brackets=brackets,
                    )
                    owner._set_value(
                        owner.theta_2[i][j][k],
                        init_solution.theta_2[i][j][k].VALUE[0],
                        brackets=brackets,
                    )
                else:
                    owner._set_value(owner.Q_r[i][j][k], 0.0, brackets=brackets)
                    owner._set_value(owner.z[i][j][k], 0, brackets=brackets)
                    owner._set_value(
                        owner.theta_1[i][j][k],
                        owner._recovery_approach_temperature(i, j),
                        brackets=brackets,
                    )
                    owner._set_value(
                        owner.theta_2[i][j][k],
                        owner._recovery_approach_temperature(i, j),
                        brackets=brackets,
                    )

    for i in range(owner.I):
        for k in range(owner.K):
            owner._set_value(
                owner.T_h[i][k],
                init_solution.T_h[i][k].VALUE[0],
                brackets=brackets,
            )
    for j in range(owner.J):
        for k in range(owner.K):
            owner._set_value(
                owner.T_c[j][k],
                init_solution.T_c[j][k].VALUE[0],
                brackets=brackets,
            )

    for i in range(owner.I):
        if owner.z_cu_allowed[i] > 0:
            owner._set_value(
                owner.Q_c[i],
                init_solution.Q_c[i].VALUE[0],
                brackets=brackets,
            )
            owner._set_value(
                owner.z_cu[i],
                init_solution.z_cu[i][0],
                brackets=brackets,
            )
        else:
            owner._set_value(owner.Q_c[i], 0.0, brackets=brackets)
            owner._set_value(owner.z_cu[i], 0, brackets=brackets)

    for j in range(owner.J):
        if owner.z_hu_allowed[j] > 0:
            owner._set_value(
                owner.Q_h[j],
                init_solution.Q_h[j].VALUE[0],
                brackets=brackets,
            )
            owner._set_value(
                owner.z_hu[j],
                init_solution.z_hu[j][0],
                brackets=brackets,
            )
        else:
            owner._set_value(owner.Q_h[j], 0.0, brackets=brackets)
            owner._set_value(owner.z_hu[j], 0, brackets=brackets)

    if owner.non_isothermal_model:
        if init_solution.non_isothermal_model:
            for k in range(owner.S):
                for j in range(owner.J):
                    for i in range(owner.I):
                        if owner.z_allowed[i][j][k] > 0:
                            owner._set_value(
                                owner.X[i][j][k],
                                init_solution.X[i][j][k].VALUE[0],
                                brackets=brackets,
                            )
                            owner._set_value(
                                owner.Y[j][i][k],
                                init_solution.Y[j][i][k].VALUE[0],
                                brackets=brackets,
                            )
                            owner._set_value(
                                owner.T_h_out_x[i][j][k],
                                init_solution.T_h_out_x[i][j][k].VALUE[0],
                                brackets=brackets,
                            )
                            owner._set_value(
                                owner.T_c_out_y[j][i][k],
                                init_solution.T_c_out_y[j][i][k].VALUE[0],
                                brackets=brackets,
                            )
                        else:
                            owner._set_value(owner.X[i][j][k], 0.0, brackets=brackets)
                            owner._set_value(owner.Y[j][i][k], 0.0, brackets=brackets)
                            owner._set_value(
                                owner.T_h_out_x[i][j][k],
                                init_solution.T_h[i][k + 1].VALUE[0],
                                brackets=brackets,
                            )
                            owner._set_value(
                                owner.T_c_out_y[j][i][k],
                                init_solution.T_c[j][k].VALUE[0],
                                brackets=brackets,
                            )
        else:
            for k in range(owner.S):
                for i in range(owner.I):
                    sum_Q_r_i = sum(
                        init_solution.Q_r[i][j][k].VALUE[0] for j in range(owner.J)
                    )
                    for j in range(owner.J):
                        sum_Q_r_j = sum(
                            init_solution.Q_r[i][j][k].VALUE[0] for i in range(owner.I)
                        )
                        if owner.z_allowed[i][j][k] > 0:
                            q_val = init_solution.Q_r[i][j][k].VALUE[0]
                            if q_val > 0.0:
                                owner._set_value(
                                    owner.X[i][j][k],
                                    q_val / sum_Q_r_i if sum_Q_r_i else 0.0,
                                    brackets=brackets,
                                )
                                owner._set_value(
                                    owner.Y[j][i][k],
                                    q_val / sum_Q_r_j if sum_Q_r_j else 0.0,
                                    brackets=brackets,
                                )
                            else:
                                owner._set_value(
                                    owner.X[i][j][k], 0.0, brackets=brackets
                                )
                                owner._set_value(
                                    owner.Y[j][i][k], 0.0, brackets=brackets
                                )
                            owner._set_value(
                                owner.T_h_out_x[i][j][k],
                                init_solution.T_h[i][k + 1].VALUE[0],
                                brackets=brackets,
                            )
                            owner._set_value(
                                owner.T_c_out_y[j][i][k],
                                init_solution.T_c[j][k].VALUE[0],
                                brackets=brackets,
                            )
                        else:
                            owner._set_value(owner.X[i][j][k], 0.0, brackets=brackets)
                            owner._set_value(owner.Y[j][i][k], 0.0, brackets=brackets)
                            owner._set_value(
                                owner.T_h_out_x[i][j][k],
                                init_solution.T_h[i][k + 1].VALUE[0],
                                brackets=brackets,
                            )
                            owner._set_value(
                                owner.T_c_out_y[j][i][k],
                                init_solution.T_c[j][k].VALUE[0],
                                brackets=brackets,
                            )


def _set_multiperiod_initial_values(owner, init_solution, *, brackets: bool) -> None:
    source_q_r = getattr(init_solution, "Q_r_by_period", None)
    source_q_c = getattr(init_solution, "Q_c_by_period", None)
    source_q_h = getattr(init_solution, "Q_h_by_period", None)
    source_t_h = getattr(init_solution, "T_h_by_period", None)
    source_t_c = getattr(init_solution, "T_c_by_period", None)
    source_theta_1 = getattr(init_solution, "theta_1_by_period", None)
    source_theta_2 = getattr(init_solution, "theta_2_by_period", None)
    if source_q_r is None:
        source_q_r = [init_solution.Q_r for _ in range(owner.N_periods)]
        source_q_c = [init_solution.Q_c for _ in range(owner.N_periods)]
        source_q_h = [init_solution.Q_h for _ in range(owner.N_periods)]
        source_t_h = [init_solution.T_h for _ in range(owner.N_periods)]
        source_t_c = [init_solution.T_c for _ in range(owner.N_periods)]
        source_theta_1 = [init_solution.theta_1 for _ in range(owner.N_periods)]
        source_theta_2 = [init_solution.theta_2 for _ in range(owner.N_periods)]

    for n in range(owner.N_periods):
        source_period_idx = min(n, len(source_q_r) - 1)
        for k in range(owner.S):
            for j in range(owner.J):
                for i in range(owner.I):
                    if owner.z_allowed[i][j][k] > 0:
                        owner._set_value(
                            owner.Q_r_by_period[n][i][j][k],
                            owner._active_binary_value(
                                source_q_r[source_period_idx][i][j][k]
                            ),
                            brackets=brackets,
                        )
                        owner._set_value(
                            owner.theta_1_by_period[n][i][j][k],
                            owner._active_binary_value(
                                source_theta_1[source_period_idx][i][j][k]
                            ),
                            brackets=brackets,
                        )
                        owner._set_value(
                            owner.theta_2_by_period[n][i][j][k],
                            owner._active_binary_value(
                                source_theta_2[source_period_idx][i][j][k]
                            ),
                            brackets=brackets,
                        )
                    else:
                        owner._set_value(
                            owner.Q_r_by_period[n][i][j][k],
                            0.0,
                            brackets=brackets,
                        )
                        approach = owner._recovery_approach_temperature(
                            i,
                            j,
                            n,
                        )
                        owner._set_value(
                            owner.theta_1_by_period[n][i][j][k],
                            approach,
                            brackets=brackets,
                        )
                        owner._set_value(
                            owner.theta_2_by_period[n][i][j][k],
                            approach,
                            brackets=brackets,
                        )

        for i in range(owner.I):
            for k in range(owner.K):
                owner._set_value(
                    owner.T_h_by_period[n][i][k],
                    owner._active_binary_value(source_t_h[source_period_idx][i][k]),
                    brackets=brackets,
                )
        for j in range(owner.J):
            for k in range(owner.K):
                owner._set_value(
                    owner.T_c_by_period[n][j][k],
                    owner._active_binary_value(source_t_c[source_period_idx][j][k]),
                    brackets=brackets,
                )
        for i in range(owner.I):
            value = (
                owner._active_binary_value(source_q_c[source_period_idx][i])
                if owner.z_cu_allowed[i] > 0
                else 0.0
            )
            owner._set_value(
                owner.Q_c_by_period[n][i],
                value,
                brackets=brackets,
            )
        for j in range(owner.J):
            value = (
                owner._active_binary_value(source_q_h[source_period_idx][j])
                if owner.z_hu_allowed[j] > 0
                else 0.0
            )
            owner._set_value(
                owner.Q_h_by_period[n][j],
                value,
                brackets=brackets,
            )

    for k in range(owner.S):
        for j in range(owner.J):
            for i in range(owner.I):
                owner._set_value(
                    owner.z[i][j][k],
                    owner._active_binary_value(init_solution.z[i][j][k]),
                    brackets=brackets,
                )
    for i in range(owner.I):
        owner._set_value(
            owner.z_cu[i],
            owner._active_binary_value(init_solution.z_cu[i]),
            brackets=brackets,
        )
    for j in range(owner.J):
        owner._set_value(
            owner.z_hu[j],
            owner._active_binary_value(init_solution.z_hu[j]),
            brackets=brackets,
        )

    source_area_r = getattr(init_solution, "area_r_shared", None)
    source_area_hu = getattr(init_solution, "area_hu_shared", None)
    source_area_cu = getattr(init_solution, "area_cu_shared", None)
    if source_area_r is not None and hasattr(owner, "area_r_shared"):
        for k in range(owner.S):
            for j in range(owner.J):
                for i in range(owner.I):
                    owner._set_value(
                        owner.area_r_shared[i][j][k],
                        owner._active_binary_value(source_area_r[i][j][k]),
                        brackets=brackets,
                    )
    if source_area_hu is not None and hasattr(owner, "area_hu_shared"):
        for j in range(owner.J):
            owner._set_value(
                owner.area_hu_shared[j],
                owner._active_binary_value(source_area_hu[j]),
                brackets=brackets,
            )
    if source_area_cu is not None and hasattr(owner, "area_cu_shared"):
        for i in range(owner.I):
            owner._set_value(
                owner.area_cu_shared[i],
                owner._active_binary_value(source_area_cu[i]),
                brackets=brackets,
            )

    if owner.non_isothermal_model and hasattr(owner, "X_by_period"):
        source_x = getattr(init_solution, "X_by_period", None)
        source_y = getattr(init_solution, "Y_by_period", None)
        source_thx = getattr(init_solution, "T_h_out_x_by_period", None)
        source_tcy = getattr(init_solution, "T_c_out_y_by_period", None)
        if source_x is None and getattr(init_solution, "non_isothermal_model", False):
            source_x = [init_solution.X for _ in range(owner.N_periods)]
            source_y = [init_solution.Y for _ in range(owner.N_periods)]
            source_thx = [init_solution.T_h_out_x for _ in range(owner.N_periods)]
            source_tcy = [init_solution.T_c_out_y for _ in range(owner.N_periods)]
        if source_x is not None:
            for n in range(owner.N_periods):
                source_period_idx = min(n, len(source_x) - 1)
                for k in range(owner.S):
                    for j in range(owner.J):
                        for i in range(owner.I):
                            if owner.z_allowed[i][j][k] > 0:
                                owner._set_value(
                                    owner.X_by_period[n][i][j][k],
                                    owner._active_binary_value(
                                        source_x[source_period_idx][i][j][k]
                                    ),
                                    brackets=brackets,
                                )
                                owner._set_value(
                                    owner.Y_by_period[n][j][i][k],
                                    owner._active_binary_value(
                                        source_y[source_period_idx][j][i][k]
                                    ),
                                    brackets=brackets,
                                )
                                owner._set_value(
                                    owner.T_h_out_x_by_period[n][i][j][k],
                                    owner._active_binary_value(
                                        source_thx[source_period_idx][i][j][k]
                                    ),
                                    brackets=brackets,
                                )
                                owner._set_value(
                                    owner.T_c_out_y_by_period[n][j][i][k],
                                    owner._active_binary_value(
                                        source_tcy[source_period_idx][j][i][k]
                                    ),
                                    brackets=brackets,
                                )
        else:
            for n in range(owner.N_periods):
                source_period_idx = min(n, len(source_q_r) - 1)
                for k in range(owner.S):
                    for i in range(owner.I):
                        hot_total = sum(
                            owner._active_binary_value(
                                source_q_r[source_period_idx][i][j][k]
                            )
                            for j in range(owner.J)
                        )
                        for j in range(owner.J):
                            cold_total = sum(
                                owner._active_binary_value(
                                    source_q_r[source_period_idx][row][j][k]
                                )
                                for row in range(owner.I)
                            )
                            duty = owner._active_binary_value(
                                source_q_r[source_period_idx][i][j][k]
                            )
                            active = owner.z_allowed[i][j][k] > 0 and duty > 0.0
                            owner._set_value(
                                owner.X_by_period[n][i][j][k],
                                duty / hot_total if active and hot_total else 0.0,
                                brackets=brackets,
                            )
                            owner._set_value(
                                owner.Y_by_period[n][j][i][k],
                                duty / cold_total if active and cold_total else 0.0,
                                brackets=brackets,
                            )
                            owner._set_value(
                                owner.T_h_out_x_by_period[n][i][j][k],
                                owner._active_binary_value(
                                    source_t_h[source_period_idx][i][k + 1]
                                ),
                                brackets=brackets,
                            )
                            owner._set_value(
                                owner.T_c_out_y_by_period[n][j][i][k],
                                owner._active_binary_value(
                                    source_t_c[source_period_idx][j][k]
                                ),
                                brackets=brackets,
                            )
