"""StageWise objective construction."""

from __future__ import annotations


def set_obj(owner) -> None:
    """Attach source StageWise objective expressions unchanged."""

    if getattr(owner, "N_periods", 1) > 1:
        if owner.minimisation_goal == "hot utility":
            owner.m.Minimize(
                owner._weighted_state_average(
                    [
                        owner.m.sum([owner.Q_h_by_period[n][j] for j in range(owner.J)])
                        for n in range(owner.N_periods)
                    ]
                )
            )
        elif owner.minimisation_goal == "cold utility":
            owner.m.Minimize(
                owner._weighted_state_average(
                    [
                        owner.m.sum([owner.Q_c_by_period[n][i] for i in range(owner.I)])
                        for n in range(owner.N_periods)
                    ]
                )
            )
        elif owner.minimisation_goal == "total utility":
            owner.m.Minimize(
                owner._weighted_state_average(
                    [
                        owner.m.sum([owner.Q_h_by_period[n][j] for j in range(owner.J)])
                        + owner.m.sum(
                            [owner.Q_c_by_period[n][i] for i in range(owner.I)]
                        )
                        for n in range(owner.N_periods)
                    ]
                )
            )
        elif owner.minimisation_goal == "utility costs":
            hot_costs = [
                owner._utility_cost_expression(
                    "hot",
                    n,
                    owner.m.sum([owner.Q_h_by_period[n][j] for j in range(owner.J)]),
                    name=f"hot_utility_period_{n}",
                )
                for n in range(owner.N_periods)
            ]
            cold_costs = [
                owner._utility_cost_expression(
                    "cold",
                    n,
                    owner.m.sum([owner.Q_c_by_period[n][i] for i in range(owner.I)]),
                    name=f"cold_utility_period_{n}",
                )
                for n in range(owner.N_periods)
            ]
            owner.m.Minimize(
                owner._weighted_state_average(
                    [hot_costs[n] + cold_costs[n] for n in range(owner.N_periods)]
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
        elif owner.minimisation_goal == "dQ/dA obj":
            owner.m.Minimize(
                owner._weighted_state_average(
                    [sum(owner.Q_h_by_period[n]) for n in range(owner.N_periods)]
                )
                - owner.HU_target
            )
        elif owner.minimisation_goal in {"total cost", "variable total cost"}:
            owner._set_total_cost_objective()
        return

    if owner.minimisation_goal == "hot utility":
        owner.m.Minimize(owner.m.sum([owner.Q_h[j] for j in range(owner.J)]))
    if owner.minimisation_goal == "cold utility":
        owner.m.Minimize(owner.m.sum([owner.Q_c[i] for i in range(owner.I)]))
    elif owner.minimisation_goal == "total utility":
        owner.m.Minimize(
            owner.m.sum([owner.Q_h[j] for j in range(owner.J)])
            + owner.m.sum([owner.Q_c[i] for i in range(owner.I)])
        )
    elif owner.minimisation_goal == "utility costs":
        hot_cost = owner._utility_cost_expression(
            "hot",
            0,
            owner.m.sum([owner.Q_h[j] for j in range(owner.J)]),
            name="hot_utility",
        )
        cold_cost = owner._utility_cost_expression(
            "cold",
            0,
            owner.m.sum([owner.Q_c[i] for i in range(owner.I)]),
            name="cold_utility",
        )
        owner.m.Minimize(hot_cost + cold_cost)
    elif owner.minimisation_goal == "heat recovery":
        owner.m.Maximize(
            owner.m.sum(
                [
                    owner.Q_r[i][j][k]
                    for i in range(owner.I)
                    for j in range(owner.J)
                    for k in range(owner.S)
                ]
            )
        )
    elif owner.minimisation_goal == "dQ/dA obj":
        owner.m.Minimize(sum(owner.Q_h) - owner.HU_target)
    elif owner.minimisation_goal in {"total cost", "variable total cost"}:
        owner._set_total_cost_objective()


def _set_total_cost_objective(owner) -> None:
    if getattr(owner, "N_periods", 1) == 1:
        owner._set_source_total_cost_objective()
        return
    owner._set_multiperiod_total_cost_objective()


def _set_source_total_cost_objective(owner) -> None:
    owner.hu_cost_total = owner.m.Intermediate(
        owner._utility_cost_expression(
            "hot",
            0,
            owner.m.sum([owner.Q_h[j] for j in range(owner.J)]),
            name="hot_utility_total_cost",
        ),
        name="Hot utility cost",
    )
    owner.cu_cost_total = owner.m.Intermediate(
        owner._utility_cost_expression(
            "cold",
            0,
            owner.m.sum([owner.Q_c[i] for i in range(owner.I)]),
            name="cold_utility_total_cost",
        ),
        name="Cold utility cost",
    )
    owner.recovery_area_cost_filtered = [
        [0 for _j in range(owner.J)] for _k in range(owner.S)
    ]
    for k in range(owner.S):
        for j in range(owner.J):
            allowed_hots = [i for i in range(owner.I) if owner.z_allowed[i][j][k] > 0]
            if sum(owner.z_allowed[i][j][k] for i in range(owner.I)) > 0:
                owner.recovery_area_cost_filtered[k][j] = owner.m.Intermediate(
                    owner.A_coeff[0]
                    * sum(
                        (
                            (
                                owner.Q_r[i][j][k]
                                / (
                                    owner.U_r[i][j]
                                    * (
                                        owner.theta_1[i][j][k]
                                        * owner.theta_2[i][j][k]
                                        * (
                                            owner.theta_1[i][j][k]
                                            + owner.theta_2[i][j][k]
                                        )
                                        / 2
                                        + 1e-3
                                    )
                                    ** (1 / 3)
                                )
                            )
                            + 1e-3
                        )
                        ** owner.A_exp[0]
                        for i in allowed_hots
                    ),
                    name=f"Recovery HX area cost in stage {k} cold {j}",
                )

    owner.hu_area_cost_total = owner.m.Intermediate(
        owner.hu_coeff[0]
        * sum(
            (
                (
                    owner.Q_h[j]
                    / (
                        owner.U_hu[j]
                        * (
                            (owner.T_hu_in[0] - owner.T_c_out[j])
                            * (owner.T_hu_out[0] - owner.T_c[j][0])
                            * (
                                (owner.T_hu_in[0] - owner.T_c_out[j])
                                + (owner.T_hu_out[0] - owner.T_c[j][0])
                            )
                            / 2
                            + 1e-3
                        )
                        ** (1 / 3)
                    )
                )
                + 1e-3
            )
            ** owner.hu_exp[0]
            for j in range(owner.J)
        ),
        name="Total hot utility HX area cost",
    )
    owner.cu_area_cost_total = owner.m.Intermediate(
        owner.cu_coeff[0]
        * sum(
            (
                (
                    owner.Q_c[i]
                    / (
                        owner.U_cu[i]
                        * (
                            (owner.T_h[i][owner.S] - owner.T_cu_out[0])
                            * (owner.T_h_out[i] - owner.T_cu_in[0])
                            * (
                                (owner.T_h[i][owner.S] - owner.T_cu_out[0])
                                + (owner.T_h_out[i] - owner.T_cu_in[0])
                            )
                            / 2
                            + 1e-3
                        )
                        ** (1 / 3)
                    )
                )
                + 1e-3
            )
            ** owner.cu_exp[0]
            for i in range(owner.I)
        ),
        name="Total cold utility HX area cost",
    )

    if owner.minimisation_goal == "total cost":
        owner.utility_unit_cost_total = owner.m.Intermediate(
            owner.hu_unit_cost[0] * sum(owner.z_hu[j] for j in range(owner.J))
            + owner.cu_unit_cost[0] * sum(owner.z_cu[i] for i in range(owner.I)),
            name="Total utility base cost",
        )
        owner.recovery_unit_cost = [
            [0 for _j in range(owner.J)] for _k in range(owner.S)
        ]
        for k in range(owner.S):
            for j in range(owner.J):
                owner.recovery_unit_cost[k][j] = owner.m.Intermediate(
                    owner.unit_cost[0] * sum(owner.z[i][j][k] for i in range(owner.I)),
                    name=f"Total recovery base cost in stage {k} cold {j}",
                )
        owner.m.Minimize(
            owner.hu_cost_total
            + owner.cu_cost_total
            + owner.utility_unit_cost_total
            + sum(
                owner.recovery_unit_cost[k][j]
                for k in range(owner.S)
                for j in range(owner.J)
            )
            + sum(
                owner.recovery_area_cost_filtered[k][j]
                for k in range(owner.S)
                for j in range(owner.J)
            )
            + owner.hu_area_cost_total
            + owner.cu_area_cost_total
        )
    elif owner.minimisation_goal == "variable total cost":
        owner.m.Minimize(
            owner.hu_cost_total
            + owner.cu_cost_total
            + sum(
                owner.recovery_area_cost_filtered[k][j]
                for k in range(owner.S)
                for j in range(owner.J)
            )
            + owner.hu_area_cost_total
            + owner.cu_area_cost_total
        )


def _set_multiperiod_total_cost_objective(owner) -> None:
    owner.hu_cost_total_by_period = [
        owner.m.Intermediate(
            owner._utility_cost_expression(
                "hot",
                n,
                owner.m.sum([owner.Q_h_by_period[n][j] for j in range(owner.J)]),
                name=f"hot_utility_total_cost_period_{n}",
            ),
            name=f"Hot utility cost state {n}",
        )
        for n in range(owner.N_periods)
    ]
    owner.cu_cost_total_by_period = [
        owner.m.Intermediate(
            owner._utility_cost_expression(
                "cold",
                n,
                owner.m.sum([owner.Q_c_by_period[n][i] for i in range(owner.I)]),
                name=f"cold_utility_total_cost_period_{n}",
            ),
            name=f"Cold utility cost state {n}",
        )
        for n in range(owner.N_periods)
    ]
    owner.operating_cost_by_state_expr = [
        owner.m.Intermediate(
            owner.hu_cost_total_by_period[n] + owner.cu_cost_total_by_period[n],
            name=f"Operating cost state {n}",
        )
        for n in range(owner.N_periods)
    ]
    owner.hu_cost_total = owner._weighted_state_average(owner.hu_cost_total_by_period)
    owner.cu_cost_total = owner._weighted_state_average(owner.cu_cost_total_by_period)
    owner.weighted_operating_cost = owner._weighted_state_average(
        owner.operating_cost_by_state_expr
    )

    owner.area_r_shared = [
        [
            [
                (
                    owner.m.Var(
                        value=0.0,
                        lb=0.0,
                        name=f"area_H{i}_to_C{j}_at_S{k}",
                    )
                    if owner.z_allowed[i][j][k] > 0
                    else owner.m.Param(
                        value=0.0,
                        name=f"area_H{i}_to_C{j}_at_S{k}",
                    )
                )
                for k in range(owner.S)
            ]
            for j in range(owner.J)
        ]
        for i in range(owner.I)
    ]
    owner.area_hu_shared = [
        (
            owner.m.Var(value=0.0, lb=0.0, name=f"area_HU_to_C{j}")
            if owner.z_hu_allowed[j] > 0
            else owner.m.Param(value=0.0, name=f"area_HU_to_C{j}")
        )
        for j in range(owner.J)
    ]
    owner.area_cu_shared = [
        (
            owner.m.Var(value=0.0, lb=0.0, name=f"area_H{i}_to_CU")
            if owner.z_cu_allowed[i] > 0
            else owner.m.Param(value=0.0, name=f"area_H{i}_to_CU")
        )
        for i in range(owner.I)
    ]

    for n in range(owner.N_periods):
        _ = [
            (
                owner.m.Equation(
                    owner.area_r_shared[i][j][k]
                    >= owner.Q_r_by_period[n][i][j][k]
                    / (
                        owner.U_r_period[n][i][j]
                        * (
                            owner.theta_1_by_period[n][i][j][k]
                            * owner.theta_2_by_period[n][i][j][k]
                            * (
                                owner.theta_1_by_period[n][i][j][k]
                                + owner.theta_2_by_period[n][i][j][k]
                            )
                            / 2
                            + 1e-3
                        )
                        ** (1 / 3)
                    )
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
                    owner.area_hu_shared[j]
                    >= owner.Q_h_by_period[n][j]
                    / (
                        owner.U_hu_period[n][j]
                        * (
                            (owner.T_hu_in_period[n][0] - owner.T_c_out_period[n][j])
                            * (
                                owner._utility_solved_outlet_temperature("hot", n, j)
                                - owner.T_c_by_period[n][j][0]
                            )
                            * (
                                (
                                    owner.T_hu_in_period[n][0]
                                    - owner.T_c_out_period[n][j]
                                )
                                + (
                                    owner._utility_solved_outlet_temperature(
                                        "hot", n, j
                                    )
                                    - owner.T_c_by_period[n][j][0]
                                )
                            )
                            / 2
                            + 1e-3
                        )
                        ** (1 / 3)
                    )
                )
                if owner.z_hu_allowed[j] > 0
                else None
            )
            for j in range(owner.J)
        ]
        _ = [
            (
                owner.m.Equation(
                    owner.area_cu_shared[i]
                    >= owner.Q_c_by_period[n][i]
                    / (
                        owner.U_cu_period[n][i]
                        * (
                            (
                                owner.T_h_by_period[n][i][owner.S]
                                - owner._utility_solved_outlet_temperature("cold", n, i)
                            )
                            * (owner.T_h_out_period[n][i] - owner.T_cu_in_period[n][0])
                            * (
                                (
                                    owner.T_h_by_period[n][i][owner.S]
                                    - owner._utility_solved_outlet_temperature(
                                        "cold", n, i
                                    )
                                )
                                + (
                                    owner.T_h_out_period[n][i]
                                    - owner.T_cu_in_period[n][0]
                                )
                            )
                            / 2
                            + 1e-3
                        )
                        ** (1 / 3)
                    )
                )
                if owner.z_cu_allowed[i] > 0
                else None
            )
            for i in range(owner.I)
        ]

    owner.recovery_area_cost_total = owner.m.Intermediate(
        owner.A_coeff[0]
        * sum(
            owner.area_r_shared[i][j][k] ** owner.A_exp[0]
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        ),
        name="Total recovery HX area cost",
    )
    owner.hu_area_cost_total = owner.m.Intermediate(
        owner.hu_coeff[0]
        * sum(owner.area_hu_shared[j] ** owner.hu_exp[0] for j in range(owner.J)),
        name="Total hot utility HX area cost",
    )
    owner.cu_area_cost_total = owner.m.Intermediate(
        owner.cu_coeff[0]
        * sum(owner.area_cu_shared[i] ** owner.cu_exp[0] for i in range(owner.I)),
        name="Total cold utility HX area cost",
    )
    owner.unit_cost_total = owner.m.Intermediate(
        owner.unit_cost[0]
        * sum(
            owner.z[i][j][k]
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        ),
        name="Total recovery base cost",
    )
    owner.utility_unit_cost_total = owner.m.Intermediate(
        owner.hu_unit_cost[0] * sum(owner.z_hu[j] for j in range(owner.J))
        + owner.cu_unit_cost[0] * sum(owner.z_cu[i] for i in range(owner.I)),
        name="Total utility base cost",
    )
    owner.capital_cost_total = owner.m.Intermediate(
        owner.unit_cost_total
        + owner.utility_unit_cost_total
        + owner.recovery_area_cost_total
        + owner.hu_area_cost_total
        + owner.cu_area_cost_total,
        name="Total shared capital cost",
    )
    owner.m.Minimize(owner.capital_cost_total + owner.weighted_operating_cost)
