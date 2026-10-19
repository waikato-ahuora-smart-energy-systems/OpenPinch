"""StageWise objective construction."""

from __future__ import annotations

from ...indexing import build_index_grid
from .._base import approach as _approach
from .._base import piecewise as _piecewise


def set_obj(owner) -> None:
    """Attach source StageWise objective expressions unchanged."""

    if getattr(owner, "N_periods", 1) > 1:
        if owner.minimisation_goal == "hot utility":
            owner.m.Minimize(
                _approach._weighted_state_average(
                    owner,
                    [
                        owner.m.sum(_utility_loads(owner.Q_h_by_period[n]))
                        for n in range(owner.N_periods)
                    ],
                )
            )
        elif owner.minimisation_goal == "cold utility":
            owner.m.Minimize(
                _approach._weighted_state_average(
                    owner,
                    [
                        owner.m.sum(_utility_loads(owner.Q_c_by_period[n]))
                        for n in range(owner.N_periods)
                    ],
                )
            )
        elif owner.minimisation_goal == "total utility":
            owner.m.Minimize(
                _approach._weighted_state_average(
                    owner,
                    [
                        owner.m.sum(_utility_loads(owner.Q_h_by_period[n]))
                        + owner.m.sum(_utility_loads(owner.Q_c_by_period[n]))
                        for n in range(owner.N_periods)
                    ],
                )
            )
        elif owner.minimisation_goal == "utility costs":
            hot_costs = [
                _utilities_cost_expression(
                    owner,
                    "hot",
                    n,
                    _utility_totals(owner, "hot", n),
                    name="hot_utility",
                )
                for n in range(owner.N_periods)
            ]
            cold_costs = [
                _utilities_cost_expression(
                    owner,
                    "cold",
                    n,
                    _utility_totals(owner, "cold", n),
                    name="cold_utility",
                )
                for n in range(owner.N_periods)
            ]
            owner.m.Minimize(
                _approach._weighted_state_average(
                    owner,
                    [hot_costs[n] + cold_costs[n] for n in range(owner.N_periods)],
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
        elif owner.minimisation_goal == "dQ/dA obj":
            owner.m.Minimize(
                _approach._weighted_state_average(
                    owner,
                    [
                        sum(_utility_loads(owner.Q_h_by_period[n]))
                        for n in range(owner.N_periods)
                    ],
                )
                - owner.HU_target
            )
        elif owner.minimisation_goal in {"total cost", "variable total cost"}:
            _set_total_cost_objective(owner)
        else:
            _raise_unsupported_goal(owner)
        return

    if owner.minimisation_goal == "hot utility":
        owner.m.Minimize(owner.m.sum(_utility_loads(owner.Q_h)))
    elif owner.minimisation_goal == "cold utility":
        owner.m.Minimize(owner.m.sum(_utility_loads(owner.Q_c)))
    elif owner.minimisation_goal == "total utility":
        owner.m.Minimize(
            owner.m.sum(_utility_loads(owner.Q_h))
            + owner.m.sum(_utility_loads(owner.Q_c))
        )
    elif owner.minimisation_goal == "utility costs":
        hot_cost = _utilities_cost_expression(
            owner, "hot", 0, _utility_totals(owner, "hot", 0), name="hot_utility"
        )
        cold_cost = _utilities_cost_expression(
            owner, "cold", 0, _utility_totals(owner, "cold", 0), name="cold_utility"
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
        owner.m.Minimize(sum(_utility_loads(owner.Q_h)) - owner.HU_target)
    elif owner.minimisation_goal in {"total cost", "variable total cost"}:
        _set_total_cost_objective(owner)
    else:
        _raise_unsupported_goal(owner)


def _utility_loads(grid) -> list:
    """Flatten a ``[stream][utility]`` duty grid into one list of loads."""
    return [load for row in grid for load in row]


def _utility_totals(owner, side: str, n: int) -> list:
    """Return each utility's total duty over its heaters or coolers in period n."""
    grid = owner.Q_h_by_period[n] if side == "hot" else owner.Q_c_by_period[n]
    count = owner.NHU if side == "hot" else owner.NCU
    return [owner.m.sum([row[u] for row in grid]) for u in range(count)]


def _utilities_cost_expression(owner, side: str, n: int, totals, *, name: str):
    """Return the summed cost expression of each utility's total duty."""
    return sum(
        _piecewise._utility_cost_expression(
            owner, side, n, total, utility=u, name=f"{name}{u}_period_{n}"
        )
        for u, total in enumerate(totals)
    )


def _heater_terminal_differences(owner, n: int, j: int, u: int, *, solved_outlet):
    """Return a heater's (hot-end, cold-end) temperature differences."""
    utility_outlet = (
        owner._utility_solved_outlet_temperature("hot", n, j, u)
        if solved_outlet
        else owner.T_hu_out_period[n][u]
    )
    return (
        owner.T_hu_in_period[n][u] - owner.T_c_out_period[n][j],
        utility_outlet - owner.T_c_by_period[n][j][0],
    )


def _cooler_terminal_differences(owner, n: int, i: int, u: int, *, solved_outlet):
    """Return a cooler's (hot-end, cold-end) temperature differences."""
    utility_outlet = (
        owner._utility_solved_outlet_temperature("cold", n, i, u)
        if solved_outlet
        else owner.T_cu_out_period[n][u]
    )
    return (
        owner.T_h_by_period[n][i][owner.S] - utility_outlet,
        owner.T_h_out_period[n][i] - owner.T_cu_in_period[n][u],
    )


def _chen_mean(delta_1, delta_2):
    """Return Chen's approximation of the log-mean temperature difference."""
    return (delta_1 * delta_2 * (delta_1 + delta_2) / 2 + 1e-3) ** (1 / 3)


def _raise_unsupported_goal(owner) -> None:
    raise ValueError(
        f"Unsupported StageWise minimisation goal: {owner.minimisation_goal!r}."
    )


def _set_total_cost_objective(owner) -> None:
    if getattr(owner, "N_periods", 1) == 1:
        _set_source_total_cost_objective(owner)
        return
    _set_multiperiod_total_cost_objective(owner)


def _set_source_total_cost_objective(owner) -> None:
    owner.hu_cost_total = owner.m.Intermediate(
        _utilities_cost_expression(
            owner,
            "hot",
            0,
            _utility_totals(owner, "hot", 0),
            name="hot_utility_total_cost",
        ),
        name="Hot utility cost",
    )
    owner.cu_cost_total = owner.m.Intermediate(
        _utilities_cost_expression(
            owner,
            "cold",
            0,
            _utility_totals(owner, "cold", 0),
            name="cold_utility_total_cost",
        ),
        name="Cold utility cost",
    )
    # Post-processing overwrites hu_cost_total and cu_cost_total with costs
    # recomputed from the duties; verify() checks the solver's own against them.
    owner.hu_cost_solver_by_period = [owner.hu_cost_total]
    owner.cu_cost_solver_by_period = [owner.cu_cost_total]
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
                    owner.Q_h[j][u]
                    / (
                        owner.U_hu[j][u]
                        * _chen_mean(
                            *_heater_terminal_differences(
                                owner, 0, j, u, solved_outlet=False
                            )
                        )
                    )
                )
                + 1e-3
            )
            ** owner.hu_exp[0]
            for j in range(owner.J)
            for u in range(owner.NHU)
            if owner.z_hu_allowed[j][u] > 0
        ),
        name="Total hot utility HX area cost",
    )
    owner.cu_area_cost_total = owner.m.Intermediate(
        owner.cu_coeff[0]
        * sum(
            (
                (
                    owner.Q_c[i][u]
                    / (
                        owner.U_cu[i][u]
                        * _chen_mean(
                            *_cooler_terminal_differences(
                                owner, 0, i, u, solved_outlet=False
                            )
                        )
                    )
                )
                + 1e-3
            )
            ** owner.cu_exp[0]
            for i in range(owner.I)
            for u in range(owner.NCU)
            if owner.z_cu_allowed[i][u] > 0
        ),
        name="Total cold utility HX area cost",
    )

    if owner.minimisation_goal == "total cost":
        owner.utility_unit_cost_total = owner.m.Intermediate(
            owner.hu_unit_cost[0] * sum(_utility_loads(owner.z_hu))
            + owner.cu_unit_cost[0] * sum(_utility_loads(owner.z_cu)),
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
            _utilities_cost_expression(
                owner,
                "hot",
                n,
                _utility_totals(owner, "hot", n),
                name="hot_utility_total_cost",
            ),
            name=f"Hot utility cost state {n}",
        )
        for n in range(owner.N_periods)
    ]
    owner.cu_cost_total_by_period = [
        owner.m.Intermediate(
            _utilities_cost_expression(
                owner,
                "cold",
                n,
                _utility_totals(owner, "cold", n),
                name="cold_utility_total_cost",
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
    owner.hu_cost_solver_by_period = owner.hu_cost_total_by_period
    owner.cu_cost_solver_by_period = owner.cu_cost_total_by_period
    owner.hu_cost_total = _approach._weighted_state_average(
        owner, owner.hu_cost_total_by_period
    )
    owner.cu_cost_total = _approach._weighted_state_average(
        owner, owner.cu_cost_total_by_period
    )
    owner.weighted_operating_cost = _approach._weighted_state_average(
        owner, owner.operating_cost_by_state_expr
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
    owner.area_hu_shared = build_index_grid(
        lambda j, u: (
            owner.m.Var(value=0.0, lb=0.0, name=f"area_HU{u}_to_C{j}")
            if owner.z_hu_allowed[j][u] > 0
            else owner.m.Param(value=0.0, name=f"area_HU{u}_to_C{j}")
        ),
        (owner.J, owner.NHU),
    )
    owner.area_cu_shared = build_index_grid(
        lambda i, u: (
            owner.m.Var(value=0.0, lb=0.0, name=f"area_H{i}_to_CU{u}")
            if owner.z_cu_allowed[i][u] > 0
            else owner.m.Param(value=0.0, name=f"area_H{i}_to_CU{u}")
        ),
        (owner.I, owner.NCU),
    )

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
                    owner.area_hu_shared[j][u]
                    >= owner.Q_h_by_period[n][j][u]
                    / (
                        owner.U_hu_period[n][j][u]
                        * _chen_mean(
                            *_heater_terminal_differences(
                                owner, n, j, u, solved_outlet=True
                            )
                        )
                    )
                )
                if owner.z_hu_allowed[j][u] > 0
                else None
            )
            for j in range(owner.J)
            for u in range(owner.NHU)
        ]
        _ = [
            (
                owner.m.Equation(
                    owner.area_cu_shared[i][u]
                    >= owner.Q_c_by_period[n][i][u]
                    / (
                        owner.U_cu_period[n][i][u]
                        * _chen_mean(
                            *_cooler_terminal_differences(
                                owner, n, i, u, solved_outlet=True
                            )
                        )
                    )
                )
                if owner.z_cu_allowed[i][u] > 0
                else None
            )
            for i in range(owner.I)
            for u in range(owner.NCU)
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
        * sum(area ** owner.hu_exp[0] for area in _utility_loads(owner.area_hu_shared)),
        name="Total hot utility HX area cost",
    )
    owner.cu_area_cost_total = owner.m.Intermediate(
        owner.cu_coeff[0]
        * sum(area ** owner.cu_exp[0] for area in _utility_loads(owner.area_cu_shared)),
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
        owner.hu_unit_cost[0] * sum(_utility_loads(owner.z_hu))
        + owner.cu_unit_cost[0] * sum(_utility_loads(owner.z_cu)),
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
