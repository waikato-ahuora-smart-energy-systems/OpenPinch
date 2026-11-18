"""StageWise objective construction."""

from __future__ import annotations

from ...indexing import build_index_grid
from .._base import approach as _approach
from .._base import piecewise as _piecewise
from . import staged_utilities as _staged

# m^2 added to a shared area under its cost exponent: a unit with no duty has
# zero area, where the cost's derivative is infinite and stops Ipopt. At
# 1e-8 the slope there was still steep enough that Ipopt's restoration phase
# failed at feasible points. The cost of 1e-4 m^2 per idle unit is negligible.
_SHARED_AREA_OFFSET = 1e-4


def set_obj(owner) -> None:
    """Attach source StageWise objective expressions unchanged."""

    if getattr(owner, "N_periods", 1) > 1:
        if owner.minimisation_goal == "hot utility":
            owner.m.Minimize(
                _approach._weighted_state_average(
                    owner,
                    [
                        owner.m.sum(_side_loads(owner, "hot", n))
                        for n in range(owner.N_periods)
                    ],
                )
            )
        elif owner.minimisation_goal == "cold utility":
            owner.m.Minimize(
                _approach._weighted_state_average(
                    owner,
                    [
                        owner.m.sum(_side_loads(owner, "cold", n))
                        for n in range(owner.N_periods)
                    ],
                )
            )
        elif owner.minimisation_goal == "total utility":
            owner.m.Minimize(
                _approach._weighted_state_average(
                    owner,
                    [
                        owner.m.sum(_side_loads(owner, "hot", n))
                        + owner.m.sum(_side_loads(owner, "cold", n))
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
                )
                for n in range(owner.N_periods)
            ]
            cold_costs = [
                _utilities_cost_expression(
                    owner,
                    "cold",
                    n,
                    _utility_totals(owner, "cold", n),
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
                    [sum(_side_loads(owner, "hot", n)) for n in range(owner.N_periods)],
                )
                - owner.HU_target
            )
        elif owner.minimisation_goal in {"total cost", "variable total cost"}:
            _set_total_cost_objective(owner)
        else:
            _raise_unsupported_goal(owner)
        return

    if owner.minimisation_goal == "hot utility":
        owner.m.Minimize(owner.m.sum(_side_loads(owner, "hot", 0)))
    elif owner.minimisation_goal == "cold utility":
        owner.m.Minimize(owner.m.sum(_side_loads(owner, "cold", 0)))
    elif owner.minimisation_goal == "total utility":
        owner.m.Minimize(
            owner.m.sum(_side_loads(owner, "hot", 0))
            + owner.m.sum(_side_loads(owner, "cold", 0))
        )
    elif owner.minimisation_goal == "utility costs":
        hot_cost = _utilities_cost_expression(
            owner, "hot", 0, _utility_totals(owner, "hot", 0)
        )
        cold_cost = _utilities_cost_expression(
            owner, "cold", 0, _utility_totals(owner, "cold", 0)
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
        owner.m.Minimize(sum(_side_loads(owner, "hot", 0)) - owner.HU_target)
    elif owner.minimisation_goal in {"total cost", "variable total cost"}:
        _set_total_cost_objective(owner)
    else:
        _raise_unsupported_goal(owner)


def _utility_loads(grid) -> list:
    """Flatten a ``[stream][utility]`` duty grid into one list of loads."""
    return [load for row in grid for load in row]


def _side_loads(owner, side: str, n: int) -> list:
    """Every heater (or cooler) duty in period n, stream-end and staged."""
    grid = owner.Q_h_by_period[n] if side == "hot" else owner.Q_c_by_period[n]
    count = owner.NHU if side == "hot" else owner.NCU
    return _utility_loads(grid) + [
        load
        for u in range(count)
        for load in _staged.staged_utility_loads(owner, side, n, u)
    ]


def _utility_totals(owner, side: str, n: int) -> list:
    """Return each utility's total duty over its heaters or coolers in period n."""
    grid = owner.Q_h_by_period[n] if side == "hot" else owner.Q_c_by_period[n]
    count = owner.NHU if side == "hot" else owner.NCU
    return [
        owner.m.sum(
            [row[u] for row in grid] + _staged.staged_utility_loads(owner, side, n, u)
        )
        for u in range(count)
    ]


def _utilities_cost_expression(owner, side: str, n: int, totals):
    """Return the summed cost expression of each utility's total duty."""
    return sum(
        _piecewise._utility_cost_expression(owner, side, n, total, utility=u)
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


_CHEN_DELTA_SMOOTHING = 1e-4


def _chen_mean(delta_1, delta_2):
    """Return Chen's approximation of the log-mean temperature difference.

    Each temperature difference goes through a smooth max with zero, so a
    heater or cooler at zero duty, whose approaches are unconstrained, cannot
    take the cube root of a negative number. For an exchanger in use the
    differences are of order 1 to 1e2 K and the smoothing changes nothing
    measurable. Smoothing the base instead squares values of up to about
    1e12 K^6, beyond what Couenne's bound tightening resolves.
    """
    delta_1 = _smooth_positive(delta_1)
    delta_2 = _smooth_positive(delta_2)
    return (delta_1 * delta_2 * (delta_1 + delta_2) / 2 + 1e-3) ** (1 / 3)


def _smooth_positive(delta):
    return 0.5 * (delta + (delta**2 + _CHEN_DELTA_SMOOTHING**2) ** 0.5)


def _raise_unsupported_goal(owner) -> None:
    raise ValueError(
        f"Unsupported StageWise minimisation goal: {owner.minimisation_goal!r}."
    )


def _set_total_cost_objective(owner) -> None:
    if getattr(owner, "staged_utilities", False):
        raise ValueError(
            "staged utilities are priced for utility goals only; "
            f"{owner.minimisation_goal!r} would cost their area as free."
        )
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
        ),
    )
    owner.cu_cost_total = owner.m.Intermediate(
        _utilities_cost_expression(
            owner,
            "cold",
            0,
            _utility_totals(owner, "cold", 0),
        ),
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
    )

    if owner.minimisation_goal == "total cost":
        owner.utility_unit_cost_total = owner.m.Intermediate(
            owner.hu_unit_cost[0] * sum(_utility_loads(owner.z_hu))
            + owner.cu_unit_cost[0] * sum(_utility_loads(owner.z_cu)),
        )
        owner.recovery_unit_cost = [
            [0 for _j in range(owner.J)] for _k in range(owner.S)
        ]
        for k in range(owner.S):
            for j in range(owner.J):
                owner.recovery_unit_cost[k][j] = owner.m.Intermediate(
                    owner.unit_cost[0] * sum(owner.z[i][j][k] for i in range(owner.I)),
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
            ),
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
            ),
        )
        for n in range(owner.N_periods)
    ]
    owner.operating_cost_by_state_expr = [
        owner.m.Intermediate(
            owner.hu_cost_total_by_period[n] + owner.cu_cost_total_by_period[n],
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
                    )
                    if owner.z_allowed[i][j][k] > 0
                    else owner.m.Param(
                        value=0.0,
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
            owner.m.Var(value=0.0, lb=0.0)
            if owner.z_hu_allowed[j][u] > 0
            else owner.m.Param(value=0.0)
        ),
        (owner.J, owner.NHU),
    )
    owner.area_cu_shared = build_index_grid(
        lambda i, u: (
            owner.m.Var(value=0.0, lb=0.0)
            if owner.z_cu_allowed[i][u] > 0
            else owner.m.Param(value=0.0)
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
            (owner.area_r_shared[i][j][k] + _SHARED_AREA_OFFSET) ** owner.A_exp[0]
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        ),
    )
    owner.hu_area_cost_total = owner.m.Intermediate(
        owner.hu_coeff[0]
        * sum(
            (area + _SHARED_AREA_OFFSET) ** owner.hu_exp[0]
            for area in _utility_loads(owner.area_hu_shared)
        ),
    )
    owner.cu_area_cost_total = owner.m.Intermediate(
        owner.cu_coeff[0]
        * sum(
            (area + _SHARED_AREA_OFFSET) ** owner.cu_exp[0]
            for area in _utility_loads(owner.area_cu_shared)
        ),
    )
    owner.unit_cost_total = owner.m.Intermediate(
        owner.unit_cost[0]
        * sum(
            owner.z[i][j][k]
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        ),
    )
    owner.utility_unit_cost_total = owner.m.Intermediate(
        owner.hu_unit_cost[0] * sum(_utility_loads(owner.z_hu))
        + owner.cu_unit_cost[0] * sum(_utility_loads(owner.z_cu)),
    )
    owner.capital_cost_total = owner.m.Intermediate(
        owner.unit_cost_total
        + owner.utility_unit_cost_total
        + owner.recovery_area_cost_total
        + owner.hu_area_cost_total
        + owner.cu_area_cost_total,
    )
    owner.m.Minimize(owner.capital_cost_total + owner.weighted_operating_cost)
