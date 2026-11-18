"""Pinch-decomposition superstructure and objective equations."""

from __future__ import annotations

from ...indexing import build_index_grid
from .._base import approach as _approach
from .._base import piecewise as _piecewise
from .._base import utility_slices as _slices
from .._base.superstructure import create_match_binaries, create_utility_duty_grids
from .._stagewise import staged_utilities as _staged


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
            )
            if owner.z_allowed[i][j][k] > 0
            else owner.m.Param(
                value=0.0,
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
            )
            if k > 0 and owner.z_i_active[i] > 0
            else owner.m.Param(
                value=owner.T_h_in_period[n][i],
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
            )
            if k < owner.S and owner.z_j_active[j] > 0
            else owner.m.Param(
                value=owner.T_c_in_period[n][j],
            )
        ),
        (owner.N_periods, owner.J, owner.K),
    )
    _staged.create_staged_utility_grids(owner)
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
                    - _staged.staged_cooler_duty(owner, n, i)
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
                    - _staged.staged_heater_duty(owner, n, j)
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
                    (
                        owner.T_h_mid_by_period[n][i][k + 1]
                        - owner.T_h_by_period[n][i][k]
                    )
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
                    (
                        owner.T_c_by_period[n][j][k + 1]
                        - owner.T_c_mid_by_period[n][j][k]
                    )
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
        # Big-M switches, with each duty's own upper bound as M, keep the
        # model linear for unsegmented streams.
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
                        owner.Q_c_by_period[n][i][u]
                        <= owner.Qtot_sh_period[n][i] * owner.z_cu[i][u]
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
                        owner.Q_h_by_period[n][j][u]
                        <= owner.Qtot_sc_period[n][j] * owner.z_hu[j][u]
                    )
                    if owner.z_hu_allowed[j][u] > 0
                    else None
                )
                for j in range(owner.J)
                for u in range(owner.NHU)
            ]

    _approach._set_multiperiod_utility_approach_equations(owner)
    _staged.set_staged_utility_equations(owner)
    _slices.set_utility_slice_coupling(owner)

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
                    (owner.T_h_by_period[n][i][k] - owner.T_c_mid_by_period[n][j][k])
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
                        owner.T_h_mid_by_period[n][i][k + 1]
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
                    sum(row[u] for row in grid)
                    + sum(_staged.staged_utility_loads(owner, side, n, u)),
                    utility=u,
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
    """Every unit binary: recovery matches, heaters and coolers."""
    return owner.m.sum(
        [
            _at(getattr(owner, name), index)
            for name, indices in _unit_index_sets(owner).items()
            for index in indices
        ]
    )


# A unit counts as used above this many times the model tolerance.
_USED_DUTY_FACTOR = 10.0


def _link_units_to_duty(owner) -> None:
    """A unit binary may be on only if the unit carries duty in some period.

    The big-M link only stops duty without its binary. When units are
    minimised, an idle binary set to 1 would satisfy a no-good cut with the
    network unchanged, so each binary also needs duty, above the tolerance
    post-processing uses to read a unit as used.
    """
    least = _USED_DUTY_FACTOR * owner.tol
    duties = {
        "z": lambda i, j, k: [owner.Q_r_by_period[n][i][j][k] for n in _periods(owner)],
        "z_hu": lambda j, u: [owner.Q_h_by_period[n][j][u] for n in _periods(owner)],
        "z_cu": lambda i, u: [owner.Q_c_by_period[n][i][u] for n in _periods(owner)],
        "z_hs": lambda j, u, k: [
            owner.Q_hs_by_period[n][j][u][k] for n in _periods(owner)
        ],
        "z_cs": lambda i, u, k: [
            owner.Q_cs_by_period[n][i][u][k] for n in _periods(owner)
        ],
    }
    for name, indices in _unit_index_sets(owner).items():
        for index in indices:
            owner.m.Equation(
                owner.m.sum(duties[name](*index))
                >= least * _at(getattr(owner, name), index)
            )


def _periods(owner) -> range:
    return range(owner.N_periods)


def _set_utility_target(owner, side: str, targets) -> None:
    duty_grids = owner.Q_h_by_period if side == "hot" else owner.Q_c_by_period
    count = owner.NHU if side == "hot" else owner.NCU
    for n in range(owner.N_periods):
        owner.m.Equation(
            _total_duty(duty_grids[n])
            + sum(
                sum(_staged.staged_utility_loads(owner, side, n, u))
                for u in range(count)
            )
            - targets[n]
            >= 0.0
        )
    if getattr(owner, "staged_utilities", False):
        _set_level_targets(owner, side)


# Relative slack on each level target, for rounding between targeting and
# the side's own duties.
_LEVEL_BAND = 1e-7


def _set_level_targets(owner, side: str) -> None:
    """Hold each utility's duty at its targeted level.

    Targeting assigns each level its duty by temperature; with staged units
    the side can place each level where it serves, so prices no longer pick
    the levels.
    """
    targets = owner.pinch_decomposition.period_targets
    duty_grids = owner.Q_h_by_period if side == "hot" else owner.Q_c_by_period
    count = owner.NHU if side == "hot" else owner.NCU
    for n in range(owner.N_periods):
        levels = (
            targets[n].hot_utility_level_targets
            if side == "hot"
            else targets[n].cold_utility_level_targets
        )
        if len(levels) != count:
            raise ValueError(
                f"PDM {side} utility level targets must match the solver's "
                f"{count} {side} utilities."
            )
        for u, level in enumerate(levels):
            duty = sum(row[u] for row in duty_grids[n]) + sum(
                _staged.staged_utility_loads(owner, side, n, u)
            )
            # A relative band, not the model tolerance: within the tolerance
            # prices would still shift up to 1e-3 kW between levels.
            relative = getattr(owner, "level_band", None)
            band = (_LEVEL_BAND if relative is None else relative) * max(
                1.0, abs(level)
            )
            owner.m.Equation(duty <= level + band)
            owner.m.Equation(duty >= level - band)


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
        _set_utility_target(owner, "hot", owner.HU_target_by_period)
    elif goal == "cold utility":
        _set_utility_target(owner, "cold", owner.CU_target_by_period)

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
        _link_units_to_duty(owner)
        if getattr(owner, "staged_utilities", False):
            # Fewer units must not move duty between levels.
            _set_level_targets(owner, "hot" if owner.pinch_loc == "above" else "cold")
        if choice and owner.utility_cost_cap is not None:
            # Relative slack for the cap's own rounding in the solver.
            owner.m.Equation(
                _utility_cost(owner) <= owner.utility_cost_cap * (1 + 1e-9) + owner.tol
            )
        owner.m.Minimize(
            _units(owner)
            if not choice or owner.utility_cost_cap is not None
            else _utility_cost(owner) + _units(owner)
        )


def _unit_index_sets(owner) -> dict[str, list[tuple[int, ...]]]:
    return {
        "z": [
            (i, j, k)
            for i in range(owner.I)
            for j in range(owner.J)
            for k in range(owner.S)
        ],
        "z_hu": [(j, u) for j in range(owner.J) for u in range(owner.NHU)],
        "z_cu": [(i, u) for i in range(owner.I) for u in range(owner.NCU)],
        # Staged heaters and coolers offered (none without staging).
        "z_hs": _staged_indices(getattr(owner, "z_hs_allowed", [])),
        "z_cs": _staged_indices(getattr(owner, "z_cs_allowed", [])),
    }


def _staged_indices(allowed) -> list[tuple[int, int, int]]:
    return [
        (s, u, k)
        for s, row in enumerate(allowed)
        for u, by_k in enumerate(row)
        for k, offered in enumerate(by_k)
        if offered
    ]


def _at(grid, index: tuple[int, ...]):
    for position in index:
        grid = grid[position]
    return grid


def used_units(owner) -> dict[str, tuple[int, ...]]:
    """Return the solved side's used-unit flags: recovery, heaters, coolers.

    Post-processing has replaced the binaries with ``[flag]`` lists of the
    units that carry duty.
    """

    return {
        name: tuple(int(_at(getattr(owner, name), index)[0]) for index in indices)
        for name, indices in _unit_index_sets(owner).items()
    }


def require_units(owner, at_least: int) -> None:
    """Require at least ``at_least`` units: matches, heaters and coolers."""

    owner.m.Equation(_units(owner) >= at_least)


def exclude_units(owner, pattern: dict[str, tuple[int, ...]]) -> None:
    """Add a no-good cut: at least one unit must differ from ``pattern``."""

    owner.m.Equation(
        sum(
            (1 - _at(getattr(owner, name), index))
            if flag
            else _at(getattr(owner, name), index)
            for name, indices in _unit_index_sets(owner).items()
            for index, flag in zip(indices, pattern[name], strict=True)
        )
        >= 1
    )
