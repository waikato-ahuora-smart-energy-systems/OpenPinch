"""Heaters and coolers between the stages of a stage-wise superstructure.

Interior stage boundary ``k`` (1 to S-1) lies between stage ``k - 1`` and
stage ``k``. A cold stream leaves stage ``k`` at ``T_c_mid[k]``, its staged
heaters take it to ``T_c[k]``, and it enters stage ``k - 1`` there. A hot
stream leaves stage ``k - 1`` at ``T_h_mid[k]``, its staged coolers take it
to ``T_h[k]``, and it enters stage ``k`` there. Where a stream has no staged
unit at a boundary, its mid temperature is the boundary temperature itself,
so a model without staged utilities is unchanged.

At most one staged heater (or cooler) runs at one boundary of one stream.
Segmented streams and segmented utilities take no staged units.
"""

from __future__ import annotations

from .._base import approach as _approach
from .._base import area as _area
from .._base import piecewise as _piecewise


def interior_boundaries(owner) -> range:
    """Stage boundaries that may hold a staged heater or cooler."""
    return range(1, owner.S)


def create_staged_utility_grids(owner) -> None:
    """Create the staged duties, binaries and mid temperatures."""
    owner.z_hs_allowed = _allowed(owner, "hot")
    owner.z_cs_allowed = _allowed(owner, "cold")
    owner.Q_hs_by_period = [
        [
            [
                [
                    _duty_variable(owner, owner.Qtot_sc_period[n][j])
                    if owner.z_hs_allowed[j][u][k]
                    else 0.0
                    for k in range(owner.K)
                ]
                for u in range(owner.NHU)
            ]
            for j in range(owner.J)
        ]
        for n in range(owner.N_periods)
    ]
    owner.Q_cs_by_period = [
        [
            [
                [
                    _duty_variable(owner, owner.Qtot_sh_period[n][i])
                    if owner.z_cs_allowed[i][u][k]
                    else 0.0
                    for k in range(owner.K)
                ]
                for u in range(owner.NCU)
            ]
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]
    owner.T_c_mid_by_period = [
        [
            [
                owner.m.Var(
                    value=owner.T_c_in_period[n][j],
                    ub=owner.T_c_out_period[n][j],
                    lb=owner.T_c_in_period[n][j],
                )
                if any(owner.z_hs_allowed[j][u][k] for u in range(owner.NHU))
                else owner.T_c_by_period[n][j][k]
                for k in range(owner.K)
            ]
            for j in range(owner.J)
        ]
        for n in range(owner.N_periods)
    ]
    owner.T_h_mid_by_period = [
        [
            [
                owner.m.Var(
                    value=owner.T_h_in_period[n][i],
                    ub=owner.T_h_in_period[n][i],
                    lb=owner.T_h_out_period[n][i],
                )
                if any(owner.z_cs_allowed[i][u][k] for u in range(owner.NCU))
                else owner.T_h_by_period[n][i][k]
                for k in range(owner.K)
            ]
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]
    owner.z_hs = _binaries(owner, owner.z_hs_allowed)
    owner.z_cs = _binaries(owner, owner.z_cs_allowed)


def staged_heater_duty(owner, n: int, j: int):
    """Total staged heater duty on cold stream ``j`` in period ``n``."""
    return sum(
        owner.Q_hs_by_period[n][j][u][k]
        for u in range(owner.NHU)
        for k in interior_boundaries(owner)
    )


def staged_cooler_duty(owner, n: int, i: int):
    """Total staged cooler duty on hot stream ``i`` in period ``n``."""
    return sum(
        owner.Q_cs_by_period[n][i][u][k]
        for u in range(owner.NCU)
        for k in interior_boundaries(owner)
    )


def staged_utility_loads(owner, side: str, n: int, u: int) -> list:
    """Every staged duty of utility ``u`` in period ``n`` (none when off)."""
    if not getattr(owner, "staged_utilities", False):
        return []
    grid = owner.Q_hs_by_period[n] if side == "hot" else owner.Q_cs_by_period[n]
    return [row[u][k] for row in grid for k in interior_boundaries(owner)]


def set_staged_utility_equations(owner) -> None:
    """Link mid temperatures to staged duties and keep each unit's approach."""
    for n in range(owner.N_periods):
        for j in range(owner.J):
            for k in interior_boundaries(owner):
                if not any(owner.z_hs_allowed[j][u][k] for u in range(owner.NHU)):
                    continue
                owner.m.Equation(
                    (owner.T_c_by_period[n][j][k] - owner.T_c_mid_by_period[n][j][k])
                    * owner.f_c_period[n][j]
                    == sum(owner.Q_hs_by_period[n][j][u][k] for u in range(owner.NHU))
                )
                for u in range(owner.NHU):
                    if owner.z_hs_allowed[j][u][k]:
                        _set_staged_heater_approach(owner, n, j, u, k)
        for i in range(owner.I):
            for k in interior_boundaries(owner):
                if not any(owner.z_cs_allowed[i][u][k] for u in range(owner.NCU)):
                    continue
                owner.m.Equation(
                    (owner.T_h_mid_by_period[n][i][k] - owner.T_h_by_period[n][i][k])
                    * owner.f_h_period[n][i]
                    == sum(owner.Q_cs_by_period[n][i][u][k] for u in range(owner.NCU))
                )
                for u in range(owner.NCU):
                    if owner.z_cs_allowed[i][u][k]:
                        _set_staged_cooler_approach(owner, n, i, u, k)
    if owner.integers:
        _set_one_unit_per_boundary(owner, owner.z_hs, owner.z_hs_allowed)
        _set_one_unit_per_boundary(owner, owner.z_cs, owner.z_cs_allowed)


def _allowed(owner, side: str) -> list:
    """Flag ``[stream][utility][boundary]`` for each staged unit offered.

    A staged unit is offered when ``staged_utilities`` is on, the stream and
    utility are not segmented, and the utility's inlet clears the stream's
    supply by its approach in every period the stream runs, so some duty is
    possible.
    """
    streams, utilities = (owner.J, owner.NHU) if side == "hot" else (owner.I, owner.NCU)
    if not getattr(owner, "staged_utilities", False):
        return [
            [[0 for _k in range(owner.K)] for _u in range(utilities)]
            for _s in range(streams)
        ]
    flags = [
        [_reachable(owner, side, s, u) for u in range(utilities)]
        for s in range(streams)
    ]
    return [
        [
            [
                int(flags[s][u] and k in interior_boundaries(owner))
                for k in range(owner.K)
            ]
            for u in range(utilities)
        ]
        for s in range(streams)
    ]


def _reachable(owner, side: str, s: int, u: int) -> bool:
    if side == "hot":
        if _piecewise._cold_parent_segmented(
            owner, s
        ) or _piecewise._utility_is_segmented(owner, "hot", u):
            return False
        return all(
            owner.T_hu_in_period[n][u] - owner.T_c_in_period[n][s]
            > _approach._hot_utility_inlet_approach_temperature(owner, s, u, n)
            + owner.tol
            for n in range(owner.N_periods)
            if _approach._stream_active(owner, "cold", n, s)
        )
    if _piecewise._hot_parent_segmented(owner, s) or _piecewise._utility_is_segmented(
        owner, "cold", u
    ):
        return False
    return all(
        owner.T_h_in_period[n][s] - owner.T_cu_in_period[n][u]
        > _approach._cold_utility_inlet_approach_temperature(owner, s, u, n) + owner.tol
        for n in range(owner.N_periods)
        if _approach._stream_active(owner, "hot", n, s)
    )


def _duty_variable(owner, upper: float):
    return owner.m.Var(value=0, ub=upper, lb=0.0)


def _binaries(owner, allowed) -> list:
    return [
        [
            [
                (
                    owner.m.Var(value=0, ub=1, lb=0, integer=True)
                    if owner.integers
                    else owner.m.Param(value=1)
                )
                if allowed[s][u][k]
                else 0.0
                for k in range(owner.K)
            ]
            for u in range(len(allowed[s]))
        ]
        for s in range(len(allowed))
    ]


def _set_staged_heater_approach(owner, n: int, j: int, u: int, k: int) -> None:
    """Utility inlet faces the heater's cold outlet, utility outlet its inlet."""
    match = owner.z_hs[j][u][k]
    duty = owner.Q_hs_by_period[n][j][u][k]
    big_m = (
        max(
            abs(owner.T_hu_in_period[n][u] - owner.T_c_in_period[n][j]),
            abs(owner.T_hu_in_period[n][u] - owner.T_c_out_period[n][j]),
            abs(owner.T_hu_out_period[n][u] - owner.T_c_in_period[n][j]),
            abs(owner.T_hu_out_period[n][u] - owner.T_c_out_period[n][j]),
        )
        + _approach._hot_utility_inlet_approach_temperature(owner, j, u, n)
        + _approach._hot_utility_outlet_approach_temperature(owner, j, u, n)
    )
    owner.m.Equation(duty <= owner.Qtot_sc_period[n][j] * match)
    owner.m.Equation(
        owner.T_hu_in_period[n][u]
        - owner.T_c_by_period[n][j][k]
        - _approach._hot_utility_inlet_approach_temperature(owner, j, u, n)
        >= -big_m * (1 - match)
    )
    owner.m.Equation(
        owner.T_hu_out_period[n][u]
        - owner.T_c_mid_by_period[n][j][k]
        - _approach._hot_utility_outlet_approach_temperature(owner, j, u, n)
        >= -big_m * (1 - match)
    )


def _set_staged_cooler_approach(owner, n: int, i: int, u: int, k: int) -> None:
    """Utility inlet faces the cooler's hot outlet, utility outlet its inlet."""
    match = owner.z_cs[i][u][k]
    duty = owner.Q_cs_by_period[n][i][u][k]
    big_m = (
        max(
            abs(owner.T_h_in_period[n][i] - owner.T_cu_in_period[n][u]),
            abs(owner.T_h_in_period[n][i] - owner.T_cu_out_period[n][u]),
            abs(owner.T_h_out_period[n][i] - owner.T_cu_in_period[n][u]),
            abs(owner.T_h_out_period[n][i] - owner.T_cu_out_period[n][u]),
        )
        + _approach._cold_utility_inlet_approach_temperature(owner, i, u, n)
        + _approach._cold_utility_outlet_approach_temperature(owner, i, u, n)
    )
    owner.m.Equation(duty <= owner.Qtot_sh_period[n][i] * match)
    owner.m.Equation(
        owner.T_h_by_period[n][i][k]
        - owner.T_cu_in_period[n][u]
        - _approach._cold_utility_inlet_approach_temperature(owner, i, u, n)
        >= -big_m * (1 - match)
    )
    owner.m.Equation(
        owner.T_h_mid_by_period[n][i][k]
        - owner.T_cu_out_period[n][u]
        - _approach._cold_utility_outlet_approach_temperature(owner, i, u, n)
        >= -big_m * (1 - match)
    )


def _set_one_unit_per_boundary(owner, binaries, allowed) -> None:
    for s, by_utility in enumerate(allowed):
        for k in interior_boundaries(owner):
            used = [
                binaries[s][u][k] for u in range(len(by_utility)) if by_utility[u][k]
            ]
            if len(used) > 1:
                owner.m.Equation(sum(used) <= 1)


def post_process_staged_utilities(owner) -> None:
    """Read solved staged duties and set their unit counts and areas.

    Sets ``Q_hs_values`` and ``Q_cs_values`` (``[period][stream][utility]
    [boundary]``), ``n_hs_units`` and ``n_cs_units``, and ``area_hs`` and
    ``area_cs`` (``[stream][utility][boundary]``, the largest over periods,
    from the exact log-mean of each period's terminal differences). With
    ``staged_utilities`` off there are no staged units: counts are 0 and the
    grids empty.
    """
    if not getattr(owner, "staged_utilities", False):
        owner.Q_hs_values = owner.Q_cs_values = []
        owner.area_hs = owner.area_cs = []
        owner.n_hs_units = owner.n_cs_units = 0
        return
    owner.Q_hs_values = _solved_duties(owner, owner.Q_hs_by_period)
    owner.Q_cs_values = _solved_duties(owner, owner.Q_cs_by_period)
    owner.n_hs_units = _used_units(owner, owner.Q_hs_values)
    owner.n_cs_units = _used_units(owner, owner.Q_cs_values)
    owner.area_hs = _design_areas(owner, owner.Q_hs_values, _heater_area)
    owner.area_cs = _design_areas(owner, owner.Q_cs_values, _cooler_area)


def with_staged_duties(owner, duties, side: str) -> list:
    """``[period][stream][utility]`` duties with the staged ones added."""
    if not getattr(owner, "staged_utilities", False):
        return duties
    staged = owner.Q_hs_values if side == "hot" else owner.Q_cs_values
    return [
        [
            [duty + sum(staged[n][s][u]) for u, duty in enumerate(row)]
            for s, row in enumerate(duties[n])
        ]
        for n in range(owner.N_periods)
    ]


def _solved_duties(owner, grid) -> list:
    return [
        [
            [[owner._active_binary_value(q) for q in by_k] for by_k in row]
            for row in rows
        ]
        for rows in grid
    ]


def _used_units(owner, duties) -> int:
    return sum(
        1
        for s, row in enumerate(duties[0])
        for u, by_k in enumerate(row)
        for k in range(len(by_k))
        if max(duties[n][s][u][k] for n in range(owner.N_periods)) > owner.tol
    )


def _design_areas(owner, duties, area) -> list:
    return [
        [
            [
                max(
                    area(owner, n, s, u, k, duties[n][s][u][k])
                    for n in range(owner.N_periods)
                )
                for k in range(len(by_k))
            ]
            for u, by_k in enumerate(row)
        ]
        for s, row in enumerate(duties[0])
    ]


def _heater_area(owner, n: int, j: int, u: int, k: int, duty: float) -> float:
    if duty <= owner.tol:
        return 0.0
    lmtd = _area._post_process_lmtd(
        owner,
        owner.T_hu_in_period[n][u]
        - owner._active_binary_value(owner.T_c_by_period[n][j][k]),
        owner._utility_solved_outlet_temperature("hot", n, j, u, duty)
        - owner._active_binary_value(owner.T_c_mid_by_period[n][j][k]),
        1.0,
        formula_allowed=False,
    )
    return _area._heat_load_area(owner, duty, owner.U_hu_period[n][j][u], lmtd)


def _cooler_area(owner, n: int, i: int, u: int, k: int, duty: float) -> float:
    if duty <= owner.tol:
        return 0.0
    lmtd = _area._post_process_lmtd(
        owner,
        owner._active_binary_value(owner.T_h_mid_by_period[n][i][k])
        - owner._utility_solved_outlet_temperature("cold", n, i, u, duty),
        owner._active_binary_value(owner.T_h_by_period[n][i][k])
        - owner.T_cu_in_period[n][u],
        1.0,
        formula_allowed=False,
    )
    return _area._heat_load_area(owner, duty, owner.U_cu_period[n][i][u], lmtd)
