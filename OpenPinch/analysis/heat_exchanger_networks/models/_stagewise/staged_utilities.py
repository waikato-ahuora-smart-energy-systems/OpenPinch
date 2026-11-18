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
from .._base import execution as _execution
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
                # A stream absent in a period (a pinch side clips it to 0 K)
                # has no approach to keep there.
                if not _approach._stream_active(owner, "cold", n, j):
                    continue
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
                if not _approach._stream_active(owner, "hot", n, i):
                    continue
                for u in range(owner.NCU):
                    if owner.z_cs_allowed[i][u][k]:
                        _set_staged_cooler_approach(owner, n, i, u, k)
    if owner.integers:
        _set_one_unit_per_boundary(owner, owner.z_hs, owner.z_hs_allowed)
        _set_one_unit_per_boundary(owner, owner.z_cs, owner.z_cs_allowed)


def _allowed(owner, side: str) -> list:
    """Flag ``[stream][utility][boundary]`` for each staged unit offered.

    A staged unit is offered when ``staged_utilities`` is on (on a pinch
    side, for heaters above the pinch and coolers below it), the stream and
    utility are not segmented, and the utility's inlet clears the stream's
    supply by its approach in every period the stream runs, so some duty is
    possible.
    """
    streams, utilities = (owner.J, owner.NHU) if side == "hot" else (owner.I, owner.NCU)
    # A pinch side has heaters only above the pinch and coolers only below.
    side_location = getattr(owner, "pinch_loc", None)
    wrong_side = side_location is not None and side_location != (
        "above" if side == "hot" else "below"
    )
    if not getattr(owner, "staged_utilities", False) or wrong_side:
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
    # As for the stream-end units, each offered binary becomes ``[flag]``:
    # whether the unit carries duty in some period.
    owner.z_hs = _used_flags(owner, owner.Q_hs_values, owner.z_hs_allowed)
    owner.z_cs = _used_flags(owner, owner.Q_cs_values, owner.z_cs_allowed)


def _used_flags(owner, duties, allowed) -> list:
    return [
        [
            [
                [
                    int(
                        max(duties[n][s][u][k] for n in range(owner.N_periods))
                        > owner.tol
                    )
                ]
                if offered
                else 0.0
                for k, offered in enumerate(by_k)
            ]
            for u, by_k in enumerate(row)
        ]
        for s, row in enumerate(allowed)
    ]


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


def staged_terminal_differences(
    owner, side: str, n: int, s: int, u: int, k: int, *, value=lambda x: x
):
    """A staged unit's (hot-end, cold-end) temperature differences.

    ``value`` reads a solved temperature; by default the model expressions
    are returned for the solver.
    """
    if side == "hot":
        return (
            owner.T_hu_in_period[n][u] - value(owner.T_c_by_period[n][s][k]),
            owner._utility_solved_outlet_temperature("hot", n, s, u)
            - value(owner.T_c_mid_by_period[n][s][k]),
        )
    return (
        value(owner.T_h_mid_by_period[n][s][k])
        - owner._utility_solved_outlet_temperature("cold", n, s, u),
        value(owner.T_h_by_period[n][s][k]) - owner.T_cu_in_period[n][u],
    )


def set_staged_initial_values(owner, source) -> None:
    """Copy a solved parent's staged units into this model.

    Each offered unit takes the parent's binary, duties and mid temperatures
    (period by period, the parent's last period where it has fewer); a unit
    the parent did not offer, or a parent without staging, gives a unit
    switched off with no duty, the mid temperature at the boundary's.
    """
    if not getattr(owner, "staged_utilities", False):
        return
    staged_source = getattr(source, "staged_utilities", False)
    for side in ("hot", "cold"):
        if side == "hot":
            allowed, flags, duties, mids, boundaries = (
                owner.z_hs_allowed,
                owner.z_hs,
                owner.Q_hs_by_period,
                owner.T_c_mid_by_period,
                owner.T_c_by_period,
            )
        else:
            allowed, flags, duties, mids, boundaries = (
                owner.z_cs_allowed,
                owner.z_cs,
                owner.Q_cs_by_period,
                owner.T_h_mid_by_period,
                owner.T_h_by_period,
            )
        prefix = "hs" if side == "hot" else "cs"
        mid_name = "T_c_mid_by_period" if side == "hot" else "T_h_mid_by_period"
        source_allowed = getattr(source, f"z_{prefix}_allowed", None)
        for s, row in enumerate(allowed):
            for u, by_k in enumerate(row):
                for k in interior_boundaries(owner):
                    if not by_k[k]:
                        continue
                    carried = bool(
                        staged_source and source_allowed and source_allowed[s][u][k]
                    )
                    _execution._set_value(
                        owner,
                        flags[s][u][k],
                        owner._active_binary_value(
                            getattr(source, f"z_{prefix}")[s][u][k]
                        )
                        if carried
                        else 0.0,
                    )
                    for n in range(owner.N_periods):
                        source_n = min(
                            n, len(getattr(source, f"Q_{prefix}_by_period", [[]])) - 1
                        )
                        _execution._set_value(
                            owner,
                            duties[n][s][u][k],
                            owner._active_binary_value(
                                getattr(source, f"Q_{prefix}_by_period")[source_n][s][
                                    u
                                ][k]
                            )
                            if carried
                            else 0.0,
                        )
        for n in range(owner.N_periods):
            for s, row in enumerate(mids[n]):
                for k in interior_boundaries(owner):
                    if row[k] is boundaries[n][s][k]:
                        continue
                    source_grid = (
                        getattr(source, mid_name)
                        if staged_source
                        else getattr(
                            source,
                            "T_c_by_period" if side == "hot" else "T_h_by_period",
                        )
                    )
                    source_n = min(n, len(source_grid) - 1)
                    _execution._set_value(
                        owner,
                        row[k],
                        owner._active_binary_value(source_grid[source_n][s][k]),
                    )


def staged_units_on(owner) -> tuple:
    """The staged units switched on, as ``(side, stream, utility, boundary)``.

    Empty without staging.
    """
    if not getattr(owner, "staged_utilities", False):
        return ()
    return tuple(
        (side, s, u, k)
        for side, flags, allowed in (
            ("hot", owner.z_hs, owner.z_hs_allowed),
            ("cold", owner.z_cs, owner.z_cs_allowed),
        )
        for s, row in enumerate(allowed)
        for u, by_k in enumerate(row)
        for k in interior_boundaries(owner)
        if by_k[k] and owner._active_binary_value(flags[s][u][k]) > 0.5
    )
