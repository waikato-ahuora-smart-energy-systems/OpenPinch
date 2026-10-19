"""Exact segment-area calculations and post-solve LMTD handling."""

from __future__ import annotations

import numpy as np

from OpenPinch.analysis.heat_transfer import compute_LMTD_from_dts

from . import piecewise as _piecewise


def _post_process_lmtd(
    model,
    delta_1: float,
    delta_2: float,
    active: float,
    *,
    formula_allowed: bool,
) -> float:
    """Return the post-solve log-mean temperature difference of one match.

    This uses the true log-mean whenever both ends are positive, including an
    end at its minimum approach (OpenHENS fell back to ``delta_1`` there, which
    misstates area at the pinch, where matches touch that bound by design).
    Equal ends give the arithmetic mean and inactive matches 0. A match with a
    non-positive end is crossed and has no log-mean: it returns 0, so the match
    gets no area and verification reports it. ``formula_allowed`` is kept for
    callers but no longer gates the formula.
    """
    del formula_allowed
    tolerance = float(getattr(model, "tol", 1e-6))
    if active <= tolerance:
        return 0.0
    if delta_1 > tolerance and delta_2 > tolerance:
        if abs(delta_1 - delta_2) <= tolerance:
            return active * 0.5 * (delta_1 + delta_2)
        return active * float(compute_LMTD_from_dts(delta_1, delta_2))
    return 0.0


def _apply_segment_recovery_areas(model, q_r) -> None:
    """Replace aggregate-CP recovery areas with ordered local slice totals."""
    if not hasattr(model, "solver_arrays"):
        return
    arrays = model.solver_arrays.arrays
    if not {"hot_segment_count", "cold_segment_count"}.issubset(arrays):
        return
    if not (
        np.any(np.asarray(arrays["hot_segment_count"]) > 1)
        or np.any(np.asarray(arrays["cold_segment_count"]) > 1)
    ):
        return

    from ...solver.piecewise import (
        duty_aligned_area_contributions,
        profile_from_solver_arrays,
    )

    contribution_grid = [
        [
            [[() for _k in range(model.S)] for _j in range(model.J)]
            for _i in range(model.I)
        ]
        for _n in range(model.N_periods)
    ]
    area_grid = [
        [
            [[0.0 for _k in range(model.S)] for _j in range(model.J)]
            for _i in range(model.I)
        ]
        for _n in range(model.N_periods)
    ]
    for n in range(model.N_periods):
        period = str(model.period_ids[n])
        for i in range(model.I):
            hot_profile = profile_from_solver_arrays(
                model.solver_arrays,
                side="hot",
                parent_index=i,
                period_index=n,
            )
            for j in range(model.J):
                cold_profile = profile_from_solver_arrays(
                    model.solver_arrays,
                    side="cold",
                    parent_index=j,
                    period_index=n,
                )
                if len(hot_profile.duties) == len(cold_profile.duties) == 1:
                    for k in range(model.S):
                        area_grid[n][i][j][k] = model.area_r_by_period[n][i][j][k]
                    continue
                for k in range(model.S):
                    duty = float(q_r[n][i][j][k])
                    if duty <= model.tol:
                        continue
                    contributions = duty_aligned_area_contributions(
                        hot_profile,
                        cold_profile,
                        duty=duty,
                        hot_inlet_temperature=model._active_binary_value(
                            model.T_h_by_period[n][i][k]
                        ),
                        cold_inlet_temperature=model._active_binary_value(
                            model.T_c_by_period[n][j][k + 1]
                        ),
                        period=period,
                        tolerance=model.tol,
                    )
                    contribution_grid[n][i][j][k] = contributions
                    area_grid[n][i][j][k] = sum(
                        contribution.area for contribution in contributions
                    )
    model.segment_area_contributions_by_period = contribution_grid
    model.area_r_by_period = area_grid
    model.area_r = [
        [
            [
                max(area_grid[n][i][j][k] for n in range(model.N_periods))
                for k in range(model.S)
            ]
            for j in range(model.J)
        ]
        for i in range(model.I)
    ]


def solved_utility_duties(model) -> tuple[list, list]:
    """Return solved heater ``[period][cold][utility]`` and cooler duties."""
    q_h = [
        [
            [model._active_binary_value(value) for value in row]
            for row in model.Q_h_by_period[n]
        ]
        for n in range(model.N_periods)
    ]
    q_c = [
        [
            [model._active_binary_value(value) for value in row]
            for row in model.Q_c_by_period[n]
        ]
        for n in range(model.N_periods)
    ]
    return q_h, q_c


def used_utility_matches(model, q_h, q_c) -> tuple[list, list]:
    """Return ``[[flag]]`` per (stream, utility): whether a period uses it."""

    def used(duties, index: int, u: int) -> list[int]:
        peak = max(duties[n][index][u] for n in range(model.N_periods))
        return [1 if peak > model.tol else 0]

    return (
        [[used(q_h, j, u) for u in range(model.NHU)] for j in range(model.J)],
        [[used(q_c, i, u) for u in range(model.NCU)] for i in range(model.I)],
    )


def recovered_utility_price(model) -> float:
    """Return the price of the utility heat one unit of recovery saves.

    Recovery saves heat from the hot and the cold utility in use; where
    several are in use, each counts by its share of the solved period-0 duty.
    """

    def price(prices, totals) -> float:
        used = sum(totals)
        if used <= model.tol:
            return float(min(prices))
        return float(sum(p * q for p, q in zip(prices, totals, strict=True)) / used)

    return price(model.hu_cost_period[0], model.Q_hu_by_utility_by_period[0]) + price(
        model.cu_cost_period[0], model.Q_cu_by_utility_by_period[0]
    )


def set_utility_lmtd_and_areas(model, q_h, q_c) -> None:
    """Set every heater's and cooler's log-mean difference and area by period.

    ``q_h`` is ``[period][cold][utility]`` and ``q_c`` ``[period][hot][utility]``;
    each (stream, utility) pair is its own unit, and its design area is the
    largest over the periods.
    """
    model.LMTD_hu_by_period = [
        [
            [
                _post_process_lmtd(
                    model,
                    *_heater_terminal_differences(model, n, j, u, q_h[n][j][u]),
                    model.z_hu[j][u][0],
                    formula_allowed=False,
                )
                for u in range(model.NHU)
            ]
            for j in range(model.J)
        ]
        for n in range(model.N_periods)
    ]
    model.area_hu_by_period = [
        [
            [
                _heat_load_area(model, q_h[n][j][u], model.U_hu_period[n][j][u], lmtd)
                for u, lmtd in enumerate(model.LMTD_hu_by_period[n][j])
            ]
            for j in range(model.J)
        ]
        for n in range(model.N_periods)
    ]
    model.LMTD_cu_by_period = [
        [
            [
                _post_process_lmtd(
                    model,
                    *_cooler_terminal_differences(model, n, i, u, q_c[n][i][u]),
                    model.z_cu[i][u][0],
                    formula_allowed=False,
                )
                for u in range(model.NCU)
            ]
            for i in range(model.I)
        ]
        for n in range(model.N_periods)
    ]
    model.area_cu_by_period = [
        [
            [
                _heat_load_area(model, q_c[n][i][u], model.U_cu_period[n][i][u], lmtd)
                for u, lmtd in enumerate(model.LMTD_cu_by_period[n][i])
            ]
            for i in range(model.I)
        ]
        for n in range(model.N_periods)
    ]
    _apply_segment_utility_areas(model, q_h, q_c)
    model.LMTD_hu = model.LMTD_hu_by_period[0]
    model.LMTD_cu = model.LMTD_cu_by_period[0]
    model.area_hu = [
        [
            max(model.area_hu_by_period[n][j][u] for n in range(model.N_periods))
            for u in range(model.NHU)
        ]
        for j in range(model.J)
    ]
    model.area_cu = [
        [
            max(model.area_cu_by_period[n][i][u] for n in range(model.N_periods))
            for u in range(model.NCU)
        ]
        for i in range(model.I)
    ]


def _heater_terminal_differences(model, n: int, j: int, u: int, duty: float):
    return (
        model.T_hu_in_period[n][u] - model.T_c_out_period[n][j],
        model._utility_solved_outlet_temperature("hot", n, j, u, duty)
        - model._active_binary_value(model.T_c_by_period[n][j][0]),
    )


def _cooler_terminal_differences(model, n: int, i: int, u: int, duty: float):
    return (
        model._active_binary_value(model.T_h_by_period[n][i][model.S])
        - model._utility_solved_outlet_temperature("cold", n, i, u, duty),
        model.T_h_out_period[n][i] - model.T_cu_in_period[n][u],
    )


def _heat_load_area(model, duty: float, coefficient: float, lmtd: float) -> float:
    if lmtd <= model.tol or duty <= model.tol:
        return 0.0
    return duty / coefficient / lmtd


def _apply_segment_utility_areas(model, q_h, q_c) -> None:
    """Use local segments for the area of heaters and coolers that have them."""
    if not hasattr(model, "solver_arrays"):
        return
    if not (
        any(
            _piecewise._solver_parent_is_segmented(model, "hot", i)
            for i in range(model.I)
        )
        or any(
            _piecewise._solver_parent_is_segmented(model, "cold", j)
            for j in range(model.J)
        )
        or _piecewise._any_utility_segmented(model, "hot")
        or _piecewise._any_utility_segmented(model, "cold")
    ):
        return
    from ...solver.piecewise import (
        duty_aligned_area_contributions,
        profile_from_solver_arrays,
        utility_thermal_profile,
    )

    identities = model.solver_arrays.utility_identities
    model.segment_area_hu_contributions_by_period = [
        [[() for _u in range(model.NHU)] for _j in range(model.J)]
        for _n in range(model.N_periods)
    ]
    model.segment_area_cu_contributions_by_period = [
        [[() for _u in range(model.NCU)] for _i in range(model.I)]
        for _n in range(model.N_periods)
    ]
    for n in range(model.N_periods):
        period = str(model.period_ids[n])
        for j in range(model.J):
            for u in range(model.NHU):
                duty = float(q_h[n][j][u])
                segmented_utility = _piecewise._utility_is_segmented(model, "hot", u)
                if duty <= model.tol or not (
                    _piecewise._solver_parent_is_segmented(model, "cold", j)
                    or segmented_utility
                ):
                    continue
                hot_utility_profile = (
                    profile_from_solver_arrays(
                        model.solver_arrays,
                        side="hot_utility",
                        parent_index=u,
                        period_index=n,
                    )
                    if segmented_utility
                    else utility_thermal_profile(
                        identity=identities["hot_utilities"][u],
                        inlet_temperature=model.T_hu_in_period[n][u],
                        outlet_temperature=model.T_hu_out_period[n][u],
                        duty=duty,
                        heat_transfer_coefficient=model.htc_hu_period[n][u],
                    )
                )
                contributions = duty_aligned_area_contributions(
                    hot_utility_profile,
                    profile_from_solver_arrays(
                        model.solver_arrays,
                        side="cold",
                        parent_index=j,
                        period_index=n,
                    ),
                    duty=duty,
                    hot_inlet_temperature=model.T_hu_in_period[n][u],
                    cold_inlet_temperature=model._active_binary_value(
                        model.T_c_by_period[n][j][0]
                    ),
                    period=period,
                    tolerance=model.tol,
                )
                model.segment_area_hu_contributions_by_period[n][j][u] = contributions
                model.area_hu_by_period[n][j][u] = sum(
                    contribution.area for contribution in contributions
                )

        for i in range(model.I):
            for u in range(model.NCU):
                duty = float(q_c[n][i][u])
                segmented_utility = _piecewise._utility_is_segmented(model, "cold", u)
                if duty <= model.tol or not (
                    _piecewise._solver_parent_is_segmented(model, "hot", i)
                    or segmented_utility
                ):
                    continue
                cold_utility_profile = (
                    profile_from_solver_arrays(
                        model.solver_arrays,
                        side="cold_utility",
                        parent_index=u,
                        period_index=n,
                    )
                    if segmented_utility
                    else utility_thermal_profile(
                        identity=identities["cold_utilities"][u],
                        inlet_temperature=model.T_cu_in_period[n][u],
                        outlet_temperature=model.T_cu_out_period[n][u],
                        duty=duty,
                        heat_transfer_coefficient=model.htc_cu_period[n][u],
                    )
                )
                contributions = duty_aligned_area_contributions(
                    profile_from_solver_arrays(
                        model.solver_arrays,
                        side="hot",
                        parent_index=i,
                        period_index=n,
                    ),
                    cold_utility_profile,
                    duty=duty,
                    hot_inlet_temperature=model._active_binary_value(
                        model.T_h_by_period[n][i][model.S]
                    ),
                    cold_inlet_temperature=model.T_cu_in_period[n][u],
                    period=period,
                    tolerance=model.tol,
                )
                model.segment_area_cu_contributions_by_period[n][i][u] = contributions
                model.area_cu_by_period[n][i][u] = sum(
                    contribution.area for contribution in contributions
                )


def _segment_exact_dqda(
    model,
    *,
    period_index: int,
    hot_parent_index: int,
    cold_parent_index: int,
    duty: float,
    hot_inlet_temperature: float,
    cold_inlet_temperature: float,
) -> float | None:
    """Return a local numerical dQ/dA from ordered segment-summed area."""
    if not (
        _piecewise._solver_parent_is_segmented(model, "hot", hot_parent_index)
        or _piecewise._solver_parent_is_segmented(model, "cold", cold_parent_index)
    ):
        return None

    from ...solver.piecewise import (
        duty_aligned_area_contributions,
        profile_from_solver_arrays,
    )

    hot_profile = profile_from_solver_arrays(
        model.solver_arrays,
        side="hot",
        parent_index=hot_parent_index,
        period_index=period_index,
    )
    cold_profile = profile_from_solver_arrays(
        model.solver_arrays,
        side="cold",
        parent_index=cold_parent_index,
        period_index=period_index,
    )
    hot_start = hot_profile.heat_at_temperature(hot_inlet_temperature)
    cold_start = cold_profile.heat_at_temperature(cold_inlet_temperature)
    maximum_duty = min(
        hot_profile.total_duty - hot_start,
        cold_profile.total_duty - cold_start,
    )
    epsilon = max(maximum_duty * 1e-5, model.tol * 10.0, 1e-6)
    lower_duty = max(0.0, duty - epsilon)
    upper_duty = min(maximum_duty, duty + epsilon)
    if upper_duty - lower_duty <= model.tol:
        return None

    def area_at(value: float) -> float:
        if value <= model.tol:
            return 0.0
        contributions = duty_aligned_area_contributions(
            hot_profile,
            cold_profile,
            duty=value,
            hot_inlet_temperature=hot_inlet_temperature,
            cold_inlet_temperature=cold_inlet_temperature,
            period=str(model.period_ids[period_index]),
            tolerance=model.tol,
        )
        return sum(contribution.area for contribution in contributions)

    try:
        area_delta = area_at(upper_duty) - area_at(lower_duty)
    except ValueError:
        return None
    if area_delta <= model.tol:
        return None
    return (upper_duty - lower_duty) / area_delta
