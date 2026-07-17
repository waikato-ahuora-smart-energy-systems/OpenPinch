"""Exact segment-area calculations and post-solve LMTD handling."""

from __future__ import annotations

import numpy as np

from OpenPinch.analysis.heat_transfer import compute_LMTD_from_dts


def _post_process_lmtd(
    model,
    delta_1: float,
    delta_2: float,
    active: float,
    *,
    formula_allowed: bool,
    fallback_delta: float | None = None,
) -> float:
    """Return source-compatible post-process LMTD.

    Heat exchanger network synthesis owns the OpenHENS active-unit and
    dTmin/tolerance gates.
    Once those gates pass, the shared OpenPinch heat-exchanger utility owns
    the positive endpoint logarithmic-mean formula.
    """

    if not formula_allowed:
        return (delta_1 if fallback_delta is None else fallback_delta) * active
    return active * float(compute_LMTD_from_dts(delta_1, delta_2))


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


def _apply_segment_utility_areas(model, q_h, q_c) -> None:
    """Use local process segments for hot- and cold-utility area totals."""
    if not hasattr(model, "solver_arrays"):
        return
    if not (
        any(model._solver_parent_is_segmented("hot", i) for i in range(model.I))
        or any(model._solver_parent_is_segmented("cold", j) for j in range(model.J))
        or model._utility_is_segmented("hot")
        or model._utility_is_segmented("cold")
    ):
        return
    from ...solver.piecewise import (
        duty_aligned_area_contributions,
        profile_from_solver_arrays,
        utility_thermal_profile,
    )

    hot_utility_identity = model.solver_arrays.utility_identities["hot_utilities"][0]
    cold_utility_identity = model.solver_arrays.utility_identities["cold_utilities"][0]
    model.segment_area_hu_contributions_by_period = [
        [() for _j in range(model.J)] for _n in range(model.N_periods)
    ]
    model.segment_area_cu_contributions_by_period = [
        [() for _i in range(model.I)] for _n in range(model.N_periods)
    ]
    for n in range(model.N_periods):
        period = str(model.period_ids[n])
        for j in range(model.J):
            duty = float(q_h[n][j])
            if duty <= model.tol or not (
                model._solver_parent_is_segmented("cold", j)
                or model._utility_is_segmented("hot")
            ):
                continue
            hot_utility_profile = (
                profile_from_solver_arrays(
                    model.solver_arrays,
                    side="hot_utility",
                    parent_index=0,
                    period_index=n,
                )
                if model._utility_is_segmented("hot")
                else utility_thermal_profile(
                    identity=hot_utility_identity,
                    inlet_temperature=model.T_hu_in_period[n][0],
                    outlet_temperature=model.T_hu_out_period[n][0],
                    duty=duty,
                    heat_transfer_coefficient=model.htc_hu_period[n][0],
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
                hot_inlet_temperature=model.T_hu_in_period[n][0],
                cold_inlet_temperature=model._active_binary_value(
                    model.T_c_by_period[n][j][0]
                ),
                period=period,
                tolerance=model.tol,
            )
            model.segment_area_hu_contributions_by_period[n][j] = contributions
            model.area_hu_by_period[n][j] = sum(
                contribution.area for contribution in contributions
            )

        for i in range(model.I):
            duty = float(q_c[n][i])
            if duty <= model.tol or not (
                model._solver_parent_is_segmented("hot", i)
                or model._utility_is_segmented("cold")
            ):
                continue
            cold_utility_profile = (
                profile_from_solver_arrays(
                    model.solver_arrays,
                    side="cold_utility",
                    parent_index=0,
                    period_index=n,
                )
                if model._utility_is_segmented("cold")
                else utility_thermal_profile(
                    identity=cold_utility_identity,
                    inlet_temperature=model.T_cu_in_period[n][0],
                    outlet_temperature=model.T_cu_out_period[n][0],
                    duty=duty,
                    heat_transfer_coefficient=model.htc_cu_period[n][0],
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
                cold_inlet_temperature=model.T_cu_in_period[n][0],
                period=period,
                tolerance=model.tol,
            )
            model.segment_area_cu_contributions_by_period[n][i] = contributions
            model.area_cu_by_period[n][i] = sum(
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
        model._solver_parent_is_segmented("hot", hot_parent_index)
        or model._solver_parent_is_segmented("cold", cold_parent_index)
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
