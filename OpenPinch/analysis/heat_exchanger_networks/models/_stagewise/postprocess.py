"""StageWise result post-processing and benefit ranking."""

from __future__ import annotations

import numpy as np

from .._base import alpha as _alpha
from .._base import approach as _approach
from .._base import area as _area
from .._base.costing import set_period_totals_and_costs


def get_post_process(owner) -> None:
    """Extract source post-process arrays after a successful solve."""

    if owner.mSuccess != 1:
        return
    owner._get_multiperiod_post_process()


def _get_multiperiod_post_process(owner) -> None:
    q_r = [
        [
            [
                [
                    owner._active_binary_value(owner.Q_r_by_period[n][i][j][k])
                    for k in range(owner.S)
                ]
                for j in range(owner.J)
            ]
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]
    q_h, q_c = _area.solved_utility_duties(owner)

    owner.z = [
        [
            [
                [
                    (
                        1
                        if max(q_r[n][i][j][k] for n in range(owner.N_periods))
                        > owner.tol
                        else 0
                    )
                ]
                for k in range(owner.S)
            ]
            for j in range(owner.J)
        ]
        for i in range(owner.I)
    ]
    owner.z_hu, owner.z_cu = _area.used_utility_matches(owner, q_h, q_c)
    owner.n_recovery_units = sum(
        owner.z[i][j][k][0]
        for k in range(owner.S)
        for j in range(owner.J)
        for i in range(owner.I)
    )
    owner.n_hu_units = sum(flag[0] for row in owner.z_hu for flag in row)
    owner.n_cu_units = sum(flag[0] for row in owner.z_cu for flag in row)
    owner.n_units = owner.n_recovery_units + owner.n_hu_units + owner.n_cu_units

    owner.LMTD_r_by_period = [
        [
            [
                [
                    _area._post_process_lmtd(
                        owner,
                        owner._active_binary_value(owner.theta_1_by_period[n][i][j][k]),
                        owner._active_binary_value(owner.theta_2_by_period[n][i][j][k]),
                        owner.z[i][j][k][0],
                        formula_allowed=(
                            abs(
                                owner._active_binary_value(
                                    owner.theta_1_by_period[n][i][j][k]
                                )
                                - owner._active_binary_value(
                                    owner.theta_2_by_period[n][i][j][k]
                                )
                            )
                            > owner.tol
                            and abs(
                                owner._active_binary_value(
                                    owner.theta_1_by_period[n][i][j][k]
                                )
                                - _approach._recovery_approach_temperature(
                                    owner, i, j, n
                                )
                            )
                            >= owner.tol
                            and abs(
                                owner._active_binary_value(
                                    owner.theta_2_by_period[n][i][j][k]
                                )
                                - _approach._recovery_approach_temperature(
                                    owner, i, j, n
                                )
                            )
                            >= owner.tol
                        ),
                    )
                    for k in range(owner.S)
                ]
                for j in range(owner.J)
            ]
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]
    owner.area_r_by_period = [
        [
            [
                [
                    (
                        q_r[n][i][j][k]
                        / owner.U_r_period[n][i][j]
                        / owner.LMTD_r_by_period[n][i][j][k]
                        if owner.LMTD_r_by_period[n][i][j][k] > owner.tol
                        and q_r[n][i][j][k] > owner.tol
                        else 0.0
                    )
                    for k in range(owner.S)
                ]
                for j in range(owner.J)
            ]
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]

    _area.set_utility_lmtd_and_areas(owner, q_h, q_c)

    owner.LMTD_r = owner.LMTD_r_by_period[0]
    owner.area_r = [
        [
            [
                max(owner.area_r_by_period[n][i][j][k] for n in range(owner.N_periods))
                for k in range(owner.S)
            ]
            for j in range(owner.J)
        ]
        for i in range(owner.I)
    ]
    _area._apply_segment_recovery_areas(owner, q_r)

    set_period_totals_and_costs(owner, q_h, q_c, q_r)
    owner.unit_cost_total = owner.unit_cost[0] * owner.n_recovery_units
    owner.utility_unit_cost_total = owner.unit_cost[0] * (
        owner.n_hu_units + owner.n_cu_units
    )

    owner.dqda = [
        [[None for _k in range(owner.S)] for _j in range(owner.J)]
        for _i in range(owner.I)
    ]
    for k in range(owner.S):
        for j in range(owner.J):
            for i in range(owner.I):
                driving_force = owner._active_binary_value(
                    owner.T_h[i][k]
                ) - owner._active_binary_value(owner.T_c[j][k + 1])
                if q_r[0][i][j][k] > 0.0 and driving_force > 0.0:
                    owner.dqda[i][j][k] = (
                        owner._active_binary_value(owner.theta_1[i][j][k])
                        * owner._active_binary_value(owner.theta_2[i][j][k])
                        * owner.U_r[i][j]
                    ) / driving_force
                elif driving_force > 0.0:
                    owner.dqda[i][j][k] = owner.U_r[i][j] * driving_force
                else:
                    owner.dqda[i][j][k] = 0.0
                exact_dqda = _area._segment_exact_dqda(
                    owner,
                    period_index=0,
                    hot_parent_index=i,
                    cold_parent_index=j,
                    duty=float(q_r[0][i][j][k]),
                    hot_inlet_temperature=owner._active_binary_value(
                        owner.T_h_by_period[0][i][k]
                    ),
                    cold_inlet_temperature=owner._active_binary_value(
                        owner.T_c_by_period[0][j][k + 1]
                    ),
                )
                if exact_dqda is not None:
                    owner.dqda[i][j][k] = exact_dqda
    alpha = owner.get_alpha_values()
    # A failed alpha solve is not cached; rank with neutral alpha instead.
    owner.alpha = alpha if alpha is not None else []
    owner.dtacda = [
        [[None for _k in range(owner.S)] for _j in range(owner.J)]
        for _i in range(owner.I)
    ]
    for k in range(owner.S):
        for j in range(owner.J):
            for i in range(owner.I):
                if owner.area_r[i][j][k] > 0.0:
                    owner.dtacda[i][j][k] = owner.dqda[i][j][k] * (
                        _area.recovered_utility_price(owner)
                    ) - (
                        owner.A_coeff[0]
                        * owner.A_exp[0]
                        * owner.area_r[i][j][k] ** (owner.A_exp[0] - 1)
                    )
                else:
                    owner.dtacda[i][j][k] = owner.dqda[i][j][k] * (
                        _area.recovered_utility_price(owner)
                    ) - (owner.A_coeff[0] * owner.A_exp[0])
    owner.alpha_dqda = [
        [
            [
                _alpha.alpha_value(owner, i, j, k) * owner.dqda[i][j][k]
                for k in range(owner.S)
            ]
            for j in range(owner.J)
        ]
        for i in range(owner.I)
    ]

    owner.TAC_model = owner.m.options.objfcnval
    owner.TAC = owner.capital_cost_value + owner.weighted_operating_cost_value


def get_lowest_benefit_HX(owner) -> list[list[int]]:
    """Return the active exchanger with the lowest source net benefit."""

    return owner.get_lowest_benefit_HX_candidates(1)


def get_lowest_benefit_HX_candidates(owner, limit: int) -> list[list[int]]:
    """Return active exchangers sorted by ascending source net benefit."""

    owner.net_benefit = np.array(
        [
            [[0.0 for _k in range(owner.S)] for _j in range(owner.J)]
            for _i in range(owner.I)
        ]
    )
    candidates: list[tuple[float, int, list[int]]] = []
    order = 0
    for k in range(owner.S):
        for j in range(owner.J):
            for i in range(owner.I):
                if owner._active_binary_value(owner.z[i][j][k]) > owner.tol:
                    owner.net_benefit[i][j][k] = owner.Q_r[i][j][k][
                        0
                    ] * _alpha.alpha_value(owner, i, j, k) * (
                        _area.recovered_utility_price(owner)
                    ) - (
                        owner.unit_cost[0]
                        + owner.A_coeff[0] * (owner.area_r[i][j][k] ** owner.A_exp[0])
                    )
                    candidates.append(
                        (float(owner.net_benefit[i][j][k]), order, [i, j, k])
                    )
                order += 1
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [position for _benefit, _order, position in candidates[: int(limit)]]


def get_max_benefit_HX(owner) -> list[list[int]]:
    """Return the inactive feasible exchanger with the highest alpha-dQ/dA."""

    return owner.get_max_benefit_HX_candidates(1)


def get_max_benefit_HX_candidates(owner, limit: int) -> list[list[int]]:
    """Return inactive feasible exchangers sorted by descending alpha-dQ/dA."""

    owner.net_benefit = np.array(
        [
            [[0.0 for _k in range(owner.S)] for _j in range(owner.J)]
            for _i in range(owner.I)
        ]
    )
    candidates: list[tuple[float, int, list[int]]] = []
    order = 0
    for k in range(owner.S):
        for j in range(owner.J):
            for i in range(owner.I):
                if (
                    owner._active_binary_value(owner.z[i][j][k]) <= owner.tol
                    and owner.alpha_dqda[i][j][k] > 0.0
                    and owner.z_feasible[i][j][k]
                ):
                    candidates.append(
                        (-float(owner.alpha_dqda[i][j][k]), order, [i, j, k])
                    )
                order += 1
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [position for _benefit, _order, position in candidates[: int(limit)]]
