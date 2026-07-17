"""StageWise result post-processing and benefit ranking."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


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
    q_h = [
        [owner._active_binary_value(owner.Q_h_by_period[n][j]) for j in range(owner.J)]
        for n in range(owner.N_periods)
    ]
    q_c = [
        [owner._active_binary_value(owner.Q_c_by_period[n][i]) for i in range(owner.I)]
        for n in range(owner.N_periods)
    ]

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
    owner.z_hu = [
        [1 if max(q_h[n][j] for n in range(owner.N_periods)) > owner.tol else 0]
        for j in range(owner.J)
    ]
    owner.z_cu = [
        [1 if max(q_c[n][i] for n in range(owner.N_periods)) > owner.tol else 0]
        for i in range(owner.I)
    ]
    owner.n_recovery_units = sum(
        owner.z[i][j][k][0]
        for k in range(owner.S)
        for j in range(owner.J)
        for i in range(owner.I)
    )
    owner.n_hu_units = sum(owner.z_hu[j][0] for j in range(owner.J))
    owner.n_cu_units = sum(owner.z_cu[i][0] for i in range(owner.I))
    owner.n_units = owner.n_recovery_units + owner.n_hu_units + owner.n_cu_units

    owner.LMTD_r_by_period = [
        [
            [
                [
                    owner._post_process_lmtd(
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
                                - owner._recovery_approach_temperature(i, j, n)
                            )
                            >= owner.tol
                            and abs(
                                owner._active_binary_value(
                                    owner.theta_2_by_period[n][i][j][k]
                                )
                                - owner._recovery_approach_temperature(i, j, n)
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

    owner.LMTD_hu_by_period = [
        [
            owner._post_process_lmtd(
                owner.T_hu_in_period[n][0] - owner.T_c_out_period[n][j],
                owner._utility_solved_outlet_temperature("hot", n, j, q_h[n][j])
                - owner._active_binary_value(owner.T_c_by_period[n][j][0]),
                owner.z_hu[j][0],
                formula_allowed=(
                    abs(
                        (owner.T_hu_in_period[n][0] - owner.T_c_out_period[n][j])
                        - (
                            owner._utility_solved_outlet_temperature(
                                "hot", n, j, q_h[n][j]
                            )
                            - owner._active_binary_value(owner.T_c_by_period[n][j][0])
                        )
                    )
                    > owner.tol
                    and owner.T_hu_in_period[n][0]
                    - owner.T_c_out_period[n][j]
                    - owner._hot_utility_inlet_approach_temperature(j, n)
                    >= owner.tol
                    and owner._utility_solved_outlet_temperature("hot", n, j, q_h[n][j])
                    - owner._active_binary_value(owner.T_c_by_period[n][j][0])
                    - owner._hot_utility_outlet_approach_temperature(j, n, q_h[n][j])
                    >= owner.tol
                ),
            )
            for j in range(owner.J)
        ]
        for n in range(owner.N_periods)
    ]
    owner.area_hu_by_period = [
        [
            (
                q_h[n][j] / owner.U_hu_period[n][j] / owner.LMTD_hu_by_period[n][j]
                if owner.LMTD_hu_by_period[n][j] > owner.tol and q_h[n][j] > owner.tol
                else 0.0
            )
            for j in range(owner.J)
        ]
        for n in range(owner.N_periods)
    ]
    owner.LMTD_cu_by_period = [
        [
            owner._post_process_lmtd(
                owner._active_binary_value(owner.T_h_by_period[n][i][owner.S])
                - owner._utility_solved_outlet_temperature("cold", n, i, q_c[n][i]),
                owner.T_h_out_period[n][i] - owner.T_cu_in_period[n][0],
                owner.z_cu[i][0],
                formula_allowed=(
                    abs(
                        (
                            owner._active_binary_value(
                                owner.T_h_by_period[n][i][owner.S]
                            )
                            - owner._utility_solved_outlet_temperature(
                                "cold", n, i, q_c[n][i]
                            )
                        )
                        - (owner.T_h_out_period[n][i] - owner.T_cu_in_period[n][0])
                    )
                    > owner.tol
                    and owner._active_binary_value(owner.T_h_by_period[n][i][owner.S])
                    - owner._utility_solved_outlet_temperature("cold", n, i, q_c[n][i])
                    - owner._cold_utility_outlet_approach_temperature(i, n, q_c[n][i])
                    >= owner.tol
                    and owner.T_h_out_period[n][i]
                    - owner.T_cu_in_period[n][0]
                    - owner._cold_utility_inlet_approach_temperature(i, n)
                    >= owner.tol
                ),
                fallback_delta=(
                    owner.T_h_out_period[n][i] - owner.T_cu_in_period[n][0]
                ),
            )
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]
    owner.area_cu_by_period = [
        [
            (
                q_c[n][i] / owner.U_cu_period[n][i] / owner.LMTD_cu_by_period[n][i]
                if owner.LMTD_cu_by_period[n][i] > owner.tol and q_c[n][i] > owner.tol
                else 0.0
            )
            for i in range(owner.I)
        ]
        for n in range(owner.N_periods)
    ]

    owner._apply_segment_utility_areas(q_h, q_c)

    owner.LMTD_r = owner.LMTD_r_by_period[0]
    owner.LMTD_hu = owner.LMTD_hu_by_period[0]
    owner.LMTD_cu = owner.LMTD_cu_by_period[0]
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
    owner._apply_segment_recovery_areas(q_r)
    owner.area_hu = [
        max(owner.area_hu_by_period[n][j] for n in range(owner.N_periods))
        for j in range(owner.J)
    ]
    owner.area_cu = [
        max(owner.area_cu_by_period[n][i] for n in range(owner.N_periods))
        for i in range(owner.I)
    ]

    owner.Q_hu_total_by_period = [sum(q_h[n]) for n in range(owner.N_periods)]
    owner.Q_cu_total_by_period = [sum(q_c[n]) for n in range(owner.N_periods)]
    owner.Q_r_total_by_period = [
        sum(
            q_r[n][i][j][k]
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        )
        for n in range(owner.N_periods)
    ]
    owner.Q_hu_total = owner._weighted_numeric_average(owner.Q_hu_total_by_period)
    owner.Q_cu_total = owner._weighted_numeric_average(owner.Q_cu_total_by_period)
    owner.Q_r_total = owner._weighted_numeric_average(owner.Q_r_total_by_period)

    owner.operating_cost_by_period = [
        owner._utility_cost_value("hot", n, owner.Q_hu_total_by_period[n])
        + owner._utility_cost_value("cold", n, owner.Q_cu_total_by_period[n])
        for n in range(owner.N_periods)
    ]
    owner.weighted_operating_cost_value = owner._weighted_numeric_average(
        owner.operating_cost_by_period
    )
    owner.capital_cost_value = (
        owner.unit_cost[0] * owner.n_units
        + owner.A_coeff[0]
        * sum(
            owner.area_r[i][j][k] ** owner.A_exp[0]
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        )
        + owner.hu_coeff[0]
        * sum(owner.area_hu[j] ** owner.hu_exp[0] for j in range(owner.J))
        + owner.cu_coeff[0]
        * sum(owner.area_cu[i] ** owner.cu_exp[0] for i in range(owner.I))
    )
    owner.hu_cost_total = owner._weighted_numeric_average(
        [
            owner._utility_cost_value("hot", n, owner.Q_hu_total_by_period[n])
            for n in range(owner.N_periods)
        ]
    )
    owner.cu_cost_total = owner._weighted_numeric_average(
        [
            owner._utility_cost_value("cold", n, owner.Q_cu_total_by_period[n])
            for n in range(owner.N_periods)
        ]
    )
    owner.recovery_area_cost_total = owner.A_coeff[0] * sum(
        owner.area_r[i][j][k] ** owner.A_exp[0]
        for k in range(owner.S)
        for j in range(owner.J)
        for i in range(owner.I)
    )
    owner.hu_area_cost_total = owner.hu_coeff[0] * sum(
        owner.area_hu[j] ** owner.hu_exp[0] for j in range(owner.J)
    )
    owner.cu_area_cost_total = owner.cu_coeff[0] * sum(
        owner.area_cu[i] ** owner.cu_exp[0] for i in range(owner.I)
    )
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
                exact_dqda = owner._segment_exact_dqda(
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
    owner.alpha = owner.get_alpha_values()
    owner.dtacda = [
        [[None for _k in range(owner.S)] for _j in range(owner.J)]
        for _i in range(owner.I)
    ]
    for k in range(owner.S):
        for j in range(owner.J):
            for i in range(owner.I):
                if owner.area_r[i][j][k] > 0.0:
                    owner.dtacda[i][j][k] = owner.dqda[i][j][k] * (
                        owner.hu_cost[0] + owner.cu_cost[0]
                    ) - (
                        owner.A_coeff[0]
                        * owner.A_exp[0]
                        * owner.area_r[i][j][k] ** (owner.A_exp[0] - 1)
                    )
                else:
                    owner.dtacda[i][j][k] = owner.dqda[i][j][k] * (
                        owner.hu_cost[0] + owner.cu_cost[0]
                    ) - (owner.A_coeff[0] * owner.A_exp[0])
    owner.alpha_dqda = [
        [
            [owner.alpha[i][j][k][0] * owner.dqda[i][j][k] for k in range(owner.S)]
            for j in range(owner.J)
        ]
        for i in range(owner.I)
    ]

    owner.TAC_model = owner.m.options.objfcnval
    owner.TAC = owner.capital_cost_value + owner.weighted_operating_cost_value


def _weighted_numeric_average(owner, values: Sequence[float]) -> float:
    return float(
        sum(
            float(owner.period_weights[n]) * float(values[n])
            for n in range(owner.N_periods)
        )
        / owner.period_weight_sum
    )


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
                    owner.net_benefit[i][j][k] = owner.Q_r[i][j][k][0] * owner.alpha[i][
                        j
                    ][k][0] * (owner.hu_cost[0] + owner.cu_cost[0]) - (
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
