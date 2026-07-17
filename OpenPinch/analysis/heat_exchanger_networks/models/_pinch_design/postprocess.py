"""Pinch-decomposition result post-processing."""

from __future__ import annotations

from collections.abc import Sequence

from ...indexing import build_index_grid
from .._stagewise.verification import _value as _scalar_value
from .preprocessing import (
    _active_period_flag,
    _area_from_heat_load,
    _lmtd_formula_allowed,
)


def get_post_process(owner) -> None:
    """Extract source PDM side arrays after a successful solve."""

    if owner.mSuccess != 1:
        return
    owner._get_multiperiod_post_process()


def _get_multiperiod_post_process(owner) -> None:
    q_r = build_index_grid(
        lambda n, i, j, k: owner._active_binary_value(owner.Q_r_by_period[n][i][j][k]),
        (owner.N_periods, owner.I, owner.J, owner.S),
    )
    q_h = build_index_grid(
        lambda n, j: owner._active_binary_value(owner.Q_h_by_period[n][j]),
        (owner.N_periods, owner.J),
    )
    q_c = build_index_grid(
        lambda n, i: owner._active_binary_value(owner.Q_c_by_period[n][i]),
        (owner.N_periods, owner.I),
    )

    owner.z = build_index_grid(
        lambda i, j, k: _active_period_flag(
            (q_r[n][i][j][k] for n in range(owner.N_periods)),
            owner.tol,
        ),
        (owner.I, owner.J, owner.S),
    )
    owner.z_hu = build_index_grid(
        lambda j: _active_period_flag(
            (q_h[n][j] for n in range(owner.N_periods)),
            owner.tol,
        ),
        (owner.J,),
    )
    owner.z_cu = build_index_grid(
        lambda i: _active_period_flag(
            (q_c[n][i] for n in range(owner.N_periods)),
            owner.tol,
        ),
        (owner.I,),
    )
    owner.n_recovery_units = sum(
        owner.z[i][j][k][0]
        for k in range(owner.S)
        for j in range(owner.J)
        for i in range(owner.I)
    )
    owner.n_hu_units = sum(owner.z_hu[j][0] for j in range(owner.J))
    owner.n_cu_units = sum(owner.z_cu[i][0] for i in range(owner.I))
    owner.n_units = owner.n_recovery_units + owner.n_hu_units + owner.n_cu_units

    def recovery_temperature_difference(
        n: int, i: int, j: int, hot_stage: int, cold_stage: int
    ) -> list[float]:
        if owner.z[i][j][cold_stage][0] <= 0:
            return [owner._recovery_approach_temperature(i, j, n)]
        return [
            owner._active_binary_value(owner.T_h_by_period[n][i][hot_stage])
            - owner._active_binary_value(owner.T_c_by_period[n][j][hot_stage])
        ]

    owner.theta_1_by_period = build_index_grid(
        lambda n, i, j, k: recovery_temperature_difference(n, i, j, k, k),
        (owner.N_periods, owner.I, owner.J, owner.S),
    )
    owner.theta_2_by_period = build_index_grid(
        lambda n, i, j, k: recovery_temperature_difference(n, i, j, k + 1, k),
        (owner.N_periods, owner.I, owner.J, owner.S),
    )
    owner.theta_1 = owner.theta_1_by_period[0]
    owner.theta_2 = owner.theta_2_by_period[0]

    owner.LMTD_r_by_period = build_index_grid(
        lambda n, i, j, k: owner._post_process_lmtd(
            owner.theta_1_by_period[n][i][j][k][0],
            owner.theta_2_by_period[n][i][j][k][0],
            owner.z[i][j][k][0],
            formula_allowed=_lmtd_formula_allowed(
                owner.theta_1_by_period[n][i][j][k][0],
                owner.theta_2_by_period[n][i][j][k][0],
                owner._recovery_approach_temperature(i, j, n),
                owner.tol,
            ),
        ),
        (owner.N_periods, owner.I, owner.J, owner.S),
    )
    owner.area_r_by_period = build_index_grid(
        lambda n, i, j, k: _area_from_heat_load(
            q_r[n][i][j][k],
            owner.U_r_period[n][i][j],
            owner.LMTD_r_by_period[n][i][j][k],
            owner.tol,
        ),
        (owner.N_periods, owner.I, owner.J, owner.S),
    )
    owner.LMTD_r = owner.LMTD_r_by_period[0]
    owner.area_r = build_index_grid(
        lambda i, j, k: max(
            owner.area_r_by_period[n][i][j][k] for n in range(owner.N_periods)
        ),
        (owner.I, owner.J, owner.S),
    )
    owner._apply_segment_recovery_areas(q_r)

    owner.LMTD_hu_by_period = build_index_grid(
        lambda n, j: owner._post_process_lmtd(
            owner.T_hu_in_period[n][0] - owner.T_c_out_period[n][j],
            owner._utility_solved_outlet_temperature("hot", n, j, q_h[n][j])
            - owner._active_binary_value(owner.T_c_by_period[n][j][0]),
            owner.z_hu[j][0],
            formula_allowed=_lmtd_formula_allowed(
                owner.T_hu_in_period[n][0] - owner.T_c_out_period[n][j],
                owner._utility_solved_outlet_temperature("hot", n, j, q_h[n][j])
                - owner._active_binary_value(owner.T_c_by_period[n][j][0]),
                owner._hot_utility_inlet_approach_temperature(j, n),
                owner.tol,
                owner._hot_utility_outlet_approach_temperature(j, n, q_h[n][j]),
            ),
        ),
        (owner.N_periods, owner.J),
    )
    owner.area_hu_by_period = build_index_grid(
        lambda n, j: _area_from_heat_load(
            q_h[n][j],
            owner.U_hu_period[n][j],
            owner.LMTD_hu_by_period[n][j],
            owner.tol,
        ),
        (owner.N_periods, owner.J),
    )

    owner.LMTD_cu_by_period = build_index_grid(
        lambda n, i: owner._post_process_lmtd(
            owner._active_binary_value(owner.T_h_by_period[n][i][owner.S])
            - owner._utility_solved_outlet_temperature("cold", n, i, q_c[n][i]),
            owner.T_h_out_period[n][i] - owner.T_cu_in_period[n][0],
            owner.z_cu[i][0],
            formula_allowed=_lmtd_formula_allowed(
                owner._active_binary_value(owner.T_h_by_period[n][i][owner.S])
                - owner._utility_solved_outlet_temperature("cold", n, i, q_c[n][i]),
                owner.T_h_out_period[n][i] - owner.T_cu_in_period[n][0],
                owner._cold_utility_outlet_approach_temperature(i, n, q_c[n][i]),
                owner.tol,
                owner._cold_utility_inlet_approach_temperature(i, n),
            ),
            fallback_delta=(owner.T_h_out_period[n][i] - owner.T_cu_in_period[n][0]),
        ),
        (owner.N_periods, owner.I),
    )
    owner.area_cu_by_period = build_index_grid(
        lambda n, i: _area_from_heat_load(
            q_c[n][i],
            owner.U_cu_period[n][i],
            owner.LMTD_cu_by_period[n][i],
            owner.tol,
        ),
        (owner.N_periods, owner.I),
    )
    owner._apply_segment_utility_areas(q_h, q_c)
    owner.LMTD_hu = owner.LMTD_hu_by_period[0]
    owner.LMTD_cu = owner.LMTD_cu_by_period[0]

    owner.area_hu = build_index_grid(
        lambda j: max(owner.area_hu_by_period[n][j] for n in range(owner.N_periods)),
        (owner.J,),
    )
    owner.area_cu = build_index_grid(
        lambda i: max(owner.area_cu_by_period[n][i] for n in range(owner.N_periods)),
        (owner.I,),
    )
    owner.Q_hu_total_by_period = build_index_grid(
        lambda n: sum(q_h[n]),
        (owner.N_periods,),
    )
    owner.Q_cu_total_by_period = build_index_grid(
        lambda n: sum(q_c[n]),
        (owner.N_periods,),
    )
    owner.Q_r_total_by_period = build_index_grid(
        lambda n: sum(
            q_r[n][i][j][k]
            for k in range(owner.S)
            for j in range(owner.J)
            for i in range(owner.I)
        ),
        (owner.N_periods,),
    )
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
    owner.TAC_model = owner.m.options.objfcnval
    owner.TAC = owner.capital_cost_value + owner.weighted_operating_cost_value


def _active_binary_value(owner, value) -> float:
    return _scalar_value(value)


def _weighted_numeric_average(owner, values: Sequence[float]) -> float:
    return float(
        sum(
            float(owner.period_weights[n]) * float(values[n])
            for n in range(owner.N_periods)
        )
        / owner.period_weight_sum
    )
