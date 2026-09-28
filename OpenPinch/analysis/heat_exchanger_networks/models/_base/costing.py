"""Period-total and cost roll-ups shared by the PDM and StageWise models."""

from __future__ import annotations

from collections.abc import Sequence


def set_period_totals_and_costs(owner, q_h, q_c, q_r) -> None:
    """Set per-period duty totals and the operating, capital and area costs."""

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
    owner.Q_hu_total = _weighted_numeric_average(owner, owner.Q_hu_total_by_period)
    owner.Q_cu_total = _weighted_numeric_average(owner, owner.Q_cu_total_by_period)
    owner.Q_r_total = _weighted_numeric_average(owner, owner.Q_r_total_by_period)

    owner.operating_cost_by_period = [
        owner._utility_cost_value("hot", n, owner.Q_hu_total_by_period[n])
        + owner._utility_cost_value("cold", n, owner.Q_cu_total_by_period[n])
        for n in range(owner.N_periods)
    ]
    owner.weighted_operating_cost_value = _weighted_numeric_average(
        owner, owner.operating_cost_by_period
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
    owner.hu_cost_total = _weighted_numeric_average(
        owner,
        [
            owner._utility_cost_value("hot", n, owner.Q_hu_total_by_period[n])
            for n in range(owner.N_periods)
        ],
    )
    owner.cu_cost_total = _weighted_numeric_average(
        owner,
        [
            owner._utility_cost_value("cold", n, owner.Q_cu_total_by_period[n])
            for n in range(owner.N_periods)
        ],
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


def _weighted_numeric_average(owner, values: Sequence[float]) -> float:
    return float(
        sum(
            float(owner.period_weights[n]) * float(values[n])
            for n in range(owner.N_periods)
        )
        / owner.period_weight_sum
    )
