"""StageWise solver-result verification."""

from __future__ import annotations

import math


def verify(owner) -> tuple[bool, list[str]]:
    """Run the source solution checks used by topology evolution."""

    failures = []
    if not _check_temperatures(owner):
        failures.append("temperature")
    if owner.minimisation_goal in {"total cost", "variable total cost"}:
        if not _check_utility_costs(owner):
            failures.append("cost")
        if not _check_area_costs(owner):
            failures.append("area")
    return len(failures) == 0, failures


def _check_temperatures(
    case,
    *,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-2,
    q_tol: float = 1e-2,
) -> bool:
    if hasattr(case, "Q_r_by_period"):
        return _check_state_temperatures(
            case,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            q_tol=q_tol,
        )

    issues = False
    for i in range(case.I):
        for j in range(case.J):
            for k in range(case.S):
                if case.Q_r[i][j][k][0] <= q_tol:
                    continue
                if case.non_isothermal_model:
                    t_h_in = case.T_h[i][k][0]
                    t_c_out_y = case.T_c_out_y[j][i][k][0]
                    theta_1 = case.theta_1[i][j][k][0]
                    if (
                        not math.isclose(
                            t_h_in,
                            t_c_out_y + theta_1,
                            rel_tol=rel_tol,
                            abs_tol=abs_tol,
                        )
                        or t_h_in < t_c_out_y
                    ):
                        issues = True

                    t_h_out = case.T_h_out_x[i][j][k][0]
                    t_c_in = case.T_c[j][k + 1][0]
                    theta_2 = case.theta_2[i][j][k][0]
                    if (
                        not math.isclose(
                            t_h_out,
                            t_c_in + theta_2,
                            rel_tol=rel_tol,
                            abs_tol=abs_tol,
                        )
                        or t_h_out < t_c_in
                    ):
                        issues = True
                else:
                    if case.T_h[i][k][0] < case.T_c[j][k][0]:
                        issues = True
                    if case.T_h[i][k + 1][0] < case.T_c[j][k + 1][0]:
                        issues = True
    return not issues


def _check_state_temperatures(
    case,
    *,
    rel_tol: float,
    abs_tol: float,
    q_tol: float,
) -> bool:
    issues = False
    for n in range(case.N_periods):
        for i in range(case.I):
            for j in range(case.J):
                for k in range(case.S):
                    if _value(case.Q_r_by_period[n][i][j][k]) <= q_tol:
                        continue
                    if case.non_isothermal_model:
                        t_h_in = _value(case.T_h_by_period[n][i][k])
                        t_c_out_y = _value(case.T_c_out_y_by_period[n][j][i][k])
                        theta_1 = _value(case.theta_1_by_period[n][i][j][k])
                        if (
                            not math.isclose(
                                t_h_in,
                                t_c_out_y + theta_1,
                                rel_tol=rel_tol,
                                abs_tol=abs_tol,
                            )
                            or t_h_in < t_c_out_y
                        ):
                            issues = True

                        t_h_out = _value(case.T_h_out_x_by_period[n][i][j][k])
                        t_c_in = _value(case.T_c_by_period[n][j][k + 1])
                        theta_2 = _value(case.theta_2_by_period[n][i][j][k])
                        if (
                            not math.isclose(
                                t_h_out,
                                t_c_in + theta_2,
                                rel_tol=rel_tol,
                                abs_tol=abs_tol,
                            )
                            or t_h_out < t_c_in
                        ):
                            issues = True
                    else:
                        if _value(case.T_h_by_period[n][i][k]) < _value(
                            case.T_c_by_period[n][j][k]
                        ):
                            issues = True
                        if _value(case.T_h_by_period[n][i][k + 1]) < _value(
                            case.T_c_by_period[n][j][k + 1]
                        ):
                            issues = True
    return not issues


def _check_utility_costs(
    case,
    *,
    rel_tol: float = 0.1,
    abs_tol: float = 1.0,
) -> bool:
    def utility_cost(side: str, period_index: int, duty: float) -> float:
        if hasattr(case, "_utility_cost_value"):
            return case._utility_cost_value(side, period_index, duty)
        prices = case.hu_cost_period if side == "hot" else case.cu_cost_period
        return float(prices[period_index][0]) * duty

    if hasattr(case, "Q_h_by_period"):
        post_hot_utility = case._weighted_numeric_average(
            [
                utility_cost(
                    "hot",
                    n,
                    sum(_value(case.Q_h_by_period[n][j]) for j in range(case.J)),
                )
                for n in range(case.N_periods)
            ]
        )
        post_cold_utility = case._weighted_numeric_average(
            [
                utility_cost(
                    "cold",
                    n,
                    sum(_value(case.Q_c_by_period[n][i]) for i in range(case.I)),
                )
                for n in range(case.N_periods)
            ]
        )
    else:
        hot_duty = sum(case.Q_h[j][0] for j in range(case.J))
        cold_duty = sum(case.Q_c[i][0] for i in range(case.I))
        if hasattr(case, "_utility_cost_value"):
            post_hot_utility = case._utility_cost_value("hot", 0, hot_duty)
            post_cold_utility = case._utility_cost_value("cold", 0, cold_duty)
        else:
            post_hot_utility = case.hu_cost[0] * hot_duty
            post_cold_utility = case.cu_cost[0] * cold_duty
    model_hot_utility = _value(case.hu_cost_total)
    model_cold_utility = _value(case.cu_cost_total)
    return math.isclose(
        post_hot_utility,
        model_hot_utility,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    ) and math.isclose(
        post_cold_utility,
        model_cold_utility,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )


def _check_area_costs(
    case,
    *,
    rel_tol: float = 0.1,
    abs_tol: float = 1.0,
    q_tol: float = 1e-2,
) -> bool:
    if getattr(case, "segment_area_contributions_by_period", None):
        recovery_area_cost = case.A_coeff[0] * sum(
            case.area_r[i][j][k] ** case.A_exp[0]
            for k in range(case.S)
            for j in range(case.J)
            for i in range(case.I)
        )
        hu_area_cost = case.hu_coeff[0] * sum(
            case.area_hu[j] ** case.hu_exp[0] for j in range(case.J)
        )
        cu_area_cost = case.cu_coeff[0] * sum(
            case.area_cu[i] ** case.cu_exp[0] for i in range(case.I)
        )
        return all(
            math.isclose(expected, actual, rel_tol=rel_tol, abs_tol=abs_tol)
            for expected, actual in (
                (recovery_area_cost, case.recovery_area_cost_total),
                (hu_area_cost, case.hu_area_cost_total),
                (cu_area_cost, case.cu_area_cost_total),
            )
        )

    if hasattr(case, "area_r_shared"):
        return _check_shared_area_costs(
            case,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            q_tol=q_tol,
        )

    for k in range(case.S):
        for j in range(case.J):
            allowed_hots = [i for i in range(case.I) if case.z_allowed[i][j][k] > 0]
            if not allowed_hots:
                continue

            total_duty = sum(case.Q_r[i][j][k][0] for i in allowed_hots)
            if abs(total_duty) <= q_tol:
                continue

            post_area_chen = (
                sum(
                    (
                        case.Q_r[hot_idx][j][k][0]
                        / (
                            case.U_r[hot_idx][j]
                            * (
                                (
                                    case.theta_1[hot_idx][j][k][0]
                                    * case.theta_2[hot_idx][j][k][0]
                                    * (
                                        case.theta_1[hot_idx][j][k][0]
                                        + case.theta_2[hot_idx][j][k][0]
                                    )
                                    / 2
                                    + 1e-3
                                )
                                ** (1 / 3)
                            )
                        )
                    )
                    ** case.A_exp[0]
                    for hot_idx in allowed_hots
                )
                * case.A_coeff[0]
            )
            model_area = _value(case.recovery_area_cost_filtered[k][j])
            if not math.isclose(
                post_area_chen,
                model_area,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            ):
                return False
    return True


def _check_shared_area_costs(
    case,
    *,
    rel_tol: float,
    abs_tol: float,
    q_tol: float,
) -> bool:
    for n in range(case.N_periods):
        for k in range(case.S):
            for j in range(case.J):
                for i in range(case.I):
                    q = _value(case.Q_r_by_period[n][i][j][k])
                    if q <= q_tol:
                        continue
                    required_area = q / (
                        case.U_r_period[n][i][j]
                        * (
                            _value(case.theta_1_by_period[n][i][j][k])
                            * _value(case.theta_2_by_period[n][i][j][k])
                            * (
                                _value(case.theta_1_by_period[n][i][j][k])
                                + _value(case.theta_2_by_period[n][i][j][k])
                            )
                            / 2
                            + 1e-3
                        )
                        ** (1 / 3)
                    )
                    if _value(case.area_r_shared[i][j][k]) + abs_tol < required_area:
                        return False
        for j in range(case.J):
            q = _value(case.Q_h_by_period[n][j])
            if q > q_tol:
                required_area = q / (
                    case.U_hu_period[n][j]
                    * (
                        (case.T_hu_in_period[n][0] - case.T_c_out_period[n][j])
                        * (
                            case._utility_solved_outlet_temperature("hot", n, j, q)
                            - _value(case.T_c_by_period[n][j][0])
                        )
                        * (
                            (case.T_hu_in_period[n][0] - case.T_c_out_period[n][j])
                            + (
                                case._utility_solved_outlet_temperature("hot", n, j, q)
                                - _value(case.T_c_by_period[n][j][0])
                            )
                        )
                        / 2
                        + 1e-3
                    )
                    ** (1 / 3)
                )
                if _value(case.area_hu_shared[j]) + abs_tol < required_area:
                    return False
        for i in range(case.I):
            q = _value(case.Q_c_by_period[n][i])
            if q > q_tol:
                required_area = q / (
                    case.U_cu_period[n][i]
                    * (
                        (
                            _value(case.T_h_by_period[n][i][case.S])
                            - case._utility_solved_outlet_temperature("cold", n, i, q)
                        )
                        * (case.T_h_out_period[n][i] - case.T_cu_in_period[n][0])
                        * (
                            (
                                _value(case.T_h_by_period[n][i][case.S])
                                - case._utility_solved_outlet_temperature(
                                    "cold", n, i, q
                                )
                            )
                            + (case.T_h_out_period[n][i] - case.T_cu_in_period[n][0])
                        )
                        / 2
                        + 1e-3
                    )
                    ** (1 / 3)
                )
                if _value(case.area_cu_shared[i]) + abs_tol < required_area:
                    return False

    recovery_area_cost = case.A_coeff[0] * sum(
        _value(case.area_r_shared[i][j][k]) ** case.A_exp[0]
        for k in range(case.S)
        for j in range(case.J)
        for i in range(case.I)
    )
    hu_area_cost = case.hu_coeff[0] * sum(
        _value(case.area_hu_shared[j]) ** case.hu_exp[0] for j in range(case.J)
    )
    cu_area_cost = case.cu_coeff[0] * sum(
        _value(case.area_cu_shared[i]) ** case.cu_exp[0] for i in range(case.I)
    )
    return (
        math.isclose(
            recovery_area_cost,
            _value(case.recovery_area_cost_total),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        and math.isclose(
            hu_area_cost,
            _value(case.hu_area_cost_total),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        and math.isclose(
            cu_area_cost,
            _value(case.cu_area_cost_total),
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
    )


def _value(value) -> float:
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value[0])
    except TypeError, IndexError, KeyError:
        pass
    for attr in ("VALUE", "value"):
        if not hasattr(value, attr):
            continue
        raw = getattr(value, attr)
        try:
            return float(raw[0])
        except TypeError, IndexError, KeyError:
            pass
        raw_value = getattr(raw, "value", raw)
        if isinstance(raw_value, int | float):
            return float(raw_value)
        try:
            return float(raw_value[0])
        except TypeError, IndexError, KeyError:
            return float(raw_value)
    return float(value)
