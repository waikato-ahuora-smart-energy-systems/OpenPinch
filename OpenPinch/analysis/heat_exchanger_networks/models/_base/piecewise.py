"""Piecewise equations and exact segment calculations for base HEN models."""

from __future__ import annotations

import numpy as np


def _register_piecewise_mapping(model, mapping) -> None:
    if mapping is not None:
        model._piecewise_active_mappings.append(mapping)


def _utility_is_segmented(model, side: str) -> bool:
    if not hasattr(model, "solver_arrays"):
        return False
    values = model.solver_arrays.arrays.get(f"{side}_utility_parent_segmented")
    return bool(values is not None and np.asarray(values, dtype=bool)[0])


def _utility_cost_expression(
    model,
    side: str,
    period_index: int,
    heat_duty,
    *,
    name: str,
):
    """Return the flat or exact piecewise utility-cost solver expression."""
    price_attr = "hu_cost_period" if side == "hot" else "cu_cost_period"
    if not model._utility_is_segmented(side):
        if hasattr(model, price_attr):
            price = getattr(model, price_attr)[period_index][0]
        else:
            price = getattr(model, "hu_cost" if side == "hot" else "cu_cost")[0]
        return price * heat_duty

    from ...solver.piecewise import (
        add_piecewise_cost_mapping,
        profile_from_solver_arrays,
    )

    profile = profile_from_solver_arrays(
        model.solver_arrays,
        side=f"{side}_utility",
        parent_index=0,
        period_index=period_index,
    )
    coordinate = model.m.Var(
        value=0.0,
        lb=0.0,
        ub=profile.total_duty,
        name=f"{name}_duty",
    )
    cost = model.m.Var(
        value=0.0,
        lb=0.0,
        ub=float(profile.cumulative_costs[-1]),
        name=f"{name}_cost",
    )
    model.m.Equation(coordinate == heat_duty)
    model._register_piecewise_mapping(
        add_piecewise_cost_mapping(
            model.m,
            coordinate,
            cost,
            profile,
            name=name,
            integer_capable=model.solver in {"apopt", "couenne"},
        )
    )
    return cost


def _utility_cost_value(
    model,
    side: str,
    period_index: int,
    heat_duty: float,
) -> float:
    """Return exact solved utility cost for reporting and verification."""
    duty = max(float(heat_duty), 0.0)
    price_attr = "hu_cost_period" if side == "hot" else "cu_cost_period"
    if not model._utility_is_segmented(side):
        return float(getattr(model, price_attr)[period_index][0]) * duty

    from ...solver.piecewise import profile_from_solver_arrays

    profile = profile_from_solver_arrays(
        model.solver_arrays,
        side=f"{side}_utility",
        parent_index=0,
        period_index=period_index,
    )
    if duty > profile.total_duty + model.tol:
        raise ValueError(
            f"Solved {side} utility duty exceeds its segmented profile capacity."
        )
    return profile.cost_at_heat(duty)


def _update_piecewise_active_segments(model) -> bool:
    changed = False
    for mapping in model._piecewise_active_mappings:
        coordinate = model._solver_value(mapping["heat_coordinate"])
        segment_index_at_heat = mapping.get(
            "segment_index_at_heat",
            mapping["profile"].segment_index_at_heat,
        )
        next_segment = segment_index_at_heat(coordinate)
        if next_segment == mapping["active_segment"]:
            continue
        for index, selector in enumerate(mapping["selectors"]):
            model._set_value(selector, 1.0 if index == next_segment else 0.0)
        mapping["active_segment"] = next_segment
        changed = True
    return changed


def _set_piecewise_stage_heat_coordinates(model) -> None:
    """Add parent cumulative-Q balances and ordered T(Q) mappings by period."""
    if not hasattr(model, "solver_arrays"):
        model._segmented_hot_parents = np.zeros(model.I, dtype=bool)
        model._segmented_cold_parents = np.zeros(model.J, dtype=bool)
        return
    arrays = model.solver_arrays.arrays
    model._set_segmented_utility_capacity_constraints()
    model._set_piecewise_utility_outlet_states()
    model._segmented_hot_parents = (
        np.asarray(arrays.get("hot_segment_count", np.ones(model.I)), dtype=int) > 1
    )
    model._segmented_cold_parents = (
        np.asarray(arrays.get("cold_segment_count", np.ones(model.J)), dtype=int) > 1
    )
    if not (
        np.any(model._segmented_hot_parents) or np.any(model._segmented_cold_parents)
    ):
        return

    from ...solver.piecewise import (
        add_piecewise_temperature_mapping,
        profile_from_solver_arrays,
    )

    integer_capable = model.solver in {"apopt", "couenne"}
    model.Q_coordinate_h_by_period = [
        [[None for _k in range(model.K)] for _i in range(model.I)]
        for _n in range(model.N_periods)
    ]
    model.Q_coordinate_c_by_period = [
        [[None for _k in range(model.K)] for _j in range(model.J)]
        for _n in range(model.N_periods)
    ]
    for n in range(model.N_periods):
        for i in range(model.I):
            if not model._segmented_hot_parents[i]:
                continue
            if (
                hasattr(model, "z_i_active_period")
                and model.z_i_active_period[n][i] <= 0
            ):
                continue
            profile = profile_from_solver_arrays(
                model.solver_arrays,
                side="hot",
                parent_index=i,
                period_index=n,
            ).clipped(model.T_h_in_period[n][i], model.T_h_out_period[n][i])
            for k in range(model.K):
                initial_q = profile.total_duty * k / max(model.S, 1)
                coordinate = (
                    model.m.Param(value=0.0, name=f"Qcoord_H{i}_B0_period{n}")
                    if k == 0
                    else model.m.Var(
                        value=initial_q,
                        lb=0.0,
                        ub=profile.total_duty,
                        name=f"Qcoord_H{i}_B{k}_period{n}",
                    )
                )
                model.Q_coordinate_h_by_period[n][i][k] = coordinate
                model._register_piecewise_mapping(
                    add_piecewise_temperature_mapping(
                        model.m,
                        coordinate,
                        model.T_h_by_period[n][i][k],
                        profile,
                        name=f"TQ_H{i}_B{k}_period{n}",
                        integer_capable=integer_capable,
                        initial_segment=profile.segment_index_at_heat(initial_q),
                    )
                )
            model.m.Equations(
                [
                    model.Q_coordinate_h_by_period[n][i][k + 1]
                    - model.Q_coordinate_h_by_period[n][i][k]
                    - sum(model.Q_r_by_period[n][i][j][k] for j in range(model.J))
                    == 0.0
                    for k in range(model.S)
                ]
            )

        for j in range(model.J):
            if not model._segmented_cold_parents[j]:
                continue
            if (
                hasattr(model, "z_j_active_period")
                and model.z_j_active_period[n][j] <= 0
            ):
                continue
            profile = profile_from_solver_arrays(
                model.solver_arrays,
                side="cold",
                parent_index=j,
                period_index=n,
            ).clipped(model.T_c_in_period[n][j], model.T_c_out_period[n][j])
            for k in range(model.K):
                initial_q = profile.total_duty * (model.S - k) / max(model.S, 1)
                coordinate = (
                    model.m.Param(value=0.0, name=f"Qcoord_C{j}_B{model.S}_period{n}")
                    if k == model.S
                    else model.m.Var(
                        value=initial_q,
                        lb=0.0,
                        ub=profile.total_duty,
                        name=f"Qcoord_C{j}_B{k}_period{n}",
                    )
                )
                model.Q_coordinate_c_by_period[n][j][k] = coordinate
                model._register_piecewise_mapping(
                    add_piecewise_temperature_mapping(
                        model.m,
                        coordinate,
                        model.T_c_by_period[n][j][k],
                        profile,
                        name=f"TQ_C{j}_B{k}_period{n}",
                        integer_capable=integer_capable,
                        initial_segment=profile.segment_index_at_heat(initial_q),
                    )
                )
            model.m.Equations(
                [
                    model.Q_coordinate_c_by_period[n][j][k]
                    - model.Q_coordinate_c_by_period[n][j][k + 1]
                    - sum(model.Q_r_by_period[n][i][j][k] for i in range(model.I))
                    == 0.0
                    for k in range(model.S)
                ]
            )


def _set_segmented_utility_capacity_constraints(model) -> None:
    """Bound selected utility load by each explicit ordered profile."""
    from ...solver.piecewise import profile_from_solver_arrays

    for n in range(model.N_periods):
        if model._utility_is_segmented("hot"):
            hot_profile = profile_from_solver_arrays(
                model.solver_arrays,
                side="hot_utility",
                parent_index=0,
                period_index=n,
            )
            model.m.Equation(sum(model.Q_h_by_period[n]) <= hot_profile.total_duty)
        if model._utility_is_segmented("cold"):
            cold_profile = profile_from_solver_arrays(
                model.solver_arrays,
                side="cold_utility",
                parent_index=0,
                period_index=n,
            )
            model.m.Equation(sum(model.Q_c_by_period[n]) <= cold_profile.total_duty)


def _set_piecewise_utility_outlet_states(model) -> None:
    """Map aggregate utility duty to outlet temperature and local ``dt_cont``."""
    from ...solver.piecewise import (
        add_piecewise_temperature_contribution_mapping,
        add_piecewise_temperature_mapping,
        profile_from_solver_arrays,
    )

    model.T_hu_solved_out_by_period = [[] for _n in range(model.N_periods)]
    model.T_cu_solved_out_by_period = [[] for _n in range(model.N_periods)]
    model.T_hu_out_cont_by_period = [[] for _n in range(model.N_periods)]
    model.T_cu_out_cont_by_period = [[] for _n in range(model.N_periods)]
    model.T_hu_in_cont_by_period = []
    model.T_cu_in_cont_by_period = []
    for n in range(model.N_periods):
        for side, loads in (
            ("hot", model.Q_h_by_period[n]),
            ("cold", model.Q_c_by_period[n]),
        ):
            scalar_contribution = float(
                (
                    model.T_hu_cont_period[n][0]
                    if side == "hot"
                    else model.T_cu_cont_period[n][0]
                )
            )
            inlet_contribution = scalar_contribution
            if model._utility_is_segmented(side):
                profile = profile_from_solver_arrays(
                    model.solver_arrays,
                    side=f"{side}_utility",
                    parent_index=0,
                    period_index=n,
                )
                inlet_contribution = float(profile.temperature_contributions[0])
            getattr(model, f"T_{side[0]}u_in_cont_by_period").append(inlet_contribution)

            solved_outlets = getattr(model, f"T_{side[0]}u_solved_out_by_period")[n]
            outlet_contributions = getattr(model, f"T_{side[0]}u_out_cont_by_period")[n]
            for match_index, load in enumerate(loads):
                if not model._utility_is_segmented(side):
                    solved_outlets.append(
                        model.T_hu_out_period[n][0]
                        if side == "hot"
                        else model.T_cu_out_period[n][0]
                    )
                    outlet_contributions.append(scalar_contribution)
                    continue

                matched_duty = (
                    model.Qtot_sc_period[n][match_index]
                    if side == "hot"
                    else model.Qtot_sh_period[n][match_index]
                )
                initial_duty = min(matched_duty / 2.0, profile.total_duty)
                coordinate = model.m.Var(
                    value=initial_duty,
                    lb=0.0,
                    ub=profile.total_duty,
                    name=(f"Qcoord_{side}_utility_M{match_index}_period{n}"),
                )
                model.m.Equation(coordinate == load)
                solved_outlet = model.m.Var(
                    value=profile.temperature_at_heat(initial_duty),
                    lb=float(
                        min(
                            profile.temperatures_in.min(),
                            profile.temperatures_out.min(),
                        )
                    ),
                    ub=float(
                        max(
                            profile.temperatures_in.max(),
                            profile.temperatures_out.max(),
                        )
                    ),
                    name=(f"T_{side}_utility_out_M{match_index}_period{n}"),
                )
                contribution_values = profile.temperature_contributions
                outlet_contribution = model.m.Var(
                    value=profile.temperature_contribution_at_heat(initial_duty),
                    lb=float(contribution_values.min()),
                    ub=float(contribution_values.max()),
                    name=(f"dTcont_{side}_utility_out_M{match_index}_period{n}"),
                )
                temperature_segment = profile.segment_index_at_heat(initial_duty)
                contribution_segment = profile.contribution_index_at_heat(initial_duty)
                model._register_piecewise_mapping(
                    add_piecewise_temperature_mapping(
                        model.m,
                        coordinate,
                        solved_outlet,
                        profile,
                        name=(f"TQ_{side}_utility_M{match_index}_period{n}"),
                        integer_capable=model.solver in {"apopt", "couenne"},
                        initial_segment=temperature_segment,
                    )
                )
                model._register_piecewise_mapping(
                    add_piecewise_temperature_contribution_mapping(
                        model.m,
                        coordinate,
                        outlet_contribution,
                        profile,
                        name=(f"dTQ_{side}_utility_M{match_index}_period{n}"),
                        initial_segment=contribution_segment,
                    )
                )
                solved_outlets.append(solved_outlet)
                outlet_contributions.append(outlet_contribution)


def _hot_parent_segmented(model, index: int) -> bool:
    return bool(getattr(model, "_segmented_hot_parents", [False] * model.I)[index])


def _cold_parent_segmented(model, index: int) -> bool:
    return bool(getattr(model, "_segmented_cold_parents", [False] * model.J)[index])


def _solver_parent_is_segmented(model, side: str, index: int) -> bool:
    if not hasattr(model, "solver_arrays"):
        return False
    counts = model.solver_arrays.arrays.get(f"{side}_segment_count")
    return counts is not None and int(counts[index]) > 1


def _parent_profile_duty(
    model,
    side: str,
    period_index: int,
    parent_index: int,
    supply_temperature: float,
    target_temperature: float,
    aggregate_cp: float,
) -> float:
    if not model._solver_parent_is_segmented(side, parent_index):
        return abs(supply_temperature - target_temperature) * aggregate_cp
    from ...solver.piecewise import profile_from_solver_arrays

    return (
        profile_from_solver_arrays(
            model.solver_arrays,
            side=side,
            parent_index=parent_index,
            period_index=period_index,
        )
        .clipped(supply_temperature, target_temperature)
        .total_duty
    )


def _recovery_heat_upper_bound(
    model,
    *,
    period_index: int,
    hot_index: int,
    cold_index: int,
    hot_total_duty: float,
    cold_total_duty: float,
    hot_cp: float,
    cold_cp: float,
) -> float:
    temperature_span = max(
        model.T_h_in_period[period_index][hot_index]
        - model.T_c_in_period[period_index][cold_index]
        - model._recovery_approach_temperature(hot_index, cold_index, period_index),
        0.0,
    )
    if model._solver_parent_is_segmented(
        "hot", hot_index
    ) or model._solver_parent_is_segmented("cold", cold_index):
        return min(hot_total_duty, cold_total_duty) if temperature_span > 0 else 0.0
    return temperature_span * min(hot_cp, cold_cp)


def _set_piecewise_match_outlet_equations(model) -> None:
    """Map non-isothermal branch outlets through parent heat coordinates."""
    if not hasattr(model, "X_by_period") or not hasattr(model, "Y_by_period"):
        return
    if not (
        np.any(getattr(model, "_segmented_hot_parents", []))
        or np.any(getattr(model, "_segmented_cold_parents", []))
    ):
        return

    from ...solver.piecewise import (
        add_piecewise_temperature_mapping,
        profile_from_solver_arrays,
    )

    integer_capable = model.solver in {"apopt", "couenne"}
    model.Q_coordinate_h_out_x_by_period = [
        [
            [[None for _k in range(model.S)] for _j in range(model.J)]
            for _i in range(model.I)
        ]
        for _n in range(model.N_periods)
    ]
    model.Q_coordinate_c_out_y_by_period = [
        [
            [[None for _k in range(model.S)] for _i in range(model.I)]
            for _j in range(model.J)
        ]
        for _n in range(model.N_periods)
    ]
    for n in range(model.N_periods):
        for i in range(model.I):
            hot_profile = (
                profile_from_solver_arrays(
                    model.solver_arrays,
                    side="hot",
                    parent_index=i,
                    period_index=n,
                ).clipped(model.T_h_in_period[n][i], model.T_h_out_period[n][i])
                if model._hot_parent_segmented(i)
                else None
            )
            for j in range(model.J):
                cold_profile = (
                    profile_from_solver_arrays(
                        model.solver_arrays,
                        side="cold",
                        parent_index=j,
                        period_index=n,
                    ).clipped(model.T_c_in_period[n][j], model.T_c_out_period[n][j])
                    if model._cold_parent_segmented(j)
                    else None
                )
                for k in range(model.S):
                    if model.z_allowed[i][j][k] <= 0:
                        continue
                    if hot_profile is not None:
                        q_in = model.Q_coordinate_h_by_period[n][i][k]
                        q_out = model.m.Var(
                            value=hot_profile.total_duty * (k + 1) / max(model.S, 1),
                            lb=0.0,
                            ub=hot_profile.total_duty,
                            name=f"Qcoord_H{i}_out_C{j}_S{k}_period{n}",
                        )
                        model.Q_coordinate_h_out_x_by_period[n][i][j][k] = q_out
                        model.m.Equation(
                            model.Q_r_by_period[n][i][j][k]
                            == model.X_by_period[n][i][j][k] * (q_out - q_in)
                        )
                        model._register_piecewise_mapping(
                            add_piecewise_temperature_mapping(
                                model.m,
                                q_out,
                                model.T_h_out_x_by_period[n][i][j][k],
                                hot_profile,
                                name=f"TQ_H{i}_out_C{j}_S{k}_period{n}",
                                integer_capable=integer_capable,
                                initial_segment=hot_profile.segment_index_at_heat(
                                    hot_profile.total_duty * (k + 1) / max(model.S, 1)
                                ),
                            )
                        )
                    if cold_profile is not None:
                        q_in = model.Q_coordinate_c_by_period[n][j][k + 1]
                        q_out = model.m.Var(
                            value=cold_profile.total_duty
                            * (model.S - k)
                            / max(model.S, 1),
                            lb=0.0,
                            ub=cold_profile.total_duty,
                            name=f"Qcoord_C{j}_out_H{i}_S{k}_period{n}",
                        )
                        model.Q_coordinate_c_out_y_by_period[n][j][i][k] = q_out
                        model.m.Equation(
                            model.Q_r_by_period[n][i][j][k]
                            == model.Y_by_period[n][j][i][k] * (q_out - q_in)
                        )
                        model._register_piecewise_mapping(
                            add_piecewise_temperature_mapping(
                                model.m,
                                q_out,
                                model.T_c_out_y_by_period[n][j][i][k],
                                cold_profile,
                                name=f"TQ_C{j}_out_H{i}_S{k}_period{n}",
                                integer_capable=integer_capable,
                                initial_segment=cold_profile.segment_index_at_heat(
                                    cold_profile.total_duty
                                    * (model.S - k)
                                    / max(model.S, 1)
                                ),
                            )
                        )
