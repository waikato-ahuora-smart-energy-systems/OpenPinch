"""Pinch-decomposition target and model preprocessing."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ...indexing import build_index_grid


def _overall_heat_transfer_coefficient(left_htc: float, right_htc: float) -> float:
    return 1 / (1 / left_htc + 1 / right_htc)


def _active_period_flag(values: Iterable[float], tolerance: float) -> list[int]:
    return [1 if max(values) > tolerance else 0]


def _lmtd_formula_allowed(
    delta_1: float,
    delta_2: float,
    approach_temperature: float,
    tolerance: float,
    second_approach_temperature: float | None = None,
) -> bool:
    second_approach = (
        approach_temperature
        if second_approach_temperature is None
        else second_approach_temperature
    )
    return (
        abs(delta_1 - delta_2) > tolerance
        and delta_1 - approach_temperature >= tolerance
        and delta_2 - second_approach >= tolerance
    )


def _area_from_heat_load(
    heat_load: float,
    overall_heat_transfer_coefficient: float,
    lmtd: float,
    tolerance: float,
) -> float:
    if lmtd <= tolerance or heat_load <= tolerance:
        return 0.0
    return heat_load / overall_heat_transfer_coefficient / lmtd


def calculate_pinch(owner) -> None:
    """Read target values from the private OpenPinch decomposition."""

    if owner.pinch_decomposition.pinch_location != owner.pinch_loc:
        raise ValueError("pinch decomposition location does not match PDM side.")
    targets = owner.pinch_decomposition.period_targets
    if len(targets) != owner.N_periods:
        raise ValueError("PDM period targets must match the model period count.")
    if tuple(target.period_id for target in targets) != tuple(owner.period_ids):
        raise ValueError("PDM period target identities must match solver arrays.")
    owner.HU_target_by_period = [target.hot_utility_target for target in targets]
    owner.CU_target_by_period = [target.cold_utility_target for target in targets]
    owner.T_pinch_by_period = [target.shifted_pinch_temperature for target in targets]
    if any(value is None for value in owner.T_pinch_by_period):
        raise ValueError(
            "PDM construction requires a shifted pinch temperature in every period."
        )
    owner.side_required = any(
        target > owner.tol
        for target in (
            owner.HU_target_by_period
            if owner.pinch_loc == "above"
            else owner.CU_target_by_period
        )
    )


def set_preprocessing(owner) -> None:
    """Pre-process PDM superstructure parameters."""

    owner._set_multiperiod_preprocessing()


def _set_multiperiod_preprocessing(owner) -> None:
    owner.I = owner.f_h_period.shape[1]
    owner.J = owner.f_c_period.shape[1]

    decomposition = owner.pinch_decomposition
    owner.T_h_in_period = np.asarray(
        decomposition.clipped_hot_supply_temperatures_by_period,
        dtype=float,
    )
    owner.T_h_out_period = np.asarray(
        decomposition.clipped_hot_target_temperatures_by_period,
        dtype=float,
    )
    owner.T_c_in_period = np.asarray(
        decomposition.clipped_cold_supply_temperatures_by_period,
        dtype=float,
    )
    owner.T_c_out_period = np.asarray(
        decomposition.clipped_cold_target_temperatures_by_period,
        dtype=float,
    )
    owner.z_i_active_period = [list(row) for row in decomposition.z_i_active_by_period]
    owner.z_j_active_period = [list(row) for row in decomposition.z_j_active_by_period]
    owner.z_i_active = list(decomposition.z_i_active)
    owner.z_j_active = list(decomposition.z_j_active)
    if decomposition.manual_stage_selection is None:
        owner.S = max(sum(owner.z_i_active), sum(owner.z_j_active))
    else:
        owner.S = decomposition.S
    owner.K = owner.S + 1
    owner.T_h_in = owner.T_h_in_period[0].copy()
    owner.T_h_out = owner.T_h_out_period[0].copy()
    owner.T_c_in = owner.T_c_in_period[0].copy()
    owner.T_c_out = owner.T_c_out_period[0].copy()

    owner.Qtot_sh_period = np.array(
        build_index_grid(
            lambda n, i: (
                owner._parent_profile_duty(
                    "hot",
                    n,
                    i,
                    owner.T_h_in_period[n][i],
                    owner.T_h_out_period[n][i],
                    owner.f_h_period[n][i],
                )
                if owner.z_i_active_period[n][i]
                else 0.0
            ),
            (owner.N_periods, owner.I),
        ),
        dtype=float,
    )
    owner.Qtot_sc_period = np.array(
        build_index_grid(
            lambda n, j: (
                owner._parent_profile_duty(
                    "cold",
                    n,
                    j,
                    owner.T_c_in_period[n][j],
                    owner.T_c_out_period[n][j],
                    owner.f_c_period[n][j],
                )
                if owner.z_j_active_period[n][j]
                else 0.0
            ),
            (owner.N_periods, owner.J),
        ),
        dtype=float,
    )
    owner.Qtot_sh = np.max(owner.Qtot_sh_period, axis=0)
    owner.Qtot_sc = np.max(owner.Qtot_sc_period, axis=0)
    owner.U_r_period = np.array(
        build_index_grid(
            lambda n, i, j: _overall_heat_transfer_coefficient(
                owner.htc_h_period[n][i],
                owner.htc_c_period[n][j],
            ),
            (owner.N_periods, owner.I, owner.J),
        ),
        dtype=float,
    )
    owner.U_hu_period = np.array(
        build_index_grid(
            lambda n, j: _overall_heat_transfer_coefficient(
                owner.htc_hu_period[n][0],
                owner.htc_c_period[n][j],
            ),
            (owner.N_periods, owner.J),
        ),
        dtype=float,
    )
    owner.U_cu_period = np.array(
        build_index_grid(
            lambda n, i: _overall_heat_transfer_coefficient(
                owner.htc_h_period[n][i],
                owner.htc_cu_period[n][0],
            ),
            (owner.N_periods, owner.I),
        ),
        dtype=float,
    )
    owner.U_r = owner.U_r_period[0].copy()
    owner.U_hu = owner.U_hu_period[0].copy()
    owner.U_cu = owner.U_cu_period[0].copy()
    owner.Q_max_period = np.array(
        build_index_grid(
            lambda n, i, j: owner._recovery_heat_upper_bound(
                period_index=n,
                hot_index=i,
                cold_index=j,
                hot_total_duty=owner.Qtot_sh_period[n][i],
                cold_total_duty=owner.Qtot_sc_period[n][j],
                hot_cp=(owner.f_h_period[n][i] * owner.z_i_active_period[n][i]),
                cold_cp=(owner.f_c_period[n][j] * owner.z_j_active_period[n][j]),
            ),
            (owner.N_periods, owner.I, owner.J),
        ),
        dtype=float,
    )
    owner.Q_max = np.max(owner.Q_max_period, axis=0)
    owner.z_feasible = build_index_grid(
        lambda i, j, _k: (
            1
            if max(owner.Q_max_period[n][i][j] for n in range(owner.N_periods))
            > owner.tol
            else 0
        ),
        (owner.I, owner.J, owner.S),
    )
    owner.z_hu_feasible = [
        1 if owner.pinch_loc == "above" and owner.z_j_active[j] > 0 else 0
        for j in range(owner.J)
    ]
    owner.z_cu_feasible = [
        1 if owner.pinch_loc == "below" and owner.z_i_active[i] > 0 else 0
        for i in range(owner.I)
    ]
