"""Migrated PDM coordinator and above/below-pinch model construction."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from ..common.indexing import build_index_grid
from ..common.solver.arrays import PreparedSolverArrays
from ..common.solver.pinch_design_decomposition import PinchDesignDecomposition
from .base import BaseHeatExchangerNetworkModel
from .stagewise import StageWiseModel
from .stagewise import _value as _scalar_value


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


class PinchDecompModel(BaseHeatExchangerNetworkModel):
    """Source-compatible private PDM slice for one pinch side."""

    def __init__(
        self,
        *,
        name: str,
        framework: Literal["PDM"],
        solver: Literal["couenne", "ipopt-pyomo", "ipopt-GEKKO", "apopt"],
        solver_arrays: PreparedSolverArrays,
        dTmin: float,
        z_restriction: list | None,
        min_dqda: float,
        minimisation_goal: Literal[
            "hot utility",
            "cold utility",
            "total utility",
            "utility costs",
            "heat recovery",
            "total cost",
            "variable total cost",
            "min units",
        ],
        non_isothermal_model: bool,
        integers: bool,
        tol: float,
        pinch_loc: Literal["above", "below"],
        pinch_decomposition: PinchDesignDecomposition,
        stage_selection: Literal["automated"] | list[int] | tuple[int, int],
        solver_options: Mapping[str, Any] | Sequence[str] | None = None,
    ) -> None:
        self.pinch_loc = pinch_loc
        self.pinch_decomposition = pinch_decomposition
        self.stage_selection = stage_selection
        super().__init__(
            name=name,
            framework=framework,
            solver=solver,
            solver_arrays=solver_arrays,
            dTmin=dTmin,
            z_restriction=z_restriction,
            min_dqda=min_dqda,
            minimisation_goal=minimisation_goal,
            non_isothermal_model=non_isothermal_model,
            integers=integers,
            tol=tol,
            solver_options=solver_options,
        )

    def setup(self) -> None:
        self.set_blank_input_parameters()
        self.get_model_parameters_from_solver_arrays()
        self.calculate_pinch()
        self.set_preprocessing()
        self.set_match_restrictions(self.z_restriction)
        self.set_stage_wise_superstructure()
        self.set_obj()

    def get_model_parameters_from_solver_arrays(self) -> None:
        super().get_model_parameters_from_solver_arrays()
        self.T_h_in_OG = self.T_h_in.copy()
        self.T_h_out_OG = self.T_h_out.copy()
        self.T_c_in_OG = self.T_c_in.copy()
        self.T_c_out_OG = self.T_c_out.copy()

    def calculate_pinch(self) -> None:
        """Read target values from the private OpenPinch decomposition."""

        if self.pinch_decomposition.pinch_location != self.pinch_loc:
            raise ValueError("pinch decomposition location does not match PDM side.")
        targets = self.pinch_decomposition.period_targets
        if len(targets) != self.N_periods:
            raise ValueError("PDM period targets must match the model period count.")
        if tuple(target.period_id for target in targets) != tuple(self.period_ids):
            raise ValueError("PDM period target identities must match solver arrays.")
        self.HU_target_by_period = [target.hot_utility_target for target in targets]
        self.CU_target_by_period = [target.cold_utility_target for target in targets]
        self.T_pinch_by_period = [
            target.shifted_pinch_temperature for target in targets
        ]
        if any(value is None for value in self.T_pinch_by_period):
            raise ValueError(
                "PDM construction requires a shifted pinch temperature in every period."
            )
        self.side_required = any(
            target > self.tol
            for target in (
                self.HU_target_by_period
                if self.pinch_loc == "above"
                else self.CU_target_by_period
            )
        )

    def set_preprocessing(self) -> None:
        """Pre-process PDM superstructure parameters."""

        self._set_multiperiod_preprocessing()

    def _set_multiperiod_preprocessing(self) -> None:
        self.I = self.f_h_period.shape[1]
        self.J = self.f_c_period.shape[1]

        decomposition = self.pinch_decomposition
        self.T_h_in_period = np.asarray(
            decomposition.clipped_hot_supply_temperatures_by_period,
            dtype=float,
        )
        self.T_h_out_period = np.asarray(
            decomposition.clipped_hot_target_temperatures_by_period,
            dtype=float,
        )
        self.T_c_in_period = np.asarray(
            decomposition.clipped_cold_supply_temperatures_by_period,
            dtype=float,
        )
        self.T_c_out_period = np.asarray(
            decomposition.clipped_cold_target_temperatures_by_period,
            dtype=float,
        )
        self.z_i_active_period = [
            list(row) for row in decomposition.z_i_active_by_period
        ]
        self.z_j_active_period = [
            list(row) for row in decomposition.z_j_active_by_period
        ]
        self.z_i_active = list(decomposition.z_i_active)
        self.z_j_active = list(decomposition.z_j_active)
        if decomposition.manual_stage_selection is None:
            self.S = max(sum(self.z_i_active), sum(self.z_j_active))
        else:
            self.S = decomposition.S
        self.K = self.S + 1
        self.T_h_in = self.T_h_in_period[0].copy()
        self.T_h_out = self.T_h_out_period[0].copy()
        self.T_c_in = self.T_c_in_period[0].copy()
        self.T_c_out = self.T_c_out_period[0].copy()

        self.Qtot_sh_period = np.array(
            build_index_grid(
                lambda n, i: (
                    self._parent_profile_duty(
                        "hot",
                        n,
                        i,
                        self.T_h_in_period[n][i],
                        self.T_h_out_period[n][i],
                        self.f_h_period[n][i],
                    )
                    if self.z_i_active_period[n][i]
                    else 0.0
                ),
                (self.N_periods, self.I),
            ),
            dtype=float,
        )
        self.Qtot_sc_period = np.array(
            build_index_grid(
                lambda n, j: (
                    self._parent_profile_duty(
                        "cold",
                        n,
                        j,
                        self.T_c_in_period[n][j],
                        self.T_c_out_period[n][j],
                        self.f_c_period[n][j],
                    )
                    if self.z_j_active_period[n][j]
                    else 0.0
                ),
                (self.N_periods, self.J),
            ),
            dtype=float,
        )
        self.Qtot_sh = np.max(self.Qtot_sh_period, axis=0)
        self.Qtot_sc = np.max(self.Qtot_sc_period, axis=0)
        self.U_r_period = np.array(
            build_index_grid(
                lambda n, i, j: _overall_heat_transfer_coefficient(
                    self.htc_h_period[n][i],
                    self.htc_c_period[n][j],
                ),
                (self.N_periods, self.I, self.J),
            ),
            dtype=float,
        )
        self.U_hu_period = np.array(
            build_index_grid(
                lambda n, j: _overall_heat_transfer_coefficient(
                    self.htc_hu_period[n][0],
                    self.htc_c_period[n][j],
                ),
                (self.N_periods, self.J),
            ),
            dtype=float,
        )
        self.U_cu_period = np.array(
            build_index_grid(
                lambda n, i: _overall_heat_transfer_coefficient(
                    self.htc_h_period[n][i],
                    self.htc_cu_period[n][0],
                ),
                (self.N_periods, self.I),
            ),
            dtype=float,
        )
        self.U_r = self.U_r_period[0].copy()
        self.U_hu = self.U_hu_period[0].copy()
        self.U_cu = self.U_cu_period[0].copy()
        self.Q_max_period = np.array(
            build_index_grid(
                lambda n, i, j: self._recovery_heat_upper_bound(
                    period_index=n,
                    hot_index=i,
                    cold_index=j,
                    hot_total_duty=self.Qtot_sh_period[n][i],
                    cold_total_duty=self.Qtot_sc_period[n][j],
                    hot_cp=(self.f_h_period[n][i] * self.z_i_active_period[n][i]),
                    cold_cp=(self.f_c_period[n][j] * self.z_j_active_period[n][j]),
                ),
                (self.N_periods, self.I, self.J),
            ),
            dtype=float,
        )
        self.Q_max = np.max(self.Q_max_period, axis=0)
        self.z_feasible = build_index_grid(
            lambda i, j, _k: (
                1
                if max(self.Q_max_period[n][i][j] for n in range(self.N_periods))
                > self.tol
                else 0
            ),
            (self.I, self.J, self.S),
        )
        self.z_hu_feasible = [
            1 if self.pinch_loc == "above" and self.z_j_active[j] > 0 else 0
            for j in range(self.J)
        ]
        self.z_cu_feasible = [
            1 if self.pinch_loc == "below" and self.z_i_active[i] > 0 else 0
            for i in range(self.I)
        ]

    def set_stage_wise_superstructure(self) -> None:
        """Create PDM variables, constraints, and binaries."""

        self._set_multiperiod_stage_wise_superstructure()

    def _set_multiperiod_stage_wise_superstructure(self) -> None:
        self.Q_r_by_period = build_index_grid(
            lambda n, i, j, k: (
                self.m.Var(
                    value=self.Q_max_period[n][i][j] / 3,
                    ub=self.Q_max_period[n][i][j],
                    lb=0.0,
                    name=f"Q_H{i}_to_C{j}_at_S{k}_period{n}",
                )
                if self.z_allowed[i][j][k] > 0
                else self.m.Param(
                    value=0.0,
                    name=f"Q_H{i}_to_C{j}_at_S{k}_period{n}",
                )
            ),
            (self.N_periods, self.I, self.J, self.S),
        )
        self.Q_c_by_period = build_index_grid(
            lambda n, i: (
                self.m.Var(
                    value=0,
                    ub=self.Qtot_sh_period[n][i],
                    lb=0.0,
                    name=f"Q_H{i}_to_CU_period{n}",
                )
                if self.z_cu_allowed[i] > 0
                else self.m.Param(value=0, name=f"Q_H{i}_to_CU_period{n}")
            ),
            (self.N_periods, self.I),
        )
        self.Q_h_by_period = build_index_grid(
            lambda n, j: (
                self.m.Var(
                    value=0,
                    ub=self.Qtot_sc_period[n][j],
                    lb=0.0,
                    name=f"Q_HU_to_C{j}_period{n}",
                )
                if self.z_hu_allowed[j] > 0
                else self.m.Param(value=0, name=f"Q_HU_to_C{j}_period{n}")
            ),
            (self.N_periods, self.J),
        )
        self.T_h_by_period = build_index_grid(
            lambda n, i, k: (
                self.m.Var(
                    value=self.T_h_in_period[n][i],
                    ub=self.T_h_in_period[n][i],
                    lb=self.T_h_out_period[n][i],
                    name=f"T_H{i}_at_B{k}_period{n}",
                )
                if k > 0 and self.z_i_active[i] > 0
                else self.m.Param(
                    value=self.T_h_in_period[n][i],
                    name=f"T_H{i}_at_B{k}_period{n}",
                )
            ),
            (self.N_periods, self.I, self.K),
        )
        self.T_c_by_period = build_index_grid(
            lambda n, j, k: (
                self.m.Var(
                    value=self.T_c_in_period[n][j],
                    ub=self.T_c_out_period[n][j],
                    lb=self.T_c_in_period[n][j],
                    name=f"T_C{j}_at_B{k}_period{n}",
                )
                if k < self.S and self.z_j_active[j] > 0
                else self.m.Param(
                    value=self.T_c_in_period[n][j],
                    name=f"T_C{j}_at_B{k}_period{n}",
                )
            ),
            (self.N_periods, self.J, self.K),
        )
        self.Q_r = self.Q_r_by_period[0]
        self.Q_c = self.Q_c_by_period[0]
        self.Q_h = self.Q_h_by_period[0]
        self.T_h = self.T_h_by_period[0]
        self.T_c = self.T_c_by_period[0]

        self._set_piecewise_stage_heat_coordinates()

        for n in range(self.N_periods):
            _ = [
                (
                    self.m.Equation(
                        self.Qtot_sh_period[n][i]
                        - sum(
                            self.Q_r_by_period[n][i][j][k]
                            for k in range(self.S)
                            for j in range(self.J)
                        )
                        - self.Q_c_by_period[n][i]
                        == 0.0
                    )
                    if self.z_i_active_period[n][i] > 0
                    else None
                )
                for i in range(self.I)
            ]
            _ = [
                (
                    self.m.Equation(
                        self.Qtot_sc_period[n][j]
                        - sum(
                            self.Q_r_by_period[n][i][j][k]
                            for k in range(self.S)
                            for i in range(self.I)
                        )
                        - self.Q_h_by_period[n][j]
                        == 0.0
                    )
                    if self.z_j_active_period[n][j] > 0
                    else None
                )
                for j in range(self.J)
            ]
            _ = [
                (
                    self.m.Equation(
                        (self.T_h_by_period[n][i][k + 1] - self.T_h_by_period[n][i][k])
                        * self.f_h_period[n][i]
                        + sum(self.Q_r_by_period[n][i][j][k] for j in range(self.J))
                        == 0
                    )
                    if self.z_i_active_period[n][i] > 0
                    and not self._hot_parent_segmented(i)
                    else None
                )
                for k in range(self.S)
                for i in range(self.I)
            ]
            _ = [
                (
                    self.m.Equation(
                        (self.T_c_by_period[n][j][k + 1] - self.T_c_by_period[n][j][k])
                        * self.f_c_period[n][j]
                        + sum(self.Q_r_by_period[n][i][j][k] for i in range(self.I))
                        == 0
                    )
                    if self.z_j_active_period[n][j] > 0
                    and not self._cold_parent_segmented(j)
                    else None
                )
                for k in range(self.S)
                for j in range(self.J)
            ]

        if self.integers:
            self.z = [
                [
                    [
                        (
                            self.m.Var(
                                value=1,
                                ub=1,
                                lb=0,
                                integer=True,
                                name=f"z_H{i}_to_C{j}_at_S{k}",
                            )
                            if self.z_allowed[i][j][k] > 0
                            else self.m.Param(value=0, name=f"z_H{i}_to_C{j}_at_S{k}")
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            self.z_cu = [
                (
                    self.m.Var(
                        value=1,
                        ub=1,
                        lb=0,
                        integer=True,
                        name=f"z_H{i}_to_CU",
                    )
                    if self.z_cu_allowed[i] > 0
                    else self.m.Param(value=0, name=f"z_H{i}_to_CU")
                )
                for i in range(self.I)
            ]
            self.z_hu = [
                (
                    self.m.Var(
                        value=1,
                        ub=1,
                        lb=0,
                        integer=True,
                        name=f"z_HU_to_C{j}",
                    )
                    if self.z_hu_allowed[j] > 0
                    else self.m.Param(value=0, name=f"z_HU_to_C{j}")
                )
                for j in range(self.J)
            ]
            for n in range(self.N_periods):
                _ = [
                    (
                        self.m.Equation(
                            self.Q_r_by_period[n][i][j][k] * (1 - self.z[i][j][k])
                            == 0.0
                        )
                        if self.z_allowed[i][j][k] > 0
                        else None
                    )
                    for k in range(self.S)
                    for j in range(self.J)
                    for i in range(self.I)
                ]
                _ = [
                    (
                        self.m.Equation(
                            self.Q_c_by_period[n][i] * (1 - self.z_cu[i]) == 0.0
                        )
                        if self.z_cu_allowed[i] > 0
                        else None
                    )
                    for i in range(self.I)
                ]
                _ = [
                    (
                        self.m.Equation(
                            self.Q_h_by_period[n][j] * (1 - self.z_hu[j]) == 0.0
                        )
                        if self.z_hu_allowed[j] > 0
                        else None
                    )
                    for j in range(self.J)
                ]
        else:
            self.z = [
                [
                    [
                        (
                            self.m.Param(value=1, name=f"z_H{i}_to_C{j}_at_S{k}")
                            if self.z_allowed[i][j][k] > 0
                            else self.m.Param(value=0, name=f"z_H{i}_to_C{j}_at_S{k}")
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            self.z_hu = [
                (
                    self.m.Param(value=1, name=f"z_HU_to_C{j}")
                    if self.z_hu_allowed[j] > 0
                    else self.m.Param(value=0, name=f"z_HU_to_C{j}")
                )
                for j in range(self.J)
            ]
            self.z_cu = [
                (
                    self.m.Param(value=1, name=f"z_H{i}_to_CU")
                    if self.z_cu_allowed[i] > 0
                    else self.m.Param(value=0, name=f"z_H{i}_to_CU")
                )
                for i in range(self.I)
            ]

        self._set_multiperiod_utility_approach_equations()

        M_ij_period = [
            [
                [
                    max(
                        abs(self.T_h_in_period[n][i] - self.T_c_in_period[n][j]),
                        abs(self.T_h_in_period[n][i] - self.T_c_out_period[n][j]),
                        abs(self.T_h_out_period[n][i] - self.T_c_in_period[n][j]),
                        abs(self.T_h_out_period[n][i] - self.T_c_out_period[n][j]),
                    )
                    + self._recovery_approach_temperature(i, j, n)
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            for n in range(self.N_periods)
        ]
        for n in range(self.N_periods):
            _ = [
                (
                    self.m.Equation(
                        (self.T_h_by_period[n][i][k] - self.T_c_by_period[n][j][k])
                        >= self._recovery_approach_temperature(i, j, n)
                        - M_ij_period[n][i][j] * (1 - self.z[i][j][k])
                    )
                    if self.z_allowed[i][j][k] > 0
                    else None
                )
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]
            _ = [
                (
                    self.m.Equation(
                        (
                            self.T_h_by_period[n][i][k + 1]
                            - self.T_c_by_period[n][j][k + 1]
                        )
                        >= self._recovery_approach_temperature(i, j, n)
                        - M_ij_period[n][i][j] * (1 - self.z[i][j][k])
                    )
                    if self.z_allowed[i][j][k] > 0
                    else None
                )
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]

        self.dqda = []
        self.alpha = []

    def set_obj(self) -> None:
        """Attach PDM objective expressions."""

        if self.minimisation_goal == "hot utility":
            for n in range(self.N_periods):
                self.m.Equation(
                    sum(self.Q_h_by_period[n]) - self.HU_target_by_period[n] >= 0.0
                )
            self.m.Minimize(
                self._weighted_state_average(
                    [sum(self.Q_h_by_period[n]) for n in range(self.N_periods)]
                )
            )
        elif self.minimisation_goal == "cold utility":
            for n in range(self.N_periods):
                self.m.Equation(
                    sum(self.Q_c_by_period[n]) - self.CU_target_by_period[n] >= 0.0
                )
            self.m.Minimize(
                self._weighted_state_average(
                    [sum(self.Q_c_by_period[n]) for n in range(self.N_periods)]
                )
            )
        elif self.minimisation_goal == "total utility":
            self.m.Minimize(
                self._weighted_state_average(
                    [
                        sum(self.Q_h_by_period[n]) + sum(self.Q_c_by_period[n])
                        for n in range(self.N_periods)
                    ]
                )
            )
        elif self.minimisation_goal == "heat recovery":
            self.m.Maximize(
                self._weighted_state_average(
                    [
                        self.m.sum(
                            [
                                self.Q_r_by_period[n][i][j][k]
                                for i in range(self.I)
                                for j in range(self.J)
                                for k in range(self.S)
                            ]
                        )
                        for n in range(self.N_periods)
                    ]
                )
            )
        elif self.minimisation_goal == "min units":
            self.m.Minimize(
                self.m.sum(
                    [
                        self.z[i][j][k]
                        for k in range(self.S)
                        for j in range(self.J)
                        for i in range(self.I)
                    ]
                )
            )

    def get_post_process(self) -> None:
        """Extract source PDM side arrays after a successful solve."""

        if self.mSuccess != 1:
            return
        self._get_multiperiod_post_process()

    def _get_multiperiod_post_process(self) -> None:
        q_r = build_index_grid(
            lambda n, i, j, k: self._active_binary_value(
                self.Q_r_by_period[n][i][j][k]
            ),
            (self.N_periods, self.I, self.J, self.S),
        )
        q_h = build_index_grid(
            lambda n, j: self._active_binary_value(self.Q_h_by_period[n][j]),
            (self.N_periods, self.J),
        )
        q_c = build_index_grid(
            lambda n, i: self._active_binary_value(self.Q_c_by_period[n][i]),
            (self.N_periods, self.I),
        )

        self.z = build_index_grid(
            lambda i, j, k: _active_period_flag(
                (q_r[n][i][j][k] for n in range(self.N_periods)),
                self.tol,
            ),
            (self.I, self.J, self.S),
        )
        self.z_hu = build_index_grid(
            lambda j: _active_period_flag(
                (q_h[n][j] for n in range(self.N_periods)),
                self.tol,
            ),
            (self.J,),
        )
        self.z_cu = build_index_grid(
            lambda i: _active_period_flag(
                (q_c[n][i] for n in range(self.N_periods)),
                self.tol,
            ),
            (self.I,),
        )
        self.n_recovery_units = sum(
            self.z[i][j][k][0]
            for k in range(self.S)
            for j in range(self.J)
            for i in range(self.I)
        )
        self.n_hu_units = sum(self.z_hu[j][0] for j in range(self.J))
        self.n_cu_units = sum(self.z_cu[i][0] for i in range(self.I))
        self.n_units = self.n_recovery_units + self.n_hu_units + self.n_cu_units

        def recovery_temperature_difference(
            n: int, i: int, j: int, hot_stage: int, cold_stage: int
        ) -> list[float]:
            if self.z[i][j][cold_stage][0] <= 0:
                return [self._recovery_approach_temperature(i, j, n)]
            return [
                self._active_binary_value(self.T_h_by_period[n][i][hot_stage])
                - self._active_binary_value(self.T_c_by_period[n][j][hot_stage])
            ]

        self.theta_1_by_period = build_index_grid(
            lambda n, i, j, k: recovery_temperature_difference(n, i, j, k, k),
            (self.N_periods, self.I, self.J, self.S),
        )
        self.theta_2_by_period = build_index_grid(
            lambda n, i, j, k: recovery_temperature_difference(n, i, j, k + 1, k),
            (self.N_periods, self.I, self.J, self.S),
        )
        self.theta_1 = self.theta_1_by_period[0]
        self.theta_2 = self.theta_2_by_period[0]

        self.LMTD_r_by_period = build_index_grid(
            lambda n, i, j, k: self._post_process_lmtd(
                self.theta_1_by_period[n][i][j][k][0],
                self.theta_2_by_period[n][i][j][k][0],
                self.z[i][j][k][0],
                formula_allowed=_lmtd_formula_allowed(
                    self.theta_1_by_period[n][i][j][k][0],
                    self.theta_2_by_period[n][i][j][k][0],
                    self._recovery_approach_temperature(i, j, n),
                    self.tol,
                ),
            ),
            (self.N_periods, self.I, self.J, self.S),
        )
        self.area_r_by_period = build_index_grid(
            lambda n, i, j, k: _area_from_heat_load(
                q_r[n][i][j][k],
                self.U_r_period[n][i][j],
                self.LMTD_r_by_period[n][i][j][k],
                self.tol,
            ),
            (self.N_periods, self.I, self.J, self.S),
        )
        self.LMTD_r = self.LMTD_r_by_period[0]
        self.area_r = build_index_grid(
            lambda i, j, k: max(
                self.area_r_by_period[n][i][j][k] for n in range(self.N_periods)
            ),
            (self.I, self.J, self.S),
        )
        self._apply_segment_recovery_areas(q_r)

        self.LMTD_hu_by_period = build_index_grid(
            lambda n, j: self._post_process_lmtd(
                self.T_hu_in_period[n][0] - self.T_c_out_period[n][j],
                self._utility_solved_outlet_temperature("hot", n, j, q_h[n][j])
                - self._active_binary_value(self.T_c_by_period[n][j][0]),
                self.z_hu[j][0],
                formula_allowed=_lmtd_formula_allowed(
                    self.T_hu_in_period[n][0] - self.T_c_out_period[n][j],
                    self._utility_solved_outlet_temperature("hot", n, j, q_h[n][j])
                    - self._active_binary_value(self.T_c_by_period[n][j][0]),
                    self._hot_utility_inlet_approach_temperature(j, n),
                    self.tol,
                    self._hot_utility_outlet_approach_temperature(j, n, q_h[n][j]),
                ),
            ),
            (self.N_periods, self.J),
        )
        self.area_hu_by_period = build_index_grid(
            lambda n, j: _area_from_heat_load(
                q_h[n][j],
                self.U_hu_period[n][j],
                self.LMTD_hu_by_period[n][j],
                self.tol,
            ),
            (self.N_periods, self.J),
        )

        self.LMTD_cu_by_period = build_index_grid(
            lambda n, i: self._post_process_lmtd(
                self._active_binary_value(self.T_h_by_period[n][i][self.S])
                - self._utility_solved_outlet_temperature("cold", n, i, q_c[n][i]),
                self.T_h_out_period[n][i] - self.T_cu_in_period[n][0],
                self.z_cu[i][0],
                formula_allowed=_lmtd_formula_allowed(
                    self._active_binary_value(self.T_h_by_period[n][i][self.S])
                    - self._utility_solved_outlet_temperature("cold", n, i, q_c[n][i]),
                    self.T_h_out_period[n][i] - self.T_cu_in_period[n][0],
                    self._cold_utility_outlet_approach_temperature(i, n, q_c[n][i]),
                    self.tol,
                    self._cold_utility_inlet_approach_temperature(i, n),
                ),
                fallback_delta=(self.T_h_out_period[n][i] - self.T_cu_in_period[n][0]),
            ),
            (self.N_periods, self.I),
        )
        self.area_cu_by_period = build_index_grid(
            lambda n, i: _area_from_heat_load(
                q_c[n][i],
                self.U_cu_period[n][i],
                self.LMTD_cu_by_period[n][i],
                self.tol,
            ),
            (self.N_periods, self.I),
        )
        self._apply_segment_utility_areas(q_h, q_c)
        self.LMTD_hu = self.LMTD_hu_by_period[0]
        self.LMTD_cu = self.LMTD_cu_by_period[0]

        self.area_hu = build_index_grid(
            lambda j: max(self.area_hu_by_period[n][j] for n in range(self.N_periods)),
            (self.J,),
        )
        self.area_cu = build_index_grid(
            lambda i: max(self.area_cu_by_period[n][i] for n in range(self.N_periods)),
            (self.I,),
        )
        self.Q_hu_total_by_period = build_index_grid(
            lambda n: sum(q_h[n]),
            (self.N_periods,),
        )
        self.Q_cu_total_by_period = build_index_grid(
            lambda n: sum(q_c[n]),
            (self.N_periods,),
        )
        self.Q_r_total_by_period = build_index_grid(
            lambda n: sum(
                q_r[n][i][j][k]
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ),
            (self.N_periods,),
        )
        self.Q_hu_total = self._weighted_numeric_average(self.Q_hu_total_by_period)
        self.Q_cu_total = self._weighted_numeric_average(self.Q_cu_total_by_period)
        self.Q_r_total = self._weighted_numeric_average(self.Q_r_total_by_period)
        self.operating_cost_by_period = [
            self._utility_cost_value("hot", n, self.Q_hu_total_by_period[n])
            + self._utility_cost_value("cold", n, self.Q_cu_total_by_period[n])
            for n in range(self.N_periods)
        ]
        self.weighted_operating_cost_value = self._weighted_numeric_average(
            self.operating_cost_by_period
        )
        self.capital_cost_value = (
            self.unit_cost[0] * self.n_units
            + self.A_coeff[0]
            * sum(
                self.area_r[i][j][k] ** self.A_exp[0]
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            )
            + self.hu_coeff[0]
            * sum(self.area_hu[j] ** self.hu_exp[0] for j in range(self.J))
            + self.cu_coeff[0]
            * sum(self.area_cu[i] ** self.cu_exp[0] for i in range(self.I))
        )
        self.hu_cost_total = self._weighted_numeric_average(
            [
                self._utility_cost_value("hot", n, self.Q_hu_total_by_period[n])
                for n in range(self.N_periods)
            ]
        )
        self.cu_cost_total = self._weighted_numeric_average(
            [
                self._utility_cost_value("cold", n, self.Q_cu_total_by_period[n])
                for n in range(self.N_periods)
            ]
        )
        self.recovery_area_cost_total = self.A_coeff[0] * sum(
            self.area_r[i][j][k] ** self.A_exp[0]
            for k in range(self.S)
            for j in range(self.J)
            for i in range(self.I)
        )
        self.hu_area_cost_total = self.hu_coeff[0] * sum(
            self.area_hu[j] ** self.hu_exp[0] for j in range(self.J)
        )
        self.cu_area_cost_total = self.cu_coeff[0] * sum(
            self.area_cu[i] ** self.cu_exp[0] for i in range(self.I)
        )
        self.TAC_model = self.m.options.objfcnval
        self.TAC = self.capital_cost_value + self.weighted_operating_cost_value

    def _active_binary_value(self, value) -> float:
        return _scalar_value(value)

    def _weighted_numeric_average(self, values: Sequence[float]) -> float:
        return float(
            sum(
                float(self.period_weights[n]) * float(values[n])
                for n in range(self.N_periods)
            )
            / self.period_weight_sum
        )

    def amalgamate_networks(
        self,
        *,
        below_case: "PinchDecompModel",
        above_case: "PinchDecompModel",
    ) -> StageWiseModel:
        """Amalgamate solved above/below-pinch side models into one network."""

        above_required = bool(above_case.side_required)
        below_required = bool(below_case.side_required)

        amalgamated = StageWiseModel(
            name="amalgamated",
            framework=self.framework,
            solver=self.solver,
            solver_arrays=self.solver_arrays,
            stages=(above_case.S if above_required else 0)
            + (below_case.S if below_required else 0),
            dTmin=self.dTmin,
            z_restriction=self.z_restriction,
            min_dqda=self.min_dqda,
            minimisation_goal="total utility",
            non_isothermal_model=self.non_isothermal_model,
            integers=True,
            tol=1e-3,
            solver_options=self.solver_options,
        )
        if (above_required and above_case.mSuccess == 0) or (
            below_required and below_case.mSuccess == 0
        ):
            raise ValueError(
                "Pinch Decomposition failed: "
                f"Above {above_case.mSuccess} Below {below_case.mSuccess} "
                f"dTmin {self.dTmin}"
            )

        if above_required and above_case.mSuccess == 1:
            amalgamated.mSuccess = above_case.mSuccess
            amalgamated.TAC = above_case.TAC
            amalgamated.solve_time = above_case.solve_time
            for i in range(self.I):
                for j in range(self.J):
                    for k in range(above_case.S):
                        self._copy_recovery_match(amalgamated, above_case, i, j, k, k)
            for i in range(self.I):
                for k in range(above_case.K):
                    for n in range(amalgamated.N_periods):
                        value = (
                            above_case.T_h_by_period[n][i][k][0]
                            if above_case.z_i_active_period[n][i] > 0
                            else amalgamated.T_h_in_period[n][i]
                        )
                        amalgamated.T_h_by_period[n][i][k].VALUE.value = [value]
            for j in range(self.J):
                for k in range(above_case.K):
                    for n in range(amalgamated.N_periods):
                        value = (
                            above_case.T_c_by_period[n][j][k][0]
                            if above_case.z_j_active_period[n][j] > 0
                            else amalgamated.T_c_out_period[n][j]
                        )
                        amalgamated.T_c_by_period[n][j][k].VALUE.value = [value]
            for j in range(self.J):
                for n in range(amalgamated.N_periods):
                    amalgamated.Q_h_by_period[n][j].VALUE.value = [
                        above_case.Q_h_by_period[n][j][0]
                    ]
                amalgamated.z_hu[j].VALUE.value = [above_case.z_hu[j][0]]
            if not below_required:
                for i in range(self.I):
                    for n in range(amalgamated.N_periods):
                        amalgamated.Q_c_by_period[n][i].VALUE.value = [0]
                    amalgamated.z_cu[i].VALUE.value = [0]
                    amalgamated.minimisation_goal = "hot utility"

        if below_required and below_case.mSuccess == 1:
            amalgamated.mSuccess = below_case.mSuccess
            amalgamated.TAC = below_case.TAC
            amalgamated.solve_time = below_case.solve_time
            if not above_required:
                above_case.S = 0
                above_case.K = 0
            for i in range(self.I):
                for j in range(self.J):
                    for k in range(above_case.S, amalgamated.S):
                        self._copy_recovery_match(
                            amalgamated,
                            below_case,
                            i,
                            j,
                            k - above_case.S,
                            k,
                        )
            for i in range(self.I):
                for k in range(above_case.K, amalgamated.K):
                    for n in range(amalgamated.N_periods):
                        if below_case.z_i_active_period[n][i] > 0:
                            value = (
                                below_case.T_h_by_period[n][i][k - above_case.K + 1][0]
                                if above_required
                                else round(below_case.T_h_by_period[n][i][k][0], 5)
                            )
                        else:
                            value = amalgamated.T_h_out_period[n][i]
                        amalgamated.T_h_by_period[n][i][k].VALUE.value = [value]
            for j in range(self.J):
                for k in range(above_case.K, amalgamated.K):
                    for n in range(amalgamated.N_periods):
                        if below_case.z_j_active_period[n][j] > 0:
                            value = (
                                below_case.T_c_by_period[n][j][k - above_case.K + 1][0]
                                if above_required
                                else round(below_case.T_c_by_period[n][j][k][0], 5)
                            )
                        else:
                            value = amalgamated.T_c_in_period[n][j]
                        amalgamated.T_c_by_period[n][j][k].VALUE.value = [value]
            for i in range(self.I):
                for n in range(amalgamated.N_periods):
                    amalgamated.Q_c_by_period[n][i].VALUE.value = [
                        round(below_case.Q_c_by_period[n][i][0], 5)
                    ]
                amalgamated.z_cu[i].VALUE.value = [below_case.z_cu[i][0]]
            if not above_required:
                for j in range(self.J):
                    for n in range(amalgamated.N_periods):
                        amalgamated.Q_h_by_period[n][j].VALUE.value = [0]
                    amalgamated.z_hu[j].VALUE.value = [0]
                    amalgamated.minimisation_goal = "cold utility"

        if (
            above_required
            and below_required
            and above_case.mSuccess == 1
            and below_case.mSuccess == 1
        ):
            amalgamated.mSuccess = 1
            amalgamated.TAC = below_case.TAC + above_case.TAC
            amalgamated.solve_time = above_case.solve_time + below_case.solve_time
            amalgamated.S = above_case.S + below_case.S

        amalgamated.K = amalgamated.S + 1
        amalgamated.z_allowed = [
            [
                [
                    (
                        1
                        if (
                            max(
                                amalgamated.Q_r_by_period[n][i][j][k][0]
                                for n in range(amalgamated.N_periods)
                            )
                            if hasattr(amalgamated, "Q_r_by_period")
                            else amalgamated.Q_r[i][j][k][0]
                        )
                        > self.tol
                        else 0
                    )
                    for k in range(amalgamated.S)
                ]
                for j in range(amalgamated.J)
            ]
            for i in range(amalgamated.I)
        ]
        return amalgamated

    def _copy_recovery_match(
        self,
        target: StageWiseModel,
        source: "PinchDecompModel",
        i: int,
        j: int,
        source_stage: int,
        target_stage: int,
    ) -> None:
        target.z[i][j][target_stage].VALUE.value = [source.z[i][j][source_stage][0]]
        if target.N_periods != source.N_periods:
            raise ValueError("PDM side period counts must match during amalgamation.")
        for n in range(target.N_periods):
            target.Q_r_by_period[n][i][j][target_stage].VALUE.value = [
                source.Q_r_by_period[n][i][j][source_stage][0]
            ]
            target.theta_1_by_period[n][i][j][target_stage].VALUE.value = [
                source.theta_1_by_period[n][i][j][source_stage][0]
            ]
            target.theta_2_by_period[n][i][j][target_stage].VALUE.value = [
                source.theta_2_by_period[n][i][j][source_stage][0]
            ]
        if source.non_isothermal_model:
            for n in range(target.N_periods):
                target.X_by_period[n][i][j][target_stage].VALUE.value = [
                    source.X_by_period[n][i][j][source_stage][0]
                ]
                target.Y_by_period[n][j][i][target_stage].VALUE.value = [
                    source.Y_by_period[n][j][i][source_stage][0]
                ]
                target.T_h_out_x_by_period[n][i][j][target_stage].VALUE.value = [
                    source.T_h_out_x_by_period[n][i][j][source_stage][0]
                ]
                target.T_c_out_y_by_period[n][j][i][target_stage].VALUE.value = [
                    source.T_c_out_y_by_period[n][j][i][source_stage][0]
                ]


__all__ = ["PinchDecompModel"]
