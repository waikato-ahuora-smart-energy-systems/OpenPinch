"""Migrated PDM coordinator and above/below-pinch model construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from ..common.solver.arrays import PreparedSolverArrays
from ..common.solver.pinch_design_snapshot import PinchDecompositionSnapshot
from .base import BaseHeatExchangerNetworkModel
from .stagewise import StageWiseModel


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
        pinch_snapshot: PinchDecompositionSnapshot,
        stage_selection: Literal["automated"] | list[int] | tuple[int, int],
        solver_options: Mapping[str, Any] | Sequence[str] | None = None,
    ) -> None:
        self.pinch_loc = pinch_loc
        self.pinch_snapshot = pinch_snapshot
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
        """Read target values from the HENS-04 private OpenPinch snapshot."""

        if self.pinch_snapshot.pinch_location != self.pinch_loc:
            raise ValueError("pinch snapshot location does not match PDM side.")
        target = self.pinch_snapshot.target
        self.HU_target = target.hot_utility_target
        self.CU_target = target.cold_utility_target
        self.T_pinch = target.shifted_pinch_temperature
        if self.T_pinch is None:
            raise ValueError("PDM construction requires a shifted pinch temperature.")

    def set_preprocessing(self) -> None:
        """Pre-process source PDM superstructure parameters unchanged."""

        self.I = len(self.f_h)
        self.J = len(self.f_c)

        snapshot = self.pinch_snapshot
        self.z_i_active = list(snapshot.z_i_active)
        self.z_j_active = list(snapshot.z_j_active)
        self.T_h_in = np.array(snapshot.clipped_hot_supply_temperatures, dtype=float)
        self.T_h_out = np.array(snapshot.clipped_hot_target_temperatures, dtype=float)
        self.T_c_in = np.array(snapshot.clipped_cold_supply_temperatures, dtype=float)
        self.T_c_out = np.array(snapshot.clipped_cold_target_temperatures, dtype=float)
        self.S = snapshot.S
        self.K = snapshot.K

        self.Qtot_sh = np.array(
            [
                (self.T_h_in[i] - self.T_h_out[i]) * self.f_h[i] * self.z_i_active[i]
                for i in range(self.I)
            ]
        )
        self.Qtot_sc = np.array(
            [
                self.f_c[j] * (self.T_c_out[j] - self.T_c_in[j]) * self.z_j_active[j]
                for j in range(self.J)
            ]
        )
        self.U_r = np.array(
            [
                [1 / (1 / self.htc_h[i] + 1 / self.htc_c[j]) for j in range(self.J)]
                for i in range(self.I)
            ]
        )
        self.U_hu = np.array(
            [1 / (1 / self.htc_hu[0] + 1 / self.htc_c[j]) for j in range(self.J)]
        )
        self.U_cu = np.array(
            [1 / (1 / self.htc_h[i] + 1 / self.htc_cu[0]) for i in range(self.I)]
        )
        self.Q_max = np.array(
            [
                [
                    max(
                        self.T_h_in[i]
                        - self.T_c_in[j]
                        - self._recovery_approach_temperature(i, j),
                        0.0,
                    )
                    * min(
                        self.f_h[i] * self.z_i_active[i],
                        self.f_c[j] * self.z_j_active[j],
                    )
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
        )
        self.z_feasible = [
            [
                [1 if self.Q_max[i][j] > self.tol else 0 for k in range(self.S)]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.z_hu_feasible = [
            1 if self.pinch_loc == "above" and self.z_j_active[j] > 0 else 0
            for j in range(self.J)
        ]
        self.z_cu_feasible = [
            1 if self.pinch_loc == "below" and self.z_i_active[i] > 0 else 0
            for i in range(self.I)
        ]

    def set_stage_wise_superstructure(self) -> None:
        """Create the source PDM variables, constraints, and binaries."""

        self.Q_r = [
            [
                [
                    (
                        self.m.Var(
                            value=self.Q_max[i][j] / 3,
                            ub=self.Q_max[i][j],
                            lb=0.0,
                            name=f"Q_H{i}_to_C{j}_at_S{k}",
                        )
                        if self.z_allowed[i][j][k] > 0
                        else self.m.Param(value=0.0, name=f"Q_H{i}_to_C{j}_at_S{k}")
                    )
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.Q_c = [
            (
                self.m.Var(
                    value=0,
                    ub=self.Qtot_sh[i],
                    lb=0.0,
                    name=f"Q_H{i}_to_CU",
                )
                if self.z_cu_allowed[i] > 0
                else self.m.Param(value=0, name=f"Q_H{i}_to_CU")
            )
            for i in range(self.I)
        ]
        self.Q_h = [
            (
                self.m.Var(
                    value=0,
                    ub=self.Qtot_sc[j],
                    lb=0.0,
                    name=f"Q_HU_to_C{j}",
                )
                if self.z_hu_allowed[j] > 0
                else self.m.Param(value=0, name=f"Q_HU_to_C{j}")
            )
            for j in range(self.J)
        ]
        self.T_h = [
            [
                (
                    self.m.Var(
                        value=self.T_h_in[i],
                        ub=self.T_h_in[i],
                        lb=self.T_h_out[i],
                        name=f"T_H{i}_at_B{k}",
                    )
                    if k > 0 and self.z_i_active[i] > 0
                    else self.m.Param(value=self.T_h_in[i], name=f"T_H{i}_at_B{k}")
                )
                for k in range(self.K)
            ]
            for i in range(self.I)
        ]
        self.T_c = [
            [
                (
                    self.m.Var(
                        value=self.T_c_in[j],
                        ub=self.T_c_out[j],
                        lb=self.T_c_in[j],
                        name=f"T_C{j}_at_B{k}",
                    )
                    if k < self.S and self.z_j_active[j] > 0
                    else self.m.Param(value=self.T_c_in[j], name=f"T_C{j}_at_B{k}")
                )
                for k in range(self.K)
            ]
            for j in range(self.J)
        ]

        _ = [
            (
                self.m.Equation(
                    self.Qtot_sh[i]
                    - sum(
                        self.Q_r[i][j][k] for k in range(self.S) for j in range(self.J)
                    )
                    - self.Q_c[i]
                    == 0.0
                )
                if self.z_i_active[i] > 0
                else None
            )
            for i in range(self.I)
        ]
        _ = [
            (
                self.m.Equation(
                    self.Qtot_sc[j]
                    - sum(
                        self.Q_r[i][j][k] for k in range(self.S) for i in range(self.I)
                    )
                    - self.Q_h[j]
                    == 0.0
                )
                if self.z_j_active[j] > 0
                else None
            )
            for j in range(self.J)
        ]
        _ = [
            (
                self.m.Equation(
                    (self.T_h[i][k + 1] - self.T_h[i][k]) * self.f_h[i]
                    + sum(self.Q_r[i][j][k] for j in range(self.J))
                    == 0
                )
                if self.z_i_active[i] > 0
                else None
            )
            for k in range(self.S)
            for i in range(self.I)
        ]
        _ = [
            (
                self.m.Equation(
                    (self.T_c[j][k + 1] - self.T_c[j][k]) * self.f_c[j]
                    + sum(self.Q_r[i][j][k] for i in range(self.I))
                    == 0
                )
                if self.z_j_active[j] > 0
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
            _ = [
                (
                    self.m.Equation(self.Q_r[i][j][k] * (1 - self.z[i][j][k]) == 0.0)
                    if self.z_allowed[i][j][k] > 0
                    else None
                )
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]
            _ = [
                (
                    self.m.Equation(self.Q_c[i] * (1 - self.z_cu[i]) == 0.0)
                    if self.z_cu_allowed[i] > 0
                    else None
                )
                for i in range(self.I)
            ]
            _ = [
                (
                    self.m.Equation(self.Q_h[j] * (1 - self.z_hu[j]) == 0.0)
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

        M_ij = [
            [
                max(
                    abs(self.T_h_in[i] - self.T_c_in[j]),
                    abs(self.T_h_in[i] - self.T_c_out[j]),
                    abs(self.T_h_out[i] - self.T_c_in[j]),
                    abs(self.T_h_out[i] - self.T_c_out[j]),
                )
                + self._recovery_approach_temperature(i, j)
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        _ = [
            (
                self.m.Equation(
                    (self.T_h[i][k] - self.T_c[j][k])
                    >= self._recovery_approach_temperature(i, j)
                    - M_ij[i][j] * (1 - self.z[i][j][k])
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
                    (self.T_h[i][k + 1] - self.T_c[j][k + 1])
                    >= self._recovery_approach_temperature(i, j)
                    - M_ij[i][j] * (1 - self.z[i][j][k])
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
        """Attach source PDM objective expressions unchanged."""

        if self.minimisation_goal == "hot utility":
            self.m.Equation(sum(self.Q_h) - self.HU_target >= 0.0)
            self.m.Minimize(sum(self.Q_h))
        elif self.minimisation_goal == "cold utility":
            self.m.Equation(sum(self.Q_c) - self.CU_target >= 0.0)
            self.m.Minimize(sum(self.Q_c))
        elif self.minimisation_goal == "total utility":
            self.m.Minimize(sum(self.Q_h) + sum(self.Q_c))
        elif self.minimisation_goal == "heat recovery":
            self.m.Maximize(
                self.m.sum(
                    [
                        self.Q_r[i][j][k]
                        for i in range(self.I)
                        for j in range(self.J)
                        for k in range(self.S)
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

        self.n_units = (
            sum(
                self.z[i][j][k][0] if self.Q_r[i][j][k][0] > self.tol else 0
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            )
            + sum(
                self.z_cu[i][0] if self.Q_c[i][0] > self.tol else 0
                for i in range(self.I)
            )
            + sum(
                self.z_hu[j][0] if self.Q_h[j][0] > self.tol else 0
                for j in range(self.J)
            )
        )
        self.n_recovery_units = sum(
            self.z[i][j][k][0] if self.Q_r[i][j][k][0] > self.tol else 0
            for k in range(self.S)
            for j in range(self.J)
            for i in range(self.I)
        )
        self.n_hu_units = sum(
            self.z_hu[j][0] if self.Q_h[j][0] > self.tol else 0 for j in range(self.J)
        )
        self.n_cu_units = sum(
            self.z_cu[i][0] if self.Q_c[i][0] > self.tol else 0 for i in range(self.I)
        )

        if self.minimisation_goal not in {"total cost", "variable total cost"}:
            self.theta_1 = [
                [
                    [
                        (
                            [self.T_h[i][k][0] - self.T_c[j][k][0]]
                            if self.z[i][j][k][0] > 0
                            else [self._recovery_approach_temperature(i, j)]
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            self.theta_2 = [
                [
                    [
                        (
                            [self.T_h[i][k + 1][0] - self.T_c[j][k + 1][0]]
                            if self.z[i][j][k][0] > 0
                            else [self._recovery_approach_temperature(i, j)]
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]

        self.LMTD_r = [
            [
                [
                    self._post_process_lmtd(
                        self.theta_1[i][j][k][0],
                        self.theta_2[i][j][k][0],
                        self.z[i][j][k][0],
                        formula_allowed=(
                            abs(self.theta_1[i][j][k][0] - self.theta_2[i][j][k][0])
                            > self.tol
                            and self.theta_1[i][j][k][0]
                            >= self._recovery_approach_temperature(i, j)
                            and self.theta_2[i][j][k][0]
                            >= self._recovery_approach_temperature(i, j)
                        ),
                    )
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.area_r = [
            [
                [
                    (
                        self.Q_r[i][j][k][0] / self.U_r[i][j] / self.LMTD_r[i][j][k]
                        if self.LMTD_r[i][j][k] > self.tol
                        and self.Q_r[i][j][k][0] > self.tol
                        else 0.0
                    )
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.LMTD_hu = [
            self._post_process_lmtd(
                self.T_hu_in[0] - self.T_c_out[j],
                self.T_hu_out[0] - self.T_c[j][0][0],
                self.z_hu[j][0],
                formula_allowed=(
                    abs(
                        (self.T_hu_in[0] - self.T_c_out[j])
                        - (self.T_hu_out[0] - self.T_c[j][0][0])
                    )
                    > self.tol
                    and self.T_hu_in[0] - self.T_c_out[j]
                    >= self._hot_utility_approach_temperature(j)
                    and self.T_hu_out[0] - self.T_c[j][0][0]
                    >= self._hot_utility_approach_temperature(j)
                ),
            )
            for j in range(self.J)
        ]
        self.area_hu = [
            (
                self.Q_h[j][0] / self.U_hu[j] / self.LMTD_hu[j]
                if self.LMTD_hu[j] > self.tol and self.Q_h[j][0] > self.tol
                else 0.0
            )
            for j in range(self.J)
        ]
        self.LMTD_cu = [
            self._post_process_lmtd(
                self.T_h[i][self.S][0] - self.T_cu_out[0],
                self.T_h_out[i] - self.T_cu_in[0],
                self.z_cu[i][0],
                formula_allowed=(
                    abs(
                        (self.T_h[i][self.S][0] - self.T_cu_out[0])
                        - (self.T_h_out[i] - self.T_cu_in[0])
                    )
                    > self.tol
                    and self.T_h[i][self.S][0] - self.T_cu_out[0]
                    >= self._cold_utility_approach_temperature(i)
                    and self.T_h_out[i] - self.T_cu_in[0]
                    >= self._cold_utility_approach_temperature(i)
                ),
                fallback_delta=self.T_h_out[i] - self.T_cu_in[0],
            )
            for i in range(self.I)
        ]
        self.area_cu = [
            (
                self.Q_c[i][0] / self.U_cu[i] / self.LMTD_cu[i]
                if self.LMTD_cu[i] > self.tol and self.Q_c[i][0] > self.tol
                else 0.0
            )
            for i in range(self.I)
        ]
        self.Q_cu_total = sum(self.Q_c[i][0] for i in range(self.I))
        self.Q_hu_total = sum(self.Q_h[j][0] for j in range(self.J))
        self.Q_r_total = sum(
            self.Q_r[i][j][k][0]
            for k in range(self.S)
            for j in range(self.J)
            for i in range(self.I)
        )
        self.dqda = [
            [
                [
                    (
                        (
                            self.theta_1[i][j][k][0]
                            * self.theta_2[i][j][k][0]
                            * self.U_r[i][j]
                        )
                        / (self.T_h[i][k][0] - self.T_c[j][k + 1][0])
                        * self.z[i][j][k][0]
                        if (self.T_h[i][k][0] - self.T_c[j][k + 1][0]) > 0.0
                        else 0.0
                    )
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.alpha = self.get_alpha_values()
        self.dtacda = [
            [[None for _k in range(self.S)] for _j in range(self.J)]
            for _i in range(self.I)
        ]
        for k in range(self.S):
            for j in range(self.J):
                for i in range(self.I):
                    if self.area_r[i][j][k] > 0.0:
                        self.dtacda[i][j][k] = self.dqda[i][j][k] * (
                            self.hu_cost[0] + self.cu_cost[0]
                        ) - (
                            self.A_coeff[0]
                            * self.A_exp[0]
                            * self.area_r[i][j][k] ** (self.A_exp[0] - 1)
                        )
                    else:
                        self.dtacda[i][j][k] = self.dqda[i][j][k] * (
                            self.hu_cost[0] + self.cu_cost[0]
                        ) - (self.A_coeff[0] * self.A_exp[0])

        self.TAC_model = self.m.options.objfcnval
        self.TAC = (
            self.hu_cost[0] * sum(self.Q_h[j][0] for j in range(self.J))
            + self.cu_cost[0] * sum(self.Q_c[i][0] for i in range(self.I))
            + self.unit_cost[0] * self.n_units
            + self.A_coeff[0]
            * sum(
                self.area_r[i][j][k] ** self.A_exp[0]
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            )
            + self.hu_coeff[0]
            * sum(self.area_hu[j] ** self.A_exp[0] for j in range(self.J))
            + self.cu_coeff[0]
            * sum(self.area_cu[i] ** self.A_exp[0] for i in range(self.I))
        )

    def amalgamate_networks(
        self,
        *,
        below_case: "PinchDecompModel",
        above_case: "PinchDecompModel",
    ) -> StageWiseModel:
        """Amalgamate solved above/below-pinch side models into one network."""

        amalgamated = StageWiseModel(
            name="amalgamated",
            framework=self.framework,
            solver=self.solver,
            solver_arrays=self.solver_arrays,
            stages=(above_case.S if above_case.HU_target > 0 else 0)
            + (below_case.S if below_case.CU_target > 0 else 0),
            dTmin=self.dTmin,
            z_restriction=self.z_restriction,
            min_dqda=self.min_dqda,
            minimisation_goal="total utility",
            non_isothermal_model=self.non_isothermal_model,
            integers=True,
            tol=1e-3,
            solver_options=self.solver_options,
        )
        if (above_case.HU_target > 0 and above_case.mSuccess == 0) or (
            below_case.CU_target > 0 and below_case.mSuccess == 0
        ):
            raise ValueError(
                "Pinch Decomposition failed: "
                f"Above {above_case.mSuccess} Below {below_case.mSuccess} "
                f"dTmin {self.dTmin}"
            )

        if above_case.HU_target > 0 and above_case.mSuccess == 1:
            amalgamated.mSuccess = above_case.mSuccess
            amalgamated.TAC = above_case.TAC
            amalgamated.solve_time = above_case.solve_time
            for i in range(self.I):
                for j in range(self.J):
                    for k in range(above_case.S):
                        self._copy_recovery_match(amalgamated, above_case, i, j, k, k)
            for i in range(self.I):
                for k in range(above_case.K):
                    value = (
                        above_case.T_h[i][k][0]
                        if above_case.z_i_active[i] > 0
                        else amalgamated.T_h_in[i]
                    )
                    amalgamated.T_h[i][k].VALUE.value = [value]
            for j in range(self.J):
                for k in range(above_case.K):
                    value = (
                        above_case.T_c[j][k][0]
                        if above_case.z_j_active[j] > 0
                        else amalgamated.T_c_out[j]
                    )
                    amalgamated.T_c[j][k].VALUE.value = [value]
            for j in range(self.J):
                amalgamated.Q_h[j].VALUE.value = [above_case.Q_h[j][0]]
                amalgamated.z_hu[j].VALUE.value = [above_case.z_hu[j][0]]
            if below_case.CU_target == 0:
                for i in range(self.I):
                    amalgamated.Q_c[i].VALUE.value = [0]
                    amalgamated.z_cu[i].VALUE.value = [0]
                    amalgamated.minimisation_goal = "hot utility"

        if below_case.CU_target > 0 and below_case.mSuccess == 1:
            amalgamated.mSuccess = below_case.mSuccess
            amalgamated.TAC = below_case.TAC
            amalgamated.solve_time = below_case.solve_time
            if above_case.HU_target == 0:
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
                    if below_case.z_i_active[i] > 0:
                        value = (
                            below_case.T_h[i][k - above_case.K + 1][0]
                            if above_case.HU_target > 0
                            else round(below_case.T_h[i][k][0], 5)
                        )
                    else:
                        value = amalgamated.T_h_out[i]
                    amalgamated.T_h[i][k].VALUE.value = [value]
            for j in range(self.J):
                for k in range(above_case.K, amalgamated.K):
                    if below_case.z_j_active[j] > 0:
                        value = (
                            below_case.T_c[j][k - above_case.K + 1][0]
                            if above_case.HU_target > 0
                            else round(below_case.T_c[j][k][0], 5)
                        )
                    else:
                        value = amalgamated.T_c_in[j]
                    amalgamated.T_c[j][k].VALUE.value = [value]
            for i in range(self.I):
                amalgamated.Q_c[i].VALUE.value = [round(below_case.Q_c[i][0], 5)]
                amalgamated.z_cu[i].VALUE.value = [below_case.z_cu[i][0]]
            if above_case.HU_target == 0:
                for j in range(self.J):
                    amalgamated.Q_h[j].VALUE.value = [0]
                    amalgamated.z_hu[j].VALUE.value = [0]
                    amalgamated.minimisation_goal = "cold utility"

        if (
            above_case.HU_target > 0
            and below_case.CU_target > 0
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
                    1 if amalgamated.Q_r[i][j][k][0] > self.tol else 0
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
        target.Q_r[i][j][target_stage].VALUE.value = [source.Q_r[i][j][source_stage][0]]
        target.theta_1[i][j][target_stage].VALUE.value = [
            source.theta_1[i][j][source_stage][0]
        ]
        target.theta_2[i][j][target_stage].VALUE.value = [
            source.theta_2[i][j][source_stage][0]
        ]
        if source.non_isothermal_model:
            target.X[i][j][target_stage].VALUE.value = [source.X[i][j][source_stage][0]]
            target.Y[j][i][target_stage].VALUE.value = [source.Y[j][i][source_stage][0]]
            target.T_h_out_x[i][j][target_stage].VALUE.value = [
                source.T_h_out_x[i][j][source_stage][0]
            ]
            target.T_c_out_y[j][i][target_stage].VALUE.value = [
                source.T_c_out_y[j][i][source_stage][0]
            ]


__all__ = ["PinchDecompModel"]
