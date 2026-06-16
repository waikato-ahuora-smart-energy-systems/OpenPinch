"""Migrated StageWise TDM/ESM equation construction.

This module is private to the problem-rooted synthesis service. The equations
mirror the source OpenHENS StageWise model while taking prepared OpenPinch
solver arrays instead of source CSV rows.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from ..array_adapter import PreparedSolverArrays
from .base import BaseHeatExchangerNetworkModel


class StageWiseModel(BaseHeatExchangerNetworkModel):
    """Source-compatible StageWise model for private TDM/ESM construction."""

    def __init__(
        self,
        *,
        name: str,
        framework: Literal["TDM", "ESM", "PDM"],
        solver: Literal["couenne", "ipopt-pyomo", "ipopt-GEKKO", "apopt"],
        solver_arrays: PreparedSolverArrays,
        stages: int,
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
            "dQ/dA obj",
        ],
        non_isothermal_model: bool,
        integers: bool,
        tol: float,
    ) -> None:
        self.stages = stages
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
            solver_options=[],
        )

    def setup(self) -> None:
        self.set_blank_input_parameters()
        self.get_model_parameters_from_solver_arrays()
        self.set_preprocessing()
        self.set_match_restrictions(self.z_restriction)
        self.set_stage_wise_superstructure()
        if self.framework == "TDM":
            self.set_dqda_equations()
        self.set_obj()

    def set_preprocessing(self) -> None:
        """Pre-process source SynHEAT superstructure parameters unchanged."""

        self.S = self.stages
        self.K = self.S + 1
        self.I = len(self.f_h)
        self.J = len(self.f_c)

        self.Qtot_sh = np.array(
            [(self.T_h_in[i] - self.T_h_out[i]) * self.f_h[i] for i in range(self.I)]
        )
        self.Qtot_sc = np.array(
            [self.f_c[j] * (self.T_c_out[j] - self.T_c_in[j]) for j in range(self.J)]
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
                    max(self.T_h_in[i] - self.T_c_in[j] - self.dTmin, 0.0)
                    * min(self.f_h[i], self.f_c[j])
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
        )
        self.M_ij = [
            [
                max(
                    abs(self.T_h_in[i] - self.T_c_in[j]),
                    abs(self.T_h_in[i] - self.T_c_out[j]),
                    abs(self.T_h_out[i] - self.T_c_in[j]),
                    abs(self.T_h_out[i] - self.T_c_out[j]),
                )
                + self.dTmin
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]

        self.z_feasible = [
            [
                [1 if self.Q_max[i][j] > self.tol else 0 for k in range(self.S)]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.z_hu_feasible = [1 for _j in range(self.J)]
        self.z_cu_feasible = [1 for _i in range(self.I)]

    def set_stage_wise_superstructure(self) -> None:
        """Create the source StageWise variables, constraints, and binaries."""

        self.Q_r = [
            [
                [
                    self.m.Var(
                        value=self.Q_max[i][j] / 3,
                        ub=self.Q_max[i][j],
                        lb=0.0,
                        name=f"Q_H{i}_to_C{j}_at_S{k}",
                    )
                    if self.z_allowed[i][j][k] > 0
                    else self.m.Param(value=0.0, name=f"Q_H{i}_to_C{j}_at_S{k}")
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.Q_c = [
            self.m.Var(
                value=0,
                ub=self.Qtot_sh[i],
                lb=0.0,
                name=f"Q_H{i}_to_CU",
            )
            if self.z_cu_allowed[i] > 0
            else self.m.Param(value=0, name=f"Q_H{i}_to_CU")
            for i in range(self.I)
        ]
        self.Q_h = [
            self.m.Var(
                value=0,
                ub=self.Qtot_sc[j],
                lb=0.0,
                name=f"Q_HU_to_C{j}",
            )
            if self.z_hu_allowed[j] > 0
            else self.m.Param(value=0, name=f"Q_HU_to_C{j}")
            for j in range(self.J)
        ]
        self.T_h = [
            [
                self.m.Var(
                    value=self.T_h_in[i],
                    ub=self.T_h_in[i],
                    lb=self.T_h_out[i],
                    name=f"T_H{i}_at_B{k}",
                )
                if k > 0
                else self.m.Param(value=self.T_h_in[i], name=f"T_H{i}_at_B{k}")
                for k in range(self.K)
            ]
            for i in range(self.I)
        ]
        self.T_c = [
            [
                self.m.Var(
                    value=self.T_c_in[j],
                    ub=self.T_c_out[j],
                    lb=self.T_c_in[j],
                    name=f"T_C{j}_at_B{k}",
                )
                if k < self.S
                else self.m.Param(value=self.T_c_in[j], name=f"T_C{j}_at_B{k}")
                for k in range(self.K)
            ]
            for j in range(self.J)
        ]
        self.theta_1 = [
            [
                [
                    self.m.Var(
                        value=self.dTmin,
                        ub=abs(self.T_h_in[i] - self.T_c_in[j]),
                        lb=self.dTmin,
                        name=f"approach_T1_H{i}_to_C{j}_at_S{k}",
                    )
                    if self.z_allowed[i][j][k] > 0
                    else self.m.Param(
                        value=self.dTmin,
                        name=f"approach_T1_H{i}_to_C{j}_at_S{k}",
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
                    self.m.Var(
                        value=self.dTmin,
                        ub=abs(self.T_h_in[i] - self.T_c_in[j]),
                        lb=self.dTmin,
                        name=f"approach_T2_H{i}_to_C{j}_at_S{k}",
                    )
                    if self.z_allowed[i][j][k] > 0
                    else self.m.Param(
                        value=self.dTmin,
                        name=f"approach_T2_H{i}_to_C{j}_at_S{k}",
                    )
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]

        self.m.Equations(
            [
                self.Qtot_sh[i]
                - self.m.sum(
                    [self.Q_r[i][j][k] for k in range(self.S) for j in range(self.J)]
                )
                - self.Q_c[i]
                == 0.0
                for i in range(self.I)
            ]
        )
        self.m.Equations(
            [
                self.Qtot_sc[j]
                - self.m.sum(
                    [self.Q_r[i][j][k] for k in range(self.S) for i in range(self.I)]
                )
                - self.Q_h[j]
                == 0.0
                for j in range(self.J)
            ]
        )
        self.m.Equations(
            [
                (self.T_h[i][k + 1] - self.T_h[i][k]) * self.f_h[i]
                + sum(self.Q_r[i][j][k] for j in range(self.J))
                == 0
                for k in range(self.S)
                for i in range(self.I)
            ]
        )
        self.m.Equations(
            [
                (self.T_c[j][k + 1] - self.T_c[j][k]) * self.f_c[j]
                + sum(self.Q_r[i][j][k] for i in range(self.I))
                == 0
                for k in range(self.S)
                for j in range(self.J)
            ]
        )

        if self.integers:
            self.z = [
                [
                    [
                        self.m.Var(
                            value=1,
                            ub=1,
                            lb=0,
                            integer=True,
                            name=f"z_H{i}_to_C{j}_at_S{k}",
                        )
                        if self.z_allowed[i][j][k] > 0
                        else self.m.Param(value=0, name=f"z_H{i}_to_C{j}_at_S{k}")
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            self.z_cu = [
                self.m.Var(
                    value=1,
                    ub=1,
                    lb=0,
                    integer=True,
                    name=f"z_H{i}_to_CU",
                )
                if self.z_cu_allowed[i] > 0
                else self.m.Param(value=0, name=f"z_H{i}_to_CU")
                for i in range(self.I)
            ]
            self.z_hu = [
                self.m.Var(
                    value=1,
                    ub=1,
                    lb=0,
                    integer=True,
                    name=f"z_HU_to_C{j}",
                )
                if self.z_hu_allowed[j] > 0
                else self.m.Param(value=0, name=f"z_HU_to_C{j}")
                for j in range(self.J)
            ]
            _ = [
                self.m.Equation(self.Q_r[i][j][k] <= self.Q_max[i][j] * self.z[i][j][k])
                if self.z_allowed[i][j][k] > 0
                else None
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]
            _ = [
                self.m.Equation(self.Q_c[i] <= self.Qtot_sh[i] * self.z_cu[i])
                if self.z_cu_allowed[i] > 0
                else None
                for i in range(self.I)
            ]
            _ = [
                self.m.Equation(self.Q_h[j] <= self.Qtot_sc[j] * self.z_hu[j])
                if self.z_hu_allowed[j] > 0
                else None
                for j in range(self.J)
            ]
        else:
            self.z = [
                [
                    [
                        self.m.Param(value=1, name=f"z_H{i}_to_C{j}_at_S{k}")
                        if self.z_allowed[i][j][k] > 0
                        else self.m.Param(value=0, name=f"z_H{i}_to_C{j}_at_S{k}")
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            self.z_hu = [
                self.m.Param(value=1, name=f"z_HU_to_C{j}")
                if self.z_hu_allowed[j] > 0
                else self.m.Param(value=0, name=f"z_HU_to_C{j}")
                for j in range(self.J)
            ]
            self.z_cu = [
                self.m.Param(value=1, name=f"z_H{i}_to_CU")
                if self.z_cu_allowed[i] > 0
                else self.m.Param(value=0, name=f"z_H{i}_to_CU")
                for i in range(self.I)
            ]

        if self.non_isothermal_model:
            self.X = [
                [
                    [
                        self.m.Var(
                            value=0.5,
                            ub=1.0,
                            lb=0.0,
                            name=f"X_H{i}_to_C{j}_at_S{k}",
                        )
                        if self.z_allowed[i][j][k] > 0
                        else self.m.Param(value=0.0, name=f"X_H{i}_to_C{j}_at_S{k}")
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            self.Y = [
                [
                    [
                        self.m.Var(
                            value=0.5,
                            ub=1.0,
                            lb=0.0,
                            name=f"Y_C{j}_to_H{i}_at_S{k}",
                        )
                        if self.z_allowed[i][j][k] > 0
                        else self.m.Param(value=0.0, name=f"Y_C{j}_to_H{i}_at_S{k}")
                        for k in range(self.S)
                    ]
                    for i in range(self.I)
                ]
                for j in range(self.J)
            ]
            self.T_h_out_x = [
                [
                    [
                        self.m.Var(
                            value=self.T_h_in[i],
                            ub=self.T_h_in[i],
                            lb=self.T_h_out[i],
                            name=f"Thout_x_H{i}_to_C{j}_at_S{k}",
                        )
                        if self.z_allowed[i][j][k] > 0
                        else self.m.Param(value=0, name=f"Tx_H{i}_to_C{j}_at_S{k}")
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            self.T_c_out_y = [
                [
                    [
                        self.m.Var(
                            value=self.T_c_in[j],
                            ub=self.T_c_out[j],
                            lb=self.T_c_in[j],
                            name=f"Tcout_y_C{j}_to_H{i}_at_S{k}",
                        )
                        if self.z_allowed[i][j][k] > 0
                        else self.m.Param(value=0, name=f"Ty_C{j}_to_H{i}_at_S{k}")
                        for k in range(self.S)
                    ]
                    for i in range(self.I)
                ]
                for j in range(self.J)
            ]
            _ = [
                self.m.Equation(
                    self.theta_1[i][j][k]
                    <= (self.T_h[i][k] - self.T_c_out_y[j][i][k])
                    + self.M_ij[i][j] * (1 - self.z[i][j][k])
                )
                if self.z_allowed[i][j][k] > 0
                else None
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]
            _ = [
                self.m.Equation(
                    self.theta_2[i][j][k]
                    <= (self.T_h_out_x[i][j][k] - self.T_c[j][k + 1])
                    + self.M_ij[i][j] * (1 - self.z[i][j][k])
                )
                if self.z_allowed[i][j][k] > 0
                else None
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]
            _ = [
                self.m.Equation(
                    self.Q_r[i][j][k]
                    - self.X[i][j][k]
                    * self.f_h[i]
                    * (self.T_h[i][k] - self.T_h_out_x[i][j][k])
                    == 0.0
                )
                if self.z_allowed[i][j][k] > 0
                else None
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]
            _ = [
                self.m.Equation(
                    self.Q_r[i][j][k]
                    - self.Y[j][i][k]
                    * self.f_c[j]
                    * (self.T_c_out_y[j][i][k] - self.T_c[j][k + 1])
                    == 0.0
                )
                if self.z_allowed[i][j][k] > 0
                else None
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]
            _ = [
                self.m.Equation(sum(self.X[i][j][k] for j in range(self.J)) == 1.0)
                if sum(self.z_allowed[i][j][k] for j in range(self.J)) > 0
                else None
                for k in range(self.S)
                for i in range(self.I)
            ]
            _ = [
                self.m.Equation(sum(self.Y[j][i][k] for i in range(self.I)) == 1.0)
                if sum(self.z_allowed[i][j][k] for i in range(self.I)) > 0
                else None
                for k in range(self.S)
                for j in range(self.J)
            ]
        else:
            _ = [
                self.m.Equation(
                    self.theta_1[i][j][k]
                    <= (self.T_h[i][k] - self.T_c[j][k])
                    + self.M_ij[i][j] * (1 - self.z[i][j][k])
                )
                if self.z_allowed[i][j][k] > 0
                else None
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]
            _ = [
                self.m.Equation(
                    self.theta_1[i][j][k]
                    >= (self.T_h[i][k] - self.T_c[j][k])
                    - self.M_ij[i][j] * (1 - self.z[i][j][k])
                )
                if self.z_allowed[i][j][k] > 0
                else None
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]
            _ = [
                self.m.Equation(
                    self.theta_2[i][j][k]
                    <= (self.T_h[i][k + 1] - self.T_c[j][k + 1])
                    + self.M_ij[i][j] * (1 - self.z[i][j][k])
                )
                if self.z_allowed[i][j][k] > 0
                else None
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]
            _ = [
                self.m.Equation(
                    self.theta_2[i][j][k]
                    >= (self.T_h[i][k + 1] - self.T_c[j][k + 1])
                    - self.M_ij[i][j] * (1 - self.z[i][j][k])
                )
                if self.z_allowed[i][j][k] > 0
                else None
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]

        self.dqda = []
        self.alpha = []

    def set_dqda_equations(self) -> None:
        """Apply the source TDM minimum dQ/dA restriction."""

        self.min_dqda_int = [
            [
                [
                    self.m.Intermediate(
                        self.min_dqda * (self.T_h[i][k] - self.T_c[j][k + 1])
                        - self.theta_1[i][j][k]
                        * self.theta_2[i][j][k]
                        * self.U_r[i][j],
                        name=f"dqda_calc_H{i}_to_C{j}_at_S{k}",
                    )
                    if self.z_allowed[i][j][k] > 0
                    else None
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.min_dQ_dA_eqn = [
            self.m.Equation(
                self.min_dqda_int[i][j][k]
                <= self.min_dqda * self.M_ij[i][j] * (1 - self.z[i][j][k])
            )
            if self.z_allowed[i][j][k] > 0
            else None
            for k in range(self.S)
            for j in range(self.J)
            for i in range(self.I)
        ]

    def set_initial_values_for_variables(
        self,
        init_solution,
        *,
        brackets: bool = False,
    ) -> None:
        """Warm-start this model from a solved parent model."""

        for k in range(self.S):
            for j in range(self.J):
                for i in range(self.I):
                    if self.z_allowed[i][j][k] > 0:
                        self._set_value(
                            self.Q_r[i][j][k],
                            init_solution.Q_r[i][j][k].VALUE[0],
                            brackets=brackets,
                        )
                        self._set_value(
                            self.z[i][j][k],
                            init_solution.z[i][j][k][0],
                            brackets=brackets,
                        )
                        self._set_value(
                            self.theta_1[i][j][k],
                            init_solution.theta_1[i][j][k].VALUE[0],
                            brackets=brackets,
                        )
                        self._set_value(
                            self.theta_2[i][j][k],
                            init_solution.theta_2[i][j][k].VALUE[0],
                            brackets=brackets,
                        )
                    else:
                        self._set_value(self.Q_r[i][j][k], 0.0, brackets=brackets)
                        self._set_value(self.z[i][j][k], 0, brackets=brackets)
                        self._set_value(
                            self.theta_1[i][j][k],
                            self.dTmin,
                            brackets=brackets,
                        )
                        self._set_value(
                            self.theta_2[i][j][k],
                            self.dTmin,
                            brackets=brackets,
                        )

        for i in range(self.I):
            for k in range(self.K):
                self._set_value(
                    self.T_h[i][k],
                    init_solution.T_h[i][k].VALUE[0],
                    brackets=brackets,
                )
        for j in range(self.J):
            for k in range(self.K):
                self._set_value(
                    self.T_c[j][k],
                    init_solution.T_c[j][k].VALUE[0],
                    brackets=brackets,
                )

        for i in range(self.I):
            if self.z_cu_allowed[i] > 0:
                self._set_value(
                    self.Q_c[i],
                    init_solution.Q_c[i].VALUE[0],
                    brackets=brackets,
                )
                self._set_value(
                    self.z_cu[i],
                    init_solution.z_cu[i][0],
                    brackets=brackets,
                )
            else:
                self._set_value(self.Q_c[i], 0.0, brackets=brackets)
                self._set_value(self.z_cu[i], 0, brackets=brackets)

        for j in range(self.J):
            if self.z_hu_allowed[j] > 0:
                self._set_value(
                    self.Q_h[j],
                    init_solution.Q_h[j].VALUE[0],
                    brackets=brackets,
                )
                self._set_value(
                    self.z_hu[j],
                    init_solution.z_hu[j][0],
                    brackets=brackets,
                )
            else:
                self._set_value(self.Q_h[j], 0.0, brackets=brackets)
                self._set_value(self.z_hu[j], 0, brackets=brackets)

        if self.non_isothermal_model:
            if init_solution.non_isothermal_model:
                for k in range(self.S):
                    for j in range(self.J):
                        for i in range(self.I):
                            if self.z_allowed[i][j][k] > 0:
                                self._set_value(
                                    self.X[i][j][k],
                                    init_solution.X[i][j][k].VALUE[0],
                                    brackets=brackets,
                                )
                                self._set_value(
                                    self.Y[j][i][k],
                                    init_solution.Y[j][i][k].VALUE[0],
                                    brackets=brackets,
                                )
                                self._set_value(
                                    self.T_h_out_x[i][j][k],
                                    init_solution.T_h_out_x[i][j][k].VALUE[0],
                                    brackets=brackets,
                                )
                                self._set_value(
                                    self.T_c_out_y[j][i][k],
                                    init_solution.T_c_out_y[j][i][k].VALUE[0],
                                    brackets=brackets,
                                )
                            else:
                                self._set_value(self.X[i][j][k], 0.0, brackets=brackets)
                                self._set_value(self.Y[j][i][k], 0.0, brackets=brackets)
                                self._set_value(
                                    self.T_h_out_x[i][j][k],
                                    init_solution.T_h[i][k + 1].VALUE[0],
                                    brackets=brackets,
                                )
                                self._set_value(
                                    self.T_c_out_y[j][i][k],
                                    init_solution.T_c[j][k].VALUE[0],
                                    brackets=brackets,
                                )
            else:
                for k in range(self.S):
                    for j in range(self.J):
                        sum_Q_r_j = sum(
                            init_solution.Q_r[i][j][k].VALUE[0] for i in range(self.I)
                        )
                    for i in range(self.I):
                        sum_Q_r_i = sum(
                            init_solution.Q_r[i][j][k].VALUE[0] for j in range(self.J)
                        )
                        if self.z_allowed[i][j][k] > 0:
                            q_val = init_solution.Q_r[i][j][k].VALUE[0]
                            if q_val > 0.0:
                                self._set_value(
                                    self.X[i][j][k],
                                    q_val / sum_Q_r_j if sum_Q_r_j else 0.0,
                                    brackets=brackets,
                                )
                                self._set_value(
                                    self.Y[j][i][k],
                                    q_val / sum_Q_r_i if sum_Q_r_i else 0.0,
                                    brackets=brackets,
                                )
                            else:
                                self._set_value(self.X[i][j][k], 0.0, brackets=brackets)
                                self._set_value(self.Y[j][i][k], 0.0, brackets=brackets)
                            self._set_value(
                                self.T_h_out_x[i][j][k],
                                init_solution.T_h[i][k + 1].VALUE[0],
                                brackets=brackets,
                            )
                            self._set_value(
                                self.T_c_out_y[j][i][k],
                                init_solution.T_c[j][k].VALUE[0],
                                brackets=brackets,
                            )
                        else:
                            self._set_value(self.X[i][j][k], 0.0, brackets=brackets)
                            self._set_value(self.Y[j][i][k], 0.0, brackets=brackets)
                            self._set_value(
                                self.T_h_out_x[i][j][k],
                                init_solution.T_h[i][k + 1].VALUE[0],
                                brackets=brackets,
                            )
                            self._set_value(
                                self.T_c_out_y[j][i][k],
                                init_solution.T_c[j][k].VALUE[0],
                                brackets=brackets,
                            )

    def set_obj(self) -> None:
        """Attach source StageWise objective expressions unchanged."""

        if self.minimisation_goal == "hot utility":
            self.m.Minimize(self.m.sum([self.Q_h[j] for j in range(self.J)]))
        if self.minimisation_goal == "cold utility":
            self.m.Minimize(self.m.sum([self.Q_c[i] for i in range(self.I)]))
        elif self.minimisation_goal == "total utility":
            self.m.Minimize(
                self.m.sum([self.Q_h[j] for j in range(self.J)])
                + self.m.sum([self.Q_c[i] for i in range(self.I)])
            )
        elif self.minimisation_goal == "utility costs":
            self.m.Minimize(
                self.hu_cost[0] * self.m.sum([self.Q_h[j] for j in range(self.J)])
                + self.cu_cost[0] * self.m.sum([self.Q_c[i] for i in range(self.I)])
            )
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
        elif self.minimisation_goal == "dQ/dA obj":
            self.m.Minimize(sum(self.Q_h) - self.HU_target)
        elif self.minimisation_goal in {"total cost", "variable total cost"}:
            self._set_total_cost_objective()

    def _set_total_cost_objective(self) -> None:
        self.hu_cost_total = self.m.Intermediate(
            self.hu_cost[0] * sum(self.Q_h[j] for j in range(self.J)),
            name="Hot utility cost",
        )
        self.cu_cost_total = self.m.Intermediate(
            self.cu_cost[0] * sum(self.Q_c[i] for i in range(self.I)),
            name="Cold utility cost",
        )
        self.recovery_area_cost_filtered = [
            [0 for _j in range(self.J)] for _k in range(self.S)
        ]
        for k in range(self.S):
            for j in range(self.J):
                allowed_hots = [i for i in range(self.I) if self.z_allowed[i][j][k] > 0]
                if sum(self.z_allowed[z][j][k] for z in range(self.I)) > 0:
                    self.recovery_area_cost_filtered[k][j] = self.m.Intermediate(
                        self.A_coeff[0]
                        * sum(
                            (
                                self.Q_r[n][j][k]
                                / (
                                    self.U_r[n][j]
                                    * (
                                        self.theta_1[n][j][k]
                                        * self.theta_2[n][j][k]
                                        * (
                                            self.theta_1[n][j][k]
                                            + self.theta_2[n][j][k]
                                        )
                                        / 2
                                        + 1e-3
                                    )
                                    ** (1 / 3)
                                )
                                + 1e-3
                            )
                            ** self.A_exp[0]
                            for n in allowed_hots
                        ),
                        name=f"Recovery HX area cost in stage {k} cold {j}",
                    )

        self.hu_area_cost_total = self.m.Intermediate(
            self.hu_coeff[0]
            * sum(
                (
                    self.Q_h[j]
                    / (
                        self.U_hu[j]
                        * (
                            (self.T_hu_in[0] - self.T_c_out[j])
                            * (self.T_hu_out[0] - self.T_c[j][0])
                            * (
                                (self.T_hu_in[0] - self.T_c_out[j])
                                + (self.T_hu_out[0] - self.T_c[j][0])
                            )
                            / 2
                            + 1e-3
                        )
                        ** (1 / 3)
                    )
                    + 1e-3
                )
                ** self.hu_exp[0]
                for j in range(self.J)
            ),
            name="Total hot utility HX area cost",
        )
        self.cu_area_cost_total = self.m.Intermediate(
            self.cu_coeff[0]
            * sum(
                (
                    self.Q_c[i]
                    / (
                        self.U_cu[i]
                        * (
                            (self.T_h[i][self.S] - self.T_cu_out[0])
                            * (self.T_h_out[i] - self.T_cu_in[0])
                            * (
                                (self.T_h[i][self.S] - self.T_cu_out[0])
                                + (self.T_h_out[i] - self.T_cu_in[0])
                            )
                            / 2
                            + 1e-3
                        )
                        ** (1 / 3)
                    )
                    + 1e-3
                )
                ** self.cu_exp[0]
                for i in range(self.I)
            ),
            name="Total cold utility HX area cost",
        )

        if self.minimisation_goal == "total cost":
            self.utility_unit_cost_total = self.m.Intermediate(
                self.hu_unit_cost[0] * sum(self.z_hu[j] for j in range(self.J))
                + self.cu_unit_cost[0] * sum(self.z_cu[i] for i in range(self.I)),
                name="Total utility base cost",
            )
            self.recovery_unit_cost = [
                [0 for _j in range(self.J)] for _k in range(self.S)
            ]
            for k in range(self.S):
                for j in range(self.J):
                    self.recovery_unit_cost[k][j] = self.m.Intermediate(
                        self.unit_cost[0] * sum(self.z[i][j][k] for i in range(self.I)),
                        name=f"Total recovery base cost in stage {k} cold {j}",
                    )
            self.m.Minimize(
                self.hu_cost_total
                + self.cu_cost_total
                + self.utility_unit_cost_total
                + sum(
                    self.recovery_unit_cost[k][j]
                    for k in range(self.S)
                    for j in range(self.J)
                )
                + sum(
                    self.recovery_area_cost_filtered[k][j]
                    for k in range(self.S)
                    for j in range(self.J)
                )
                + self.hu_area_cost_total
                + self.cu_area_cost_total
            )
        elif self.minimisation_goal == "variable total cost":
            self.m.Minimize(
                self.hu_cost_total
                + self.cu_cost_total
                + sum(
                    self.recovery_area_cost_filtered[k][j]
                    for k in range(self.S)
                    for j in range(self.J)
                )
                + self.hu_area_cost_total
                + self.cu_area_cost_total
            )

    def get_post_process(self) -> None:
        """Extract source post-process arrays after a successful solve."""

        if self.mSuccess != 1:
            return

        self.z = [
            [
                [[1] if self.Q_r[i][j][k][0] > self.tol else [0] for k in range(self.S)]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.n_recovery_units = sum(
            self.z[i][j][k][0] if self.Q_r[i][j][k][0] > self.tol else 0
            for k in range(self.S)
            for j in range(self.J)
            for i in range(self.I)
        )
        self.z_hu = [[1] if self.Q_h[j][0] > self.tol else [0] for j in range(self.J)]
        self.n_hu_units = sum(
            self.z_hu[j][0] if self.Q_h[j][0] > self.tol else 0 for j in range(self.J)
        )
        self.z_cu = [[1] if self.Q_c[i][0] > self.tol else [0] for i in range(self.I)]
        self.n_cu_units = sum(
            self.z_cu[i][0] if self.Q_c[i][0] > self.tol else 0 for i in range(self.I)
        )
        self.n_units = self.n_recovery_units + self.n_hu_units + self.n_cu_units

        self.LMTD_r = [
            [
                [
                    self.z[i][j][k][0]
                    * (self.theta_1[i][j][k][0] - self.theta_2[i][j][k][0])
                    / math.log(self.theta_1[i][j][k][0] / self.theta_2[i][j][k][0])
                    if (
                        abs(self.theta_1[i][j][k][0] - self.theta_2[i][j][k][0])
                        > self.tol
                        and abs(self.theta_1[i][j][k][0] - self.dTmin) >= self.tol
                        and abs(self.theta_2[i][j][k][0] - self.dTmin) >= self.tol
                    )
                    else self.theta_1[i][j][k][0] * self.z[i][j][k][0]
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.area_r = [
            [
                [
                    self.Q_r[i][j][k][0] / self.U_r[i][j] / self.LMTD_r[i][j][k]
                    if self.LMTD_r[i][j][k] > self.tol
                    and self.Q_r[i][j][k][0] > self.tol
                    else 0.0
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.LMTD_hu = [
            self.z_hu[j][0]
            * (
                (self.T_hu_in[0] - self.T_c_out[j])
                - (self.T_hu_out[0] - self.T_c[j][0][0])
            )
            / math.log(
                (self.T_hu_in[0] - self.T_c_out[j])
                / (self.T_hu_out[0] - self.T_c[j][0][0])
            )
            if (
                abs(
                    (self.T_hu_in[0] - self.T_c_out[j])
                    - (self.T_hu_out[0] - self.T_c[j][0][0])
                )
                > self.tol
                and (self.T_hu_in[0] - self.T_c_out[j] - self.dTmin) >= self.tol
                and (self.T_hu_out[0] - self.T_c[j][0][0] - self.dTmin) >= self.tol
            )
            else (self.T_hu_in[0] - self.T_c_out[j]) * self.z_hu[j][0]
            for j in range(self.J)
        ]
        self.area_hu = [
            self.Q_h[j][0] / self.U_hu[j] / self.LMTD_hu[j]
            if self.LMTD_hu[j] > self.tol and self.Q_h[j][0] > self.tol
            else 0.0
            for j in range(self.J)
        ]
        self.LMTD_cu = [
            self.z_cu[i][0]
            * (
                (self.T_h[i][self.S][0] - self.T_cu_out[0])
                - (self.T_h_out[i] - self.T_cu_in[0])
            )
            / math.log(
                (self.T_h[i][self.S][0] - self.T_cu_out[0])
                / (self.T_h_out[i] - self.T_cu_in[0])
            )
            if (
                abs(
                    (self.T_h[i][self.S][0] - self.T_cu_out[0])
                    - (self.T_h_out[i] - self.T_cu_in[0])
                )
                > self.tol
                and (self.T_h[i][self.S][0] - self.T_cu_out[0] - self.dTmin) >= self.tol
                and (self.T_h_out[i] - self.T_cu_in[0] - self.dTmin) >= self.tol
            )
            else (self.T_h_out[i] - self.T_cu_in[0]) * self.z_cu[i][0]
            for i in range(self.I)
        ]
        self.area_cu = [
            self.Q_c[i][0] / self.U_cu[i] / self.LMTD_cu[i]
            if self.LMTD_cu[i] > self.tol and self.Q_c[i][0] > self.tol
            else 0.0
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
        self.alpha = self.get_alpha_values()
        self.dqda = [
            [[None for _k in range(self.S)] for _j in range(self.J)]
            for _i in range(self.I)
        ]
        self.dtacda = [
            [[None for _k in range(self.S)] for _j in range(self.J)]
            for _i in range(self.I)
        ]
        for k in range(self.S):
            for j in range(self.J):
                for i in range(self.I):
                    if (
                        self.Q_r[i][j][k][0] > 0
                        and (self.T_h[i][k][0] - self.T_c[j][k + 1][0]) > 0.0
                    ):
                        self.dqda[i][j][k] = (
                            self.theta_1[i][j][k][0]
                            * self.theta_2[i][j][k][0]
                            * self.U_r[i][j]
                        ) / (self.T_h[i][k][0] - self.T_c[j][k + 1][0])
                    elif (self.T_h[i][k][0] - self.T_c[j][k + 1][0]) > 0.0:
                        self.dqda[i][j][k] = self.U_r[i][j] * (
                            self.T_h[i][k][0] - self.T_c[j][k + 1][0]
                        )
                    else:
                        self.dqda[i][j][k] = 0

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
        self.alpha_dqda = [
            [
                [self.alpha[i][j][k][0] * self.dqda[i][j][k] for k in range(self.S)]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]

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


__all__ = ["StageWiseModel"]
