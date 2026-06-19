"""Migrated StageWise TDM/ESM equation construction.

This module is private to the problem-rooted synthesis service. The equations
mirror the source OpenHENS StageWise model while taking prepared OpenPinch
solver arrays instead of source CSV rows.
"""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from ..array_adapter import PreparedSolverArrays
from .base import BaseHeatExchangerNetworkModel

logger = logging.getLogger(__name__)


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
        solver_options: Mapping[str, Any] | Sequence[str] | None = None,
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
            solver_options=solver_options,
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

    def get_net_benefit_evolution(
        self,
        print_output: bool,
        max_depth: int = 5,
    ):
        """Evolve topology using the source add/remove net-benefit heuristic."""

        if self.mSuccess != 1:
            logger.warning("Initial model was not successful; skipping evolution.")
            return self

        model = self
        best_model = self

        for unit in range(max_depth):
            logger.debug(
                "Evolution step %s/%s: current TAC %s",
                unit + 1,
                max_depth,
                getattr(model, "TAC", None),
            )
            model_minus_one = self.get_n_minus_one_evolution(
                print_output=print_output,
                unit=unit,
                prev_case=model,
            )
            model_plus_one = self.get_n_plus_one_evolution(
                print_output=print_output,
                unit=unit,
                prev_case=model,
            )

            model = self._select_best_candidate(model, model_minus_one, model_plus_one)
            if model is None:
                logger.debug("No viable evolution model found.")
                break

            if model.TAC < best_model.TAC:
                best_model = model
                logger.debug(
                    "New best evolution model %s with TAC %.6f",
                    model.name,
                    best_model.TAC,
                )

        if best_model.mSuccess and best_model.TAC < self.TAC:
            self._update_with_best_model(best_model)
        else:
            logger.debug("No evolution improvement found over original model.")
        self.m.cleanup()
        return self

    def _select_best_candidate(
        self,
        current_model,
        model_minus_one,
        model_plus_one,
    ):
        """Select the source plus/minus evolution candidate for the next step."""

        del current_model
        if not model_minus_one.mSuccess and not model_plus_one.mSuccess:
            return None
        if model_minus_one.mSuccess and not model_plus_one.mSuccess:
            return model_minus_one
        if not model_minus_one.mSuccess and model_plus_one.mSuccess:
            return model_plus_one
        logger.debug(
            "TAC comparison -1: %.6f, +1: %.6f",
            model_minus_one.TAC,
            model_plus_one.TAC,
        )
        return min(
            [model_minus_one, model_plus_one],
            key=lambda candidate: candidate.TAC,
        )

    def _update_with_best_model(self, best_model) -> None:
        """Adopt the selected evolved topology while retaining this model object."""

        best_model.verify()
        self.alpha = [
            [
                [best_model.alpha[i][j][k] for k in range(self.S)]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.z_allowed = [
            [
                [best_model.z_allowed[i][j][k] for k in range(self.S)]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.set_initial_values_for_variables(best_model, brackets=True)

        self.hu_cost_total = copy.deepcopy(best_model.hu_cost_total)
        self.cu_cost_total = copy.deepcopy(best_model.cu_cost_total)
        self.recovery_area_cost_filtered = copy.deepcopy(
            best_model.recovery_area_cost_filtered
        )
        self.hu_area_cost_total = copy.deepcopy(best_model.hu_area_cost_total)
        self.cu_area_cost_total = copy.deepcopy(best_model.cu_area_cost_total)
        self.get_post_process()

    def get_n_minus_one_evolution(self, print_output: bool, unit: int, prev_case):
        """Build and solve the source minus-one topology evolution candidate."""

        low_pos = prev_case.get_lowest_benefit_HX()
        z_allowed_removed = copy.deepcopy(prev_case.z)
        for i, j, k in low_pos:
            logger.debug("worst selected position i,j,k %s", [i, j, k])
            if isinstance(z_allowed_removed[0][0][0], int):
                z_allowed_removed[i][j][k] = 0
            else:
                z_allowed_removed[i][j][k][0] = 0

        logger.debug(
            "number in z_allowed_removed %s",
            _count_allowed_matches(z_allowed_removed),
        )
        model_minus_one = StageWiseModel(
            name=f"{self.name}-n_minus 1 evolution model {unit}",
            framework=prev_case.framework,
            solver="ipopt-pyomo",
            solver_arrays=prev_case.solver_arrays,
            stages=prev_case.stages,
            dTmin=prev_case.dTmin,
            z_restriction=[z_allowed_removed, None, None],
            min_dqda=prev_case.min_dqda,
            minimisation_goal=prev_case.minimisation_goal,
            non_isothermal_model=prev_case.non_isothermal_model,
            integers=False,
            tol=1e-3,
            solver_options=self.solver_options,
        )

        for i, j, k in low_pos:
            model_minus_one.Q_r[i][j][k].VALUE.value = 0.0
            model_minus_one.z[i][j][k].VALUE.value = 0
            model_minus_one.theta_1[i][j][k].VALUE.value = self.dTmin
            model_minus_one.theta_2[i][j][k].VALUE.value = self.dTmin

        model_minus_one.optimise(print_output=print_output)
        return model_minus_one

    def get_n_plus_one_evolution(self, print_output: bool, unit: int, prev_case):
        """Build and solve the source plus-one topology evolution candidate."""

        high_pos = prev_case.get_max_benefit_HX()
        z_allowed_added = copy.deepcopy(prev_case.z)
        for i, j, k in high_pos:
            logger.debug("best non-selected position i,j,k %s", [i, j, k])
            if isinstance(z_allowed_added[0][0][0], int):
                z_allowed_added[i][j][k] = 1
            else:
                z_allowed_added[i][j][k][0] = 1

        logger.debug(
            "number in z_allowed_added %s",
            _count_allowed_matches(z_allowed_added),
        )
        model_plus_one = StageWiseModel(
            name=f"{self.name}-n_plus 1 evolution model {unit}",
            framework=prev_case.framework,
            solver="ipopt-pyomo",
            solver_arrays=prev_case.solver_arrays,
            stages=prev_case.stages,
            dTmin=prev_case.dTmin,
            z_restriction=[z_allowed_added, None, None],
            min_dqda=prev_case.min_dqda,
            minimisation_goal=prev_case.minimisation_goal,
            non_isothermal_model=prev_case.non_isothermal_model,
            integers=False,
            tol=1e-3,
            solver_options=self.solver_options,
        )

        for i, j, k in high_pos:
            model_plus_one.z[i][j][k].VALUE.value = 1

        model_plus_one.optimise(print_output=print_output)
        return model_plus_one

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
                    self._post_process_lmtd(
                        self.theta_1[i][j][k][0],
                        self.theta_2[i][j][k][0],
                        self.z[i][j][k][0],
                        formula_allowed=(
                            abs(
                                self.theta_1[i][j][k][0]
                                - self.theta_2[i][j][k][0]
                            )
                            > self.tol
                            and abs(self.theta_1[i][j][k][0] - self.dTmin)
                            >= self.tol
                            and abs(self.theta_2[i][j][k][0] - self.dTmin)
                            >= self.tol
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
                    and (self.T_hu_in[0] - self.T_c_out[j] - self.dTmin)
                    >= self.tol
                    and (self.T_hu_out[0] - self.T_c[j][0][0] - self.dTmin)
                    >= self.tol
                ),
            )
            for j in range(self.J)
        ]
        self.area_hu = [
            self.Q_h[j][0] / self.U_hu[j] / self.LMTD_hu[j]
            if self.LMTD_hu[j] > self.tol and self.Q_h[j][0] > self.tol
            else 0.0
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
                    and (self.T_h[i][self.S][0] - self.T_cu_out[0] - self.dTmin)
                    >= self.tol
                    and (self.T_h_out[i] - self.T_cu_in[0] - self.dTmin)
                    >= self.tol
                ),
                fallback_delta=self.T_h_out[i] - self.T_cu_in[0],
            )
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

    def get_lowest_benefit_HX(self) -> list[list[int]]:
        """Return the active exchanger with the lowest source net benefit."""

        self.net_benefit = np.array(
            [
                [[0.0 for _k in range(self.S)] for _j in range(self.J)]
                for _i in range(self.I)
            ]
        )
        smallest_net_benefit = float("inf")
        low_pos: list[list[int]] = []
        for k in range(self.S):
            for j in range(self.J):
                for i in range(self.I):
                    if self.z[i][j][k][0] == 1:
                        self.net_benefit[i][j][k] = (
                            self.Q_r[i][j][k][0]
                            * self.alpha[i][j][k][0]
                            * (self.hu_cost[0] + self.cu_cost[0])
                            - (
                                self.unit_cost[0]
                                + self.A_coeff[0]
                                * (self.area_r[i][j][k] ** self.A_exp[0])
                            )
                        )
                        if self.net_benefit[i][j][k] < smallest_net_benefit:
                            smallest_net_benefit = self.net_benefit[i][j][k]
                            low_pos = [[i, j, k]]
        return low_pos

    def get_max_benefit_HX(self) -> list[list[int]]:
        """Return the inactive feasible exchanger with the highest alpha-dQ/dA."""

        self.net_benefit = np.array(
            [
                [[0.0 for _k in range(self.S)] for _j in range(self.J)]
                for _i in range(self.I)
            ]
        )
        highest_net_benefit = 0.0
        high_pos: list[list[int]] = []
        for k in range(self.S):
            for j in range(self.J):
                for i in range(self.I):
                    if (
                        self.alpha_dqda[i][j][k] > highest_net_benefit
                        and self.z_feasible[i][j][k]
                    ):
                        highest_net_benefit = self.alpha_dqda[i][j][k]
                        high_pos = [[i, j, k]]
        return high_pos

    def verify(self) -> tuple[bool, list[str]]:
        """Run the source solution checks used by topology evolution."""

        failures = []
        if not _check_temperatures(self):
            failures.append("temperature")
        if self.minimisation_goal in {"total cost", "variable total cost"}:
            if not _check_utility_costs(self):
                failures.append("cost")
            if not _check_area_costs(self):
                failures.append("area")
        return len(failures) == 0, failures


__all__ = ["StageWiseModel"]


def _count_allowed_matches(values) -> int:
    count = 0
    for layer in values:
        for row in layer:
            for element in row:
                if isinstance(element, int):
                    count += 1 if element == 1 else 0
                else:
                    count += 1 if element[0] == 1 else 0
    return count


def _check_temperatures(
    case,
    *,
    rel_tol: float = 1e-3,
    abs_tol: float = 1e-2,
    q_tol: float = 1e-2,
) -> bool:
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


def _check_utility_costs(
    case,
    *,
    rel_tol: float = 0.1,
    abs_tol: float = 1.0,
) -> bool:
    post_hot_utility = case.hu_cost[0] * sum(case.Q_h[j][0] for j in range(case.J))
    post_cold_utility = case.cu_cost[0] * sum(case.Q_c[i][0] for i in range(case.I))
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
    for k in range(case.S):
        for j in range(case.J):
            allowed_hots = [
                i for i in range(case.I) if case.z_allowed[i][j][k] > 0
            ]
            if not allowed_hots:
                continue

            total_duty = sum(case.Q_r[i][j][k][0] for i in allowed_hots)
            if abs(total_duty) <= q_tol:
                continue

            post_area_chen = (
                sum(
                    (
                        case.Q_r[n][j][k][0]
                        / (
                            case.U_r[n][j]
                            * (
                                (
                                    case.theta_1[n][j][k][0]
                                    * case.theta_2[n][j][k][0]
                                    * (
                                        case.theta_1[n][j][k][0]
                                        + case.theta_2[n][j][k][0]
                                    )
                                    / 2
                                    + 1e-3
                                )
                                ** (1 / 3)
                            )
                        )
                    )
                    ** case.A_exp[0]
                    for n in allowed_hots
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


def _value(value) -> float:
    if isinstance(value, int | float):
        return float(value)
    return float(value.value[0])
