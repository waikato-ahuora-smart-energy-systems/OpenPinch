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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..common.solver.arrays import PreparedSolverArrays
from .base import BaseHeatExchangerNetworkModel

logger = logging.getLogger(__name__)


@dataclass
class _EvolutionCandidateSpec:
    kind: Literal["minus", "plus"]
    unit: int
    branch_index: int
    rank: int
    prev_case: Any
    position: tuple[int, int, int]
    z_allowed: list
    signature: tuple[tuple[int, int, int], ...]


@dataclass
class _EvolutionBranchState:
    model: Any
    best_tac: float
    stale_depths: int = 0


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
                    max(
                        self.T_h_in[i]
                        - self.T_c_in[j]
                        - self._recovery_approach_temperature(i, j),
                        0.0,
                    )
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
                + self._recovery_approach_temperature(i, j)
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
                    if k > 0
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
                    if k < self.S
                    else self.m.Param(value=self.T_c_in[j], name=f"T_C{j}_at_B{k}")
                )
                for k in range(self.K)
            ]
            for j in range(self.J)
        ]
        self.theta_1 = [
            [
                [
                    (
                        self.m.Var(
                            value=self._recovery_approach_temperature(i, j),
                            ub=abs(self.T_h_in[i] - self.T_c_in[j]),
                            lb=self._recovery_approach_temperature(i, j),
                            name=f"approach_T1_H{i}_to_C{j}_at_S{k}",
                        )
                        if self.z_allowed[i][j][k] > 0
                        else self.m.Param(
                            value=self._recovery_approach_temperature(i, j),
                            name=f"approach_T1_H{i}_to_C{j}_at_S{k}",
                        )
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
                        self.m.Var(
                            value=self._recovery_approach_temperature(i, j),
                            ub=abs(self.T_h_in[i] - self.T_c_in[j]),
                            lb=self._recovery_approach_temperature(i, j),
                            name=f"approach_T2_H{i}_to_C{j}_at_S{k}",
                        )
                        if self.z_allowed[i][j][k] > 0
                        else self.m.Param(
                            value=self._recovery_approach_temperature(i, j),
                            name=f"approach_T2_H{i}_to_C{j}_at_S{k}",
                        )
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
                    self.m.Equation(
                        self.Q_r[i][j][k] <= self.Q_max[i][j] * self.z[i][j][k]
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
                    self.m.Equation(self.Q_c[i] <= self.Qtot_sh[i] * self.z_cu[i])
                    if self.z_cu_allowed[i] > 0
                    else None
                )
                for i in range(self.I)
            ]
            _ = [
                (
                    self.m.Equation(self.Q_h[j] <= self.Qtot_sc[j] * self.z_hu[j])
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

        if self.non_isothermal_model:
            self.X = [
                [
                    [
                        (
                            self.m.Var(
                                value=0.5,
                                ub=1.0,
                                lb=0.0,
                                name=f"X_H{i}_to_C{j}_at_S{k}",
                            )
                            if self.z_allowed[i][j][k] > 0
                            else self.m.Param(value=0.0, name=f"X_H{i}_to_C{j}_at_S{k}")
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            self.Y = [
                [
                    [
                        (
                            self.m.Var(
                                value=0.5,
                                ub=1.0,
                                lb=0.0,
                                name=f"Y_C{j}_to_H{i}_at_S{k}",
                            )
                            if self.z_allowed[i][j][k] > 0
                            else self.m.Param(value=0.0, name=f"Y_C{j}_to_H{i}_at_S{k}")
                        )
                        for k in range(self.S)
                    ]
                    for i in range(self.I)
                ]
                for j in range(self.J)
            ]
            self.T_h_out_x = [
                [
                    [
                        (
                            self.m.Var(
                                value=self.T_h_in[i],
                                ub=self.T_h_in[i],
                                lb=self.T_h_out[i],
                                name=f"Thout_x_H{i}_to_C{j}_at_S{k}",
                            )
                            if self.z_allowed[i][j][k] > 0
                            else self.m.Param(value=0, name=f"Tx_H{i}_to_C{j}_at_S{k}")
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            self.T_c_out_y = [
                [
                    [
                        (
                            self.m.Var(
                                value=self.T_c_in[j],
                                ub=self.T_c_out[j],
                                lb=self.T_c_in[j],
                                name=f"Tcout_y_C{j}_to_H{i}_at_S{k}",
                            )
                            if self.z_allowed[i][j][k] > 0
                            else self.m.Param(value=0, name=f"Ty_C{j}_to_H{i}_at_S{k}")
                        )
                        for k in range(self.S)
                    ]
                    for i in range(self.I)
                ]
                for j in range(self.J)
            ]
            _ = [
                (
                    self.m.Equation(
                        self.theta_1[i][j][k]
                        <= (self.T_h[i][k] - self.T_c_out_y[j][i][k])
                        + self.M_ij[i][j] * (1 - self.z[i][j][k])
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
                        self.theta_2[i][j][k]
                        <= (self.T_h_out_x[i][j][k] - self.T_c[j][k + 1])
                        + self.M_ij[i][j] * (1 - self.z[i][j][k])
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
                        self.Q_r[i][j][k]
                        - self.X[i][j][k]
                        * self.f_h[i]
                        * (self.T_h[i][k] - self.T_h_out_x[i][j][k])
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
                        self.Q_r[i][j][k]
                        - self.Y[j][i][k]
                        * self.f_c[j]
                        * (self.T_c_out_y[j][i][k] - self.T_c[j][k + 1])
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
                        self.m.sum([self.X[i][j][k] for j in range(self.J)]) == 1.0
                    )
                    if sum(self.z_allowed[i][j][k] for j in range(self.J)) > 0
                    else None
                )
                for k in range(self.S)
                for i in range(self.I)
            ]
            _ = [
                (
                    self.m.Equation(
                        self.m.sum([self.Y[j][i][k] for i in range(self.I)]) == 1.0
                    )
                    if sum(self.z_allowed[i][j][k] for i in range(self.I)) > 0
                    else None
                )
                for k in range(self.S)
                for j in range(self.J)
            ]
        else:
            _ = [
                (
                    self.m.Equation(
                        self.theta_1[i][j][k]
                        <= (self.T_h[i][k] - self.T_c[j][k])
                        + self.M_ij[i][j] * (1 - self.z[i][j][k])
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
                        self.theta_1[i][j][k]
                        >= (self.T_h[i][k] - self.T_c[j][k])
                        - self.M_ij[i][j] * (1 - self.z[i][j][k])
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
                        self.theta_2[i][j][k]
                        <= (self.T_h[i][k + 1] - self.T_c[j][k + 1])
                        + self.M_ij[i][j] * (1 - self.z[i][j][k])
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
                        self.theta_2[i][j][k]
                        >= (self.T_h[i][k + 1] - self.T_c[j][k + 1])
                        - self.M_ij[i][j] * (1 - self.z[i][j][k])
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

    def set_dqda_equations(self) -> None:
        """Apply the source TDM minimum dQ/dA restriction."""

        self.min_dqda_int = [
            [
                [
                    (
                        self.m.Intermediate(
                            self.min_dqda * (self.T_h[i][k] - self.T_c[j][k + 1])
                            - self.theta_1[i][j][k]
                            * self.theta_2[i][j][k]
                            * self.U_r[i][j],
                            name=f"dqda_calc_H{i}_to_C{j}_at_S{k}",
                        )
                        if self.z_allowed[i][j][k] > 0
                        else None
                    )
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.min_dQ_dA_eqn = [
            (
                self.m.Equation(
                    self.min_dqda_int[i][j][k]
                    <= self.min_dqda * self.M_ij[i][j] * (1 - self.z[i][j][k])
                )
                if self.z_allowed[i][j][k] > 0
                else None
            )
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
                            self._recovery_approach_temperature(i, j),
                            brackets=brackets,
                        )
                        self._set_value(
                            self.theta_2[i][j][k],
                            self._recovery_approach_temperature(i, j),
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
                    for i in range(self.I):
                        sum_Q_r_i = sum(
                            init_solution.Q_r[i][j][k].VALUE[0] for j in range(self.J)
                        )
                        allowed_js = [
                            j for j in range(self.J) if self.z_allowed[i][j][k] > 0
                        ]
                        for j in range(self.J):
                            if self.z_allowed[i][j][k] <= 0:
                                self._set_value(self.X[i][j][k], 0.0, brackets=brackets)
                                continue
                            q_val = init_solution.Q_r[i][j][k].VALUE[0]
                            if sum_Q_r_i > self.tol:
                                value = q_val / sum_Q_r_i
                            else:
                                value = 1.0 / len(allowed_js) if allowed_js else 0.0
                            self._set_value(
                                self.X[i][j][k],
                                value,
                                brackets=brackets,
                            )
                    for j in range(self.J):
                        sum_Q_r_j = sum(
                            init_solution.Q_r[i][j][k].VALUE[0] for i in range(self.I)
                        )
                        allowed_is = [
                            i for i in range(self.I) if self.z_allowed[i][j][k] > 0
                        ]
                        for i in range(self.I):
                            if self.z_allowed[i][j][k] <= 0:
                                self._set_value(self.Y[j][i][k], 0.0, brackets=brackets)
                                continue
                            q_val = init_solution.Q_r[i][j][k].VALUE[0]
                            if sum_Q_r_j > self.tol:
                                value = q_val / sum_Q_r_j
                            else:
                                value = 1.0 / len(allowed_is) if allowed_is else 0.0
                            self._set_value(
                                self.Y[j][i][k],
                                value,
                                brackets=brackets,
                            )
                    for i in range(self.I):
                        for j in range(self.J):
                            if self.z_allowed[i][j][k] > 0:
                                q_val = init_solution.Q_r[i][j][k].VALUE[0]
                                self._set_value(
                                    self.T_h_out_x[i][j][k],
                                    (
                                        init_solution.T_h[i][k + 1].VALUE[0]
                                        if q_val > self.tol
                                        else init_solution.T_h[i][k].VALUE[0]
                                    ),
                                    brackets=brackets,
                                )
                                self._set_value(
                                    self.T_c_out_y[j][i][k],
                                    (
                                        init_solution.T_c[j][k].VALUE[0]
                                        if q_val > self.tol
                                        else init_solution.T_c[j][k + 1].VALUE[0]
                                    ),
                                    brackets=brackets,
                                )
                            else:
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
        n_ad_branches: int = 1,
        n_rm_branches: int = 1,
        max_parallel: int = 1,
        no_improvement_patience: int | None = None,
    ):
        """Evolve topology using branched add/remove net-benefit heuristics."""

        if self.mSuccess != 1:
            logger.warning("Initial model was not successful; skipping evolution.")
            return self

        n_ad_branches = max(1, int(n_ad_branches))
        n_rm_branches = max(1, int(n_rm_branches))
        max_parallel = max(1, int(max_parallel))
        if no_improvement_patience is not None:
            no_improvement_patience = max(1, int(no_improvement_patience))
        frontier = [_EvolutionBranchState(model=self, best_tac=float(self.TAC))]
        best_model = self

        for unit in range(max_depth):
            logger.debug(
                "Evolution step %s/%s: active branches %s, best TAC %s",
                unit + 1,
                max_depth,
                len(frontier),
                getattr(best_model, "TAC", None),
            )
            specs = self._evolution_candidate_specs(
                frontier,
                unit=unit,
                n_ad_branches=n_ad_branches,
                n_rm_branches=n_rm_branches,
            )
            if not specs:
                logger.debug("No evolution candidate topologies found.")
                break

            solved_candidates = self._solve_evolution_candidates(
                specs,
                print_output=print_output,
                max_parallel=max_parallel,
            )
            next_frontier: list[_EvolutionBranchState] = []
            for spec, candidate in solved_candidates:
                if not _is_usable_evolution_candidate(candidate):
                    continue
                parent_state = frontier[spec.branch_index]
                candidate_tac = float(candidate.TAC)
                if candidate_tac < parent_state.best_tac:
                    branch_state = _EvolutionBranchState(
                        model=candidate,
                        best_tac=candidate_tac,
                        stale_depths=0,
                    )
                else:
                    branch_state = _EvolutionBranchState(
                        model=candidate,
                        best_tac=parent_state.best_tac,
                        stale_depths=parent_state.stale_depths + 1,
                    )
                if (
                    no_improvement_patience is not None
                    and branch_state.stale_depths >= no_improvement_patience
                ):
                    logger.debug(
                        "Pruning EVM branch %s after %s non-improving steps.",
                        spec.branch_index,
                        branch_state.stale_depths,
                    )
                    continue
                next_frontier.append(branch_state)
                if candidate.TAC < best_model.TAC:
                    best_model = candidate
            frontier = next_frontier
            if not frontier:
                logger.debug("No viable evolution model found.")
                break

            if best_model is not self:
                logger.debug(
                    "New best evolution model %s with TAC %.6f",
                    best_model.name,
                    best_model.TAC,
                )

        if best_model.mSuccess and best_model.TAC < self.TAC:
            self._update_with_best_model(best_model)
        else:
            logger.debug("No evolution improvement found over original model.")
        self.m.cleanup()
        return self

    def _evolution_candidate_specs(
        self,
        frontier: Sequence[_EvolutionBranchState],
        *,
        unit: int,
        n_ad_branches: int,
        n_rm_branches: int,
    ) -> list[_EvolutionCandidateSpec]:
        specs: list[_EvolutionCandidateSpec] = []
        seen_signatures: set[tuple[tuple[int, int, int], ...]] = set()
        for branch_index, branch_state in enumerate(frontier):
            prev_case = branch_state.model
            for rank, position in enumerate(
                prev_case.get_lowest_benefit_HX_candidates(n_rm_branches),
                start=1,
            ):
                spec = self._evolution_candidate_spec(
                    kind="minus",
                    unit=unit,
                    branch_index=branch_index,
                    rank=rank,
                    prev_case=prev_case,
                    position=position,
                    z_value=0,
                    seen_signatures=seen_signatures,
                )
                if spec is not None:
                    specs.append(spec)
            for rank, position in enumerate(
                prev_case.get_max_benefit_HX_candidates(n_ad_branches),
                start=1,
            ):
                spec = self._evolution_candidate_spec(
                    kind="plus",
                    unit=unit,
                    branch_index=branch_index,
                    rank=rank,
                    prev_case=prev_case,
                    position=position,
                    z_value=1,
                    seen_signatures=seen_signatures,
                )
                if spec is not None:
                    specs.append(spec)
        return specs

    def _evolution_candidate_spec(
        self,
        *,
        kind: Literal["minus", "plus"],
        unit: int,
        branch_index: int,
        rank: int,
        prev_case,
        position: Sequence[int],
        z_value: int,
        seen_signatures: set[tuple[tuple[int, int, int], ...]],
    ) -> _EvolutionCandidateSpec | None:
        if len(position) != 3:
            return None
        candidate_position = tuple(int(index) for index in position)
        z_allowed = self._z_allowed_with_candidate(
            prev_case,
            position=candidate_position,
            value=z_value,
        )
        signature = self._topology_signature_from_z(z_allowed)
        if signature in seen_signatures:
            logger.debug(
                "Skipping duplicate EVM topology at depth %s from branch %s",
                unit,
                branch_index,
            )
            return None
        seen_signatures.add(signature)
        return _EvolutionCandidateSpec(
            kind=kind,
            unit=unit,
            branch_index=branch_index,
            rank=rank,
            prev_case=prev_case,
            position=candidate_position,
            z_allowed=z_allowed,
            signature=signature,
        )

    def _solve_evolution_candidates(
        self,
        specs: Sequence[_EvolutionCandidateSpec],
        *,
        print_output: bool,
        max_parallel: int,
    ) -> list[tuple[_EvolutionCandidateSpec, Any]]:
        if max_parallel <= 1 or len(specs) <= 1:
            return [
                (spec, candidate)
                for spec in specs
                if (candidate := self._solve_evolution_candidate(spec, print_output))
                is not None
            ]

        candidates: list[tuple[_EvolutionCandidateSpec, Any]] = []
        with ThreadPoolExecutor(max_workers=min(max_parallel, len(specs))) as pool:
            futures = {
                pool.submit(self._solve_evolution_candidate, spec, print_output): spec
                for spec in specs
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    candidate = future.result()
                except Exception as exc:
                    logger.debug(
                        "Discarding failed EVM %s branch %s rank %s: %s",
                        spec.kind,
                        spec.branch_index,
                        spec.rank,
                        exc,
                    )
                    continue
                if candidate is not None:
                    candidates.append((spec, candidate))
        return candidates

    def _solve_evolution_candidate(
        self,
        spec: _EvolutionCandidateSpec,
        print_output: bool,
    ):
        try:
            if spec.kind == "minus":
                return self._build_and_solve_n_minus_one_evolution(
                    print_output=print_output,
                    unit=spec.unit,
                    prev_case=spec.prev_case,
                    position=spec.position,
                    z_allowed_removed=spec.z_allowed,
                    branch_label=self._evolution_branch_label(spec),
                )
            return self._build_and_solve_n_plus_one_evolution(
                print_output=print_output,
                unit=spec.unit,
                prev_case=spec.prev_case,
                position=spec.position,
                z_allowed_added=spec.z_allowed,
                branch_label=self._evolution_branch_label(spec),
            )
        except Exception as exc:
            logger.debug(
                "EVM %s branch %s rank %s failed: %s",
                spec.kind,
                spec.branch_index,
                spec.rank,
                exc,
            )
            return None

    def _evolution_branch_label(self, spec: _EvolutionCandidateSpec) -> str:
        return f"{spec.unit}-b{spec.branch_index}-{spec.kind}{spec.rank}"

    def _select_best_candidate(
        self,
        current_model,
        model_minus_one,
        model_plus_one,
    ):
        """Select the source plus/minus evolution candidate for the next step."""

        del current_model
        minus_usable = _is_usable_evolution_candidate(model_minus_one)
        plus_usable = _is_usable_evolution_candidate(model_plus_one)
        if not minus_usable and not plus_usable:
            return None
        if minus_usable and not plus_usable:
            return model_minus_one
        if not minus_usable and plus_usable:
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
            [[best_model.alpha[i][j][k] for k in range(self.S)] for j in range(self.J)]
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

        candidates = prev_case.get_lowest_benefit_HX_candidates(1)
        if not candidates:
            return None
        position = tuple(candidates[0])
        z_allowed_removed = self._z_allowed_with_candidate(
            prev_case,
            position=position,
            value=0,
        )
        return self._build_and_solve_n_minus_one_evolution(
            print_output=print_output,
            unit=unit,
            prev_case=prev_case,
            position=position,
            z_allowed_removed=z_allowed_removed,
        )

    def _build_and_solve_n_minus_one_evolution(
        self,
        *,
        print_output: bool,
        unit: int,
        prev_case,
        position: Sequence[int],
        z_allowed_removed: list,
        branch_label: str | None = None,
    ):
        """Build and solve one minus-one topology evolution candidate."""

        i, j, k = (int(index) for index in position)
        logger.debug("worst selected position i,j,k %s", [i, j, k])
        logger.debug(
            "number in z_allowed_removed %s",
            _count_allowed_matches(z_allowed_removed),
        )
        model_minus_one = StageWiseModel(
            name=(
                f"{self.name}-n_minus 1 evolution model "
                f"{branch_label if branch_label is not None else unit}"
            ),
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

        model_minus_one.Q_r[i][j][k].VALUE.value = 0.0
        model_minus_one.z[i][j][k].VALUE.value = 0
        approach = self._recovery_approach_temperature(i, j)
        model_minus_one.theta_1[i][j][k].VALUE.value = approach
        model_minus_one.theta_2[i][j][k].VALUE.value = approach

        model_minus_one.optimise(print_output=print_output)
        return model_minus_one

    def get_n_plus_one_evolution(self, print_output: bool, unit: int, prev_case):
        """Build and solve the source plus-one topology evolution candidate."""

        candidates = prev_case.get_max_benefit_HX_candidates(1)
        if not candidates:
            return None
        position = tuple(candidates[0])
        z_allowed_added = self._z_allowed_with_candidate(
            prev_case,
            position=position,
            value=1,
        )
        return self._build_and_solve_n_plus_one_evolution(
            print_output=print_output,
            unit=unit,
            prev_case=prev_case,
            position=position,
            z_allowed_added=z_allowed_added,
        )

    def _build_and_solve_n_plus_one_evolution(
        self,
        *,
        print_output: bool,
        unit: int,
        prev_case,
        position: Sequence[int],
        z_allowed_added: list,
        branch_label: str | None = None,
    ):
        """Build and solve one plus-one topology evolution candidate."""

        i, j, k = (int(index) for index in position)
        logger.debug("best non-selected position i,j,k %s", [i, j, k])
        logger.debug(
            "number in z_allowed_added %s",
            _count_allowed_matches(z_allowed_added),
        )
        model_plus_one = StageWiseModel(
            name=(
                f"{self.name}-n_plus 1 evolution model "
                f"{branch_label if branch_label is not None else unit}"
            ),
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

        model_plus_one.z[i][j][k].VALUE.value = 1

        model_plus_one.optimise(print_output=print_output)
        return model_plus_one

    def _z_allowed_with_candidate(
        self,
        prev_case,
        *,
        position: Sequence[int],
        value: int,
    ) -> list:
        z_allowed = copy.deepcopy(prev_case.z)
        i, j, k = (int(index) for index in position)
        self._set_recovery_binary_value(z_allowed, (i, j, k), value)
        return z_allowed

    def _set_recovery_binary_value(
        self,
        z_values: list,
        position: tuple[int, int, int],
        value: int,
    ) -> None:
        i, j, k = position
        element = z_values[i][j][k]
        if isinstance(element, (int, float)):
            z_values[i][j][k] = int(value)
            return
        try:
            element[0] = int(value)
            return
        except TypeError:
            pass
        except IndexError:
            pass
        if hasattr(element, "VALUE"):
            element.VALUE.value = int(value)
            return
        z_values[i][j][k] = int(value)

    def _topology_signature_from_z(
        self,
        z_values: Sequence[Sequence[Sequence[Any]]],
    ) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (i, j, k)
            for k in range(self.S)
            for j in range(self.J)
            for i in range(self.I)
            if self._active_binary_value(z_values[i][j][k]) > self.tol
        )

    def _active_binary_value(self, value) -> float:
        try:
            return float(value[0])
        except TypeError, IndexError:
            pass
        if hasattr(value, "VALUE"):
            return float(value.VALUE[0])
        return float(value)

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
            self.hu_cost[0] * self.m.sum([self.Q_h[j] for j in range(self.J)]),
            name="Hot utility cost",
        )
        self.cu_cost_total = self.m.Intermediate(
            self.cu_cost[0] * self.m.sum([self.Q_c[i] for i in range(self.I)]),
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
                            abs(self.theta_1[i][j][k][0] - self.theta_2[i][j][k][0])
                            > self.tol
                            and abs(
                                self.theta_1[i][j][k][0]
                                - self._recovery_approach_temperature(i, j)
                            )
                            >= self.tol
                            and abs(
                                self.theta_2[i][j][k][0]
                                - self._recovery_approach_temperature(i, j)
                            )
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
                    and (
                        self.T_hu_in[0]
                        - self.T_c_out[j]
                        - self._hot_utility_approach_temperature(j)
                    )
                    >= self.tol
                    and (
                        self.T_hu_out[0]
                        - self.T_c[j][0][0]
                        - self._hot_utility_approach_temperature(j)
                    )
                    >= self.tol
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
                    and (
                        self.T_h[i][self.S][0]
                        - self.T_cu_out[0]
                        - self._cold_utility_approach_temperature(i)
                    )
                    >= self.tol
                    and (
                        self.T_h_out[i]
                        - self.T_cu_in[0]
                        - self._cold_utility_approach_temperature(i)
                    )
                    >= self.tol
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

        return self.get_lowest_benefit_HX_candidates(1)

    def get_lowest_benefit_HX_candidates(self, limit: int) -> list[list[int]]:
        """Return active exchangers sorted by ascending source net benefit."""

        self.net_benefit = np.array(
            [
                [[0.0 for _k in range(self.S)] for _j in range(self.J)]
                for _i in range(self.I)
            ]
        )
        candidates: list[tuple[float, int, list[int]]] = []
        order = 0
        for k in range(self.S):
            for j in range(self.J):
                for i in range(self.I):
                    if self._active_binary_value(self.z[i][j][k]) > self.tol:
                        self.net_benefit[i][j][k] = self.Q_r[i][j][k][0] * self.alpha[
                            i
                        ][j][k][0] * (self.hu_cost[0] + self.cu_cost[0]) - (
                            self.unit_cost[0]
                            + self.A_coeff[0] * (self.area_r[i][j][k] ** self.A_exp[0])
                        )
                        candidates.append(
                            (float(self.net_benefit[i][j][k]), order, [i, j, k])
                        )
                    order += 1
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [position for _benefit, _order, position in candidates[: int(limit)]]

    def get_max_benefit_HX(self) -> list[list[int]]:
        """Return the inactive feasible exchanger with the highest alpha-dQ/dA."""

        return self.get_max_benefit_HX_candidates(1)

    def get_max_benefit_HX_candidates(self, limit: int) -> list[list[int]]:
        """Return inactive feasible exchangers sorted by descending alpha-dQ/dA."""

        self.net_benefit = np.array(
            [
                [[0.0 for _k in range(self.S)] for _j in range(self.J)]
                for _i in range(self.I)
            ]
        )
        candidates: list[tuple[float, int, list[int]]] = []
        order = 0
        for k in range(self.S):
            for j in range(self.J):
                for i in range(self.I):
                    if (
                        self._active_binary_value(self.z[i][j][k]) <= self.tol
                        and self.alpha_dqda[i][j][k] > 0.0
                        and self.z_feasible[i][j][k]
                    ):
                        candidates.append(
                            (-float(self.alpha_dqda[i][j][k]), order, [i, j, k])
                        )
                    order += 1
        candidates.sort(key=lambda item: (item[0], item[1]))
        return [position for _benefit, _order, position in candidates[: int(limit)]]

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


def _is_usable_evolution_candidate(candidate) -> bool:
    if candidate is None:
        return False
    if not candidate.mSuccess:
        return False
    verify = getattr(candidate, "verify", None)
    if not callable(verify):
        return True
    is_valid, reasons = verify()
    if not is_valid:
        logger.debug(
            "Discarding invalid evolution candidate %s: %s",
            getattr(candidate, "name", None),
            ", ".join(reasons),
        )
    return is_valid


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
            allowed_hots = [i for i in range(case.I) if case.z_allowed[i][j][k] > 0]
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
