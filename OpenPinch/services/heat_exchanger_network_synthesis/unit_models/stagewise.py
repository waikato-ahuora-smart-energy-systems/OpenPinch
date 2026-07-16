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
from typing import Any, Literal

import numpy as np

from ..common.indexing import build_index_grid
from ..common.solver.arrays import PreparedSolverArrays
from ._stagewise.evolution import (
    _EvolutionBranchState,
    _EvolutionCandidateSpec,
)
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
        """Pre-process SynHEAT superstructure parameters for all states."""

        self.S = self.stages
        self.K = self.S + 1
        self.I = self.f_h_period.shape[1]
        self.J = self.f_c_period.shape[1]

        self.Qtot_sh_period = np.array(
            [
                [
                    self._parent_profile_duty(
                        "hot",
                        n,
                        i,
                        self.T_h_in_period[n][i],
                        self.T_h_out_period[n][i],
                        self.f_h_period[n][i],
                    )
                    for i in range(self.I)
                ]
                for n in range(self.N_periods)
            ],
            dtype=float,
        )
        self.Qtot_sc_period = np.array(
            [
                [
                    self._parent_profile_duty(
                        "cold",
                        n,
                        j,
                        self.T_c_in_period[n][j],
                        self.T_c_out_period[n][j],
                        self.f_c_period[n][j],
                    )
                    for j in range(self.J)
                ]
                for n in range(self.N_periods)
            ],
            dtype=float,
        )
        self.Qtot_sh = np.max(self.Qtot_sh_period, axis=0)
        self.Qtot_sc = np.max(self.Qtot_sc_period, axis=0)

        self.U_r_period = np.array(
            [
                [
                    [
                        1 / (1 / self.htc_h_period[n][i] + 1 / self.htc_c_period[n][j])
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]
                for n in range(self.N_periods)
            ],
            dtype=float,
        )
        self.U_hu_period = np.array(
            [
                [
                    1 / (1 / self.htc_hu_period[n][0] + 1 / self.htc_c_period[n][j])
                    for j in range(self.J)
                ]
                for n in range(self.N_periods)
            ],
            dtype=float,
        )
        self.U_cu_period = np.array(
            [
                [
                    1 / (1 / self.htc_h_period[n][i] + 1 / self.htc_cu_period[n][0])
                    for i in range(self.I)
                ]
                for n in range(self.N_periods)
            ],
            dtype=float,
        )
        self.U_r = self.U_r_period[0].copy()
        self.U_hu = self.U_hu_period[0].copy()
        self.U_cu = self.U_cu_period[0].copy()

        self.Q_max_period = np.array(
            [
                [
                    [
                        max(
                            self._recovery_heat_upper_bound(
                                period_index=n,
                                hot_index=i,
                                cold_index=j,
                                hot_total_duty=self.Qtot_sh_period[n][i],
                                cold_total_duty=self.Qtot_sc_period[n][j],
                                hot_cp=self.f_h_period[n][i],
                                cold_cp=self.f_c_period[n][j],
                            ),
                            0.0,
                        )
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]
                for n in range(self.N_periods)
            ],
            dtype=float,
        )
        self.Q_max = np.max(self.Q_max_period, axis=0)
        self.M_ij_period = np.array(
            [
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
            ],
            dtype=float,
        )
        self.M_ij = np.max(self.M_ij_period, axis=0)

        self.z_feasible = [
            [
                [
                    (
                        1
                        if max(
                            self.Q_max_period[n][i][j] for n in range(self.N_periods)
                        )
                        > self.tol
                        else 0
                    )
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.z_hu_feasible = [1 for _j in range(self.J)]
        self.z_cu_feasible = [1 for _i in range(self.I)]

    def set_stage_wise_superstructure(self) -> None:
        """Create StageWise variables, constraints, and binaries."""

        self._set_multiperiod_stage_wise_superstructure()

    def _set_multiperiod_stage_wise_superstructure(self) -> None:
        """Create shared topology with state-indexed operating variables."""

        self.Q_r_by_period = [
            [
                [
                    [
                        (
                            self.m.Var(
                                value=self.Q_max_period[n][i][j] / 3,
                                ub=self.Q_max_period[n][i][j],
                                lb=0.0,
                                name=(f"Q_H{i}_to_C{j}_at_S{k}_period{n}"),
                            )
                            if self.z_allowed[i][j][k] > 0
                            else self.m.Param(
                                value=0.0,
                                name=f"Q_H{i}_to_C{j}_at_S{k}_period{n}",
                            )
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            for n in range(self.N_periods)
        ]
        self.Q_c_by_period = [
            [
                (
                    self.m.Var(
                        value=0,
                        ub=self.Qtot_sh_period[n][i],
                        lb=0.0,
                        name=f"Q_H{i}_to_CU_period{n}",
                    )
                    if self.z_cu_allowed[i] > 0
                    else self.m.Param(value=0, name=f"Q_H{i}_to_CU_period{n}")
                )
                for i in range(self.I)
            ]
            for n in range(self.N_periods)
        ]
        self.Q_h_by_period = [
            [
                (
                    self.m.Var(
                        value=0,
                        ub=self.Qtot_sc_period[n][j],
                        lb=0.0,
                        name=f"Q_HU_to_C{j}_period{n}",
                    )
                    if self.z_hu_allowed[j] > 0
                    else self.m.Param(value=0, name=f"Q_HU_to_C{j}_period{n}")
                )
                for j in range(self.J)
            ]
            for n in range(self.N_periods)
        ]
        self.T_h_by_period = [
            [
                [
                    (
                        self.m.Var(
                            value=self.T_h_in_period[n][i],
                            ub=self.T_h_in_period[n][i],
                            lb=self.T_h_out_period[n][i],
                            name=f"T_H{i}_at_B{k}_period{n}",
                        )
                        if k > 0
                        else self.m.Param(
                            value=self.T_h_in_period[n][i],
                            name=f"T_H{i}_at_B{k}_period{n}",
                        )
                    )
                    for k in range(self.K)
                ]
                for i in range(self.I)
            ]
            for n in range(self.N_periods)
        ]
        self.T_c_by_period = [
            [
                [
                    (
                        self.m.Var(
                            value=self.T_c_in_period[n][j],
                            ub=self.T_c_out_period[n][j],
                            lb=self.T_c_in_period[n][j],
                            name=f"T_C{j}_at_B{k}_period{n}",
                        )
                        if k < self.S
                        else self.m.Param(
                            value=self.T_c_in_period[n][j],
                            name=f"T_C{j}_at_B{k}_period{n}",
                        )
                    )
                    for k in range(self.K)
                ]
                for j in range(self.J)
            ]
            for n in range(self.N_periods)
        ]
        self.theta_1_by_period = [
            [
                [
                    [
                        (
                            self.m.Var(
                                value=self._recovery_approach_temperature(
                                    i,
                                    j,
                                    n,
                                ),
                                ub=abs(
                                    self.T_h_in_period[n][i] - self.T_c_in_period[n][j]
                                ),
                                lb=self._recovery_approach_temperature(
                                    i,
                                    j,
                                    n,
                                ),
                                name=(f"approach_T1_H{i}_to_C{j}_at_S{k}_period{n}"),
                            )
                            if self.z_allowed[i][j][k] > 0
                            else self.m.Param(
                                value=self._recovery_approach_temperature(
                                    i,
                                    j,
                                    n,
                                ),
                                name=(f"approach_T1_H{i}_to_C{j}_at_S{k}_period{n}"),
                            )
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            for n in range(self.N_periods)
        ]
        self.theta_2_by_period = [
            [
                [
                    [
                        (
                            self.m.Var(
                                value=self._recovery_approach_temperature(
                                    i,
                                    j,
                                    n,
                                ),
                                ub=abs(
                                    self.T_h_in_period[n][i] - self.T_c_in_period[n][j]
                                ),
                                lb=self._recovery_approach_temperature(
                                    i,
                                    j,
                                    n,
                                ),
                                name=(f"approach_T2_H{i}_to_C{j}_at_S{k}_period{n}"),
                            )
                            if self.z_allowed[i][j][k] > 0
                            else self.m.Param(
                                value=self._recovery_approach_temperature(
                                    i,
                                    j,
                                    n,
                                ),
                                name=(f"approach_T2_H{i}_to_C{j}_at_S{k}_period{n}"),
                            )
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            for n in range(self.N_periods)
        ]

        self.Q_r = self.Q_r_by_period[0]
        self.Q_c = self.Q_c_by_period[0]
        self.Q_h = self.Q_h_by_period[0]
        self.T_h = self.T_h_by_period[0]
        self.T_c = self.T_c_by_period[0]
        self.theta_1 = self.theta_1_by_period[0]
        self.theta_2 = self.theta_2_by_period[0]

        self._set_piecewise_stage_heat_coordinates()

        for n in range(self.N_periods):
            self.m.Equations(
                [
                    self.Qtot_sh_period[n][i]
                    - self.m.sum(
                        [
                            self.Q_r_by_period[n][i][j][k]
                            for k in range(self.S)
                            for j in range(self.J)
                        ]
                    )
                    - self.Q_c_by_period[n][i]
                    == 0.0
                    for i in range(self.I)
                ]
            )
            self.m.Equations(
                [
                    self.Qtot_sc_period[n][j]
                    - self.m.sum(
                        [
                            self.Q_r_by_period[n][i][j][k]
                            for k in range(self.S)
                            for i in range(self.I)
                        ]
                    )
                    - self.Q_h_by_period[n][j]
                    == 0.0
                    for j in range(self.J)
                ]
            )
            self.m.Equations(
                [
                    (self.T_h_by_period[n][i][k + 1] - self.T_h_by_period[n][i][k])
                    * self.f_h_period[n][i]
                    + sum(self.Q_r_by_period[n][i][j][k] for j in range(self.J))
                    == 0
                    for k in range(self.S)
                    for i in range(self.I)
                    if not self._hot_parent_segmented(i)
                ]
            )
            self.m.Equations(
                [
                    (self.T_c_by_period[n][j][k + 1] - self.T_c_by_period[n][j][k])
                    * self.f_c_period[n][j]
                    + sum(self.Q_r_by_period[n][i][j][k] for i in range(self.I))
                    == 0
                    for k in range(self.S)
                    for j in range(self.J)
                    if not self._cold_parent_segmented(j)
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
            for n in range(self.N_periods):
                _ = [
                    (
                        self.m.Equation(
                            self.Q_r_by_period[n][i][j][k]
                            <= self.Q_max_period[n][i][j] * self.z[i][j][k]
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
                            self.Q_c_by_period[n][i]
                            <= self.Qtot_sh_period[n][i] * self.z_cu[i]
                        )
                        if self.z_cu_allowed[i] > 0
                        else None
                    )
                    for i in range(self.I)
                ]
                _ = [
                    (
                        self.m.Equation(
                            self.Q_h_by_period[n][j]
                            <= self.Qtot_sc_period[n][j] * self.z_hu[j]
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

        if self.non_isothermal_model:
            self._set_multiperiod_non_isothermal_equations()
        else:
            self._set_multiperiod_isothermal_approach_equations()

        self.dqda = []
        self.alpha = []

    def _set_multiperiod_isothermal_approach_equations(self) -> None:
        for n in range(self.N_periods):
            _ = [
                (
                    self.m.Equation(
                        self.theta_1_by_period[n][i][j][k]
                        <= (self.T_h_by_period[n][i][k] - self.T_c_by_period[n][j][k])
                        + self.M_ij_period[n][i][j] * (1 - self.z[i][j][k])
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
                        self.theta_1_by_period[n][i][j][k]
                        >= (self.T_h_by_period[n][i][k] - self.T_c_by_period[n][j][k])
                        - self.M_ij_period[n][i][j] * (1 - self.z[i][j][k])
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
                        self.theta_2_by_period[n][i][j][k]
                        <= (
                            self.T_h_by_period[n][i][k + 1]
                            - self.T_c_by_period[n][j][k + 1]
                        )
                        + self.M_ij_period[n][i][j] * (1 - self.z[i][j][k])
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
                        self.theta_2_by_period[n][i][j][k]
                        >= (
                            self.T_h_by_period[n][i][k + 1]
                            - self.T_c_by_period[n][j][k + 1]
                        )
                        - self.M_ij_period[n][i][j] * (1 - self.z[i][j][k])
                    )
                    if self.z_allowed[i][j][k] > 0
                    else None
                )
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]

    def _set_multiperiod_non_isothermal_equations(self) -> None:
        self.X_by_period = [
            [
                [
                    [
                        (
                            self.m.Var(
                                value=0.5,
                                ub=1.0,
                                lb=0.0,
                                name=f"X_H{i}_to_C{j}_at_S{k}_period{n}",
                            )
                            if self.z_allowed[i][j][k] > 0
                            else self.m.Param(
                                value=0.0,
                                name=f"X_H{i}_to_C{j}_at_S{k}_period{n}",
                            )
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            for n in range(self.N_periods)
        ]
        self.Y_by_period = [
            [
                [
                    [
                        (
                            self.m.Var(
                                value=0.5,
                                ub=1.0,
                                lb=0.0,
                                name=f"Y_C{j}_to_H{i}_at_S{k}_period{n}",
                            )
                            if self.z_allowed[i][j][k] > 0
                            else self.m.Param(
                                value=0.0,
                                name=f"Y_C{j}_to_H{i}_at_S{k}_period{n}",
                            )
                        )
                        for k in range(self.S)
                    ]
                    for i in range(self.I)
                ]
                for j in range(self.J)
            ]
            for n in range(self.N_periods)
        ]
        self.T_h_out_x_by_period = [
            [
                [
                    [
                        (
                            self.m.Var(
                                value=self.T_h_in_period[n][i],
                                ub=self.T_h_in_period[n][i],
                                lb=self.T_h_out_period[n][i],
                                name=f"Thout_x_H{i}_to_C{j}_at_S{k}_period{n}",
                            )
                            if self.z_allowed[i][j][k] > 0
                            else self.m.Param(
                                value=0,
                                name=f"Tx_H{i}_to_C{j}_at_S{k}_period{n}",
                            )
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            for n in range(self.N_periods)
        ]
        self.T_c_out_y_by_period = [
            [
                [
                    [
                        (
                            self.m.Var(
                                value=self.T_c_in_period[n][j],
                                ub=self.T_c_out_period[n][j],
                                lb=self.T_c_in_period[n][j],
                                name=f"Tcout_y_C{j}_to_H{i}_at_S{k}_period{n}",
                            )
                            if self.z_allowed[i][j][k] > 0
                            else self.m.Param(
                                value=0,
                                name=f"Ty_C{j}_to_H{i}_at_S{k}_period{n}",
                            )
                        )
                        for k in range(self.S)
                    ]
                    for i in range(self.I)
                ]
                for j in range(self.J)
            ]
            for n in range(self.N_periods)
        ]
        self.X = self.X_by_period[0]
        self.Y = self.Y_by_period[0]
        self.T_h_out_x = self.T_h_out_x_by_period[0]
        self.T_c_out_y = self.T_c_out_y_by_period[0]

        self._set_piecewise_match_outlet_equations()

        for n in range(self.N_periods):
            _ = [
                (
                    self.m.Equation(
                        self.theta_1_by_period[n][i][j][k]
                        <= (
                            self.T_h_by_period[n][i][k]
                            - self.T_c_out_y_by_period[n][j][i][k]
                        )
                        + self.M_ij_period[n][i][j] * (1 - self.z[i][j][k])
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
                        self.theta_2_by_period[n][i][j][k]
                        <= (
                            self.T_h_out_x_by_period[n][i][j][k]
                            - self.T_c_by_period[n][j][k + 1]
                        )
                        + self.M_ij_period[n][i][j] * (1 - self.z[i][j][k])
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
                        self.Q_r_by_period[n][i][j][k]
                        - self.X_by_period[n][i][j][k]
                        * self.f_h_period[n][i]
                        * (
                            self.T_h_by_period[n][i][k]
                            - self.T_h_out_x_by_period[n][i][j][k]
                        )
                        == 0.0
                    )
                    if self.z_allowed[i][j][k] > 0 and not self._hot_parent_segmented(i)
                    else None
                )
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]
            _ = [
                (
                    self.m.Equation(
                        self.Q_r_by_period[n][i][j][k]
                        - self.Y_by_period[n][j][i][k]
                        * self.f_c_period[n][j]
                        * (
                            self.T_c_out_y_by_period[n][j][i][k]
                            - self.T_c_by_period[n][j][k + 1]
                        )
                        == 0.0
                    )
                    if self.z_allowed[i][j][k] > 0
                    and not self._cold_parent_segmented(j)
                    else None
                )
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]
            _ = [
                (
                    self.m.Equation(
                        self.m.sum(
                            [self.X_by_period[n][i][j][k] for j in range(self.J)]
                        )
                        == 1.0
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
                        self.m.sum(
                            [self.Y_by_period[n][j][i][k] for i in range(self.I)]
                        )
                        == 1.0
                    )
                    if sum(self.z_allowed[i][j][k] for i in range(self.I)) > 0
                    else None
                )
                for k in range(self.S)
                for j in range(self.J)
            ]

    def set_dqda_equations(self) -> None:
        """Apply the source TDM minimum dQ/dA restriction."""

        if getattr(self, "N_periods", 1) > 1:
            self.min_dqda_int = build_index_grid(
                lambda n, i, j, k: (
                    self.m.Intermediate(
                        self.min_dqda
                        * (
                            self.T_h_by_period[n][i][k]
                            - self.T_c_by_period[n][j][k + 1]
                        )
                        - self.theta_1_by_period[n][i][j][k]
                        * self.theta_2_by_period[n][i][j][k]
                        * self.U_r_period[n][i][j],
                        name=f"dqda_calc_H{i}_to_C{j}_at_S{k}_period{n}",
                    )
                    if self.z_allowed[i][j][k] > 0
                    else None
                ),
                (self.N_periods, self.I, self.J, self.S),
            )
            self.min_dQ_dA_eqn = [
                (
                    self.m.Equation(
                        self.min_dqda_int[n][i][j][k]
                        <= self.min_dqda
                        * self.M_ij_period[n][i][j]
                        * (1 - self.z[i][j][k])
                    )
                    if self.z_allowed[i][j][k] > 0
                    else None
                )
                for n in range(self.N_periods)
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]
            return

        self.min_dqda_int = build_index_grid(
            lambda i, j, k: (
                self.m.Intermediate(
                    self.min_dqda * (self.T_h[i][k] - self.T_c[j][k + 1])
                    - self.theta_1[i][j][k] * self.theta_2[i][j][k] * self.U_r[i][j],
                    name=f"dqda_calc_H{i}_to_C{j}_at_S{k}",
                )
                if self.z_allowed[i][j][k] > 0
                else None
            ),
            (self.I, self.J, self.S),
        )
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

        if getattr(self, "N_periods", 1) > 1 and hasattr(self, "Q_r_by_period"):
            self._set_multiperiod_initial_values(init_solution, brackets=brackets)
            return

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
                        for j in range(self.J):
                            sum_Q_r_j = sum(
                                init_solution.Q_r[i][j][k].VALUE[0]
                                for i in range(self.I)
                            )
                            if self.z_allowed[i][j][k] > 0:
                                q_val = init_solution.Q_r[i][j][k].VALUE[0]
                                if q_val > 0.0:
                                    self._set_value(
                                        self.X[i][j][k],
                                        q_val / sum_Q_r_i if sum_Q_r_i else 0.0,
                                        brackets=brackets,
                                    )
                                    self._set_value(
                                        self.Y[j][i][k],
                                        q_val / sum_Q_r_j if sum_Q_r_j else 0.0,
                                        brackets=brackets,
                                    )
                                else:
                                    self._set_value(
                                        self.X[i][j][k], 0.0, brackets=brackets
                                    )
                                    self._set_value(
                                        self.Y[j][i][k], 0.0, brackets=brackets
                                    )
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

    def _set_multiperiod_initial_values(self, init_solution, *, brackets: bool) -> None:
        source_q_r = getattr(init_solution, "Q_r_by_period", None)
        source_q_c = getattr(init_solution, "Q_c_by_period", None)
        source_q_h = getattr(init_solution, "Q_h_by_period", None)
        source_t_h = getattr(init_solution, "T_h_by_period", None)
        source_t_c = getattr(init_solution, "T_c_by_period", None)
        source_theta_1 = getattr(init_solution, "theta_1_by_period", None)
        source_theta_2 = getattr(init_solution, "theta_2_by_period", None)
        if source_q_r is None:
            source_q_r = [init_solution.Q_r for _ in range(self.N_periods)]
            source_q_c = [init_solution.Q_c for _ in range(self.N_periods)]
            source_q_h = [init_solution.Q_h for _ in range(self.N_periods)]
            source_t_h = [init_solution.T_h for _ in range(self.N_periods)]
            source_t_c = [init_solution.T_c for _ in range(self.N_periods)]
            source_theta_1 = [init_solution.theta_1 for _ in range(self.N_periods)]
            source_theta_2 = [init_solution.theta_2 for _ in range(self.N_periods)]

        for n in range(self.N_periods):
            source_period_idx = min(n, len(source_q_r) - 1)
            for k in range(self.S):
                for j in range(self.J):
                    for i in range(self.I):
                        if self.z_allowed[i][j][k] > 0:
                            self._set_value(
                                self.Q_r_by_period[n][i][j][k],
                                self._active_binary_value(
                                    source_q_r[source_period_idx][i][j][k]
                                ),
                                brackets=brackets,
                            )
                            self._set_value(
                                self.theta_1_by_period[n][i][j][k],
                                self._active_binary_value(
                                    source_theta_1[source_period_idx][i][j][k]
                                ),
                                brackets=brackets,
                            )
                            self._set_value(
                                self.theta_2_by_period[n][i][j][k],
                                self._active_binary_value(
                                    source_theta_2[source_period_idx][i][j][k]
                                ),
                                brackets=brackets,
                            )
                        else:
                            self._set_value(
                                self.Q_r_by_period[n][i][j][k],
                                0.0,
                                brackets=brackets,
                            )
                            approach = self._recovery_approach_temperature(
                                i,
                                j,
                                n,
                            )
                            self._set_value(
                                self.theta_1_by_period[n][i][j][k],
                                approach,
                                brackets=brackets,
                            )
                            self._set_value(
                                self.theta_2_by_period[n][i][j][k],
                                approach,
                                brackets=brackets,
                            )

            for i in range(self.I):
                for k in range(self.K):
                    self._set_value(
                        self.T_h_by_period[n][i][k],
                        self._active_binary_value(source_t_h[source_period_idx][i][k]),
                        brackets=brackets,
                    )
            for j in range(self.J):
                for k in range(self.K):
                    self._set_value(
                        self.T_c_by_period[n][j][k],
                        self._active_binary_value(source_t_c[source_period_idx][j][k]),
                        brackets=brackets,
                    )
            for i in range(self.I):
                value = (
                    self._active_binary_value(source_q_c[source_period_idx][i])
                    if self.z_cu_allowed[i] > 0
                    else 0.0
                )
                self._set_value(
                    self.Q_c_by_period[n][i],
                    value,
                    brackets=brackets,
                )
            for j in range(self.J):
                value = (
                    self._active_binary_value(source_q_h[source_period_idx][j])
                    if self.z_hu_allowed[j] > 0
                    else 0.0
                )
                self._set_value(
                    self.Q_h_by_period[n][j],
                    value,
                    brackets=brackets,
                )

        for k in range(self.S):
            for j in range(self.J):
                for i in range(self.I):
                    self._set_value(
                        self.z[i][j][k],
                        self._active_binary_value(init_solution.z[i][j][k]),
                        brackets=brackets,
                    )
        for i in range(self.I):
            self._set_value(
                self.z_cu[i],
                self._active_binary_value(init_solution.z_cu[i]),
                brackets=brackets,
            )
        for j in range(self.J):
            self._set_value(
                self.z_hu[j],
                self._active_binary_value(init_solution.z_hu[j]),
                brackets=brackets,
            )

        source_area_r = getattr(init_solution, "area_r_shared", None)
        source_area_hu = getattr(init_solution, "area_hu_shared", None)
        source_area_cu = getattr(init_solution, "area_cu_shared", None)
        if source_area_r is not None and hasattr(self, "area_r_shared"):
            for k in range(self.S):
                for j in range(self.J):
                    for i in range(self.I):
                        self._set_value(
                            self.area_r_shared[i][j][k],
                            self._active_binary_value(source_area_r[i][j][k]),
                            brackets=brackets,
                        )
        if source_area_hu is not None and hasattr(self, "area_hu_shared"):
            for j in range(self.J):
                self._set_value(
                    self.area_hu_shared[j],
                    self._active_binary_value(source_area_hu[j]),
                    brackets=brackets,
                )
        if source_area_cu is not None and hasattr(self, "area_cu_shared"):
            for i in range(self.I):
                self._set_value(
                    self.area_cu_shared[i],
                    self._active_binary_value(source_area_cu[i]),
                    brackets=brackets,
                )

        if self.non_isothermal_model and hasattr(self, "X_by_period"):
            source_x = getattr(init_solution, "X_by_period", None)
            source_y = getattr(init_solution, "Y_by_period", None)
            source_thx = getattr(init_solution, "T_h_out_x_by_period", None)
            source_tcy = getattr(init_solution, "T_c_out_y_by_period", None)
            if source_x is None and getattr(
                init_solution, "non_isothermal_model", False
            ):
                source_x = [init_solution.X for _ in range(self.N_periods)]
                source_y = [init_solution.Y for _ in range(self.N_periods)]
                source_thx = [init_solution.T_h_out_x for _ in range(self.N_periods)]
                source_tcy = [init_solution.T_c_out_y for _ in range(self.N_periods)]
            if source_x is not None:
                for n in range(self.N_periods):
                    source_period_idx = min(n, len(source_x) - 1)
                    for k in range(self.S):
                        for j in range(self.J):
                            for i in range(self.I):
                                if self.z_allowed[i][j][k] > 0:
                                    self._set_value(
                                        self.X_by_period[n][i][j][k],
                                        self._active_binary_value(
                                            source_x[source_period_idx][i][j][k]
                                        ),
                                        brackets=brackets,
                                    )
                                    self._set_value(
                                        self.Y_by_period[n][j][i][k],
                                        self._active_binary_value(
                                            source_y[source_period_idx][j][i][k]
                                        ),
                                        brackets=brackets,
                                    )
                                    self._set_value(
                                        self.T_h_out_x_by_period[n][i][j][k],
                                        self._active_binary_value(
                                            source_thx[source_period_idx][i][j][k]
                                        ),
                                        brackets=brackets,
                                    )
                                    self._set_value(
                                        self.T_c_out_y_by_period[n][j][i][k],
                                        self._active_binary_value(
                                            source_tcy[source_period_idx][j][i][k]
                                        ),
                                        brackets=brackets,
                                    )
            else:
                for n in range(self.N_periods):
                    source_period_idx = min(n, len(source_q_r) - 1)
                    for k in range(self.S):
                        for i in range(self.I):
                            hot_total = sum(
                                self._active_binary_value(
                                    source_q_r[source_period_idx][i][j][k]
                                )
                                for j in range(self.J)
                            )
                            for j in range(self.J):
                                cold_total = sum(
                                    self._active_binary_value(
                                        source_q_r[source_period_idx][row][j][k]
                                    )
                                    for row in range(self.I)
                                )
                                duty = self._active_binary_value(
                                    source_q_r[source_period_idx][i][j][k]
                                )
                                active = self.z_allowed[i][j][k] > 0 and duty > 0.0
                                self._set_value(
                                    self.X_by_period[n][i][j][k],
                                    duty / hot_total if active and hot_total else 0.0,
                                    brackets=brackets,
                                )
                                self._set_value(
                                    self.Y_by_period[n][j][i][k],
                                    duty / cold_total if active and cold_total else 0.0,
                                    brackets=brackets,
                                )
                                self._set_value(
                                    self.T_h_out_x_by_period[n][i][j][k],
                                    self._active_binary_value(
                                        source_t_h[source_period_idx][i][k + 1]
                                    ),
                                    brackets=brackets,
                                )
                                self._set_value(
                                    self.T_c_out_y_by_period[n][j][i][k],
                                    self._active_binary_value(
                                        source_t_c[source_period_idx][j][k]
                                    ),
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
        if (
            n_ad_branches == 1
            and n_rm_branches == 1
            and no_improvement_patience is None
        ):
            return self._get_source_net_benefit_evolution(
                print_output=print_output,
                max_depth=max_depth,
            )
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
                parent_period = frontier[spec.branch_index]
                candidate_tac = float(candidate.TAC)
                if candidate_tac < parent_period.best_tac:
                    branch_period = _EvolutionBranchState(
                        model=candidate,
                        best_tac=candidate_tac,
                        stale_depths=0,
                    )
                else:
                    branch_period = _EvolutionBranchState(
                        model=candidate,
                        best_tac=parent_period.best_tac,
                        stale_depths=parent_period.stale_depths + 1,
                    )
                if (
                    no_improvement_patience is not None
                    and branch_period.stale_depths >= no_improvement_patience
                ):
                    logger.debug(
                        "Pruning EVM branch %s after %s non-improving steps.",
                        spec.branch_index,
                        branch_period.stale_depths,
                    )
                    continue
                next_frontier.append(branch_period)
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

    def _get_source_net_benefit_evolution(
        self,
        *,
        print_output: bool,
        max_depth: int,
    ):
        """Run the original OpenHENS single-path add/remove evolution search."""

        model = self
        best_model = self
        for unit in range(max_depth):
            logger.debug(
                "Evolution step %s/%s - Current TAC: %s",
                unit + 1,
                max_depth,
                model.TAC,
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
            model = self._select_source_best_candidate(
                model,
                model_minus_one,
                model_plus_one,
            )
            if model is None:
                logger.debug("No viable model found; ending evolution.")
                break
            if model.TAC < best_model.TAC:
                best_model = model
                logger.debug(
                    "New best model: %s found with TAC: %.6f",
                    model.name,
                    best_model.TAC,
                )

        if best_model.mSuccess and best_model.TAC < self.TAC:
            self._update_with_best_model(best_model)
        else:
            logger.debug("No improvement found over original model.")
        self.m.cleanup()
        return self

    def _select_source_best_candidate(
        self,
        current_model,
        model_minus_one,
        model_plus_one,
    ):
        """Select the next OpenHENS tier-1 evolution candidate by success and TAC."""

        del current_model
        minus_success = bool(getattr(model_minus_one, "mSuccess", 0))
        plus_success = bool(getattr(model_plus_one, "mSuccess", 0))
        if not minus_success and not plus_success:
            return None
        if minus_success and not plus_success:
            return model_minus_one
        if not minus_success and plus_success:
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
        for branch_index, branch_period in enumerate(frontier):
            prev_case = branch_period.model
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
        self.alpha = build_index_grid(
            lambda i, j, k: best_model.alpha[i][j][k],
            (self.I, self.J, self.S),
        )
        self.z_allowed = build_index_grid(
            lambda i, j, k: best_model.z_allowed[i][j][k],
            (self.I, self.J, self.S),
        )
        self.set_initial_values_for_variables(best_model, brackets=True)

        self.hu_cost_total = copy.deepcopy(best_model.hu_cost_total)
        self.cu_cost_total = copy.deepcopy(best_model.cu_cost_total)
        if hasattr(best_model, "recovery_area_cost_filtered"):
            self.recovery_area_cost_filtered = copy.deepcopy(
                best_model.recovery_area_cost_filtered
            )
        if hasattr(best_model, "recovery_area_cost_total"):
            self.recovery_area_cost_total = copy.deepcopy(
                best_model.recovery_area_cost_total
            )
        if hasattr(best_model, "capital_cost_total"):
            self.capital_cost_total = copy.deepcopy(best_model.capital_cost_total)
        if hasattr(best_model, "weighted_operating_cost"):
            self.weighted_operating_cost = copy.deepcopy(
                best_model.weighted_operating_cost
            )
        if hasattr(best_model, "area_r_shared"):
            self.area_r_shared = copy.deepcopy(best_model.area_r_shared)
        if hasattr(best_model, "area_hu_shared"):
            self.area_hu_shared = copy.deepcopy(best_model.area_hu_shared)
        if hasattr(best_model, "area_cu_shared"):
            self.area_cu_shared = copy.deepcopy(best_model.area_cu_shared)
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
        return _value(value)

    def set_obj(self) -> None:
        """Attach source StageWise objective expressions unchanged."""

        if getattr(self, "N_periods", 1) > 1:
            if self.minimisation_goal == "hot utility":
                self.m.Minimize(
                    self._weighted_state_average(
                        [
                            self.m.sum(
                                [self.Q_h_by_period[n][j] for j in range(self.J)]
                            )
                            for n in range(self.N_periods)
                        ]
                    )
                )
            elif self.minimisation_goal == "cold utility":
                self.m.Minimize(
                    self._weighted_state_average(
                        [
                            self.m.sum(
                                [self.Q_c_by_period[n][i] for i in range(self.I)]
                            )
                            for n in range(self.N_periods)
                        ]
                    )
                )
            elif self.minimisation_goal == "total utility":
                self.m.Minimize(
                    self._weighted_state_average(
                        [
                            self.m.sum(
                                [self.Q_h_by_period[n][j] for j in range(self.J)]
                            )
                            + self.m.sum(
                                [self.Q_c_by_period[n][i] for i in range(self.I)]
                            )
                            for n in range(self.N_periods)
                        ]
                    )
                )
            elif self.minimisation_goal == "utility costs":
                hot_costs = [
                    self._utility_cost_expression(
                        "hot",
                        n,
                        self.m.sum([self.Q_h_by_period[n][j] for j in range(self.J)]),
                        name=f"hot_utility_period_{n}",
                    )
                    for n in range(self.N_periods)
                ]
                cold_costs = [
                    self._utility_cost_expression(
                        "cold",
                        n,
                        self.m.sum([self.Q_c_by_period[n][i] for i in range(self.I)]),
                        name=f"cold_utility_period_{n}",
                    )
                    for n in range(self.N_periods)
                ]
                self.m.Minimize(
                    self._weighted_state_average(
                        [hot_costs[n] + cold_costs[n] for n in range(self.N_periods)]
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
            elif self.minimisation_goal == "dQ/dA obj":
                self.m.Minimize(
                    self._weighted_state_average(
                        [sum(self.Q_h_by_period[n]) for n in range(self.N_periods)]
                    )
                    - self.HU_target
                )
            elif self.minimisation_goal in {"total cost", "variable total cost"}:
                self._set_total_cost_objective()
            return

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
            hot_cost = self._utility_cost_expression(
                "hot",
                0,
                self.m.sum([self.Q_h[j] for j in range(self.J)]),
                name="hot_utility",
            )
            cold_cost = self._utility_cost_expression(
                "cold",
                0,
                self.m.sum([self.Q_c[i] for i in range(self.I)]),
                name="cold_utility",
            )
            self.m.Minimize(hot_cost + cold_cost)
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
        if getattr(self, "N_periods", 1) == 1:
            self._set_source_total_cost_objective()
            return
        self._set_multiperiod_total_cost_objective()

    def _set_source_total_cost_objective(self) -> None:
        self.hu_cost_total = self.m.Intermediate(
            self._utility_cost_expression(
                "hot",
                0,
                self.m.sum([self.Q_h[j] for j in range(self.J)]),
                name="hot_utility_total_cost",
            ),
            name="Hot utility cost",
        )
        self.cu_cost_total = self.m.Intermediate(
            self._utility_cost_expression(
                "cold",
                0,
                self.m.sum([self.Q_c[i] for i in range(self.I)]),
                name="cold_utility_total_cost",
            ),
            name="Cold utility cost",
        )
        self.recovery_area_cost_filtered = [
            [0 for _j in range(self.J)] for _k in range(self.S)
        ]
        for k in range(self.S):
            for j in range(self.J):
                allowed_hots = [i for i in range(self.I) if self.z_allowed[i][j][k] > 0]
                if sum(self.z_allowed[i][j][k] for i in range(self.I)) > 0:
                    self.recovery_area_cost_filtered[k][j] = self.m.Intermediate(
                        self.A_coeff[0]
                        * sum(
                            (
                                (
                                    self.Q_r[i][j][k]
                                    / (
                                        self.U_r[i][j]
                                        * (
                                            self.theta_1[i][j][k]
                                            * self.theta_2[i][j][k]
                                            * (
                                                self.theta_1[i][j][k]
                                                + self.theta_2[i][j][k]
                                            )
                                            / 2
                                            + 1e-3
                                        )
                                        ** (1 / 3)
                                    )
                                )
                                + 1e-3
                            )
                            ** self.A_exp[0]
                            for i in allowed_hots
                        ),
                        name=f"Recovery HX area cost in stage {k} cold {j}",
                    )

        self.hu_area_cost_total = self.m.Intermediate(
            self.hu_coeff[0]
            * sum(
                (
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

    def _set_multiperiod_total_cost_objective(self) -> None:
        self.hu_cost_total_by_period = [
            self.m.Intermediate(
                self._utility_cost_expression(
                    "hot",
                    n,
                    self.m.sum([self.Q_h_by_period[n][j] for j in range(self.J)]),
                    name=f"hot_utility_total_cost_period_{n}",
                ),
                name=f"Hot utility cost state {n}",
            )
            for n in range(self.N_periods)
        ]
        self.cu_cost_total_by_period = [
            self.m.Intermediate(
                self._utility_cost_expression(
                    "cold",
                    n,
                    self.m.sum([self.Q_c_by_period[n][i] for i in range(self.I)]),
                    name=f"cold_utility_total_cost_period_{n}",
                ),
                name=f"Cold utility cost state {n}",
            )
            for n in range(self.N_periods)
        ]
        self.operating_cost_by_state_expr = [
            self.m.Intermediate(
                self.hu_cost_total_by_period[n] + self.cu_cost_total_by_period[n],
                name=f"Operating cost state {n}",
            )
            for n in range(self.N_periods)
        ]
        self.hu_cost_total = self._weighted_state_average(self.hu_cost_total_by_period)
        self.cu_cost_total = self._weighted_state_average(self.cu_cost_total_by_period)
        self.weighted_operating_cost = self._weighted_state_average(
            self.operating_cost_by_state_expr
        )

        self.area_r_shared = [
            [
                [
                    (
                        self.m.Var(
                            value=0.0,
                            lb=0.0,
                            name=f"area_H{i}_to_C{j}_at_S{k}",
                        )
                        if self.z_allowed[i][j][k] > 0
                        else self.m.Param(
                            value=0.0,
                            name=f"area_H{i}_to_C{j}_at_S{k}",
                        )
                    )
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.area_hu_shared = [
            (
                self.m.Var(value=0.0, lb=0.0, name=f"area_HU_to_C{j}")
                if self.z_hu_allowed[j] > 0
                else self.m.Param(value=0.0, name=f"area_HU_to_C{j}")
            )
            for j in range(self.J)
        ]
        self.area_cu_shared = [
            (
                self.m.Var(value=0.0, lb=0.0, name=f"area_H{i}_to_CU")
                if self.z_cu_allowed[i] > 0
                else self.m.Param(value=0.0, name=f"area_H{i}_to_CU")
            )
            for i in range(self.I)
        ]

        for n in range(self.N_periods):
            _ = [
                (
                    self.m.Equation(
                        self.area_r_shared[i][j][k]
                        >= self.Q_r_by_period[n][i][j][k]
                        / (
                            self.U_r_period[n][i][j]
                            * (
                                self.theta_1_by_period[n][i][j][k]
                                * self.theta_2_by_period[n][i][j][k]
                                * (
                                    self.theta_1_by_period[n][i][j][k]
                                    + self.theta_2_by_period[n][i][j][k]
                                )
                                / 2
                                + 1e-3
                            )
                            ** (1 / 3)
                        )
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
                        self.area_hu_shared[j]
                        >= self.Q_h_by_period[n][j]
                        / (
                            self.U_hu_period[n][j]
                            * (
                                (self.T_hu_in_period[n][0] - self.T_c_out_period[n][j])
                                * (
                                    self._utility_solved_outlet_temperature("hot", n, j)
                                    - self.T_c_by_period[n][j][0]
                                )
                                * (
                                    (
                                        self.T_hu_in_period[n][0]
                                        - self.T_c_out_period[n][j]
                                    )
                                    + (
                                        self._utility_solved_outlet_temperature(
                                            "hot", n, j
                                        )
                                        - self.T_c_by_period[n][j][0]
                                    )
                                )
                                / 2
                                + 1e-3
                            )
                            ** (1 / 3)
                        )
                    )
                    if self.z_hu_allowed[j] > 0
                    else None
                )
                for j in range(self.J)
            ]
            _ = [
                (
                    self.m.Equation(
                        self.area_cu_shared[i]
                        >= self.Q_c_by_period[n][i]
                        / (
                            self.U_cu_period[n][i]
                            * (
                                (
                                    self.T_h_by_period[n][i][self.S]
                                    - self._utility_solved_outlet_temperature(
                                        "cold", n, i
                                    )
                                )
                                * (
                                    self.T_h_out_period[n][i]
                                    - self.T_cu_in_period[n][0]
                                )
                                * (
                                    (
                                        self.T_h_by_period[n][i][self.S]
                                        - self._utility_solved_outlet_temperature(
                                            "cold", n, i
                                        )
                                    )
                                    + (
                                        self.T_h_out_period[n][i]
                                        - self.T_cu_in_period[n][0]
                                    )
                                )
                                / 2
                                + 1e-3
                            )
                            ** (1 / 3)
                        )
                    )
                    if self.z_cu_allowed[i] > 0
                    else None
                )
                for i in range(self.I)
            ]

        self.recovery_area_cost_total = self.m.Intermediate(
            self.A_coeff[0]
            * sum(
                self.area_r_shared[i][j][k] ** self.A_exp[0]
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ),
            name="Total recovery HX area cost",
        )
        self.hu_area_cost_total = self.m.Intermediate(
            self.hu_coeff[0]
            * sum(self.area_hu_shared[j] ** self.hu_exp[0] for j in range(self.J)),
            name="Total hot utility HX area cost",
        )
        self.cu_area_cost_total = self.m.Intermediate(
            self.cu_coeff[0]
            * sum(self.area_cu_shared[i] ** self.cu_exp[0] for i in range(self.I)),
            name="Total cold utility HX area cost",
        )
        self.unit_cost_total = self.m.Intermediate(
            self.unit_cost[0]
            * sum(
                self.z[i][j][k]
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ),
            name="Total recovery base cost",
        )
        self.utility_unit_cost_total = self.m.Intermediate(
            self.hu_unit_cost[0] * sum(self.z_hu[j] for j in range(self.J))
            + self.cu_unit_cost[0] * sum(self.z_cu[i] for i in range(self.I)),
            name="Total utility base cost",
        )
        self.capital_cost_total = self.m.Intermediate(
            self.unit_cost_total
            + self.utility_unit_cost_total
            + self.recovery_area_cost_total
            + self.hu_area_cost_total
            + self.cu_area_cost_total,
            name="Total shared capital cost",
        )
        self.m.Minimize(self.capital_cost_total + self.weighted_operating_cost)

    def get_post_process(self) -> None:
        """Extract source post-process arrays after a successful solve."""

        if self.mSuccess != 1:
            return
        self._get_multiperiod_post_process()

    def _get_multiperiod_post_process(self) -> None:
        q_r = [
            [
                [
                    [
                        self._active_binary_value(self.Q_r_by_period[n][i][j][k])
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            for n in range(self.N_periods)
        ]
        q_h = [
            [self._active_binary_value(self.Q_h_by_period[n][j]) for j in range(self.J)]
            for n in range(self.N_periods)
        ]
        q_c = [
            [self._active_binary_value(self.Q_c_by_period[n][i]) for i in range(self.I)]
            for n in range(self.N_periods)
        ]

        self.z = [
            [
                [
                    [
                        (
                            1
                            if max(q_r[n][i][j][k] for n in range(self.N_periods))
                            > self.tol
                            else 0
                        )
                    ]
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.z_hu = [
            [1 if max(q_h[n][j] for n in range(self.N_periods)) > self.tol else 0]
            for j in range(self.J)
        ]
        self.z_cu = [
            [1 if max(q_c[n][i] for n in range(self.N_periods)) > self.tol else 0]
            for i in range(self.I)
        ]
        self.n_recovery_units = sum(
            self.z[i][j][k][0]
            for k in range(self.S)
            for j in range(self.J)
            for i in range(self.I)
        )
        self.n_hu_units = sum(self.z_hu[j][0] for j in range(self.J))
        self.n_cu_units = sum(self.z_cu[i][0] for i in range(self.I))
        self.n_units = self.n_recovery_units + self.n_hu_units + self.n_cu_units

        self.LMTD_r_by_period = [
            [
                [
                    [
                        self._post_process_lmtd(
                            self._active_binary_value(
                                self.theta_1_by_period[n][i][j][k]
                            ),
                            self._active_binary_value(
                                self.theta_2_by_period[n][i][j][k]
                            ),
                            self.z[i][j][k][0],
                            formula_allowed=(
                                abs(
                                    self._active_binary_value(
                                        self.theta_1_by_period[n][i][j][k]
                                    )
                                    - self._active_binary_value(
                                        self.theta_2_by_period[n][i][j][k]
                                    )
                                )
                                > self.tol
                                and abs(
                                    self._active_binary_value(
                                        self.theta_1_by_period[n][i][j][k]
                                    )
                                    - self._recovery_approach_temperature(i, j, n)
                                )
                                >= self.tol
                                and abs(
                                    self._active_binary_value(
                                        self.theta_2_by_period[n][i][j][k]
                                    )
                                    - self._recovery_approach_temperature(i, j, n)
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
            for n in range(self.N_periods)
        ]
        self.area_r_by_period = [
            [
                [
                    [
                        (
                            q_r[n][i][j][k]
                            / self.U_r_period[n][i][j]
                            / self.LMTD_r_by_period[n][i][j][k]
                            if self.LMTD_r_by_period[n][i][j][k] > self.tol
                            and q_r[n][i][j][k] > self.tol
                            else 0.0
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            for n in range(self.N_periods)
        ]

        self.LMTD_hu_by_period = [
            [
                self._post_process_lmtd(
                    self.T_hu_in_period[n][0] - self.T_c_out_period[n][j],
                    self._utility_solved_outlet_temperature("hot", n, j, q_h[n][j])
                    - self._active_binary_value(self.T_c_by_period[n][j][0]),
                    self.z_hu[j][0],
                    formula_allowed=(
                        abs(
                            (self.T_hu_in_period[n][0] - self.T_c_out_period[n][j])
                            - (
                                self._utility_solved_outlet_temperature(
                                    "hot", n, j, q_h[n][j]
                                )
                                - self._active_binary_value(self.T_c_by_period[n][j][0])
                            )
                        )
                        > self.tol
                        and self.T_hu_in_period[n][0]
                        - self.T_c_out_period[n][j]
                        - self._hot_utility_inlet_approach_temperature(j, n)
                        >= self.tol
                        and self._utility_solved_outlet_temperature(
                            "hot", n, j, q_h[n][j]
                        )
                        - self._active_binary_value(self.T_c_by_period[n][j][0])
                        - self._hot_utility_outlet_approach_temperature(j, n, q_h[n][j])
                        >= self.tol
                    ),
                )
                for j in range(self.J)
            ]
            for n in range(self.N_periods)
        ]
        self.area_hu_by_period = [
            [
                (
                    q_h[n][j] / self.U_hu_period[n][j] / self.LMTD_hu_by_period[n][j]
                    if self.LMTD_hu_by_period[n][j] > self.tol and q_h[n][j] > self.tol
                    else 0.0
                )
                for j in range(self.J)
            ]
            for n in range(self.N_periods)
        ]
        self.LMTD_cu_by_period = [
            [
                self._post_process_lmtd(
                    self._active_binary_value(self.T_h_by_period[n][i][self.S])
                    - self._utility_solved_outlet_temperature("cold", n, i, q_c[n][i]),
                    self.T_h_out_period[n][i] - self.T_cu_in_period[n][0],
                    self.z_cu[i][0],
                    formula_allowed=(
                        abs(
                            (
                                self._active_binary_value(
                                    self.T_h_by_period[n][i][self.S]
                                )
                                - self._utility_solved_outlet_temperature(
                                    "cold", n, i, q_c[n][i]
                                )
                            )
                            - (self.T_h_out_period[n][i] - self.T_cu_in_period[n][0])
                        )
                        > self.tol
                        and self._active_binary_value(self.T_h_by_period[n][i][self.S])
                        - self._utility_solved_outlet_temperature(
                            "cold", n, i, q_c[n][i]
                        )
                        - self._cold_utility_outlet_approach_temperature(
                            i, n, q_c[n][i]
                        )
                        >= self.tol
                        and self.T_h_out_period[n][i]
                        - self.T_cu_in_period[n][0]
                        - self._cold_utility_inlet_approach_temperature(i, n)
                        >= self.tol
                    ),
                    fallback_delta=(
                        self.T_h_out_period[n][i] - self.T_cu_in_period[n][0]
                    ),
                )
                for i in range(self.I)
            ]
            for n in range(self.N_periods)
        ]
        self.area_cu_by_period = [
            [
                (
                    q_c[n][i] / self.U_cu_period[n][i] / self.LMTD_cu_by_period[n][i]
                    if self.LMTD_cu_by_period[n][i] > self.tol and q_c[n][i] > self.tol
                    else 0.0
                )
                for i in range(self.I)
            ]
            for n in range(self.N_periods)
        ]

        self._apply_segment_utility_areas(q_h, q_c)

        self.LMTD_r = self.LMTD_r_by_period[0]
        self.LMTD_hu = self.LMTD_hu_by_period[0]
        self.LMTD_cu = self.LMTD_cu_by_period[0]
        self.area_r = [
            [
                [
                    max(
                        self.area_r_by_period[n][i][j][k] for n in range(self.N_periods)
                    )
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self._apply_segment_recovery_areas(q_r)
        self.area_hu = [
            max(self.area_hu_by_period[n][j] for n in range(self.N_periods))
            for j in range(self.J)
        ]
        self.area_cu = [
            max(self.area_cu_by_period[n][i] for n in range(self.N_periods))
            for i in range(self.I)
        ]

        self.Q_hu_total_by_period = [sum(q_h[n]) for n in range(self.N_periods)]
        self.Q_cu_total_by_period = [sum(q_c[n]) for n in range(self.N_periods)]
        self.Q_r_total_by_period = [
            sum(
                q_r[n][i][j][k]
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            )
            for n in range(self.N_periods)
        ]
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
        self.unit_cost_total = self.unit_cost[0] * self.n_recovery_units
        self.utility_unit_cost_total = self.unit_cost[0] * (
            self.n_hu_units + self.n_cu_units
        )

        self.dqda = [
            [[None for _k in range(self.S)] for _j in range(self.J)]
            for _i in range(self.I)
        ]
        for k in range(self.S):
            for j in range(self.J):
                for i in range(self.I):
                    driving_force = self._active_binary_value(
                        self.T_h[i][k]
                    ) - self._active_binary_value(self.T_c[j][k + 1])
                    if q_r[0][i][j][k] > 0.0 and driving_force > 0.0:
                        self.dqda[i][j][k] = (
                            self._active_binary_value(self.theta_1[i][j][k])
                            * self._active_binary_value(self.theta_2[i][j][k])
                            * self.U_r[i][j]
                        ) / driving_force
                    elif driving_force > 0.0:
                        self.dqda[i][j][k] = self.U_r[i][j] * driving_force
                    else:
                        self.dqda[i][j][k] = 0.0
                    exact_dqda = self._segment_exact_dqda(
                        period_index=0,
                        hot_parent_index=i,
                        cold_parent_index=j,
                        duty=float(q_r[0][i][j][k]),
                        hot_inlet_temperature=self._active_binary_value(
                            self.T_h_by_period[0][i][k]
                        ),
                        cold_inlet_temperature=self._active_binary_value(
                            self.T_c_by_period[0][j][k + 1]
                        ),
                    )
                    if exact_dqda is not None:
                        self.dqda[i][j][k] = exact_dqda
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
        self.alpha_dqda = [
            [
                [self.alpha[i][j][k][0] * self.dqda[i][j][k] for k in range(self.S)]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]

        self.TAC_model = self.m.options.objfcnval
        self.TAC = self.capital_cost_value + self.weighted_operating_cost_value

    def _weighted_numeric_average(self, values: Sequence[float]) -> float:
        return float(
            sum(
                float(self.period_weights[n]) * float(values[n])
                for n in range(self.N_periods)
            )
            / self.period_weight_sum
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
