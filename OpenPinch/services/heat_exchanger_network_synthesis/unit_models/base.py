"""Base setup for migrated heat exchanger network equation kernels."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ....utils.heat_exchanger import compute_LMTD_from_dts
from ..common.solver import backend
from ..common.solver.arrays import PreparedSolverArrays

logger = logging.getLogger(__name__)


class BaseHeatExchangerNetworkModel(ABC):
    """Shared private state for migrated PDM/TDM/ESM equation models.

    The constructor mirrors the source OpenHENS solver defaults, but it accepts
    OpenPinch-prepared solver arrays instead of a CSV path. This layer owns the
    guarded GEKKO backend setup, source-shaped array normalization, inherited
    topology restrictions, common diagnostics, and helper equations that are
    stable across the moved private ``PinchDecompModel`` and ``StageWiseModel``.
    HENS-08 still owns topology evolution and stage-reduction behavior; those
    remain outside the base contract.
    """

    def __init__(
        self,
        name: str,
        framework: Literal["PDM", "TDM", "ESM"],
        solver: Literal["couenne", "ipopt-pyomo", "ipopt-GEKKO", "apopt"],
        solver_arrays: PreparedSolverArrays,
        dTmin: float,
        z_restriction: list | None,
        min_dqda: float,
        minimisation_goal: Literal[
            "hot utility",
            "total utility",
            "utility costs",
            "heat recovery",
            "total cost",
            "variable total cost",
        ],
        non_isothermal_model: bool,
        integers: bool,
        tol: float,
        solver_options: Mapping[str, Any] | Sequence[str] | None = None,
        import_file: Path | None = None,
    ) -> None:
        self.name = name
        self.framework = framework
        self.solver = solver
        self.solver_arrays = solver_arrays
        self.import_file = import_file
        self.dTmin = dTmin
        self.z_restriction = z_restriction
        self.min_dqda = min_dqda
        self.minimisation_goal = minimisation_goal
        self.non_isothermal_model = non_isothermal_model
        self.integers = integers
        self.tol = tol
        self.solver_options = solver_options

        self.solve_time = None
        self.solver_run = None

        self.setup_model()
        self.setup()

    def setup_model(self) -> None:
        """Create and configure the GEKKO model behind optional guards."""

        self.m = backend.create_gekko_model(remote=False)
        self.mSuccess: int = 0
        self.solver_run = backend.configure_gekko_solver(
            self.m,
            self.solver,
            solver_options=self.solver_options,
        )

    @abstractmethod
    def setup(self) -> None:
        """Create concrete equation variables, constraints, and objective."""

    @abstractmethod
    def set_preprocessing(self) -> None:
        """Populate model dimensions and derived solver constants."""

    @abstractmethod
    def set_stage_wise_superstructure(self) -> None:
        """Create the stage-wise superstructure in concrete model slices."""

    @abstractmethod
    def set_obj(self) -> None:
        """Attach the concrete objective formula unchanged from OpenHENS."""

    @abstractmethod
    def get_post_process(self) -> None:
        """Extract solved arrays after a successful concrete solve."""

    def _solver_value(self, value: Any) -> float:
        try:
            return float(value[0])
        except TypeError, IndexError, KeyError:
            return float(value)

    def _set_value(
        self, variable: Any, value: float, *, brackets: bool = False
    ) -> None:
        """Assign GEKKO values while preserving source bound-clamping behavior."""

        if type(variable).__name__ == "GKVariable":
            value = max(variable.lower, min(variable.upper, value))
            variable.VALUE.value = [value] if brackets else value
            return
        if type(variable).__name__ == "GKParameter":
            variable.VALUE.value = [value] if brackets else value
            return

    def _post_process_lmtd(
        self,
        delta_1: float,
        delta_2: float,
        active: float,
        *,
        formula_allowed: bool,
        fallback_delta: float | None = None,
    ) -> float:
        """Return source-compatible post-process LMTD.

        Heat exchanger network synthesis owns the OpenHENS active-unit and
        dTmin/tolerance gates.
        Once those gates pass, the shared OpenPinch heat-exchanger utility owns
        the positive endpoint logarithmic-mean formula.
        """

        if not formula_allowed:
            return (delta_1 if fallback_delta is None else fallback_delta) * active
        return active * float(compute_LMTD_from_dts(delta_1, delta_2))

    def get_alpha_values(self) -> list:
        """Calculate source alpha flow-on values in a post-optimisation solve."""

        if self.alpha != []:
            return self.alpha

        model = backend.create_gekko_model(remote=False)
        model.options.IMODE = 1
        model.options.SOLVER = 1
        self.set_alpha_dqda_equations(m=model, postoptimisation=True)
        try:
            model.solve(disp=False)
        except Exception:
            pass
        return self.alpha

    def set_alpha_dqda_equations(
        self,
        *,
        m: Any | None = None,
        postoptimisation: bool = False,
    ) -> None:
        """Move the source alpha and dQ/dA equations without changing formulas."""

        if postoptimisation:
            if m is None:
                raise ValueError("postoptimisation alpha equations require a model.")
            if self.non_isothermal_model:
                self.P_h = [
                    [
                        [
                            (
                                (self.T_h[i][k][0] - self.T_h_out_x[i][j][k][0])
                                / (self.T_h[i][k][0] - self.T_c[j][k + 1][0])
                                if self.T_h[i][k][0] > self.T_c[j][k + 1][0]
                                else 0.0
                            )
                            for k in range(self.S)
                        ]
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]
                self.P_c = [
                    [
                        [
                            (
                                (self.T_c_out_y[j][i][k][0] - self.T_c[j][k + 1][0])
                                / (self.T_h[i][k][0] - self.T_c[j][k + 1][0])
                                if self.T_h[i][k][0] > self.T_c[j][k + 1][0]
                                else 0.0
                            )
                            for k in range(self.S)
                        ]
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]
            else:
                self.P_h = [
                    [
                        [
                            (
                                (self.T_h[i][k][0] - self.T_h[i][k + 1][0])
                                / (self.T_h[i][k][0] - self.T_c[j][k + 1][0])
                                if self.T_h[i][k][0] > self.T_c[j][k + 1][0]
                                else 0.0
                            )
                            for k in range(self.S)
                        ]
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]
                self.P_c = [
                    [
                        [
                            (
                                (self.T_c[j][k][0] - self.T_c[j][k + 1][0])
                                / (self.T_h[i][k][0] - self.T_c[j][k + 1][0])
                                if self.T_h[i][k][0] > self.T_c[j][k + 1][0]
                                else 0.0
                            )
                            for k in range(self.S)
                        ]
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]

            self.Sum_Qr_is = [
                [
                    [sum(self.Q_r[i][j][k][0] for j in range(self.J))]
                    for k in range(self.S)
                ]
                for i in range(self.I)
            ]
            self.Sum_Qr_js = [
                [
                    [sum(self.Q_r[i][j][k][0] for i in range(self.I))]
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            self.beta_h = [
                [
                    [
                        (
                            self.Q_r[i][j][k][0] / self.Sum_Qr_is[i][k][0]
                            if self.Sum_Qr_is[i][k][0] > 0.0
                            else 0.0
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            self.beta_c = [
                [
                    [
                        (
                            self.Q_r[i][j][k][0] / self.Sum_Qr_js[j][k][0]
                            if self.Sum_Qr_js[j][k][0] > 0.0
                            else 0.0
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            self.z_i = [
                [
                    sum(self.z[i][j][k][0] for i in range(self.I))
                    / (sum(self.z[i][j][k][0] for i in range(self.I)) + 1e-9)
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            self.z_j = [
                [
                    sum(self.z[i][j][k][0] for j in range(self.J))
                    / (sum(self.z[i][j][k][0] for j in range(self.J)) + 1e-9)
                    for k in range(self.S)
                ]
                for i in range(self.I)
            ]
        else:
            m = self.m
            if self.non_isothermal_model:
                self.P_h = [
                    [
                        [
                            m.Intermediate(
                                (self.T_h[i][k] - self.T_h_out_x[i][j][k])
                                * self.z[i][j][k]
                                / (
                                    (self.T_h[i][k] - self.T_c[j][k + 1] - 1)
                                    * self.z[i][j][k]
                                    + 1
                                )
                            )
                            for k in range(self.S)
                        ]
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]
                self.P_c = [
                    [
                        [
                            m.Intermediate(
                                (self.T_c_out_y[j][i][k] - self.T_c[j][k + 1])
                                * self.z[i][j][k]
                                / (
                                    (self.T_h[i][k] - self.T_c[j][k + 1] - 1)
                                    * self.z[i][j][k]
                                    + 1
                                )
                            )
                            for k in range(self.S)
                        ]
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]
            else:
                self.P_h = [
                    [
                        [
                            m.Intermediate(
                                (self.T_h[i][k] - self.T_h[i][k + 1])
                                * self.z[i][j][k]
                                / (
                                    (self.T_h[i][k] - self.T_c[j][k + 1] - 1)
                                    * self.z[i][j][k]
                                    + 1
                                )
                            )
                            for k in range(self.S)
                        ]
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]
                self.P_c = [
                    [
                        [
                            m.Intermediate(
                                (self.T_c[j][k] - self.T_c[j][k + 1])
                                * self.z[i][j][k]
                                / (
                                    (self.T_h[i][k] - self.T_c[j][k + 1] - 1)
                                    * self.z[i][j][k]
                                    + 1
                                )
                            )
                            for k in range(self.S)
                        ]
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]

            self.Sum_Qr_j = [
                [
                    m.Intermediate(sum(self.Q_r[i][j][k] for j in range(self.J)))
                    for k in range(self.S)
                ]
                for i in range(self.I)
            ]
            self.Sum_Qr_i = [
                [
                    m.Intermediate(sum(self.Q_r[i][j][k] for i in range(self.I)))
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            self.beta_h = [
                [
                    [
                        m.Intermediate(
                            self.Q_r[i][j][k]
                            / (self.Sum_Qr_j[i][k] + 1 - self.z[i][j][k])
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            self.beta_c = [
                [
                    [
                        m.Intermediate(
                            self.Q_r[i][j][k]
                            / (self.Sum_Qr_i[j][k] + 1 - self.z[i][j][k])
                        )
                        for k in range(self.S)
                    ]
                    for j in range(self.J)
                ]
                for i in range(self.I)
            ]
            self.z_i = [
                [
                    m.Intermediate(
                        sum(self.z[i][j][k] for i in range(self.I))
                        / (sum(self.z[i][j][k] for i in range(self.I)) + 1e-9)
                    )
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            self.z_j = [
                [
                    m.Intermediate(
                        sum(self.z[i][j][k] for j in range(self.J))
                        / (sum(self.z[i][j][k] for j in range(self.J)) + 1e-9)
                    )
                    for k in range(self.S)
                ]
                for i in range(self.I)
            ]

        self.alpha = [
            [
                [
                    m.Var(
                        value=0.0,
                        ub=1.0,
                        lb=-1.0,
                        name=f"alpha_H{i}_to_C{j}_at_S{k}",
                    )
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.gamma_h = [
            [
                [
                    m.Var(
                        value=0.5,
                        ub=1.0,
                        lb=-1.0,
                        name=f"gamma_h_H{i}_to_C{j}_at_S{k}",
                    )
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.gamma_c = [
            [
                [
                    m.Var(
                        value=0.5,
                        ub=1.0,
                        lb=-1.0,
                        name=f"gamma_c_H{i}_to_C{j}_at_S{k}",
                    )
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]

        self.gamma_h_eqn = []
        self.gamma_c_eqn = []
        for k in range(self.S):
            for j in range(self.J):
                for i in range(self.I):
                    if k + 1 >= self.S:
                        self.gamma_h_eqn.append(
                            [m.Equation(self.gamma_h[i][j][k] == 0.0)]
                        )
                        self.gamma_c_eqn.append(
                            [
                                m.Equation(
                                    self.gamma_c[i][j][k]
                                    == sum(
                                        self.beta_c[i0][j][k - 1]
                                        * self.P_c[i0][j][k - 1]
                                        * self.alpha[i0][j][k - 1]
                                        for i0 in range(self.I)
                                    )
                                    + (1 - self.z_i[j][k - 1])
                                    * self.gamma_c[i][j][k - 1]
                                )
                            ]
                        )
                    elif k - 1 < 0:
                        self.gamma_h_eqn.append(
                            [
                                m.Equation(
                                    self.gamma_h[i][j][k]
                                    == sum(
                                        self.beta_h[i][j0][k + 1]
                                        * self.P_h[i][j0][k + 1]
                                        * self.alpha[i][j0][k + 1]
                                        for j0 in range(self.J)
                                    )
                                    + (1 - self.z_j[i][k + 1])
                                    * self.gamma_h[i][j][k + 1]
                                )
                            ]
                        )
                        self.gamma_c_eqn.append(
                            [m.Equation(self.gamma_c[i][j][k] == 0.0)]
                        )
                    else:
                        self.gamma_h_eqn.append(
                            [
                                m.Equation(
                                    self.gamma_h[i][j][k]
                                    == sum(
                                        self.beta_h[i][j0][k + 1]
                                        * self.P_h[i][j0][k + 1]
                                        * self.alpha[i][j0][k + 1]
                                        for j0 in range(self.J)
                                    )
                                    + (1 - self.z_j[i][k + 1])
                                    * self.gamma_h[i][j][k + 1]
                                )
                            ]
                        )
                        self.gamma_c_eqn.append(
                            [
                                m.Equation(
                                    self.gamma_c[i][j][k]
                                    == sum(
                                        self.beta_c[i0][j][k - 1]
                                        * self.P_c[i0][j][k - 1]
                                        * self.alpha[i0][j][k - 1]
                                        for i0 in range(self.I)
                                    )
                                    + (1 - self.z_i[j][k - 1])
                                    * self.gamma_c[i][j][k - 1]
                                )
                            ]
                        )

        self.alpha_eqn = [
            m.Equation(
                self.alpha[i][j][k]
                == (1 - 0.5 * (self.gamma_h[i][j][k] + self.gamma_c[i][j][k]))
            )
            for k in range(self.S)
            for j in range(self.J)
            for i in range(self.I)
            if postoptimisation or self.z_allowed[i][j][k] > 0
        ]
        if not postoptimisation:
            self.alpha_dQ_dA_eqn = [
                (
                    m.Equation(
                        (
                            self.min_dqda * (self.T_h[i][k] - self.T_c[j][k + 1])
                            - self.alpha[i][j][k]
                            * self.theta_1[i][j][k]
                            * self.theta_2[i][j][k]
                            * self.U_r[i][j]
                        )
                        * self.z[i][j][k]
                        <= 0.0
                    )
                    if self.z_allowed[i][j][k] > 0
                    else None
                )
                for k in range(self.S)
                for j in range(self.J)
                for i in range(self.I)
            ]

    def set_blank_input_parameters(self) -> None:
        """Initialize the solver-array attributes expected by source equations."""

        self.T_h_in = np.array([], dtype=float)
        self.T_h_out = np.array([], dtype=float)
        self.f_h = np.array([], dtype=float)
        self.htc_h = np.array([], dtype=float)
        self.h_cost = np.array([], dtype=float)
        self.hot_names = np.array([], dtype=str)
        self.T_h_cont = np.array([], dtype=float)

        self.T_c_in = np.array([], dtype=float)
        self.T_c_out = np.array([], dtype=float)
        self.f_c = np.array([], dtype=float)
        self.htc_c = np.array([], dtype=float)
        self.c_cost = np.array([], dtype=float)
        self.cold_names = np.array([], dtype=str)
        self.T_c_cont = np.array([], dtype=float)

        self.hu_cost = np.array([], dtype=float)
        self.hu_unit_cost = np.array([], dtype=float)
        self.hu_coeff = np.array([], dtype=float)
        self.T_hu_in = np.array([], dtype=float)
        self.T_hu_out = np.array([], dtype=float)
        self.T_hu_cont = np.array([], dtype=float)
        self.htc_hu = np.array([], dtype=float)
        self.hu_exp = np.array([], dtype=float)

        self.cu_cost = np.array([], dtype=float)
        self.cu_unit_cost = np.array([], dtype=float)
        self.cu_coeff = np.array([], dtype=float)
        self.T_cu_in = np.array([], dtype=float)
        self.T_cu_out = np.array([], dtype=float)
        self.T_cu_cont = np.array([], dtype=float)
        self.htc_cu = np.array([], dtype=float)
        self.cu_exp = np.array([], dtype=float)

        self.unit_cost = np.array([], dtype=float)
        self.A_coeff = np.array([], dtype=float)
        self.A_exp = np.array([], dtype=float)
        self.period_ids = np.array(["0"], dtype=str)
        self.period_weights = np.array([1.0], dtype=float)
        self.N_periods = 1
        self.period_weight_sum = 1.0

    def get_model_parameters_from_solver_arrays(self) -> None:
        """Populate model attributes from the OpenPinch private array adapter."""

        for name, values in self.solver_arrays.arrays.items():
            setattr(self, name, np.array(values, copy=True))
        self._normalise_state_arrays()
        self._set_minimum_approach_temperatures()

    def _normalise_state_arrays(self) -> None:
        """Validate the explicit operating-period axis used by HEN models."""

        if "period_ids" not in self.solver_arrays.arrays:
            raise ValueError("period_ids is required for HEN model setup.")
        if "period_weights" not in self.solver_arrays.arrays:
            raise ValueError("period_weights is required for HEN model setup.")

        self.period_ids = np.asarray(self.period_ids, dtype=str)
        self.period_weights = np.asarray(self.period_weights, dtype=float)
        self.N_periods = int(len(self.period_ids))
        if self.N_periods <= 0:
            raise ValueError("HEN model construction requires at least one state.")
        if len(self.period_weights) != self.N_periods:
            raise ValueError("HEN period weight count must match period_id count.")
        if not np.isfinite(self.period_weights).all():
            raise ValueError("HEN period weights must be finite.")
        self.period_weight_sum = float(np.sum(self.period_weights))
        if self.period_weight_sum <= 0.0:
            raise ValueError("HEN period weights must have a positive sum.")

        for base_name in (
            "T_h_in",
            "T_h_out",
            "f_h",
            "htc_h",
            "h_cost",
            "T_h_cont",
            "T_c_in",
            "T_c_out",
            "f_c",
            "htc_c",
            "c_cost",
            "T_c_cont",
            "T_hu_in",
            "T_hu_out",
            "htc_hu",
            "hu_cost",
            "T_hu_cont",
            "T_cu_in",
            "T_cu_out",
            "htc_cu",
            "cu_cost",
            "T_cu_cont",
        ):
            period_name = f"{base_name}_period"
            values = np.asarray(getattr(self, period_name, []), dtype=float)
            if values.size == 0:
                raise ValueError(f"{period_name} is required for HEN model setup.")
            if values.ndim != 2:
                raise ValueError(f"{period_name} must be indexed by operating period.")
            if values.shape[0] != self.N_periods:
                raise ValueError(
                    f"{period_name} has {values.shape[0]} state rows; "
                    f"expected {self.N_periods}."
                )
            setattr(self, period_name, values)
            setattr(self, base_name, values[0].copy())

    def _set_minimum_approach_temperatures(self) -> None:
        """Derive pair-specific approach limits from stream contributions."""

        self.dT_r_period = np.array(
            [
                [
                    [
                        self.T_h_cont_period[n][i] + self.T_c_cont_period[n][j]
                        for j in range(len(self.T_c_cont_period[n]))
                    ]
                    for i in range(len(self.T_h_cont_period[n]))
                ]
                for n in range(self.N_periods)
            ],
            dtype=float,
        )
        self.dT_hu_period = np.array(
            [
                [
                    (
                        self.T_hu_cont_period[n][0]
                        if len(self.T_hu_cont_period[n])
                        else self.dTmin / 2.0
                    )
                    + self.T_c_cont_period[n][j]
                    for j in range(len(self.T_c_cont_period[n]))
                ]
                for n in range(self.N_periods)
            ],
            dtype=float,
        )
        self.dT_cu_period = np.array(
            [
                [
                    self.T_h_cont_period[n][i]
                    + (
                        self.T_cu_cont_period[n][0]
                        if len(self.T_cu_cont_period[n])
                        else self.dTmin / 2.0
                    )
                    for i in range(len(self.T_h_cont_period[n]))
                ]
                for n in range(self.N_periods)
            ],
            dtype=float,
        )
        self.dT_r = self.dT_r_period[0].copy()
        self.dT_hu = self.dT_hu_period[0].copy()
        self.dT_cu = self.dT_cu_period[0].copy()

    def _recovery_approach_temperature(
        self,
        i: int,
        j: int,
        period_idx: int = 0,
    ) -> float:
        if not hasattr(self, "dT_r"):
            return float(self.dTmin)
        if hasattr(self, "dT_r_period"):
            return float(self.dT_r_period[period_idx][i][j])
        return float(self.dT_r[i][j])

    def _hot_utility_approach_temperature(
        self,
        j: int,
        period_idx: int = 0,
    ) -> float:
        if not hasattr(self, "dT_hu"):
            return float(self.dTmin)
        if hasattr(self, "dT_hu_period"):
            return float(self.dT_hu_period[period_idx][j])
        return float(self.dT_hu[j])

    def _cold_utility_approach_temperature(
        self,
        i: int,
        period_idx: int = 0,
    ) -> float:
        if not hasattr(self, "dT_cu"):
            return float(self.dTmin)
        if hasattr(self, "dT_cu_period"):
            return float(self.dT_cu_period[period_idx][i])
        return float(self.dT_cu[i])

    def _weighted_state_average(self, values: Sequence[Any]) -> Any:
        """Return ``sum_s(w_s * value_s) / sum_s(w_s)`` for GEKKO expressions."""

        return (
            sum(
                float(self.period_weights[n]) * values[n] for n in range(self.N_periods)
            )
            / self.period_weight_sum
        )

    def set_match_restrictions(self, restrictions) -> None:
        """Apply inherited topology restrictions in the source array shape."""

        if restrictions is None:
            restrictions = [None, None, None]
        z_restriction, zhu_restriction, zcu_restriction = (
            restrictions[0],
            restrictions[1],
            restrictions[2],
        )

        if z_restriction is not None:
            if isinstance(z_restriction[0][0][0], int):
                self.z_allowed = [
                    [
                        [
                            1 if z_restriction[i][j][k] > self.tol else 0
                            for k in range(self.S)
                        ]
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]
            elif isinstance(z_restriction[0][0][0], list):
                self.z_allowed = [
                    [
                        [
                            1 if z_restriction[i][j][k][0] > self.tol else 0
                            for k in range(self.S)
                        ]
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]
            elif type(z_restriction[0][0][0]).__name__ in {
                "GKVariable",
                "GKParameter",
            }:
                self.z_allowed = [
                    [
                        [
                            1 if z_restriction[i][j][k][0] > self.tol else 0
                            for k in range(self.S)
                        ]
                        for j in range(self.J)
                    ]
                    for i in range(self.I)
                ]
            else:
                raise ValueError("Invalid restriction type")
        else:
            self.z_allowed = self.z_feasible

        if zhu_restriction is not None:
            if isinstance(zhu_restriction[0], int):
                self.z_hu_allowed = [
                    1 if zhu_restriction[j] > self.tol else 0 for j in range(self.J)
                ]
            else:
                self.z_hu_allowed = [
                    1 if zhu_restriction[j][0] > self.tol else 0 for j in range(self.J)
                ]
        else:
            self.z_hu_allowed = self.z_hu_feasible

        if zcu_restriction is not None:
            if isinstance(zcu_restriction[0], int):
                self.z_cu_allowed = [
                    1 if zcu_restriction[i] > self.tol else 0 for i in range(self.I)
                ]
            else:
                self.z_cu_allowed = [
                    1 if zcu_restriction[i][0] > self.tol else 0 for i in range(self.I)
                ]
        else:
            self.z_cu_allowed = self.z_cu_feasible

    def optimise(self, print_output: bool) -> None:
        """Solve the concrete model and extract plain result data on success."""

        self.solver_run = backend.solve_gekko_model(
            self.m,
            solver_name=self.solver,
            disp=False,
            debug=0,
        )
        self.solve_time = self.solver_run.solve_time

        if self.solver_run.failure_reason is not None:
            self.mSuccess = 0
            logger.error("[Failed] [model: %s] [path: %s]", self.name, self.m._path)
        elif self.m.options.SOLVESTATUS == 1:
            if self.m.options.objfcnval + self.tol < 0:
                self.mSuccess = 0
                self.solver_run = backend.SolverRun(
                    name=self.solver_run.name,
                    extension=self.solver_run.extension,
                    status=self.solver_run.status,
                    objective_value=self.solver_run.objective_value,
                    solve_time=self.solver_run.solve_time,
                    failure_reason="negative objective value",
                )
                logger.error(
                    "[Failed] [model: %s] [path: %s]",
                    self.name,
                    self.m._path,
                )
            else:
                self.mSuccess = self.m.options.SOLVESTATUS
                logger.info(
                    "[Success] [model: %s] [path: %s]",
                    self.name,
                    self.m._path,
                )
        else:
            self.mSuccess = self.m.options.SOLVESTATUS
            logger.error(
                "[Failed] [model: %s] [path: %s] [status: %s]",
                self.name,
                self.m._path,
                self.m.options.SOLVESTATUS,
            )

        if self.mSuccess:
            self.get_post_process()
            if print_output:
                self.output_to_cmd_line()

    def output_to_cmd_line(self) -> None:
        """Emit the same solved-array diagnostics as the source base model."""

        if self.mSuccess != 1:
            return
        logger.info("Successful Solve.Path %s name %s", self.m._path, self.name)
        logger.info("Objective 0: %s", self.m.options.objfcnval)
        logger.info("Objective 1: %s", self.TAC)
        logger.info("Total Units: %s", self.n_units)
        logger.info("Total Recovery Units: %s", self.n_recovery_units)
        logger.info("T hot: %s", self.T_h)
        logger.info("T cold: %s", self.T_c)
        logger.info("theta 1: %s", self.theta_1)
        logger.info("theta 2: %s", self.theta_2)
        logger.info("Heat recovery Q: %s", self.Q_r)
        logger.info("Heat recovery z: %s", self.z)
        logger.info("Heat recovery LMTD: %s", self.LMTD_r)
        logger.info("Heat recovery area: %s", self.area_r)
        logger.info("Q_r total: %s", self.Q_r_total)
        logger.info("Cold utility Q: %s", self.Q_c)
        logger.info("Cold utility z: %s", self.z_cu)
        logger.info("Cold utility LMTD: %s", self.LMTD_cu)
        logger.info("Cold utility area: %s", self.area_cu)
        logger.info("Q_cu total: %s", self.Q_cu_total)
        logger.info("Hot utility Q: %s", self.Q_h)
        logger.info("Hot utility z: %s", self.z_hu)
        logger.info("Hot utility LMTD: %s", self.LMTD_hu)
        logger.info("Hot utility area: %s", self.area_hu)
        logger.info("Q_hu total: %s", self.Q_hu_total)
