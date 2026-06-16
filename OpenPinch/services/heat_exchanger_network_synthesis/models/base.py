"""Base setup for migrated HEN equation kernels."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import numpy as np

from ..array_adapter import PreparedSolverArrays
from . import backend

logger = logging.getLogger(__name__)


class BaseHeatExchangerNetworkModel(ABC):
    """Shared mutable state for future PDM/TDM/ESM equation model slices.

    The constructor mirrors the source OpenHENS solver defaults, but it accepts
    OpenPinch-prepared solver arrays instead of a CSV path. Concrete equation
    classes are intentionally left to later migration slices.
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
        solver_options: list[str] | None = None,
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
        self.solver_options = list(solver_options or [])

        self.solve_time = None
        self.solver_run = None

        self.setup_model()
        self.setup()

    def setup_model(self) -> None:
        """Create and configure the GEKKO model behind optional guards."""

        self.m = backend.create_gekko_model(remote=False)
        self.mSuccess: int = 0
        self.solver_run = backend.configure_gekko_solver(self.m, self.solver)

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
        self.htc_hu = np.array([], dtype=float)
        self.hu_exp = np.array([], dtype=float)

        self.cu_cost = np.array([], dtype=float)
        self.cu_unit_cost = np.array([], dtype=float)
        self.cu_coeff = np.array([], dtype=float)
        self.T_cu_in = np.array([], dtype=float)
        self.T_cu_out = np.array([], dtype=float)
        self.htc_cu = np.array([], dtype=float)
        self.cu_exp = np.array([], dtype=float)

        self.unit_cost = np.array([], dtype=float)
        self.A_coeff = np.array([], dtype=float)
        self.A_exp = np.array([], dtype=float)

    def get_model_parameters_from_solver_arrays(self) -> None:
        """Populate model attributes from the OpenPinch private array adapter."""

        for name, values in self.solver_arrays.arrays.items():
            setattr(self, name, np.array(values, copy=True))

    def set_match_restrictions(self, restrictions) -> None:
        """Apply inherited topology restrictions in the source array shape."""

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
                    1 if zhu_restriction[j] > self.tol else 0
                    for j in range(self.J)
                ]
            else:
                self.z_hu_allowed = [
                    1 if zhu_restriction[j][0] > self.tol else 0
                    for j in range(self.J)
                ]
        else:
            self.z_hu_allowed = self.z_hu_feasible

        if zcu_restriction is not None:
            if isinstance(zcu_restriction[0], int):
                self.z_cu_allowed = [
                    1 if zcu_restriction[i] > self.tol else 0
                    for i in range(self.I)
                ]
            else:
                self.z_cu_allowed = [
                    1 if zcu_restriction[i][0] > self.tol else 0
                    for i in range(self.I)
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
