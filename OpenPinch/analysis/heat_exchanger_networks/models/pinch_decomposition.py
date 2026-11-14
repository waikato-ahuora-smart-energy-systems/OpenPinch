"""Pinch-decomposition heat-exchanger-network model coordinator."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from ..solver import backend
from ..solver.arrays import PreparedSolverArrays
from ..solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from ..solver.pinch_design_decomposition import PinchDesignDecomposition
from ._pinch_design import amalgamation as _amalgamation
from ._pinch_design import equations as _equations
from ._pinch_design import postprocess as _postprocess
from ._pinch_design import preprocessing as _preprocessing
from .base import BaseHeatExchangerNetworkModel
from .stagewise import StageWiseModel

logger = logging.getLogger(__name__)


class PinchDecompModel(BaseHeatExchangerNetworkModel):
    """Source-compatible private PDM slice for one pinch side."""

    def __init__(
        self,
        *,
        name: str,
        framework: Literal["PDM"],
        solver: Literal["couenne", "highs", "ipopt-pyomo"],
        solver_arrays: PreparedSolverArrays,
        dTmin: float,
        z_restriction: list | None,
        min_dqda: float,
        minimisation_goal: Literal[
            "hot utility",
            "cold utility",
            "total utility",
            "heat recovery",
            "min units",
        ],
        non_isothermal_model: bool,
        integers: bool,
        tol: float,
        pinch_loc: Literal["above", "below"],
        pinch_decomposition: PinchDesignDecomposition,
        stage_selection: Literal["automated"] | list[int] | tuple[int, int],
        solver_options: Mapping[str, Any] | Sequence[str] | None = None,
        utility_cost_cap: float | None = None,
    ) -> None:
        self.pinch_loc = pinch_loc
        # With a cap, "min units" minimises the unit count alone, keeping
        # utility cost at or below the cap instead of adding it in.
        self.utility_cost_cap = utility_cost_cap
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
        self._solve_nonlinear_models_with_couenne()

    def used_units(self) -> dict[str, tuple[int, ...]]:
        """Return the solved side's used recovery, heater and cooler units."""

        return _equations.used_units(self)

    def require_units(self, at_least: int) -> None:
        """Require at least ``at_least`` units in this unsolved side."""

        _equations.require_units(self, at_least)

    def exclude_units(self, pattern: dict[str, tuple[int, ...]]) -> None:
        """Cut ``pattern`` from this unsolved side with a no-good constraint."""

        _equations.exclude_units(self, pattern)

    def _solve_nonlinear_models_with_couenne(self) -> None:
        """Send a model HiGHS cannot take to Couenne, or SCIP without it.

        HiGHS options do not apply to the new solver, except a time limit,
        which is carried over. Couenne is an optional extra; SCIP comes with
        the synthesis extra.
        """

        if self.solver != "highs" or self.m.is_linear():
            return
        solver = "couenne"
        try:
            backend.require_solver_backend(solver)
        except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
            solver = backend.FALLBACK_SOLVER
        logger.warning(
            "[%s] The pinch design model is not linear, so %s solves it "
            "instead of HiGHS.",
            self.name,
            solver,
        )
        options = self.solver_options
        limit = (
            options.get(backend.TIME_LIMIT_OPTIONS["highs"])
            if isinstance(options, dict)
            else None
        )
        self.solver = solver
        self.solver_options = (
            None if limit is None else {backend.TIME_LIMIT_OPTIONS[solver]: limit}
        )
        self.solver_run = backend.configure_pyomo_solver(
            self.m, self.solver, solver_options=self.solver_options
        )

    def get_model_parameters_from_solver_arrays(self) -> None:
        super().get_model_parameters_from_solver_arrays()
        self.T_h_in_OG = self.T_h_in.copy()
        self.T_h_out_OG = self.T_h_out.copy()
        self.T_c_in_OG = self.T_c_in.copy()
        self.T_c_out_OG = self.T_c_out.copy()

    def calculate_pinch(self) -> None:
        """Read target values from the private OpenPinch decomposition."""

        return _preprocessing.calculate_pinch(self)

    def set_preprocessing(self) -> None:
        """Pre-process PDM superstructure parameters."""

        return _preprocessing.set_preprocessing(self)

    def set_stage_wise_superstructure(self) -> None:
        """Create PDM variables, constraints, and binaries."""

        return _equations.set_stage_wise_superstructure(self)

    def _set_multiperiod_stage_wise_superstructure(self) -> None:
        """Delegate _set_multiperiod_stage_wise_superstructure to its owner helper."""

        return _equations._set_multiperiod_stage_wise_superstructure(self)

    def set_obj(self) -> None:
        """Attach PDM objective expressions."""

        return _equations.set_obj(self)

    def get_post_process(self) -> None:
        """Extract source PDM side arrays after a successful solve."""

        return _postprocess.get_post_process(self)

    def _get_multiperiod_post_process(self) -> None:
        """Delegate _get_multiperiod_post_process to its owner helper."""

        return _postprocess._get_multiperiod_post_process(self)

    def _active_binary_value(self, value) -> float:
        """Delegate _active_binary_value to its owner helper."""

        return _postprocess._active_binary_value(self, value)

    def amalgamate_networks(
        self, *, below_case: "PinchDecompModel", above_case: "PinchDecompModel"
    ) -> StageWiseModel:
        """Amalgamate solved above/below-pinch side models into one network."""

        return _amalgamation.amalgamate_networks(
            self, below_case=below_case, above_case=above_case
        )


__all__ = ["PinchDecompModel"]
