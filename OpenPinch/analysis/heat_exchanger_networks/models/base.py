"""Base setup for migrated heat exchanger network equation kernels."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from ..solver.arrays import PreparedSolverArrays
from ._base import alpha as _alpha
from ._base import approach as _approach
from ._base import area as _area
from ._base import execution as _execution
from ._base import parameters as _parameters
from ._base import piecewise as _piecewise

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
            "cold utility",
            "total utility",
            "utility costs",
            "heat recovery",
            "total cost",
            "variable total cost",
            "dQ/dA obj",
            "min units",
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
        self._piecewise_active_mappings: list[dict[str, Any]] = []

        self.setup_model()
        self.setup()

    def setup_model(self) -> None:
        "Create and configure the GEKKO model behind optional guards."
        return _execution.setup_model(self)

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
        return _execution._solver_value(self, value)

    def _set_value(
        self, variable: Any, value: float, *, brackets: bool = False
    ) -> None:
        "Assign GEKKO values while preserving source bound-clamping behavior."
        return _execution._set_value(self, variable, value, brackets=brackets)

    def _post_process_lmtd(
        self,
        delta_1: float,
        delta_2: float,
        active: float,
        *,
        formula_allowed: bool,
        fallback_delta: float | None = None,
    ) -> float:
        """Return source-compatible post-process LMTD."""
        return _area._post_process_lmtd(
            self,
            delta_1,
            delta_2,
            active,
            formula_allowed=formula_allowed,
            fallback_delta=fallback_delta,
        )

    def _utility_cost_value(
        self,
        side: str,
        period_index: int,
        heat_duty: float,
    ) -> float:
        "Return exact solved utility cost for reporting and verification."
        return _piecewise._utility_cost_value(self, side, period_index, heat_duty)

    def get_alpha_values(self) -> list:
        "Calculate source alpha flow-on values in a post-optimisation solve."
        return _alpha.get_alpha_values(self)

    def set_alpha_dqda_equations(
        self,
        *,
        m: Any | None = None,
        postoptimisation: bool = False,
    ) -> None:
        "Move the source alpha and dQ/dA equations without changing formulas."
        return _alpha.set_alpha_dqda_equations(
            self, m=m, postoptimisation=postoptimisation
        )

    def set_blank_input_parameters(self) -> None:
        "Initialize the solver-array attributes expected by source equations."
        return _parameters.set_blank_input_parameters(self)

    def get_model_parameters_from_solver_arrays(self) -> None:
        "Populate model attributes from the OpenPinch private array adapter."
        return _parameters.get_model_parameters_from_solver_arrays(self)

    def _normalise_state_arrays(self) -> None:
        "Validate the explicit operating-period axis used by HEN models."
        return _parameters._normalise_state_arrays(self)

    def _set_minimum_approach_temperatures(self) -> None:
        "Derive pair-specific approach limits from stream contributions."
        return _approach._set_minimum_approach_temperatures(self)

    def _recovery_approach_temperature(
        self,
        i: int,
        j: int,
        period_idx: int = 0,
    ) -> float:
        return _approach._recovery_approach_temperature(self, i, j, period_idx)

    def _hot_utility_inlet_approach_temperature(
        self,
        j: int,
        period_idx: int = 0,
    ) -> float:
        return _approach._hot_utility_inlet_approach_temperature(self, j, period_idx)

    def _hot_utility_outlet_approach_temperature(
        self,
        j: int,
        period_idx: int = 0,
        heat_duty: float | None = None,
    ):
        return _approach._hot_utility_outlet_approach_temperature(
            self, j, period_idx, heat_duty
        )

    def _cold_utility_inlet_approach_temperature(
        self,
        i: int,
        period_idx: int = 0,
    ) -> float:
        return _approach._cold_utility_inlet_approach_temperature(self, i, period_idx)

    def _cold_utility_outlet_approach_temperature(
        self,
        i: int,
        period_idx: int = 0,
        heat_duty: float | None = None,
    ):
        return _approach._cold_utility_outlet_approach_temperature(
            self, i, period_idx, heat_duty
        )

    def _utility_outlet_temperature_contribution(
        self,
        side: str,
        period_idx: int,
        match_index: int,
        heat_duty: float | None = None,
    ):
        return _approach._utility_outlet_temperature_contribution(
            self, side, period_idx, match_index, heat_duty
        )

    def _utility_solved_outlet_temperature(
        self,
        side: str,
        period_idx: int,
        match_index: int,
        heat_duty: float | None = None,
    ):
        return _approach._utility_solved_outlet_temperature(
            self, side, period_idx, match_index, heat_duty
        )

    def _weighted_state_average(self, values: Sequence[Any]) -> Any:
        "Return ``sum_s(w_s * value_s) / sum_s(w_s)`` for GEKKO expressions."
        return _approach._weighted_state_average(self, values)

    def set_match_restrictions(self, restrictions) -> None:
        "Apply inherited topology restrictions in the source array shape."
        return _approach.set_match_restrictions(self, restrictions)

    def optimise(self, print_output: bool) -> None:
        """Delegate solver execution with explicit model state."""
        _execution.optimise(self, print_output)

    def output_to_cmd_line(self) -> None:
        "Emit the same solved-array diagnostics as the source base model."
        return _execution.output_to_cmd_line(self)
