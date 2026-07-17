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

    def _apply_segment_recovery_areas(self, q_r) -> None:
        "Replace aggregate-CP recovery areas with ordered local slice totals."
        return _area._apply_segment_recovery_areas(self, q_r)

    def _apply_segment_utility_areas(self, q_h, q_c) -> None:
        "Use local process segments for hot- and cold-utility area totals."
        return _area._apply_segment_utility_areas(self, q_h, q_c)

    def _segment_exact_dqda(
        self,
        *,
        period_index: int,
        hot_parent_index: int,
        cold_parent_index: int,
        duty: float,
        hot_inlet_temperature: float,
        cold_inlet_temperature: float,
    ) -> float | None:
        "Return a local numerical dQ/dA from ordered segment-summed area."
        return _area._segment_exact_dqda(
            self,
            period_index=period_index,
            hot_parent_index=hot_parent_index,
            cold_parent_index=cold_parent_index,
            duty=duty,
            hot_inlet_temperature=hot_inlet_temperature,
            cold_inlet_temperature=cold_inlet_temperature,
        )

    def _register_piecewise_mapping(self, mapping) -> None:
        return _piecewise._register_piecewise_mapping(self, mapping)

    def _utility_is_segmented(self, side: str) -> bool:
        return _piecewise._utility_is_segmented(self, side)

    def _utility_cost_expression(
        self,
        side: str,
        period_index: int,
        heat_duty,
        *,
        name: str,
    ):
        "Return the flat or exact piecewise utility-cost solver expression."
        return _piecewise._utility_cost_expression(
            self, side, period_index, heat_duty, name=name
        )

    def _utility_cost_value(
        self,
        side: str,
        period_index: int,
        heat_duty: float,
    ) -> float:
        "Return exact solved utility cost for reporting and verification."
        return _piecewise._utility_cost_value(self, side, period_index, heat_duty)

    def _update_piecewise_active_segments(self) -> bool:
        return _piecewise._update_piecewise_active_segments(self)

    def _set_piecewise_stage_heat_coordinates(self) -> None:
        "Add parent cumulative-Q balances and ordered T(Q) mappings by period."
        return _piecewise._set_piecewise_stage_heat_coordinates(self)

    def _set_segmented_utility_capacity_constraints(self) -> None:
        "Bound selected utility load by each explicit ordered profile."
        return _piecewise._set_segmented_utility_capacity_constraints(self)

    def _set_piecewise_utility_outlet_states(self) -> None:
        "Map aggregate utility duty to outlet temperature and local ``dt_cont``."
        return _piecewise._set_piecewise_utility_outlet_states(self)

    def _hot_parent_segmented(self, index: int) -> bool:
        return _piecewise._hot_parent_segmented(self, index)

    def _cold_parent_segmented(self, index: int) -> bool:
        return _piecewise._cold_parent_segmented(self, index)

    def _solver_parent_is_segmented(self, side: str, index: int) -> bool:
        return _piecewise._solver_parent_is_segmented(self, side, index)

    def _parent_profile_duty(
        self,
        side: str,
        period_index: int,
        parent_index: int,
        supply_temperature: float,
        target_temperature: float,
        aggregate_cp: float,
    ) -> float:
        return _piecewise._parent_profile_duty(
            self,
            side,
            period_index,
            parent_index,
            supply_temperature,
            target_temperature,
            aggregate_cp,
        )

    def _recovery_heat_upper_bound(
        self,
        *,
        period_index: int,
        hot_index: int,
        cold_index: int,
        hot_total_duty: float,
        cold_total_duty: float,
        hot_cp: float,
        cold_cp: float,
    ) -> float:
        return _piecewise._recovery_heat_upper_bound(
            self,
            period_index=period_index,
            hot_index=hot_index,
            cold_index=cold_index,
            hot_total_duty=hot_total_duty,
            cold_total_duty=cold_total_duty,
            hot_cp=hot_cp,
            cold_cp=cold_cp,
        )

    def _set_piecewise_match_outlet_equations(self) -> None:
        "Map non-isothermal branch outlets through parent heat coordinates."
        return _piecewise._set_piecewise_match_outlet_equations(self)

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

    def _utility_max_temperature_contribution(
        self,
        side: str,
        period_idx: int,
    ) -> float:
        return _approach._utility_max_temperature_contribution(self, side, period_idx)

    def _set_multiperiod_utility_approach_equations(self) -> None:
        "Constrain both utility terminals with local segment contributions."
        return _approach._set_multiperiod_utility_approach_equations(self)

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
