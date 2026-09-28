"""StageWise heat-exchanger-network model coordinator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from ..solver.arrays import PreparedSolverArrays
from ._stagewise import equations as _equations
from ._stagewise import evolution as _evolution
from ._stagewise import objectives as _objectives
from ._stagewise import postprocess as _postprocess
from ._stagewise import setup as _setup
from ._stagewise import verification as _verification
from ._stagewise import warm_start as _warm_start
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

        return _setup.set_preprocessing(self)

    def set_stage_wise_superstructure(self) -> None:
        """Create StageWise variables, constraints, and binaries."""

        return _equations.set_stage_wise_superstructure(self)

    def _set_multiperiod_stage_wise_superstructure(self) -> None:
        """Create shared topology with state-indexed operating variables."""

        return _equations._set_multiperiod_stage_wise_superstructure(self)

    def set_dqda_equations(self) -> None:
        """Apply the source TDM minimum dQ/dA restriction."""

        return _equations.set_dqda_equations(self)

    def set_initial_values_for_variables(
        self, init_solution, *, brackets: bool = False
    ) -> None:
        """Warm-start this model from a solved parent model."""

        return _warm_start.set_initial_values_for_variables(
            self, init_solution, brackets=brackets
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

        return _evolution.get_net_benefit_evolution(
            self,
            print_output,
            max_depth,
            n_ad_branches,
            n_rm_branches,
            max_parallel,
            no_improvement_patience,
        )

    def get_n_minus_one_evolution(self, print_output: bool, unit: int, prev_case):
        """Build and solve the source minus-one topology evolution candidate."""

        return _evolution.get_n_minus_one_evolution(self, print_output, unit, prev_case)

    def get_n_plus_one_evolution(self, print_output: bool, unit: int, prev_case):
        """Build and solve the source plus-one topology evolution candidate."""

        return _evolution.get_n_plus_one_evolution(self, print_output, unit, prev_case)

    def _active_binary_value(self, value) -> float:
        """Delegate _active_binary_value to its owner helper."""

        return _evolution._active_binary_value(self, value)

    def set_obj(self) -> None:
        """Attach source StageWise objective expressions unchanged."""

        return _objectives.set_obj(self)

    def get_post_process(self) -> None:
        """Extract source post-process arrays after a successful solve."""

        return _postprocess.get_post_process(self)

    def _get_multiperiod_post_process(self) -> None:
        """Delegate _get_multiperiod_post_process to its owner helper."""

        return _postprocess._get_multiperiod_post_process(self)

    def get_lowest_benefit_HX(self) -> list[list[int]]:
        """Return the active exchanger with the lowest source net benefit."""

        return _postprocess.get_lowest_benefit_HX(self)

    def get_lowest_benefit_HX_candidates(self, limit: int) -> list[list[int]]:
        """Return active exchangers sorted by ascending source net benefit."""

        return _postprocess.get_lowest_benefit_HX_candidates(self, limit)

    def get_max_benefit_HX(self) -> list[list[int]]:
        """Return the inactive feasible exchanger with the highest alpha-dQ/dA."""

        return _postprocess.get_max_benefit_HX(self)

    def get_max_benefit_HX_candidates(self, limit: int) -> list[list[int]]:
        """Return inactive feasible exchangers sorted by descending alpha-dQ/dA."""

        return _postprocess.get_max_benefit_HX_candidates(self, limit)

    def verify(self) -> tuple[bool, list[str]]:
        """Run the source solution checks used by topology evolution."""

        return _verification.verify(self)


__all__ = ["StageWiseModel"]
