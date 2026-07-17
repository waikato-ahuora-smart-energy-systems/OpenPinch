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
from ._stagewise.evolution import (
    _EvolutionBranchState,
    _EvolutionCandidateSpec,
)
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

    def _set_multiperiod_isothermal_approach_equations(self) -> None:
        """Build multiperiod isothermal approach equations."""

        return _equations._set_multiperiod_isothermal_approach_equations(self)

    def _set_multiperiod_non_isothermal_equations(self) -> None:
        """Delegate _set_multiperiod_non_isothermal_equations to its owner helper."""

        return _equations._set_multiperiod_non_isothermal_equations(self)

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

    def _set_multiperiod_initial_values(self, init_solution, *, brackets: bool) -> None:
        """Delegate _set_multiperiod_initial_values to its owner helper."""

        return _warm_start._set_multiperiod_initial_values(
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

    def _get_source_net_benefit_evolution(self, *, print_output: bool, max_depth: int):
        """Run the original OpenHENS single-path add/remove evolution search."""

        return _evolution._get_source_net_benefit_evolution(
            self, print_output=print_output, max_depth=max_depth
        )

    def _select_source_best_candidate(
        self, current_model, model_minus_one, model_plus_one
    ):
        """Select the next OpenHENS tier-1 evolution candidate by success and TAC."""

        return _evolution._select_source_best_candidate(
            self, current_model, model_minus_one, model_plus_one
        )

    def _evolution_candidate_specs(
        self,
        frontier: Sequence[_EvolutionBranchState],
        *,
        unit: int,
        n_ad_branches: int,
        n_rm_branches: int,
    ) -> list[_EvolutionCandidateSpec]:
        """Delegate _evolution_candidate_specs to its owner helper."""

        return _evolution._evolution_candidate_specs(
            self,
            frontier,
            unit=unit,
            n_ad_branches=n_ad_branches,
            n_rm_branches=n_rm_branches,
        )

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
        """Delegate _evolution_candidate_spec to its owner helper."""

        return _evolution._evolution_candidate_spec(
            self,
            kind=kind,
            unit=unit,
            branch_index=branch_index,
            rank=rank,
            prev_case=prev_case,
            position=position,
            z_value=z_value,
            seen_signatures=seen_signatures,
        )

    def _solve_evolution_candidates(
        self,
        specs: Sequence[_EvolutionCandidateSpec],
        *,
        print_output: bool,
        max_parallel: int,
    ) -> list[tuple[_EvolutionCandidateSpec, Any]]:
        """Delegate _solve_evolution_candidates to its owner helper."""

        return _evolution._solve_evolution_candidates(
            self, specs, print_output=print_output, max_parallel=max_parallel
        )

    def _solve_evolution_candidate(
        self, spec: _EvolutionCandidateSpec, print_output: bool
    ):
        """Delegate _solve_evolution_candidate to its owner helper."""

        return _evolution._solve_evolution_candidate(self, spec, print_output)

    def _evolution_branch_label(self, spec: _EvolutionCandidateSpec) -> str:
        """Delegate _evolution_branch_label to its owner helper."""

        return _evolution._evolution_branch_label(self, spec)

    def _select_best_candidate(self, current_model, model_minus_one, model_plus_one):
        """Select the source plus/minus evolution candidate for the next step."""

        return _evolution._select_best_candidate(
            self, current_model, model_minus_one, model_plus_one
        )

    def _update_with_best_model(self, best_model) -> None:
        """Adopt the selected evolved topology while retaining this model object."""

        return _evolution._update_with_best_model(self, best_model)

    def get_n_minus_one_evolution(self, print_output: bool, unit: int, prev_case):
        """Build and solve the source minus-one topology evolution candidate."""

        return _evolution.get_n_minus_one_evolution(self, print_output, unit, prev_case)

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

        return _evolution._build_and_solve_n_minus_one_evolution(
            self,
            print_output=print_output,
            unit=unit,
            prev_case=prev_case,
            position=position,
            z_allowed_removed=z_allowed_removed,
            branch_label=branch_label,
        )

    def get_n_plus_one_evolution(self, print_output: bool, unit: int, prev_case):
        """Build and solve the source plus-one topology evolution candidate."""

        return _evolution.get_n_plus_one_evolution(self, print_output, unit, prev_case)

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

        return _evolution._build_and_solve_n_plus_one_evolution(
            self,
            print_output=print_output,
            unit=unit,
            prev_case=prev_case,
            position=position,
            z_allowed_added=z_allowed_added,
            branch_label=branch_label,
        )

    def _z_allowed_with_candidate(
        self, prev_case, *, position: Sequence[int], value: int
    ) -> list:
        """Delegate _z_allowed_with_candidate to its owner helper."""

        return _evolution._z_allowed_with_candidate(
            self, prev_case, position=position, value=value
        )

    def _set_recovery_binary_value(
        self, z_values: list, position: tuple[int, int, int], value: int
    ) -> None:
        """Delegate _set_recovery_binary_value to its owner helper."""

        return _evolution._set_recovery_binary_value(self, z_values, position, value)

    def _topology_signature_from_z(
        self, z_values: Sequence[Sequence[Sequence[Any]]]
    ) -> tuple[tuple[int, int, int], ...]:
        """Delegate _topology_signature_from_z to its owner helper."""

        return _evolution._topology_signature_from_z(self, z_values)

    def _active_binary_value(self, value) -> float:
        """Delegate _active_binary_value to its owner helper."""

        return _evolution._active_binary_value(self, value)

    def set_obj(self) -> None:
        """Attach source StageWise objective expressions unchanged."""

        return _objectives.set_obj(self)

    def _set_total_cost_objective(self) -> None:
        """Delegate _set_total_cost_objective to its owner helper."""

        return _objectives._set_total_cost_objective(self)

    def _set_source_total_cost_objective(self) -> None:
        """Delegate _set_source_total_cost_objective to its owner helper."""

        return _objectives._set_source_total_cost_objective(self)

    def _set_multiperiod_total_cost_objective(self) -> None:
        """Delegate _set_multiperiod_total_cost_objective to its owner helper."""

        return _objectives._set_multiperiod_total_cost_objective(self)

    def get_post_process(self) -> None:
        """Extract source post-process arrays after a successful solve."""

        return _postprocess.get_post_process(self)

    def _get_multiperiod_post_process(self) -> None:
        """Delegate _get_multiperiod_post_process to its owner helper."""

        return _postprocess._get_multiperiod_post_process(self)

    def _weighted_numeric_average(self, values: Sequence[float]) -> float:
        """Delegate _weighted_numeric_average to its owner helper."""

        return _postprocess._weighted_numeric_average(self, values)

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
