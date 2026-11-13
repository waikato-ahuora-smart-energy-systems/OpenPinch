"""Internal heat exchanger network problem shell behind the synthesis service."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Any, Literal, Mapping, Sequence

from ....contracts.synthesis.result import HeatExchangerNetworkSynthesisResult
from ....domain.enums import HeatExchangerNetworkDesignMethod
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ..extraction.service import (
    extract_heat_exchanger_network,
    extract_network_synthesis_result,
)
from ..solver import backend
from ..solver.arrays import PreparedSolverArrays
from ..solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
)
from ..solver.pinch_design_decomposition import PinchDesignDecomposition
from ._base.execution import _assign_value, _set_value, _value
from ._pinch_design.equations import _unit_index_sets as _equations_unit_indices
from .stage_packing import add_recovery_stage_packing_constraints
from .stagewise import StageWiseModel

FrameworkName = Literal["PDM", "TDM", "ESM"]

logger = logging.getLogger(__name__)


class ModelSliceUnavailableError(NotImplementedError):
    """Raised when a later migration slice is asked to run too early."""


@dataclass
class InternalHeatExchangerNetworkProblem:
    """OpenPinch-owned replacement for source ``HeatExchangerNetworkProblem``.

    This object is private solver state. HENS-07 constructs moved PDM and
    StageWise models from OpenPinch-prepared solver arrays and emits OpenPinch
    network/result data at the extraction boundary. HENS-08 still owns
    stage reduction and topology evolution.
    """

    solver_arrays: PreparedSolverArrays
    name: str = ""
    framework: FrameworkName = "TDM"
    solver: str = "couenne"
    dTmin: float = 0.1
    # The synthesis task's dt_cont multiplier, recorded on extracted networks.
    dt_cont_multiplier: float | None = None
    min_dqda: float = 0.0
    z_restriction: list | None = None
    minimisation_goal: str = "hot utility"
    non_isothermal_model: bool = False
    integers: bool = True
    parent: "InternalHeatExchangerNetworkProblem | None" = None
    tol: float = 1e-3
    solver_options: Mapping[str, Any] | Sequence[str] | None = None
    stage_selection: str | list[str] = "automated"
    stages: int | None = None
    synthesis_task_id: str | None = None
    pinch_decompositions: Mapping[str, PinchDesignDecomposition] | None = None
    # PDM only: how many extra networks to collect at the fewest units and at
    # one more unit. Each becomes a problem in ``pdm_variants``.
    pdm_network_limits: tuple[int, int] = (0, 0)
    pdm_variants: list["InternalHeatExchangerNetworkProblem"] = field(
        default_factory=list, init=False
    )
    # PDM only: above 0, the heuristic pinch design method builds up to this
    # many networks instead of the MILP sides; extra ones go in
    # ``pdm_variants``.
    heuristic_network_limit: int = 0
    heuristic_report: dict[str, Any] = field(default_factory=dict, init=False)

    def load_model(
        self,
        *,
        model_factories: Mapping[str, Any] | None = None,
    ) -> None:
        """Construct the private PDM or StageWise model for this task."""

        self._model_factories = model_factories
        self.args = {
            "name": self.name,
            "framework": self.framework,
            "solver": self.solver,
            "solver_arrays": self.solver_arrays,
            "dTmin": self.dTmin,
            "min_dqda": self.min_dqda,
            "z_restriction": self.z_restriction,
            "minimisation_goal": self.minimisation_goal,
            "non_isothermal_model": self.non_isothermal_model,
            "integers": self.integers,
            "tol": self.tol,
            "solver_options": self.solver_options,
        }
        if self.framework == "PDM":
            self._build_pdm(model_factories=model_factories)
            return
        self._build_stage_wise(model_factories=model_factories)

    def get_solution(
        self,
        *,
        print_output: bool = True,
        evolution: bool | None = None,
        evolution_n_ad_branches: int = 1,
        evolution_n_rm_branches: int = 1,
        evolution_max_parallel: int = 1,
        evolution_no_improvement_patience: int | None = None,
        evolution_beam_width: int = 4,
        model_factories: Mapping[str, Any] | None = None,
    ) -> Any:
        """Load, solve, and return the private solved model for this task."""

        unreduced_case = None
        try:
            self.load_model(model_factories=model_factories)
            if self.framework == "PDM" and self.heuristic_network_limit > 0:
                self._solve_heuristic_pdm()
            elif self.framework == "PDM":
                self._solve_pdm(print_output=print_output)
            else:
                self.case.optimise(print_output=print_output)
            if self.framework in {"PDM", "TDM"}:
                unreduced_case = self.case
                self.case = self.remove_unused_stages(self.case)
            if (
                self.framework == "PDM"
                and self.heuristic_network_limit <= 0
                and self.case.mSuccess == 1
            ):
                self.pdm_variants = self._solve_pdm_variants(model_factories)
            if evolution:
                evolved = self.case.get_net_benefit_evolution(
                    print_output=print_output,
                    n_ad_branches=evolution_n_ad_branches,
                    n_rm_branches=evolution_n_rm_branches,
                    max_parallel=evolution_max_parallel,
                    no_improvement_patience=evolution_no_improvement_patience,
                    beam_width=evolution_beam_width,
                )
                if evolved is not None:
                    self.case = evolved
            return self.case
        except ValueError as exc:
            self.solution_failure_reason = str(exc)
            return None
        finally:
            # Results live on the model variables now; drop every run
            # directory this solve created, including unsolved PDM sides and
            # the never-solved amalgamated network.
            for model in (
                getattr(self, "above", None),
                getattr(self, "below", None),
                unreduced_case,
                getattr(self, "case", None),
            ):
                _cleanup_model(model)

    def extract_network(self, *, run_id: str) -> HeatExchangerNetwork:
        """Convert the solved private case into an OpenPinch network result."""

        return extract_heat_exchanger_network(
            self.case,
            self.solver_arrays,
            run_id=run_id,
            task_id=self.synthesis_task_id,
            method=self._result_method(),
            dt_cont_multiplier=self.dt_cont_multiplier,
            stage_count=getattr(
                self.case, "stages", getattr(self.case, "S", self.stages)
            ),
        )

    def extract_result(
        self,
        *,
        run_id: str,
        problem_id: str | None = None,
        workspace_variant: str | None = None,
        period_id: str | None = None,
    ) -> HeatExchangerNetworkSynthesisResult:
        """Return the serializable result data for the service boundary."""

        return extract_network_synthesis_result(
            self.case,
            self.solver_arrays,
            run_id=run_id,
            task_id=self.synthesis_task_id,
            problem_id=problem_id,
            workspace_variant=workspace_variant,
            period_id=period_id,
            solver_name=self.solver,
            method=self._result_method(),
            stage_count=getattr(self.case, "S", self.stages),
        )

    def _result_method(self) -> str:
        """The method extracted networks are labelled with."""

        if self.framework == "PDM" and self.heuristic_network_limit > 0:
            return HeatExchangerNetworkDesignMethod.HeuristicPinchDesign.value
        return self.framework

    def _build_pdm(
        self,
        *,
        model_factories: Mapping[str, Any] | None,
    ) -> None:
        """Construct source PDM above/below models with private decompositions."""

        decompositions = dict(self.pinch_decompositions or {})
        if "above" not in decompositions or "below" not in decompositions:
            raise ValueError(
                "PDM construction requires above and below pinch decomposition "
                "records from the OpenPinch targeting boundary."
            )
        factory = self._model_factory(
            model_factories,
            "pinch_design_method",
            default="PinchDecompModel",
        )
        base_args = dict(self.args)
        self.above = factory(
            **(
                base_args
                | {
                    "name": f"above pinch {self.dTmin}",
                    "pinch_loc": "above",
                    "minimisation_goal": "hot utility",
                    "pinch_decomposition": decompositions["above"],
                    "stage_selection": self.stage_selection,
                }
            )
        )
        self.below = factory(
            **(
                base_args
                | {
                    "name": f"below pinch {self.dTmin}",
                    "pinch_loc": "below",
                    "minimisation_goal": "cold utility",
                    "pinch_decomposition": decompositions["below"],
                    "stage_selection": self.stage_selection,
                }
            )
        )

    def _build_stage_wise(
        self,
        *,
        model_factories: Mapping[str, Any] | None,
    ) -> None:
        """Construct a TDM/ESM model from explicit or parent stage state."""

        stages = self.stages
        if stages is None and self.parent is not None:
            stages = getattr(
                self.parent.case,
                "stages",
                getattr(self.parent.case, "S", None),
            )
        if stages is None:
            raise ValueError(
                "Stage-wise models require an explicit stage count or a parent "
                "solution."
            )
        factory = self._model_factory(
            model_factories,
            "stagewise",
            default="StageWiseModel",
        )
        self.case = factory(**self.args, stages=stages)
        if self.parent is not None:
            self.case.set_initial_values_for_variables(self.parent.case)

    def _solve_pdm(self, *, print_output: bool = True) -> None:
        """Solve PDM sides and amalgamate them before stage reduction."""

        if self.above.side_required:
            self.above.optimise(print_output=print_output)
            self.above = self._fewest_units_side("above", self.above)
        if self.below.side_required:
            self.below.optimise(print_output=print_output)
            self.below = self._fewest_units_side("below", self.below)
        self.case = self.above.amalgamate_networks(
            below_case=self.below,
            above_case=self.above,
        )
        self._carry_unconverged_side_solve()
        self.case.get_post_process()
        if print_output:
            self.case.output_to_cmd_line()
        self.args.update(
            {
                "name": f"PDM amalgamated {self.dTmin}",
                "minimisation_goal": "total utility",
                "tol": self.tol,
            }
        )

    def _fewest_units_side(self, name: str, side):
        """Re-solve a side for the fewest units among the matches it used.

        Minimum utility cost leaves the binaries free once the targets are
        met, so an integer solver may keep matches the network does not need
        (thousands of sub-kilowatt matches on a 64-stream case). The side's
        energy balance holds its utility at the target, and its own solution
        is feasible on its own matches, so the re-solve keeps the utility and
        can only drop units. The original side is kept if it fails.
        """

        if side.mSuccess != 1 or self.solver not in backend.INTEGER_SOLVERS:
            return side
        factory = self._model_factory(
            getattr(self, "_model_factories", None),
            "pinch_design_method",
            default="PinchDecompModel",
        )
        sparse = factory(
            **(
                dict(self.args)
                | {
                    "name": f"{name} pinch {self.dTmin} fewest units",
                    "pinch_loc": name,
                    "minimisation_goal": "min units",
                    "utility_cost_cap": side.solver_run.objective_value,
                    "solver_options": _with_time_limit(
                        self.solver, self.args.get("solver_options")
                    ),
                    "pinch_decomposition": self.pinch_decompositions[name],
                    "stage_selection": self.stage_selection,
                }
            )
        )
        _start_from_side(sparse, side)
        sparse.optimise(print_output=False)
        if sparse.mSuccess != 1:
            return side
        # Stopping at the time limit leaves a network at the same, proven,
        # minimum utility: the side keeps the first solve's convergence.
        sparse.solver_run = replace(
            sparse.solver_run,
            termination_condition=side.solver_run.termination_condition,
        )
        _cleanup_model(side)
        return sparse

    def _solve_pdm_variants(
        self, model_factories: Mapping[str, Any] | None
    ) -> list["InternalHeatExchangerNetworkProblem"]:
        """Collect extra networks from minimum-unit side topologies.

        Each required side is re-solved for the fewest units, with no-good
        cuts excluding its minimum-utility topology and every one found, until
        a side passes one unit over its minimum. Side pairs with the fewest
        total units, then one more, become variants up to
        ``pdm_network_limits``. Needs a solver that keeps integers.
        """

        at_minimum, one_more = self.pdm_network_limits
        if at_minimum + one_more <= 0 or self.solver not in backend.INTEGER_SOLVERS:
            return []
        limit = at_minimum + one_more
        sides = {
            name: self._minimum_unit_sides(name, side, limit, model_factories)
            for name, side in (("above", self.above), ("below", self.below))
        }
        pairs = sorted(
            (
                (_side_units(above) + _side_units(below), index_a + index_b, a, b)
                for index_a, (a, above) in enumerate(enumerate(sides["above"]))
                for index_b, (b, below) in enumerate(enumerate(sides["below"]))
                if (a, b) != (0, 0)
            ),
            key=lambda pair: pair[:2] + (pair[2], pair[3]),
        )
        fewest = min(
            min(_side_units(side) for side in sides["above"])
            + min(_side_units(side) for side in sides["below"]),
            pairs[0][0] if pairs else 0,
        )
        chosen = [pair for pair in pairs if pair[0] == fewest][:at_minimum] + [
            pair for pair in pairs if pair[0] == fewest + 1
        ][:one_more]
        from ..reporting.ranking import network_structure_signature

        seen = {
            network_structure_signature(self.extract_network(run_id="pdm-variants"))
        }
        variants = []
        for _units, _rank, a, b in chosen:
            variant = self._pdm_variant(sides["above"][a], sides["below"][b])
            if variant is None:
                continue
            signature = network_structure_signature(
                variant.extract_network(run_id="pdm-variants")
            )
            if signature not in seen:
                seen.add(signature)
                variants.append(variant)
        return variants

    def _minimum_unit_sides(
        self,
        name: str,
        primary,
        limit: int,
        model_factories: Mapping[str, Any] | None,
    ) -> list:
        """The primary side, then its minimum-unit and one-more-unit sides."""

        if not primary.side_required or primary.mSuccess != 1:
            return [primary]
        factory = self._model_factory(
            model_factories,
            "pinch_design_method",
            default="PinchDecompModel",
        )
        excluded = [primary.used_units()]
        found = []
        fewest = None
        while len(found) < limit:
            side = factory(
                **(
                    dict(self.args)
                    | {
                        "name": f"{name} pinch {self.dTmin} min units {len(found)}",
                        "pinch_loc": name,
                        "minimisation_goal": "min units",
                        "utility_cost_cap": primary.solver_run.objective_value,
                        "pinch_decomposition": self.pinch_decompositions[name],
                        "stage_selection": self.stage_selection,
                    }
                )
            )
            if not hasattr(side, "recovery_stage_active"):
                # Without packing, topologies that differ only in which
                # stages are empty amalgamate into the same network.
                add_recovery_stage_packing_constraints(side)
            for pattern in excluded:
                side.exclude_units(pattern)
            side.optimise(print_output=False)
            if side.mSuccess != 1:
                break
            fewest = side.n_units if fewest is None else fewest
            if side.n_units > fewest + 1:
                break
            found.append(side)
            excluded.append(side.used_units())
        return [primary, *found]

    def _solve_heuristic_pdm(self) -> None:
        """Realise heuristic pinch design proposals as networks.

        Each required side's proposals are solved for minimum utility with
        only their matches, heaters and coolers allowed; a side that fails is
        dropped. Side pairs, fewest proposal ranks first, are amalgamated,
        stage-reduced and verified like the extra PDM networks, and repeats
        are dropped, up to ``heuristic_network_limit`` networks. The first
        becomes this problem's case, the others ``pdm_variants``.
        """

        from ..heuristic.enumeration import propose_side_topologies
        from ..heuristic.side_streams import pinch_side
        from ..reporting.ranking import network_structure_signature

        limit = self.heuristic_network_limit
        report: dict[str, Any] = {"proposed": {}, "realised": {}}
        sides = {}
        for name in ("above", "below"):
            template = getattr(self, name)
            if not template.side_required:
                sides[name] = [template]
                continue
            proposals = propose_side_topologies(pinch_side(template), limit=limit)
            realised = [
                side
                for rank, proposal in enumerate(proposals)
                if (side := self._realise_side(name, proposal, rank)) is not None
            ]
            report["proposed"][name] = len(proposals)
            report["realised"][name] = len(realised)
            sides[name] = realised
        pairs = sorted(
            product(enumerate(sides["above"]), enumerate(sides["below"])),
            key=lambda pair: (pair[0][0] + pair[1][0], pair[0][0]),
        )
        networks: list[InternalHeatExchangerNetworkProblem] = []
        seen: set = set()
        failed = repeated = 0
        for (_a, above), (_b, below) in pairs:
            if len(networks) >= limit:
                break
            variant = self._pdm_variant(above, below)
            if variant is None or not variant.case.verify()[0]:
                failed += 1
                continue
            signature = network_structure_signature(
                variant.extract_network(run_id="heuristic-pdm")
            )
            if signature in seen:
                repeated += 1
                continue
            seen.add(signature)
            networks.append(variant)
        # Results live on the model variables: drop every side's run directory,
        # and the unsolved templates' that the realised sides replace.
        for side in sides["above"] + sides["below"] + [self.above, self.below]:
            _cleanup_model(side)
        report.update(
            pairs=len(pairs), failed=failed, repeated=repeated, networks=len(networks)
        )
        self.heuristic_report = report
        logger.info("[%s] heuristic pinch design: %s", self.name, report)
        if not networks:
            raise ValueError(
                "No heuristic pinch design proposal could be realised as a network."
            )
        first, *rest = networks
        self.above, self.below, self.case = first.above, first.below, first.case
        self.pdm_variants = rest

    def _realise_side(self, name: str, proposal, rank: int):
        """Solve one side proposal, or return ``None`` if it cannot be."""

        template = getattr(self, name)
        stages = proposal.stages
        factory = self._model_factory(
            getattr(self, "_model_factories", None),
            "pinch_design_method",
            default="PinchDecompModel",
        )
        try:
            side = factory(
                **(
                    dict(self.args)
                    | {
                        "name": f"{name} pinch {self.dTmin} heuristic {rank}",
                        "pinch_loc": name,
                        "minimisation_goal": (
                            "hot utility" if name == "above" else "cold utility"
                        ),
                        "pinch_decomposition": self.pinch_decompositions[
                            name
                        ].model_copy(
                            update={
                                "S": stages,
                                "K": stages + 1,
                                "manual_stage_selection": (stages, stages),
                            }
                        ),
                        "stage_selection": [stages, stages],
                        "z_restriction": proposal.z_restriction(
                            hot_count=template.I,
                            cold_count=template.J,
                            hot_utilities=template.NHU,
                            cold_utilities=template.NCU,
                        ),
                    }
                )
            )
        except ValueError as exc:
            logger.debug("Heuristic %s proposal %d not built: %s", name, rank, exc)
            return None
        side.optimise(print_output=False)
        if side.mSuccess != 1:
            _cleanup_model(side)
            return None
        # The amalgamated network is built with the above side's restriction,
        # which must be the problem's, not this side's stage grid.
        side.z_restriction = self.args["z_restriction"]
        return side

    def _pdm_variant(
        self, above, below
    ) -> "InternalHeatExchangerNetworkProblem | None":
        variant = copy.copy(self)
        variant.above, variant.below = above, below
        variant.pdm_variants = []
        try:
            variant.case = above.amalgamate_networks(below_case=below, above_case=above)
        except ValueError:
            return None
        unreduced = variant.case
        try:
            variant._carry_unconverged_side_solve()
            variant.case.get_post_process()
            variant.case = variant.remove_unused_stages(variant.case)
        finally:
            _cleanup_model(unreduced)
            _cleanup_model(variant.case)
        return variant if variant.case.mSuccess == 1 else None

    def _carry_unconverged_side_solve(self) -> None:
        # The amalgamated model is built, not solved, so its own solver run
        # says nothing about the sides' convergence.
        for side in (self.above, self.below):
            side_run = getattr(side, "solver_run", None)
            if side.side_required and not getattr(side_run, "converged", True):
                case_run = getattr(self.case, "solver_run", None)
                self.case.solver_run = (
                    side_run
                    if case_run is None
                    else replace(
                        case_run,
                        termination_condition=side_run.termination_condition,
                    )
                )
                return

    def remove_unused_stages(self, case) -> StageWiseModel:
        """Apply the source stage-utilisation reduction after PDM/TDM solves."""

        if case.mSuccess != 1:
            return case

        active_stages = self._get_active_stages(case)
        if len(active_stages) == case.S:
            return case

        peak_recovery = _peak_recovery_duties(case)
        active_locations = [
            [[None for _k in range(len(active_stages))] for _j in range(case.J)]
            for _i in range(case.I)
        ]
        for new_k, old_k in enumerate(active_stages):
            for i in range(case.I):
                for j in range(case.J):
                    active_locations[i][j][new_k] = (
                        1 if peak_recovery[i][j][old_k] > case.tol else 0
                    )

        heaters_in_use = _used_in_any_period(case, "Q_h")
        coolers_in_use = _used_in_any_period(case, "Q_c")
        try:
            # HiGHS for a linear reduced model, Ipopt for the others.
            backend.require_solver_backend("highs")
            solver = "highs"
        except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
            solver = "ipopt-pyomo"
        f_case = StageWiseModel(
            name=f"reduced-{case.name}",
            framework=case.framework,
            solver=solver,
            solver_arrays=case.solver_arrays,
            stages=len(active_stages),
            dTmin=case.dTmin,
            # Relaxed: an allowed heater or cooler is fixed in use, so keep
            # only those the solved case uses in some period.
            z_restriction=[
                active_locations,
                heaters_in_use,
                coolers_in_use,
            ],
            min_dqda=case.min_dqda,
            minimisation_goal=case.minimisation_goal,
            non_isothermal_model=case.non_isothermal_model,
            integers=False,
            tol=case.tol,
        )

        for new_k, old_k in enumerate(active_stages):
            for i in range(case.I):
                for j in range(case.J):
                    q_val = (
                        _value(case.Q_r[i][j][old_k])
                        if _value(case.Q_r[i][j][old_k]) > case.tol
                        else 0.0
                    )
                    _assign(f_case.Q_r[i][j][new_k], q_val)
                    # Shared across periods: a match used in any period is on.
                    _assign_binary(f_case.z[i][j][new_k], active_locations[i][j][new_k])
                    _assign(
                        f_case.theta_1[i][j][new_k], _value(case.theta_1[i][j][old_k])
                    )
                    _assign(
                        f_case.theta_2[i][j][new_k], _value(case.theta_2[i][j][old_k])
                    )

                    if case.non_isothermal_model:
                        _assign(f_case.X[i][j][new_k], _value(case.X[i][j][old_k]))
                        _assign(f_case.Y[j][i][new_k], _value(case.Y[j][i][old_k]))
                        _assign(
                            f_case.T_h_out_x[i][j][new_k],
                            _value(case.T_h_out_x[i][j][old_k]),
                        )
                        _assign(
                            f_case.T_c_out_y[j][i][new_k],
                            _value(case.T_c_out_y[j][i][old_k]),
                        )

        reduced_boundaries = [0] + [stage + 1 for stage in active_stages]
        for i in range(case.I):
            for new_k, old_k in enumerate(reduced_boundaries):
                _assign(f_case.T_h[i][new_k], _value(case.T_h[i][old_k]))

        for j in range(case.J):
            for new_k, old_k in enumerate(reduced_boundaries):
                _assign(f_case.T_c[j][new_k], _value(case.T_c[j][old_k]))

        for duties, flags, source, in_use in (
            (f_case.Q_c, f_case.z_cu, case.Q_c, coolers_in_use),
            (f_case.Q_h, f_case.z_hu, case.Q_h, heaters_in_use),
        ):
            for index, row in enumerate(source):
                for u, duty in enumerate(row):
                    _assign(duties[index][u], _value(duty))
                    _assign_binary(flags[index][u], in_use[index][u])

        if f_case.solver == "highs" and not f_case.m.is_linear():
            f_case.solver = "ipopt-pyomo"
            f_case.solver_run = backend.configure_pyomo_solver(f_case.m, f_case.solver)
        f_case.TAC_model = case.TAC_model
        f_case.TAC = case.TAC
        f_case.solve_time = case.solve_time
        f_case.mSuccess = case.mSuccess

        f_case.optimise(print_output=False)
        # Reduction only simplifies the solved network: keep the original
        # unless the reduced one solves and passes the same checks.
        if f_case.mSuccess != 1 or not f_case.verify()[0]:
            return case
        return f_case

    def _get_active_stages(self, case) -> list[int]:
        peak_recovery = _peak_recovery_duties(case)
        q_total = sum(
            peak_recovery[i][j][k]
            for i in range(case.I)
            for j in range(case.J)
            for k in range(case.S)
        )
        if q_total <= 0.0:
            # No recovery to apportion: keep the stage layout unchanged.
            return list(range(case.S))
        q_per_stage = [
            sum(peak_recovery[i][j][k] for i in range(case.I) for j in range(case.J))
            for k in range(case.S)
        ]
        threshold = self._utilisation_threshold(case.S)
        return [
            k
            for k, q_stage in enumerate(q_per_stage)
            if q_stage / q_total * 100 >= threshold
        ]

    def _utilisation_threshold(self, stage_count: int) -> float:
        if stage_count <= 3:
            return 0.1
        if stage_count <= 5:
            return 5.0
        if stage_count <= 8:
            return 8.0
        if stage_count <= 10:
            return 10.0
        return 0.0

    def _model_factory(
        self,
        model_factories: Mapping[str, Any] | None,
        factory_name: str,
        *,
        default: str,
    ) -> Any:
        if model_factories and factory_name in model_factories:
            return model_factories[factory_name]
        if default == "PinchDecompModel":
            from .pinch_decomposition import PinchDecompModel

            return PinchDecompModel
        if default == "StageWiseModel":
            return StageWiseModel
        raise ValueError(f"Unknown heat exchanger network model factory {default!r}.")


def _start_from_side(sparse, side) -> None:
    """Start ``sparse`` at ``side``'s solution with its unused units off.

    Both models have the same superstructure, so their variables line up.
    Fixing the unused binaries at zero keeps the re-solve to the side's own
    matches, and the solution is a feasible warm start for it.
    """

    for target, source in zip(
        sparse.m.model.variables.values(),
        side.m.model.variables.values(),
        strict=True,
    ):
        target.set_value(source.value, skip_validation=True)
    # Binaries the solver left off, read from the copied values: a match
    # with a tiny duty is still on, so the start is exactly feasible.
    for grid, indices in _equations_unit_indices(sparse).items():
        for index in indices:
            binary = getattr(sparse, grid)
            for position in index:
                binary = binary[position]
            if not binary.fixed and (binary.value or 0.0) < 0.5:
                binary.fix(0)
    sparse.m.warm_start = True


def _with_time_limit(solver: str, options) -> Any:
    """The PDM solver options with the fewest-units time limit added.

    A limit the user set is kept, as are options given as option lines.
    """

    name = backend.TIME_LIMIT_OPTIONS.get(solver)
    if name is None or not (options is None or isinstance(options, Mapping)):
        return options
    return {name: backend.FEWEST_UNITS_TIME_LIMIT, **dict(options or {})}


def _side_units(side) -> int:
    """A side's unit count; a side not on this side of the pinch has none."""

    return int(side.n_units) if side.side_required and side.mSuccess == 1 else 0


def _peak_recovery_duties(case) -> list:
    """Return each match's largest duty over the periods, ``[hot][cold][stage]``.

    ``Q_r`` is period 0's; a stage or match used only in a later period
    still belongs in the reduced network.
    """
    by_period = getattr(case, "Q_r_by_period", None) or [case.Q_r]
    return [
        [
            [
                max(_value(period[i][j][k]) for period in by_period)
                for k in range(case.S)
            ]
            for j in range(case.J)
        ]
        for i in range(case.I)
    ]


def _used_in_any_period(case, name: str) -> list[list[int]]:
    """Flag each (stream, utility) unit with duty in any period.

    ``Q_h`` and ``Q_c`` are period 0's ``[stream][utility]`` duties.
    """
    by_period = getattr(case, f"{name}_by_period", None) or [getattr(case, name)]
    return [
        [
            1 if any(_value(period[index][u]) > case.tol for period in by_period) else 0
            for u in range(len(row))
        ]
        for index, row in enumerate(by_period[0])
    ]


def _cleanup_model(model: Any) -> None:
    if model is not None:
        backend.cleanup_model_run_directory(getattr(model, "m", None))


def _assign(variable: Any, value: float) -> None:
    _set_value(None, variable, value)


def _assign_binary(variable: Any, value: float) -> None:
    _assign_value(variable, 1 if value > 0 else 0)
