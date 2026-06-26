"""Task executor contracts and implementations for HEN synthesis."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Protocol, Sequence

from .....lib.schemas.synthesis import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
    SynthesisMethod,
)
from ...unit_models.problem import InternalHeatExchangerNetworkProblem
from ..errors import WorkflowContractError
from ..solver.arrays import PreparedSolverArrays, problem_to_solver_arrays
from .pathways import pathways_from_metadata, tier_evm_branch_breadth


def _process_pool(max_workers: int) -> ProcessPoolExecutor:
    """Create solver workers with inherited local-package import state."""

    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("fork"),
    )


class SynthesisExecutor(Protocol):
    """Internal executor protocol used by fake and future local solvers."""

    def execute(
        self,
        tasks: Sequence[HeatExchangerNetworkSynthesisTask],
        *,
        problem,
        parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
        max_parallel: int,
    ) -> tuple[HeatExchangerNetworkSynthesisTaskOutcome, ...]:
        """Execute one task stage and return serializable task outcomes."""


class LocalSynthesisExecutor:
    """Internal executor for moved PDM/TDM/ESM model slices."""

    def __init__(
        self,
        *,
        print_output: bool = False,
        evolution: bool = False,
        model_factories: dict[str, Any] | None = None,
        worker_pool_factory: Callable[[int], object] | None = None,
    ) -> None:
        self.print_output = print_output
        self.evolution = evolution
        self.model_factories = model_factories
        self.worker_pool_factory = worker_pool_factory or _process_pool
        self.executed_tasks: list[HeatExchangerNetworkSynthesisTask] = []
        self.stage_order: list[SynthesisMethod] = []
        self.problems_by_task_id: dict[str, InternalHeatExchangerNetworkProblem] = {}

    def execute(
        self,
        tasks: Sequence[HeatExchangerNetworkSynthesisTask],
        *,
        problem,
        parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
        max_parallel: int,
    ) -> tuple[HeatExchangerNetworkSynthesisTaskOutcome, ...]:
        if tasks:
            self.stage_order.append(tasks[0].method)

        worker_count = max(1, min(int(max_parallel), len(tasks)))
        if worker_count > 1 and _can_build_tasks_in_workers(tasks, parent_outcomes):
            solve_inputs = (
                (
                    task,
                    problem,
                    self.print_output,
                    self.model_factories,
                    _evolution_options(problem, task, max_parallel=1),
                )
                for task in tasks
            )
            with self.worker_pool_factory(worker_count) as pool:
                solved_results = tuple(
                    pool.map(_build_and_solve_root_task, solve_inputs)
                )
            return self._collect_outcomes(tasks, solved_results, {})

        built: list[
            tuple[
                HeatExchangerNetworkSynthesisTask,
                InternalHeatExchangerNetworkProblem,
                dict[str, Any] | None,
            ]
        ] = []
        failed: dict[str, HeatExchangerNetworkSynthesisTaskOutcome] = {}
        for task in tasks:
            try:
                task_model_factories = self._model_factories_for_task(task, problem)
                built.append(
                    (
                        task,
                        self._build_problem(
                            task,
                            problem=problem,
                            parent_outcomes=parent_outcomes,
                        ),
                        task_model_factories,
                    )
                )
            except Exception as exc:
                if task.task_id is not None:
                    failed[task.task_id] = _failed_task_outcome(task, str(exc))

        worker_count = max(1, min(int(max_parallel), len(built)))
        branch_parallel = max(1, int(max_parallel)) if worker_count == 1 else 1
        solve_inputs = (
            (
                task,
                internal_problem,
                self.print_output,
                task_model_factories,
                _evolution_options(problem, task, max_parallel=branch_parallel),
            )
            for task, internal_problem, task_model_factories in built
        )
        if worker_count == 1:
            solved_results = tuple(
                _solve_built_task(payload) for payload in solve_inputs
            )
        else:
            with self.worker_pool_factory(worker_count) as pool:
                solved_results = tuple(pool.map(_solve_built_task, solve_inputs))

        return self._collect_outcomes(tasks, solved_results, failed)

    def _collect_outcomes(
        self,
        tasks: Sequence[HeatExchangerNetworkSynthesisTask],
        solved_results: Sequence[
            tuple[
                HeatExchangerNetworkSynthesisTask,
                HeatExchangerNetworkSynthesisTaskOutcome,
                InternalHeatExchangerNetworkProblem | None,
            ]
        ],
        failed: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
    ) -> tuple[HeatExchangerNetworkSynthesisTaskOutcome, ...]:
        results_by_task_id = {
            task.task_id: (task, outcome, internal_problem)
            for task, outcome, internal_problem in solved_results
            if task.task_id is not None
        }

        outcomes: list[HeatExchangerNetworkSynthesisTaskOutcome] = []
        for task in tasks:
            result = results_by_task_id.get(task.task_id)
            if result is None:
                outcome = failed.get(task.task_id)
                if outcome is None:
                    outcome = _failed_task_outcome(
                        task,
                        "solver worker did not return a result",
                    )
                internal_problem = None
            else:
                _task, outcome, internal_problem = result
            self.executed_tasks.append(task)
            if (
                internal_problem is not None
                and outcome.status == "success"
                and task.task_id is not None
            ):
                self.problems_by_task_id[task.task_id] = internal_problem
            outcomes.append(outcome)
        return tuple(outcomes)

    def _build_problem(
        self,
        task: HeatExchangerNetworkSynthesisTask,
        *,
        problem,
        parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
    ) -> InternalHeatExchangerNetworkProblem:
        parent_problem = self._parent_problem(task, parent_outcomes)
        dTmin = _legacy_task_dTmin(task)
        arrays = problem_to_solver_arrays(problem, dTmin)
        stage_selection = _legacy_pdm_stage_selection(problem, task)
        decompositions = None
        if task.method == "pinch_design_method":
            from ..solver.pinch_design_decomposition import (
                build_pinch_design_decomposition,
            )

            decompositions = {
                "above": build_pinch_design_decomposition(
                    problem,
                    task.approach_temperature,
                    pinch_location="above",
                    stage_selection=stage_selection,
                ),
                "below": build_pinch_design_decomposition(
                    problem,
                    task.approach_temperature,
                    pinch_location="below",
                    stage_selection=stage_selection,
                ),
            }

        return InternalHeatExchangerNetworkProblem(
            solver_arrays=arrays,
            name=_legacy_task_name(task),
            framework=_legacy_framework(task.method),
            solver=_solver_for_task(problem, task.method),
            dTmin=dTmin,
            min_dqda=(
                0.0
                if task.derivative_threshold is None
                else float(task.derivative_threshold)
            ),
            z_restriction=_legacy_z_restriction(task, arrays),
            minimisation_goal=_legacy_objective(task.method),
            non_isothermal_model=task.method == "network_evolution_method",
            integers=task.method != "network_evolution_method",
            parent=parent_problem,
            tol=_solve_tolerance(problem),
            solver_options=_solver_options_for_task(problem, task),
            stage_selection=stage_selection,
            stages=task.stage_count,
            synthesis_task_id=task.task_id,
            pinch_decompositions=decompositions,
        )

    def _parent_problem(
        self,
        task: HeatExchangerNetworkSynthesisTask,
        parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
    ) -> InternalHeatExchangerNetworkProblem | None:
        if task.parent_task_id is None:
            return None
        parent_outcome = parent_outcomes.get(task.parent_task_id)
        if parent_outcome is None:
            raise WorkflowContractError(
                f"Missing parent outcome for task {task.task_id}: {task.parent_task_id}"
            )
        if parent_outcome.status != "success":
            raise WorkflowContractError(
                f"Cannot build task {task.task_id} from failed parent "
                f"{task.parent_task_id}."
            )
        parent_problem = self.problems_by_task_id.get(task.parent_task_id)
        if parent_problem is None:
            raise WorkflowContractError(
                f"Successful parent task {task.parent_task_id} has no private "
                "solver problem for StageWise warm-start."
            )
        return parent_problem

    def _model_factories_for_task(
        self,
        task: HeatExchangerNetworkSynthesisTask,
        problem,
    ) -> dict[str, Any] | None:
        factories = dict(self.model_factories or {})
        scope = _stage_packing_scope(problem)
        pdm_mode = str(task.settings.get("pdm_mode", ""))
        if task.method == "pinch_design_method" and (
            pdm_mode == "compact" or (not pdm_mode and scope in {"pdm", "all"})
        ):
            factories.setdefault(
                "pinch_design_method",
                _stage_packed_pdm_factory(),
            )
        if task.method == "thermal_derivative_method" and scope in {"tdm", "all"}:
            factories.setdefault("stagewise", _stage_packed_stagewise_factory())
        return factories or None


def _solve_built_task(
    payload: tuple[Any, ...],
) -> tuple[
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
    InternalHeatExchangerNetworkProblem | None,
]:
    if len(payload) == 4:
        task, internal_problem, print_output, model_factories = payload
        evolution_options = {
            "n_ad_branches": 1,
            "n_rm_branches": 1,
            "max_parallel": 1,
            "no_improvement_patience": None,
        }
    else:
        task, internal_problem, print_output, model_factories, evolution_options = (
            payload
        )
    try:
        solve_kwargs: dict[str, Any] = {
            "print_output": print_output,
            "evolution": task.method == "network_evolution_method",
            "model_factories": model_factories,
        }
        if task.method == "network_evolution_method" and evolution_options != {
            "n_ad_branches": 1,
            "n_rm_branches": 1,
            "max_parallel": 1,
            "no_improvement_patience": None,
        }:
            solve_kwargs.update(
                {
                    "evolution_n_ad_branches": evolution_options["n_ad_branches"],
                    "evolution_n_rm_branches": evolution_options["n_rm_branches"],
                    "evolution_max_parallel": evolution_options["max_parallel"],
                    "evolution_no_improvement_patience": evolution_options[
                        "no_improvement_patience"
                    ],
                }
            )
        solved = internal_problem.get_solution(**solve_kwargs)
        if solved is None or getattr(solved, "mSuccess", 0) != 1:
            reason = getattr(
                internal_problem,
                "solution_failure_reason",
                "solver did not return a successful heat exchanger network model",
            )
            return task, _failed_task_outcome(task, reason), None
        verify = getattr(solved, "verify", None)
        if callable(verify):
            is_valid, reasons = verify()
            if not is_valid:
                reason = "verification failed: " + ", ".join(
                    str(reason) for reason in reasons
                )
                internal_problem.solution_failure_reason = reason
                return task, _failed_task_outcome(task, reason), internal_problem

        network = internal_problem.extract_network(run_id=task.run_id)
        return (
            task,
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=task,
                status="success",
                network=network,
                objective_value=network.total_annual_cost,
                solver_status=_solver_status(internal_problem),
            ),
            internal_problem,
        )
    except Exception as exc:
        return task, _failed_task_outcome(task, str(exc)), None


def _build_and_solve_root_task(
    payload: tuple[
        HeatExchangerNetworkSynthesisTask,
        Any,
        bool,
        dict[str, Any] | None,
        dict[str, int | None],
    ],
) -> tuple[
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
    InternalHeatExchangerNetworkProblem | None,
]:
    task, problem, print_output, model_factories, evolution_options = payload
    executor = LocalSynthesisExecutor(
        print_output=print_output,
        model_factories=model_factories,
    )
    task_model_factories = executor._model_factories_for_task(task, problem)
    try:
        internal_problem = executor._build_problem(
            task,
            problem=problem,
            parent_outcomes={},
        )
    except Exception as exc:
        return task, _failed_task_outcome(task, str(exc)), None
    return _solve_built_task(
        (
            task,
            internal_problem,
            print_output,
            task_model_factories,
            evolution_options,
        )
    )


def _can_build_tasks_in_workers(
    tasks: Sequence[HeatExchangerNetworkSynthesisTask],
    parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
) -> bool:
    return not parent_outcomes and all(task.parent_task_id is None for task in tasks)


def _failed_task_outcome(
    task: HeatExchangerNetworkSynthesisTask,
    reason: str,
) -> HeatExchangerNetworkSynthesisTaskOutcome:
    return HeatExchangerNetworkSynthesisTaskOutcome(
        task=task,
        status="failed",
        solver_status="failed",
        error=reason or "heat exchanger network synthesis task failed",
    )


def _legacy_framework(method: SynthesisMethod) -> str:
    return {
        "pinch_design_method": "PDM",
        "thermal_derivative_method": "TDM",
        "network_evolution_method": "ESM",
    }[method]


def _legacy_objective(method: SynthesisMethod) -> str:
    if method == "network_evolution_method":
        return "variable total cost"
    return "hot utility"


def _legacy_task_dTmin(task: HeatExchangerNetworkSynthesisTask) -> float:
    if task.method in {"thermal_derivative_method", "network_evolution_method"}:
        return 0.1
    return float(task.approach_temperature)


def _legacy_task_name(task: HeatExchangerNetworkSynthesisTask) -> str:
    if task.method == "pinch_design_method":
        return f"P-+--PDM-{task.approach_temperature}"
    if task.method == "thermal_derivative_method":
        stages = task.stage_count if task.stage_count is not None else "unknown"
        return f"P-S{stages}--TDM-{task.derivative_threshold}"
    stages = task.stage_count if task.stage_count is not None else "unknown"
    return f"P-S{stages}-Synheat-Iso-NLP"


def _solver_for_task(problem, method: SynthesisMethod) -> str:
    hens = problem.master_zone.config.hens
    if method == "pinch_design_method":
        return str(hens.solver_pdm)
    if method == "thermal_derivative_method":
        return str(hens.solver_tdm)
    return str(hens.solver_evm)


def _solver_options_for_task(
    problem,
    task: HeatExchangerNetworkSynthesisTask,
) -> dict[str, Any]:
    method = task.method
    hens = problem.master_zone.config.hens
    if method == "pinch_design_method":
        base_options = dict(hens.solver_options_pdm)
    elif method == "thermal_derivative_method":
        base_options = dict(hens.solver_options_tdm)
    else:
        base_options = dict(hens.solver_options_evm)

    task_options = task.settings.get("solver_options")
    if task_options is None:
        return base_options
    if not isinstance(task_options, dict):
        raise WorkflowContractError("task solver_options setting must be a dict.")
    return {**base_options, **task_options}


def _evolution_options(
    problem,
    task: HeatExchangerNetworkSynthesisTask,
    *,
    max_parallel: int,
) -> dict[str, int | None]:
    hens = problem.master_zone.config.hens
    tier = int(getattr(hens, "synthesis_quality_tier", 1))
    task_settings = task.settings or {}
    pathway_options = _pathway_evolution_options(task)
    return {
        "n_ad_branches": int(
            task_settings.get(
                "evolution_n_ad_branches",
                pathway_options.get(
                    "n_ad_branches",
                    _branch_breadth(
                        getattr(hens, "evm_n_ad_branches", None),
                        tier=tier,
                    ),
                ),
            )
        ),
        "n_rm_branches": int(
            task_settings.get(
                "evolution_n_rm_branches",
                pathway_options.get(
                    "n_rm_branches",
                    _branch_breadth(
                        getattr(hens, "evm_n_rm_branches", None),
                        tier=tier,
                    ),
                ),
            )
        ),
        "max_parallel": max(1, int(max_parallel)),
        "no_improvement_patience": _optional_int(
            task_settings.get(
                "evolution_no_improvement_patience",
                pathway_options.get("no_improvement_patience"),
            )
        ),
    }


def _branch_breadth(value: int | None, *, tier: int) -> int:
    if value is not None:
        return max(1, int(value))
    return tier_evm_branch_breadth(tier)


def _pathway_evolution_options(
    task: HeatExchangerNetworkSynthesisTask,
) -> dict[str, int | None]:
    pathways = pathways_from_metadata(task.metadata)
    if not pathways:
        return {}
    first = pathways[0]
    return {
        "n_ad_branches": first.evm_n_ad_branches,
        "n_rm_branches": first.evm_n_rm_branches,
        "no_improvement_patience": first.evm_no_improvement_patience,
    }


def _optional_int(value) -> int | None:
    if value is None:
        return None
    return int(value)


def _stage_packing_scope(problem) -> str:
    scope = str(getattr(problem.master_zone.config.hens, "stage_packing", "auto"))
    if scope == "auto":
        return "none"
    return scope


def _stage_packed_pdm_factory():
    from ...unit_models.packed_pinch_design import StagePackedPinchDecompModel

    return StagePackedPinchDecompModel


def _stage_packed_stagewise_factory():
    from ...unit_models.packed_stagewise import StagePackedStageWiseModel

    return StagePackedStageWiseModel


def _solve_tolerance(problem) -> float:
    return float(problem.master_zone.config.hens.solve_tolerance)


def _legacy_pdm_stage_selection(
    problem,
    task: HeatExchangerNetworkSynthesisTask | None = None,
) -> str | list[int]:
    if task is not None:
        task_stage_selection = task.settings.get("stage_selection")
        if task_stage_selection is not None:
            selection = [int(value) for value in task_stage_selection]
            if len(selection) == 2:
                return selection
    selection = [
        int(value) for value in problem.master_zone.config.hens.stage_selection
    ]
    if len(selection) == 2:
        return selection
    return "automated"


def _legacy_z_restriction(
    task: HeatExchangerNetworkSynthesisTask,
    arrays: PreparedSolverArrays,
) -> list:
    if not task.topology_restrictions:
        return [None, None, None]

    hot_axis = arrays.axis_maps["hot_process_streams"]
    cold_axis = arrays.axis_maps["cold_process_streams"]
    stage_count = task.stage_count or max(
        restriction.stage for restriction in task.topology_restrictions
    )
    recovery = [
        [[[0.0] for _k in range(stage_count)] for _j in range(len(cold_axis))]
        for _i in range(len(hot_axis))
    ]
    for restriction in task.topology_restrictions:
        i = hot_axis[restriction.source_stream]
        j = cold_axis[restriction.sink_stream]
        k = restriction.stage - 1
        recovery[i][j][k] = [float(restriction.duty)]
    return [recovery, None, None]


def _solver_status(problem: InternalHeatExchangerNetworkProblem) -> str:
    solver_run = getattr(problem.case, "solver_run", None)
    failure_reason = getattr(solver_run, "failure_reason", None)
    if failure_reason:
        return str(failure_reason)
    status = getattr(solver_run, "status", None)
    if status is not None:
        return str(status)
    return "success"


__all__ = [
    "LocalSynthesisExecutor",
    "SynthesisExecutor",
    "_failed_task_outcome",
]
