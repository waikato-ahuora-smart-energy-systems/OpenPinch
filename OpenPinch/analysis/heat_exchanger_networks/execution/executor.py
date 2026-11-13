"""Task executor contracts and implementations for HEN synthesis."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Protocol, Sequence

from ....contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from ....domain.enums import HeatExchangerNetworkDesignMethod
from ....domain.progress import emit as emit_progress
from ....optimisation.execution import _main_module_reruns_in_workers
from ...parallelism import supported_workers
from ..errors import WorkflowContractError
from ..models.problem import InternalHeatExchangerNetworkProblem
from ..solver.arrays import (
    PreparedSolverArrays,
    nominal_approach_temperature,
    problem_to_solver_arrays,
)
from .search import resolve_search

logger = logging.getLogger(__name__)

_PINCH_DESIGN_METHODS = (
    HeatExchangerNetworkDesignMethod.PinchDesign,
    HeatExchangerNetworkDesignMethod.HeuristicPinchDesign,
)
# Heuristic networks per multiplier when a task does not set its own cap.
_HEURISTIC_NETWORKS = 10


def _process_pool(max_workers: int) -> ProcessPoolExecutor:
    """Create solver workers with the interpreter's default process context."""
    if _main_module_reruns_in_workers():
        logger.warning(
            "Solving HEN tasks serially: the main script has no main guard, "
            "so every worker process would rerun it. Put the script's entry "
            'code under `if __name__ == "__main__":` to use worker processes.'
        )
        raise NotImplementedError("the main script has no main guard")
    return ProcessPoolExecutor(max_workers=max_workers)


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
        self.stage_order: list[HeatExchangerNetworkDesignMethod] = []
        self.problems_by_task_id: dict[str, InternalHeatExchangerNetworkProblem] = {}
        # Lowest total annual cost solved so far, across stages, for progress.
        self.best_objective: float | None = None

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
            self._report_progress(tasks, 0)

        max_parallel = supported_workers(int(max_parallel))
        worker_count = max(1, min(max_parallel, len(tasks)))
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
            solved_results = self._reported(
                tasks,
                self._map(_build_and_solve_root_task, solve_inputs, worker_count),
                0,
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
                logger.debug(
                    "Failed to build HEN synthesis task %s: %s",
                    task.task_id,
                    exc,
                    exc_info=True,
                )
                if task.task_id is not None:
                    failed[task.task_id] = _failed_task_outcome(task, str(exc))

        worker_count = max(1, min(max_parallel, len(built)))
        branch_parallel = max(1, max_parallel) if worker_count == 1 else 1
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
        # Tasks that failed to build count as done.
        done = len(tasks) - len(built)
        solved_results = self._reported(
            tasks, self._map(_solve_built_task, solve_inputs, worker_count), done
        )

        return self._collect_outcomes(tasks, solved_results, failed)

    def _map(self, func, inputs, worker_count: int):
        """Yield ``func`` over ``inputs``, in a worker pool when one can start."""
        if worker_count > 1:
            try:
                pool = self.worker_pool_factory(worker_count)
            except NotImplementedError as exc:
                logger.debug("Solving HEN tasks serially: %s", exc)
            else:
                with pool:
                    yield from pool.map(func, inputs)
                return
        for worker_args in inputs:
            yield func(worker_args)

    def _reported(
        self,
        tasks: Sequence[HeatExchangerNetworkSynthesisTask],
        solved_results: Any,
        done: int,
    ) -> tuple:
        """The solved results, reporting progress as each one arrives."""
        collected = []
        for result in solved_results:
            collected.append(result)
            outcome = result[1]
            if outcome.status == "success" and outcome.objective_value is not None:
                if self.best_objective is None:
                    self.best_objective = outcome.objective_value
                else:
                    self.best_objective = min(
                        self.best_objective, outcome.objective_value
                    )
            self._report_progress(tasks, done + len(collected))
        if not collected and done:
            # Every task failed to build: the stage is finished.
            self._report_progress(tasks, done)
        return tuple(collected)

    def _report_progress(
        self, tasks: Sequence[HeatExchangerNetworkSynthesisTask], done: int
    ) -> None:
        emit_progress(
            analysis="hen_synthesis",
            stage=str(tasks[0].method),
            done=done,
            total=len(tasks),
            unit="tasks",
            best_objective=self.best_objective,
            objective="total annual cost",
        )

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
            self._record(task, outcome, internal_problem, outcomes)
            if internal_problem is not None and outcome.status == "success":
                outcomes.extend(self._pdm_variant_outcomes(task, internal_problem))
        return tuple(outcomes)

    def _record(
        self,
        task: HeatExchangerNetworkSynthesisTask,
        outcome: HeatExchangerNetworkSynthesisTaskOutcome,
        internal_problem: InternalHeatExchangerNetworkProblem | None,
        outcomes: list[HeatExchangerNetworkSynthesisTaskOutcome],
    ) -> None:
        self.executed_tasks.append(task)
        if (
            internal_problem is not None
            and outcome.status == "success"
            and task.task_id is not None
        ):
            self.problems_by_task_id[task.task_id] = internal_problem
        outcomes.append(outcome)

    def _pdm_variant_outcomes(
        self,
        task: HeatExchangerNetworkSynthesisTask,
        internal_problem: InternalHeatExchangerNetworkProblem,
    ) -> list[HeatExchangerNetworkSynthesisTaskOutcome]:
        """Give each extra PDM network its own task, outcome and problem.

        Each variant is a seed for network evolution, which warm-starts from
        the private problem registered under the variant's task id.
        """

        outcomes: list[HeatExchangerNetworkSynthesisTaskOutcome] = []
        variants = getattr(internal_problem, "pdm_variants", None) or []
        internal_problem.pdm_variants = []
        for index, variant in enumerate(variants, start=1):
            variant_task = _pdm_variant_task(task, index, _variant_key(task))
            variant.synthesis_task_id = variant_task.task_id
            outcome = _solved_outcome(variant_task, variant, variant.case)
            self._record(variant_task, outcome, variant, outcomes)
        return outcomes

    def _build_problem(
        self,
        task: HeatExchangerNetworkSynthesisTask,
        *,
        problem,
        parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
    ) -> InternalHeatExchangerNetworkProblem:
        parent_problem = self._parent_problem(task, parent_outcomes)
        config = problem.master_zone.config
        nominal_approach = nominal_approach_temperature(config, task.dt_cont_multiplier)
        dTmin = _task_model_approach_temperature(
            task,
            nominal_approach,
            min_approach_share=config.hens.min_exchanger_approach_share,
        )
        arrays = problem_to_solver_arrays(problem, dTmin)
        stage_selection = _pdm_stage_selection(problem, task)
        decompositions = None
        is_evolution = task.method == HeatExchangerNetworkDesignMethod.NetworkEvolution
        is_heuristic = (
            task.method == HeatExchangerNetworkDesignMethod.HeuristicPinchDesign
        )
        if task.method in _PINCH_DESIGN_METHODS:
            from ..solver.pinch_design_decomposition import (
                build_pinch_design_decomposition,
            )

            decompositions = {
                "above": build_pinch_design_decomposition(
                    problem,
                    nominal_approach,
                    pinch_location="above",
                    stage_selection=stage_selection,
                ),
                "below": build_pinch_design_decomposition(
                    problem,
                    nominal_approach,
                    pinch_location="below",
                    stage_selection=stage_selection,
                ),
            }

        return InternalHeatExchangerNetworkProblem(
            solver_arrays=arrays,
            name=_task_model_name(task),
            framework=_model_framework(task.method),
            solver=_solver_for_task(problem, task.method),
            dTmin=dTmin,
            dt_cont_multiplier=task.dt_cont_multiplier,
            min_dqda=(
                0.0
                if task.derivative_threshold is None
                else float(task.derivative_threshold)
            ),
            z_restriction=_topology_restriction_values(task, arrays),
            minimisation_goal=_model_objective(task.method),
            non_isothermal_model=is_evolution,
            integers=not is_evolution,
            parent=parent_problem,
            tol=_solve_tolerance(problem),
            solver_options=_solver_options_for_task(problem, task),
            stage_selection=stage_selection,
            stages=task.stage_count,
            synthesis_task_id=task.task_id,
            pinch_decompositions=decompositions,
            pdm_network_limits=(
                (
                    int(problem.master_zone.config.hens.pdm_min_unit_networks),
                    int(problem.master_zone.config.hens.pdm_extra_unit_networks),
                )
                if decompositions is not None and not is_heuristic
                else (0, 0)
            ),
            heuristic_network_limit=(
                int(task.settings.get("heuristic_networks", _HEURISTIC_NETWORKS))
                if is_heuristic
                else 0
            ),
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
        if task.method == HeatExchangerNetworkDesignMethod.PinchDesign and scope in {
            "pdm",
            "all",
        }:
            factories.setdefault(
                "pinch_design_method",
                _stage_packed_pdm_factory(),
            )
        if (
            task.method == HeatExchangerNetworkDesignMethod.ThermalDerivative
            and scope in {"tdm", "all"}
        ):
            factories.setdefault("stagewise", _stage_packed_stagewise_factory())
        return factories or None


def _solve_built_task(
    worker_args: tuple[Any, ...],
) -> tuple[
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
    InternalHeatExchangerNetworkProblem | None,
]:
    if len(worker_args) == 4:
        task, internal_problem, print_output, model_factories = worker_args
        evolution_options = {
            "n_ad_branches": 1,
            "n_rm_branches": 1,
            "max_parallel": 1,
            "no_improvement_patience": None,
            "beam_width": 4,
        }
    else:
        task, internal_problem, print_output, model_factories, evolution_options = (
            worker_args
        )
    is_evolution = task.method == HeatExchangerNetworkDesignMethod.NetworkEvolution
    try:
        solve_kwargs: dict[str, Any] = {
            "print_output": print_output,
            "evolution": is_evolution,
            "model_factories": model_factories,
        }
        if is_evolution and evolution_options != {
            "n_ad_branches": 1,
            "n_rm_branches": 1,
            "max_parallel": 1,
            "no_improvement_patience": None,
            "beam_width": 4,
        }:
            solve_kwargs.update(
                {
                    "evolution_n_ad_branches": evolution_options["n_ad_branches"],
                    "evolution_n_rm_branches": evolution_options["n_rm_branches"],
                    "evolution_max_parallel": evolution_options["max_parallel"],
                    "evolution_no_improvement_patience": evolution_options[
                        "no_improvement_patience"
                    ],
                    "evolution_beam_width": evolution_options.get("beam_width", 4),
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
        return task, _solved_outcome(task, internal_problem, solved), internal_problem
    except Exception as exc:
        logger.debug(
            "HEN synthesis task %s failed to solve: %s",
            task.task_id,
            exc,
            exc_info=True,
        )
        return task, _failed_task_outcome(task, str(exc)), None


def _solved_outcome(
    task: HeatExchangerNetworkSynthesisTask,
    internal_problem: InternalHeatExchangerNetworkProblem,
    solved: Any,
) -> HeatExchangerNetworkSynthesisTaskOutcome:
    """Verify a solved model and extract its network as the task outcome."""

    verify = getattr(solved, "verify", None)
    if callable(verify):
        is_valid, reasons = verify()
        if not is_valid:
            reason = "verification failed: " + ", ".join(
                str(reason) for reason in reasons
            )
            internal_problem.solution_failure_reason = reason
            return _failed_task_outcome(task, reason)

    network = internal_problem.extract_network(run_id=task.run_id)
    # An unconverged solve stays a success so downstream tasks can start
    # from it; demote_infeasible_outcomes keeps it out of the ranking.
    solver_run = getattr(solved, "solver_run", None)
    diagnostics = (
        {}
        if getattr(solver_run, "converged", True)
        else {"unconverged_termination": str(solver_run.termination_condition)}
    )
    return HeatExchangerNetworkSynthesisTaskOutcome(
        task=task,
        status="success",
        network=network,
        objective_value=network.total_annual_cost,
        solver_status=_solver_status(internal_problem),
        diagnostics=diagnostics,
    )


def _pdm_variant_task(
    task: HeatExchangerNetworkSynthesisTask, index: int, key: str = "pdm_variant"
) -> HeatExchangerNetworkSynthesisTask:
    """The task record of a pinch design task's ``index``-th extra network."""

    return HeatExchangerNetworkSynthesisTask(
        **task.model_dump(exclude={"task_id", "settings"}),
        settings={**task.settings, key: index},
    )


def _variant_key(task: HeatExchangerNetworkSynthesisTask) -> str:
    """The task-settings key that numbers a task's extra networks."""

    if task.method == HeatExchangerNetworkDesignMethod.HeuristicPinchDesign:
        return "heuristic_variant"
    return "pdm_variant"


def _build_and_solve_root_task(
    worker_args: tuple[
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
    task, problem, print_output, model_factories, evolution_options = worker_args
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
        logger.debug(
            "Failed to build HEN synthesis root task %s: %s",
            task.task_id,
            exc,
            exc_info=True,
        )
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


def _model_framework(method: HeatExchangerNetworkDesignMethod) -> str:
    return {
        HeatExchangerNetworkDesignMethod.PinchDesign: "PDM",
        HeatExchangerNetworkDesignMethod.HeuristicPinchDesign: "PDM",
        HeatExchangerNetworkDesignMethod.ThermalDerivative: "TDM",
        HeatExchangerNetworkDesignMethod.NetworkEvolution: "ESM",
    }[method]


def _model_objective(method: HeatExchangerNetworkDesignMethod) -> str:
    if method == HeatExchangerNetworkDesignMethod.NetworkEvolution:
        return "variable total cost"
    return "hot utility"


def _task_model_approach_temperature(
    task: HeatExchangerNetworkSynthesisTask,
    nominal_approach: float,
    *,
    min_approach_share: float = 0.0,
) -> float:
    """Return the model's minimum approach temperature.

    ``nominal_approach`` is the task multiplier's internal approach,
    ``2 * THERMAL_DT_CONT * dt_cont_multiplier``. The pinch design step uses
    it. The thermal derivative and evolution steps minimise cost down to
    0.1 K, or to ``min_approach_share`` (``HENS_MIN_EXCHANGER_APPROACH_SHARE``)
    times it if larger.
    """
    if task.method in {
        HeatExchangerNetworkDesignMethod.ThermalDerivative,
        HeatExchangerNetworkDesignMethod.NetworkEvolution,
    }:
        return max(0.1, min_approach_share * float(nominal_approach))
    return float(nominal_approach)


def _task_model_name(task: HeatExchangerNetworkSynthesisTask) -> str:
    if task.method == HeatExchangerNetworkDesignMethod.PinchDesign:
        return f"P-+--PDM-m{task.dt_cont_multiplier:g}"
    if task.method == HeatExchangerNetworkDesignMethod.HeuristicPinchDesign:
        return f"P-+--HPDM-m{task.dt_cont_multiplier:g}"
    if task.method == HeatExchangerNetworkDesignMethod.ThermalDerivative:
        stages = task.stage_count if task.stage_count is not None else "unknown"
        return f"P-S{stages}--TDM-{task.derivative_threshold}"
    stages = task.stage_count if task.stage_count is not None else "unknown"
    return f"P-S{stages}-Synheat-Iso-NLP"


def _solver_for_task(problem, method: HeatExchangerNetworkDesignMethod) -> str:
    hens = problem.master_zone.config.hens
    if method in _PINCH_DESIGN_METHODS:
        return str(hens.solver_pdm)
    if method == HeatExchangerNetworkDesignMethod.ThermalDerivative:
        return str(hens.solver_tdm)
    return str(hens.solver_evm)


def _solver_options_for_task(
    problem,
    task: HeatExchangerNetworkSynthesisTask,
) -> dict[str, Any]:
    method = task.method
    hens = problem.master_zone.config.hens
    if method in _PINCH_DESIGN_METHODS:
        base_options = dict(hens.solver_options_pdm)
    elif method == HeatExchangerNetworkDesignMethod.ThermalDerivative:
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
    search = resolve_search(hens)
    task_settings = task.settings or {}
    return {
        "n_ad_branches": int(
            task_settings.get("evolution_n_ad_branches", search.evm_n_ad_branches)
        ),
        "n_rm_branches": int(
            task_settings.get("evolution_n_rm_branches", search.evm_n_rm_branches)
        ),
        "max_parallel": max(1, int(max_parallel)),
        "no_improvement_patience": _optional_int(
            task_settings.get("evolution_no_improvement_patience")
        ),
        "beam_width": max(1, int(getattr(hens, "evm_beam_width", 4) or 4)),
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
    from ..models.packed_pinch_design import StagePackedPinchDecompModel

    return StagePackedPinchDecompModel


def _stage_packed_stagewise_factory():
    from ..models.packed_stagewise import StagePackedStageWiseModel

    return StagePackedStageWiseModel


def _solve_tolerance(problem) -> float:
    return float(problem.master_zone.config.hens.solve_tolerance)


def _pdm_stage_selection(
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


def _topology_restriction_values(
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
