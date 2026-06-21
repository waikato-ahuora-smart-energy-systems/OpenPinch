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
        worker_pool_factory: Callable[[int], object] | None = None,
    ) -> None:
        self.print_output = print_output
        self.evolution = evolution
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

        built: list[
            tuple[
                HeatExchangerNetworkSynthesisTask,
                InternalHeatExchangerNetworkProblem,
            ]
        ] = []
        failed: dict[str, HeatExchangerNetworkSynthesisTaskOutcome] = {}
        for task in tasks:
            try:
                built.append(
                    (
                        task,
                        self._build_problem(
                            task,
                            problem=problem,
                            parent_outcomes=parent_outcomes,
                        ),
                    )
                )
            except Exception as exc:
                if task.task_id is not None:
                    failed[task.task_id] = _failed_task_outcome(task, str(exc))

        worker_count = max(1, min(int(max_parallel), len(built)))
        solve_inputs = (
            (task, internal_problem, self.print_output)
            for task, internal_problem in built
        )
        if worker_count == 1:
            solved_results = tuple(
                _solve_built_task(payload) for payload in solve_inputs
            )
        else:
            with self.worker_pool_factory(worker_count) as pool:
                solved_results = tuple(pool.map(_solve_built_task, solve_inputs))

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
        stage_selection = _legacy_pdm_stage_selection(problem)
        snapshots = None
        if task.method == "pinch_design_method":
            from ..solver.pinch_design_snapshot import (
                build_pinch_design_method_snapshot,
            )

            snapshots = {
                "above": build_pinch_design_method_snapshot(
                    problem,
                    task.approach_temperature,
                    pinch_location="above",
                    stage_selection=stage_selection,
                ),
                "below": build_pinch_design_method_snapshot(
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
            min_dqda=0.0
            if task.derivative_threshold is None
            else float(task.derivative_threshold),
            z_restriction=_legacy_z_restriction(task, arrays),
            minimisation_goal=_legacy_objective(task.method),
            non_isothermal_model=task.method == "network_evolution_method",
            integers=task.method != "network_evolution_method",
            parent=parent_problem,
            tol=_solve_tolerance(problem),
            solver_options=_solver_options_for_task(problem, task.method),
            stage_selection=stage_selection,
            stages=task.stage_count,
            synthesis_task_id=task.task_id,
            pinch_snapshots=snapshots,
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


def _solve_built_task(
    payload: tuple[
        HeatExchangerNetworkSynthesisTask,
        InternalHeatExchangerNetworkProblem,
        bool,
    ],
) -> tuple[
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
    InternalHeatExchangerNetworkProblem | None,
]:
    task, internal_problem, print_output = payload
    try:
        solved = internal_problem.get_solution(
            print_output=print_output,
            evolution=task.method == "network_evolution_method",
        )
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
    if task.method == "pinch_design_method":
        return float(task.approach_temperature)
    return 0.1


def _legacy_task_name(task: HeatExchangerNetworkSynthesisTask) -> str:
    if task.method == "pinch_design_method":
        return f"P-+--PDM-{task.approach_temperature}"
    if task.method == "thermal_derivative_method":
        stages = task.stage_count if task.stage_count is not None else "unknown"
        return f"P-S{stages}--TDM-{task.derivative_threshold}"
    stages = task.stage_count if task.stage_count is not None else "unknown"
    return f"P-S{stages}-Synheat-Iso-NLP"


def _solver_for_task(problem, method: SynthesisMethod) -> str:
    config = problem.master_zone.config
    if method == "pinch_design_method":
        return str(config.HENS_PDM_SOLVER)
    if method == "thermal_derivative_method":
        return str(config.HENS_TDM_SOLVER)
    return str(config.HENS_ESM_SOLVER)


def _solver_options_for_task(problem, method: SynthesisMethod) -> dict[str, Any]:
    config = problem.master_zone.config
    if method == "pinch_design_method":
        return dict(config.HENS_PDM_SOLVER_OPTIONS)
    if method == "thermal_derivative_method":
        return dict(config.HENS_TDM_SOLVER_OPTIONS)
    return dict(config.HENS_ESM_SOLVER_OPTIONS)


def _solve_tolerance(problem) -> float:
    return float(problem.master_zone.config.HENS_SOLVE_TOLERANCE)


def _legacy_pdm_stage_selection(problem) -> str | list[int]:
    selection = [
        int(value) for value in problem.master_zone.config.HENS_STAGE_SELECTION
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
