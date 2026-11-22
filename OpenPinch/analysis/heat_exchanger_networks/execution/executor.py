"""Task executor contracts and implementations for HEN synthesis."""

from __future__ import annotations

import contextvars
import logging
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from functools import cached_property
from typing import Any, Callable, Protocol, Sequence

from ....contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from ....domain.enums import HeatExchangerNetworkDesignMethod, ZoneType
from ....domain.progress import emit as emit_progress
from ....optimisation.execution import _main_module_blocks_workers
from ...parallelism import supported_workers
from ..errors import WorkflowContractError
from ..models._stagewise.evolution import reporting_improvements
from ..models.problem import InternalHeatExchangerNetworkProblem
from ..results.record import admit, recording_active
from ..site.levels import site_utility_system, utility_capacities
from ..solver.arrays import (
    PreparedSolverArrays,
    annual_cost_scales,
    nominal_approach_temperature,
    problem_to_solver_arrays,
)
from .search import resolve_search
from .settings import configured_stage_solver, heuristic_solver

logger = logging.getLogger(__name__)

_PINCH_DESIGN_METHODS = (
    HeatExchangerNetworkDesignMethod.PinchDesign,
    HeatExchangerNetworkDesignMethod.HeuristicPinchDesign,
)
# Heuristic networks per multiplier when a task does not set its own cap.
_HEURISTIC_NETWORKS = 50


def _process_pool(max_workers: int) -> ProcessPoolExecutor:
    """Create solver workers with the interpreter's default process context."""
    if reason := _main_module_blocks_workers():
        logger.warning("Solving HEN tasks serially: %s", reason)
        raise NotImplementedError(reason)
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

    # Offers each outcome to the run's records as it is solved, so the stage
    # need not offer them again.
    admits_outcomes = True

    def __init__(
        self,
        *,
        print_output: bool = False,
        evolution: bool = False,
        model_factories: dict[str, Any] | None = None,
        worker_pool_factory: Callable[[int], object] | None = None,
        handoff_queue_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.print_output = print_output
        self.evolution = evolution
        self.model_factories = model_factories
        self.worker_pool_factory = worker_pool_factory or _process_pool
        # The queue workers hand admitted networks back on; by default a
        # multiprocessing manager's queue, started with the pool.
        self.handoff_queue_factory = handoff_queue_factory
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

        shared = _ZoneCopiedOnce(problem)
        max_parallel = supported_workers(int(max_parallel))
        worker_count = max(1, min(max_parallel, len(tasks)))
        pool = self._start_pool(worker_count, max_parallel)
        if pool is not None and _can_build_tasks_in_workers(tasks, parent_outcomes):
            # Workers get the problem itself; each copies the zone once.
            solve_inputs = (
                (
                    task,
                    problem,
                    self.print_output,
                    self.model_factories,
                    _evolution_options(shared, task, max_parallel=1),
                )
                for task in tasks
            )
            solved_results = self._reported(
                tasks,
                self._run(pool, _build_and_solve_root_task, solve_inputs),
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
                task_model_factories = self._model_factories_for_task(task, shared)
                built.append(
                    (
                        task,
                        self._build_problem(
                            task,
                            problem=shared,
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
        if pool is not None and worker_count == 1:
            # At most one task built: solve here, closing the unused pool
            # unless a session keeps it for later stages.
            with self._using(pool):
                pass
            pool = None
        branch_parallel = max(1, max_parallel) if worker_count == 1 else 1
        # Serial tasks, including those of a pool that could not start, share
        # the evolution topologies they solve. Tasks in worker processes do
        # not: what they shared would depend on which tasks a worker ran.
        solved_topologies: dict | None = {} if pool is None else None
        solve_inputs = (
            (
                task,
                internal_problem,
                self.print_output,
                task_model_factories,
                _evolution_options(shared, task, max_parallel=branch_parallel)
                | {"solved_topologies": solved_topologies},
            )
            for task, internal_problem, task_model_factories in built
        )
        # Tasks that failed to build count as done.
        done = len(tasks) - len(built)
        solved_results = self._reported(
            tasks, self._run(pool, _solve_built_task, solve_inputs), done
        )

        return self._collect_outcomes(tasks, solved_results, failed)

    # Whether a session is open, and the pool it keeps for its stages.
    _session_open = False
    _session_pool = None

    @contextmanager
    def session(self):
        """Keep one worker pool for every stage executed inside.

        Worker processes take seconds to start, so a run of several stages
        starts its pool once, at the first stage that needs one, and closes
        it when the session ends. Outside a session each ``execute`` call
        starts and closes its own pool.
        """
        self._session_open = True
        try:
            yield self
        finally:
            self._session_open = False
            pool, self._session_pool = self._session_pool, None
            if pool is not None:
                pool.__exit__(None, None, None)

    def _start_pool(self, worker_count: int, session_workers: int | None = None):
        """A pool of ``worker_count`` workers, or ``None`` to run serially.

        In a session the pool is the session's, started on first use with
        ``session_workers`` workers (``worker_count`` when not given) so it
        can serve the later stages too.
        """
        if worker_count <= 1:
            return None
        if self._session_pool is not None:
            return self._session_pool
        workers = (
            (session_workers or worker_count) if self._session_open else worker_count
        )
        try:
            pool = self.worker_pool_factory(workers)
        except NotImplementedError as exc:
            logger.debug("Solving HEN tasks serially: %s", exc)
            return None
        if self._session_open:
            pool.__enter__()
            self._session_pool = pool
        return pool

    @contextmanager
    def _using(self, pool):
        """Use ``pool`` for one stage.

        A stage's own pool is closed after it. A session's pool stays open
        for the stages to come, unless the stage fails in it: then it is
        closed and dropped, so the next stage starts a new one.
        """
        if pool is not self._session_pool:
            with pool:
                yield
            return
        try:
            yield
        except BaseException:
            self._session_pool = None
            pool.__exit__(None, None, None)
            raise

    def _run(self, pool, func, inputs):
        """Yield ``func`` over ``inputs``, in ``pool`` unless it is ``None``.

        Workers hand the networks their searches admit to this process, which
        offers them to the run's records as they arrive. Pooled results come
        back as their tasks finish, not in input order, so a task's outcome
        reaches the records without waiting for slower tasks before it; the
        caller restores input order.
        """
        if pool is None:
            for worker_args in inputs:
                yield func(worker_args)
            return
        if not recording_active():
            # Nothing listens: workers have nothing to hand off.
            with self._using(pool):
                yield from _as_completed(pool, func, inputs)
            return
        with self._handoff() as handoff, self._using(pool):
            yield from _as_completed(
                pool,
                _handed_off,
                ((func, handoff, worker_args) for worker_args in inputs),
            )

    @contextmanager
    def _handoff(self):
        """A queue workers put outcomes on; a thread here admits them."""
        with ExitStack() as stack:
            if self.handoff_queue_factory is None:
                manager = stack.enter_context(multiprocessing.Manager())
                handoff = manager.Queue()
            else:
                handoff = self.handoff_queue_factory()
            errors: list[BaseException] = []

            def drain() -> None:
                while (outcome := handoff.get()) is not None:
                    try:
                        admit(outcome)
                    except BaseException as error:
                        errors.append(error)

            # The drain thread sees the caller's records.
            thread = threading.Thread(
                target=contextvars.copy_context().run, args=(drain,), daemon=True
            )
            thread.start()
            try:
                yield handoff
            finally:
                handoff.put(None)
                thread.join()
            if errors:
                raise errors[0]

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
            admit(outcome)
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
            try:
                outcome = _solved_outcome(variant_task, variant, variant.case)
            except (ValueError, WorkflowContractError) as exc:
                # One extra network that cannot be extracted or verified does
                # not lose the others or the main network.
                logger.warning(
                    "Extra network %s dropped: %s", variant_task.task_id, exc
                )
                continue
            self._record(variant_task, outcome, variant, outcomes)
            admit(outcome)
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
        arrays = problem_to_solver_arrays(
            problem,
            _task_array_approach_temperature(
                task,
                nominal_approach,
                min_approach_share=config.hens.min_exchanger_approach_share,
            ),
        )
        # A site seed's task holds each process's levels within the process.
        site_processes = task.settings.get("site_processes")
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
            approach_share=(
                float(config.hens.min_exchanger_approach_share)
                if task.method
                in {
                    HeatExchangerNetworkDesignMethod.ThermalDerivative,
                    HeatExchangerNetworkDesignMethod.NetworkEvolution,
                }
                else None
            ),
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
            staged_utilities=site_processes is not None
            or (
                decompositions is not None
                and _utility_placement(problem) == "temperature"
                and _has_utility_levels(decompositions["above"])
            ),
            utility_slicing=bool(problem.master_zone.config.hens.utility_slices),
            site_processes=site_processes,
            site_level_holds=str(problem.master_zone.config.hens.site_level_holds),
            site_utility_system=(
                None
                if site_processes is None
                else site_utility_system(
                    arrays,
                    capacities=utility_capacities(problem.master_zone),
                    letdown_price=_site_letdown_price(problem.master_zone.config),
                )
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
    evolution_options = dict(evolution_options)
    solved_topologies = evolution_options.pop("solved_topologies", None)
    try:
        solve_kwargs: dict[str, Any] = {
            "print_output": print_output,
            "evolution": is_evolution,
            "model_factories": model_factories,
        }
        if is_evolution and solved_topologies is not None:
            solve_kwargs["evolution_solved_topologies"] = solved_topologies
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
        with reporting_improvements(
            lambda candidate: _hand_off_candidate(task, internal_problem, candidate)
        ):
            solved = internal_problem.get_solution(**solve_kwargs)
        if solved is None or getattr(solved, "mSuccess", 0) != 1:
            return task, _unsolved_outcome(task, internal_problem, solved), None
        return task, _solved_outcome(task, internal_problem, solved), internal_problem
    except Exception as exc:
        logger.debug(
            "HEN synthesis task %s failed to solve: %s",
            task.task_id,
            exc,
            exc_info=True,
        )
        return task, _failed_task_outcome(task, str(exc)), None


# Where a worker sends the outcomes it admits; unset, they are admitted here.
_CANDIDATE_SINK: ContextVar[Callable[[Any], None] | None] = ContextVar(
    "openpinch_hen_candidate_sink", default=None
)


def _as_completed(pool, func, inputs):
    """``func`` over ``inputs`` in ``pool``, each result as its task finishes.

    When the caller stops early (a solve raised, or the generator is
    closed), the solves still waiting for a worker are cancelled, so the
    pool's shutdown waits only for those already running.
    """
    futures = [pool.submit(func, worker_args) for worker_args in inputs]
    try:
        for future in as_completed(futures):
            yield future.result()
    finally:
        for future in futures:
            future.cancel()


def _handed_off(worker_args: tuple[Callable, Any, Any]):
    """Run ``func`` in a worker, sending its admitted outcomes to ``handoff``."""
    func, handoff, inner_args = worker_args
    token = _CANDIDATE_SINK.set(handoff.put)
    try:
        return func(inner_args)
    finally:
        _CANDIDATE_SINK.reset(token)


def _hand_off_candidate(
    task: HeatExchangerNetworkSynthesisTask,
    internal_problem: InternalHeatExchangerNetworkProblem,
    candidate: Any,
) -> None:
    """Offer one improved search candidate to the run's records."""
    try:
        network = internal_problem.extract_candidate_network(
            candidate, run_id=task.run_id
        )
    except (ValueError, WorkflowContractError) as exc:
        logger.warning("Evolution candidate of %s dropped: %s", task.task_id, exc)
        return
    outcome = HeatExchangerNetworkSynthesisTaskOutcome(
        task=task,
        status="success",
        network=network,
        objective_value=network.total_annual_cost,
        solver_status=str(
            getattr(getattr(candidate, "solver_run", None), "status", "success")
        ),
    )
    sink = _CANDIDATE_SINK.get()
    if sink is None:
        admit(outcome)
    else:
        sink(outcome)


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


class _ZoneCopiedOnce:
    """A prepared problem whose ``master_zone`` is copied on first read only.

    ``PinchProblem.master_zone`` deep-copies the whole zone tree on every
    read, and building one stage's tasks reads it many times for settings
    and solver arrays, none of which change it. Readers share this copy;
    the pinch design decomposition makes its own before changing a zone.
    Every other attribute, such as the validated input records the solver
    arrays order streams by, is the problem's own.
    """

    def __init__(self, problem) -> None:
        self._problem = problem

    def __getattr__(self, name: str):
        if name == "_problem":
            raise AttributeError(name)
        return getattr(self._problem, name)

    @cached_property
    def master_zone(self):
        return self._problem.master_zone

    @property
    def results(self):
        return self._problem.results

    @property
    def project_name(self) -> str:
        return self._problem.project_name


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
    problem = _ZoneCopiedOnce(problem)
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


def _unsolved_outcome(
    task: HeatExchangerNetworkSynthesisTask,
    internal_problem: Any,
    solved: Any,
) -> HeatExchangerNetworkSynthesisTaskOutcome:
    """The failed outcome of a task whose solve returned no solved model.

    An evolution whose seed model failed evolved nothing; the outcome keeps
    that as ``diagnostics["evolution_skipped"]`` so results assembly warns.
    """

    reason = getattr(
        internal_problem,
        "solution_failure_reason",
        "solver did not return a successful heat exchanger network model",
    )
    outcome = _failed_task_outcome(task, reason)
    if getattr(solved, "evolution_skipped", False):
        outcome = outcome.model_copy(
            update={"diagnostics": {"evolution_skipped": reason}}
        )
    return outcome


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


# The smallest nominal approach a task's arrays are built at, K: a share of 0
# leaves nothing of the contributions to scale, and the arrays need a
# positive value to say so.
_LEAST_ARRAY_APPROACH = 1e-6


def _task_array_approach_temperature(
    task: HeatExchangerNetworkSynthesisTask,
    nominal_approach: float,
    *,
    min_approach_share: float = 0.0,
) -> float:
    """Return the nominal approach a task's arrays scale contributions to.

    Pinch design scales each contribution by the task's multiplier. The
    thermal derivative and evolution steps scale it by the multiplier times
    ``min_approach_share``, so a match is held to that share of its pinch
    design approach; the models floor each pair at 0.1 K
    (``pair_minimum_approach``), which a nominal floor here would not do for
    a pair above the reference contribution.
    """
    if task.method in {
        HeatExchangerNetworkDesignMethod.ThermalDerivative,
        HeatExchangerNetworkDesignMethod.NetworkEvolution,
    }:
        return max(_LEAST_ARRAY_APPROACH, min_approach_share * float(nominal_approach))
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
    if method == HeatExchangerNetworkDesignMethod.HeuristicPinchDesign:
        return heuristic_solver(configured_stage_solver(hens, "HENS_SOLVER_PDM"))
    return configured_stage_solver(hens, _SOLVER_OPTIONS[_stage(method)])


# A task's solver can differ from its stage's configured one (HiGHS for the
# heuristic, or SCIP for a missing default); the stage's options are for the
# configured solver only.
def _configured_solver(hens, method: HeatExchangerNetworkDesignMethod) -> str:
    if method in _PINCH_DESIGN_METHODS:
        return str(hens.solver_pdm)
    if method == HeatExchangerNetworkDesignMethod.ThermalDerivative:
        return str(hens.solver_tdm)
    return str(hens.solver_evm)


_SOLVER_OPTIONS = {
    "pdm": "HENS_SOLVER_PDM",
    "tdm": "HENS_SOLVER_TDM",
    "evm": "HENS_SOLVER_EVM",
}


def _stage(method: HeatExchangerNetworkDesignMethod) -> str:
    if method in _PINCH_DESIGN_METHODS:
        return "pdm"
    if method == HeatExchangerNetworkDesignMethod.ThermalDerivative:
        return "tdm"
    return "evm"


def _solver_options_for_task(
    problem,
    task: HeatExchangerNetworkSynthesisTask,
) -> dict[str, Any]:
    method = task.method
    hens = problem.master_zone.config.hens
    if _solver_for_task(problem, method) != _configured_solver(hens, method):
        # Every option given, for the method or the task, is for the
        # configured solver.
        return {}
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


def _site_letdown_price(config) -> float:
    """``HENS_SITE_LETDOWN_PRICE`` ($/MWh) as an annual price ($/kW/y), as
    the solver arrays price utilities."""
    price_scale, _capital_scale = annual_cost_scales(config.costing)
    return float(config.hens.site_letdown_price) * price_scale


def _solve_tolerance(problem) -> float:
    return float(problem.master_zone.config.hens.solve_tolerance)


def _pdm_stage_selection(
    problem,
    task: HeatExchangerNetworkSynthesisTask | None = None,
) -> str | list[int]:
    """The pinch design stage counts above and below the pinch, or
    ``"automated"``: the task's own pair (a stage pair candidate), else
    ``HENS_PDM_STAGES``. A heuristic pinch design task sets its own stages,
    so the option does not reach it."""
    if task is not None:
        task_stage_selection = task.settings.get("stage_selection")
        if task_stage_selection is not None:
            selection = [int(value) for value in task_stage_selection]
            if len(selection) == 2:
                return selection
        if task.method == HeatExchangerNetworkDesignMethod.HeuristicPinchDesign:
            return "automated"
    pdm_stages = problem.master_zone.config.hens.pdm_stages
    if pdm_stages is not None:
        return [int(value) for value in pdm_stages]
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


def _utility_placement(problem) -> str:
    """``HENS_UTILITY_PLACEMENT``, with ``"auto"`` resolved by the design's scope.

    A design covering one process zone (a zone designed in isolation, or a
    problem of one process) places utility by price; one covering several,
    a site design, by temperature, as site targeting does.
    """
    placement = problem.master_zone.config.hens.utility_placement
    if placement != "auto":
        return placement
    return "temperature" if _process_zone_count(problem.master_zone) > 1 else "price"


def _process_zone_count(zone) -> int:
    own = 1 if str(getattr(zone, "type", "")) == ZoneType.P.value else 0
    return own + sum(
        _process_zone_count(child) for child in (zone.subzones or {}).values()
    )


def _has_utility_levels(decomposition) -> bool:
    """Whether targeting splits utility between levels on either side.

    With one hot and one cold utility there is no level to hold, and the
    pinch design sides keep their stream-end heaters and coolers.
    """
    return (
        len(decomposition.hot_utility_identities) > 1
        or len(decomposition.cold_utility_identities) > 1
    )
