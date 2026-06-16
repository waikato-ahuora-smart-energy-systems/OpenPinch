"""Problem-rooted HEN synthesis workflow orchestration.

The workflow in this module is intentionally internal to OpenPinch. Public
execution starts at ``PinchProblem.design`` or ``PinchWorkspace.solve_variant``;
task records here are durable workflow metadata, not alternate owners of case,
stream, utility, or configuration state.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, Sequence

from ...classes.heat_exchanger import (
    HeatExchanger,
    HeatExchangerKind,
    HeatExchangerStreamRole,
)
from ...classes.heat_exchanger_network import HeatExchangerNetwork
from ...lib.schemas.synthesis import (
    HeatExchangerNetworkSynthesisManifest,
    HeatExchangerNetworkSynthesisResult,
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
    HeatExchangerNetworkTopologyRestriction,
    SynthesisMethod,
)

_METHOD_SEQUENCE: tuple[SynthesisMethod, ...] = (
    "pinch_decomposition",
    "topology_design",
    "energy_stage_refinement",
)


class WorkflowContractError(RuntimeError):
    """Raised when task fan-out would violate the synthesis workflow contract."""


@dataclass(frozen=True)
class SynthesisWorkflowSettings:
    """Resolved HEN controls read from a prepared OpenPinch problem."""

    run_id: str
    approach_temperatures: tuple[float, ...]
    derivative_thresholds: tuple[float, ...]
    stage_selection: tuple[int, ...]
    method_sequence: tuple[SynthesisMethod, ...]
    output_formats: tuple[str, ...]
    solve_tolerance: float
    best_solutions_to_save: int
    max_parallel: int
    pdm_solver: str
    tdm_solver: str
    esm_solver: str
    problem_id: str | None = None
    workspace_variant: str | None = None
    state_id: str | None = None

    def solver_for(self, method: SynthesisMethod | None) -> str | None:
        """Return the configured solver name for one workflow method."""
        if method == "pinch_decomposition":
            return self.pdm_solver
        if method == "topology_design":
            return self.tdm_solver
        if method == "energy_stage_refinement":
            return self.esm_solver
        return None


@dataclass(frozen=True)
class SynthesisWorkflowResult:
    """Executed task graph plus accepted design payload."""

    tasks: tuple[HeatExchangerNetworkSynthesisTask, ...]
    outcomes: tuple[HeatExchangerNetworkSynthesisTaskOutcome, ...]
    accepted_result: HeatExchangerNetworkSynthesisResult
    total_run_time: float


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


class FakeSynthesisExecutor:
    """Deterministic no-solver executor for workflow and cache tests."""

    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = set(failures or ())
        self.stage_order: list[SynthesisMethod] = []
        self.executed_tasks: list[HeatExchangerNetworkSynthesisTask] = []

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

        outcomes: list[HeatExchangerNetworkSynthesisTaskOutcome] = []
        for task in tasks:
            if task.parent_task_id is not None:
                parent = parent_outcomes.get(task.parent_task_id)
                if parent is None or parent.status != "success":
                    raise WorkflowContractError(
                        "downstream HEN tasks require successful parent outcomes"
                    )

            self.executed_tasks.append(task)
            if task.task_id in self.failures:
                outcomes.append(
                    HeatExchangerNetworkSynthesisTaskOutcome(
                        task=task,
                        status="failed",
                        solver_status="fake-failed",
                        error="configured fake executor failure",
                    )
                )
                continue

            network = _fake_network(problem, task)
            outcomes.append(
                HeatExchangerNetworkSynthesisTaskOutcome(
                    task=task,
                    status="success",
                    network=network,
                    objective_value=network.total_annual_cost,
                    solver_status="fake-optimal",
                )
            )
        return tuple(outcomes)


def workflow_settings_from_problem(
    problem,
    *,
    state_id: str | None = None,
    workspace_variant: str | None = None,
) -> SynthesisWorkflowSettings:
    """Read persistent HEN controls from a prepared problem configuration."""
    zone = problem.master_zone
    if zone is None:
        raise RuntimeError("HEN synthesis requires a loaded PinchProblem.")
    config = zone.config
    return SynthesisWorkflowSettings(
        run_id=str(config.HENS_RUN_ID),
        approach_temperatures=tuple(
            float(value) for value in config.HENS_APPROACH_TEMPERATURES
        ),
        derivative_thresholds=tuple(
            float(value) for value in config.HENS_DERIVATIVE_THRESHOLDS
        ),
        stage_selection=tuple(int(value) for value in config.HENS_STAGE_SELECTION),
        method_sequence=tuple(config.HENS_METHOD_SEQUENCE),
        output_formats=tuple(config.HENS_OUTPUT_FORMATS),
        solve_tolerance=float(config.HENS_SOLVE_TOLERANCE),
        best_solutions_to_save=int(config.HENS_BEST_SOLUTIONS_TO_SAVE),
        max_parallel=int(config.HENS_MAX_PARALLEL),
        pdm_solver=str(config.HENS_PDM_SOLVER),
        tdm_solver=str(config.HENS_TDM_SOLVER),
        esm_solver=str(config.HENS_ESM_SOLVER),
        problem_id=problem.project_name,
        workspace_variant=workspace_variant,
        state_id=state_id,
    )


def _execute_synthesis_workflow(
    problem,
    settings: SynthesisWorkflowSettings,
    *,
    executor: SynthesisExecutor | None = None,
) -> SynthesisWorkflowResult:
    """Generate, fake-execute, and collect the PDM -> TDM -> ESM task graph."""
    if settings.method_sequence != _METHOD_SEQUENCE:
        raise WorkflowContractError(
            "HENS_METHOD_SEQUENCE must preserve pinch_decomposition -> "
            "topology_design -> energy_stage_refinement for this migration slice."
        )

    executor = executor or FakeSynthesisExecutor()
    start = perf_counter()

    pdm_tasks = build_pinch_decomposition_tasks(settings)
    pdm_outcomes = executor.execute(
        pdm_tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=settings.max_parallel,
    )

    pdm_outcome_map = _outcome_map(pdm_outcomes)
    tdm_tasks = build_topology_design_tasks(settings, pdm_outcomes)
    tdm_outcomes = executor.execute(
        tdm_tasks,
        problem=problem,
        parent_outcomes=pdm_outcome_map,
        max_parallel=settings.max_parallel,
    )

    upstream_outcomes = pdm_outcome_map | _outcome_map(tdm_outcomes)
    esm_tasks = build_energy_stage_refinement_tasks(settings, tdm_outcomes)
    esm_outcomes = executor.execute(
        esm_tasks,
        problem=problem,
        parent_outcomes=upstream_outcomes,
        max_parallel=settings.max_parallel,
    )

    tasks = tuple(pdm_tasks + tdm_tasks + esm_tasks)
    outcomes = tuple(pdm_outcomes + tdm_outcomes + esm_outcomes)
    return SynthesisWorkflowResult(
        tasks=tasks,
        outcomes=outcomes,
        accepted_result=build_synthesis_result(settings, tasks, outcomes),
        total_run_time=perf_counter() - start,
    )


def build_pinch_decomposition_tasks(
    settings: SynthesisWorkflowSettings,
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate root PDM tasks by sweeping configured approach temperatures."""
    return tuple(
        HeatExchangerNetworkSynthesisTask(
            run_id=settings.run_id,
            method="pinch_decomposition",
            approach_temperature=approach_temperature,
            problem_id=settings.problem_id,
            workspace_variant=settings.workspace_variant,
            state_id=settings.state_id,
        )
        for approach_temperature in settings.approach_temperatures
    )


def build_topology_design_tasks(
    settings: SynthesisWorkflowSettings,
    pdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Fan successful PDM topologies out over derivative thresholds."""
    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    for outcome in pdm_outcomes:
        if not _successful_method(outcome, "pinch_decomposition"):
            continue
        restrictions = required_topology_restrictions_from_outcome(
            outcome,
            "topology_design",
        )
        stage_count = _required_stage_count(outcome, "topology_design")
        for derivative_threshold in settings.derivative_thresholds:
            tasks.append(
                HeatExchangerNetworkSynthesisTask(
                    run_id=settings.run_id,
                    method="topology_design",
                    approach_temperature=outcome.task.approach_temperature,
                    derivative_threshold=derivative_threshold,
                    stage_count=stage_count,
                    problem_id=settings.problem_id,
                    workspace_variant=settings.workspace_variant,
                    state_id=settings.state_id,
                    parent_task_id=outcome.task.task_id,
                    topology_restrictions=restrictions,
                )
            )
    return tuple(tasks)


def build_energy_stage_refinement_tasks(
    settings: SynthesisWorkflowSettings,
    tdm_outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> tuple[HeatExchangerNetworkSynthesisTask, ...]:
    """Generate one ESM refinement task for each successful TDM topology."""
    tasks: list[HeatExchangerNetworkSynthesisTask] = []
    for outcome in tdm_outcomes:
        if not _successful_method(outcome, "topology_design"):
            continue
        restrictions = required_topology_restrictions_from_outcome(
            outcome,
            "energy_stage_refinement",
        )
        stage_count = _required_stage_count(outcome, "energy_stage_refinement")
        tasks.append(
            HeatExchangerNetworkSynthesisTask(
                run_id=settings.run_id,
                method="energy_stage_refinement",
                approach_temperature=outcome.task.approach_temperature,
                derivative_threshold=outcome.task.derivative_threshold,
                stage_count=stage_count,
                problem_id=settings.problem_id,
                workspace_variant=settings.workspace_variant,
                state_id=settings.state_id,
                parent_task_id=outcome.task.task_id,
                topology_restrictions=restrictions,
            )
        )
    return tuple(tasks)


def required_topology_restrictions_from_outcome(
    outcome: HeatExchangerNetworkSynthesisTaskOutcome,
    downstream_method: SynthesisMethod,
) -> tuple[HeatExchangerNetworkTopologyRestriction, ...]:
    """Return downstream topology restrictions or fail the workflow contract."""
    if outcome.network is None:
        raise WorkflowContractError(
            f"Successful {outcome.task.method} task {outcome.task.task_id} cannot "
            f"spawn {downstream_method} tasks without a HeatExchangerNetwork."
        )

    restrictions = tuple(
        HeatExchangerNetworkTopologyRestriction(
            source_stream=exchanger.source_stream,
            sink_stream=exchanger.sink_stream,
            stage=exchanger.stage,
            duty=exchanger.duty,
        )
        for exchanger in outcome.network.exchangers
        if exchanger.kind is HeatExchangerKind.RECOVERY
        and exchanger.active
        and exchanger.match_allowed
        and exchanger.stage is not None
    )
    if not restrictions:
        raise WorkflowContractError(
            f"Successful {outcome.task.method} task {outcome.task.task_id} cannot "
            f"spawn {downstream_method} tasks without recovery topology restrictions."
        )
    return restrictions


def build_synthesis_result(
    settings: SynthesisWorkflowSettings,
    tasks: Sequence[HeatExchangerNetworkSynthesisTask],
    outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> HeatExchangerNetworkSynthesisResult:
    """Convert accepted task outcomes into the canonical design payload."""
    accepted = _accepted_outcome(outcomes)
    if accepted.network is None:
        raise WorkflowContractError("Accepted HEN outcome must include a network.")

    network = accepted.network.model_copy(
        update={
            "run_id": settings.run_id,
            "task_id": accepted.task.task_id,
            "state_id": settings.state_id,
            "method": accepted.task.method,
            "stage_count": accepted.network.stage_count
            or accepted.task.stage_count,
        }
    )
    objective_values = {
        key: value
        for key, value in {
            "total_annual_cost": network.total_annual_cost,
            "utility_cost": network.utility_cost,
            "capital_cost": network.capital_cost,
        }.items()
        if value is not None
    }
    manifest = HeatExchangerNetworkSynthesisManifest(
        run_id=settings.run_id,
        approach_temperatures=settings.approach_temperatures,
        derivative_thresholds=settings.derivative_thresholds,
        stage_selection=settings.stage_selection,
        method_sequence=settings.method_sequence,
        export_formats=settings.output_formats,
        solve_tolerance=settings.solve_tolerance,
        best_solutions_to_save=settings.best_solutions_to_save,
        task_ids=tuple(task.task_id for task in tasks if task.task_id is not None),
        problem_id=settings.problem_id,
        workspace_variant=settings.workspace_variant,
        state_id=settings.state_id,
    )
    return HeatExchangerNetworkSynthesisResult(
        network=network,
        run_id=settings.run_id,
        task_id=accepted.task.task_id,
        problem_id=settings.problem_id,
        workspace_variant=settings.workspace_variant,
        state_id=settings.state_id,
        solver_name=settings.solver_for(accepted.task.method),
        solver_status=accepted.solver_status,
        method=accepted.task.method,
        stage_count=network.stage_count,
        objective_values=objective_values,
        task_outcomes=tuple(outcomes),
        manifest=manifest,
    )


def _accepted_outcome(
    outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> HeatExchangerNetworkSynthesisTaskOutcome:
    for method in reversed(_METHOD_SEQUENCE):
        candidates = [
            outcome
            for outcome in outcomes
            if outcome.status == "success" and outcome.task.method == method
        ]
        if candidates:
            return min(
                candidates,
                key=lambda outcome: float("inf")
                if outcome.objective_value is None
                else outcome.objective_value,
            )
    raise WorkflowContractError("HEN synthesis produced no successful task outcomes.")


def _successful_method(
    outcome: HeatExchangerNetworkSynthesisTaskOutcome,
    method: SynthesisMethod,
) -> bool:
    return outcome.status == "success" and outcome.task.method == method


def _required_stage_count(
    outcome: HeatExchangerNetworkSynthesisTaskOutcome,
    downstream_method: SynthesisMethod,
) -> int:
    stage_count = (
        outcome.network.stage_count
        if outcome.network is not None
        else outcome.task.stage_count
    )
    if stage_count is None:
        raise WorkflowContractError(
            f"Successful {outcome.task.method} task {outcome.task.task_id} cannot "
            f"spawn {downstream_method} tasks without a stage count."
        )
    return int(stage_count)


def _outcome_map(
    outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> dict[str, HeatExchangerNetworkSynthesisTaskOutcome]:
    return {
        outcome.task.task_id: outcome
        for outcome in outcomes
        if outcome.task.task_id is not None
    }


def _fake_network(
    problem,
    task: HeatExchangerNetworkSynthesisTask,
) -> HeatExchangerNetwork:
    zone = problem.master_zone
    if zone is None:
        raise RuntimeError("Fake HEN execution requires a prepared root Zone.")

    hot_key, hot_stream = _first_stream(zone.hot_streams.items(), "hot process")
    cold_key, cold_stream = _first_stream(zone.cold_streams.items(), "cold process")
    hot_utility_key, _hot_utility = _first_stream(
        zone.hot_utilities.items(),
        "hot utility",
    )
    cold_utility_key, _cold_utility = _first_stream(
        zone.cold_utilities.items(),
        "cold utility",
    )
    stage_count = task.stage_count or int(zone.config.HENS_STAGE_SELECTION[0])
    duty = _fake_duty(task)
    utility_cost = round(duty * 0.35, 6)
    capital_cost = round(duty * 1.15, 6)
    total_annual_cost = round(utility_cost + capital_cost, 6)

    exchangers = (
        HeatExchanger(
            exchanger_id=f"{task.task_id}-recovery",
            kind=HeatExchangerKind.RECOVERY,
            source_stream=hot_key,
            sink_stream=cold_key,
            source_stream_role=HeatExchangerStreamRole.PROCESS,
            sink_stream_role=HeatExchangerStreamRole.PROCESS,
            stage=stage_count,
            duty=duty,
            area=round(duty / 10.0, 6),
            approach_temperatures=(task.approach_temperature,),
            source_inlet_temperature=_temperature(hot_stream.t_supply, "K"),
            source_outlet_temperature=_temperature(hot_stream.t_target, "K"),
            sink_inlet_temperature=_temperature(cold_stream.t_supply, "K"),
            sink_outlet_temperature=_temperature(cold_stream.t_target, "K"),
            capital_cost=capital_cost,
            total_annual_cost=total_annual_cost,
        ),
        HeatExchanger(
            exchanger_id=f"{task.task_id}-hot-utility",
            kind=HeatExchangerKind.HOT_UTILITY,
            source_stream=hot_utility_key,
            sink_stream=cold_key,
            source_stream_role=HeatExchangerStreamRole.UTILITY,
            sink_stream_role=HeatExchangerStreamRole.PROCESS,
            duty=round(duty * 0.1, 6),
            operating_cost=round(utility_cost * 0.5, 6),
        ),
        HeatExchanger(
            exchanger_id=f"{task.task_id}-cold-utility",
            kind=HeatExchangerKind.COLD_UTILITY,
            source_stream=hot_key,
            sink_stream=cold_utility_key,
            source_stream_role=HeatExchangerStreamRole.PROCESS,
            sink_stream_role=HeatExchangerStreamRole.UTILITY,
            duty=round(duty * 0.08, 6),
            operating_cost=round(utility_cost * 0.5, 6),
        ),
    )
    return HeatExchangerNetwork(
        exchangers=exchangers,
        run_id=task.run_id,
        task_id=task.task_id,
        state_id=task.state_id,
        method=task.method,
        stage_count=stage_count,
        objective_value=total_annual_cost,
        total_annual_cost=total_annual_cost,
        utility_cost=utility_cost,
        capital_cost=capital_cost,
        summary_metrics={
            "recovery_unit_count": 1,
            "hot_utility_unit_count": 1,
            "cold_utility_unit_count": 1,
            "approach_temperature": task.approach_temperature,
            "derivative_threshold": task.derivative_threshold,
        },
    )


def _first_stream(items, label: str):
    try:
        return next(iter(items))
    except StopIteration as exc:
        raise ValueError(
            f"HEN synthesis requires at least one {label} stream."
        ) from exc


def _fake_duty(task: HeatExchangerNetworkSynthesisTask) -> float:
    derivative = 0.0 if task.derivative_threshold is None else task.derivative_threshold
    stage = 0 if task.stage_count is None else task.stage_count
    return round(
        1000.0 + task.approach_temperature * 10.0 + derivative * 100.0 + stage,
        6,
    )


def _temperature(value, unit: str) -> float | None:
    if value is None:
        return None
    if hasattr(value, "to"):
        return float(value.to(unit).value)
    return float(value)
