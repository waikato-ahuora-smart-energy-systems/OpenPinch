"""HENS-05 workflow, result-cache, export, and API-boundary tests."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import OpenPinch
import OpenPinch.analysis.heat_exchanger_networks as hens_package
import OpenPinch.analysis.heat_exchanger_networks.service as service_module
import OpenPinch.analysis.heat_exchanger_networks.targeting.open_hens_method as workflow_module
from OpenPinch.analysis.heat_exchanger_networks.errors import (
    WorkflowContractError,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.executor import (
    LocalSynthesisExecutor,
    _topology_restriction_values,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
    FakeSynthesisExecutor,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting.exports import (
    export_heat_exchanger_network_synthesis_results,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting.ranking import (
    network_structure_signature,
)
from OpenPinch.analysis.heat_exchanger_networks.results.assembly import (
    build_synthesis_result,
    demote_infeasible_outcomes,
)
from OpenPinch.analysis.heat_exchanger_networks.service import (
    heat_exchanger_network_synthesis_service,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
    require_solver_binary,
    require_synthesis_dependency,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.network_evolution_method import (
    build_network_evolution_method_tasks,
    build_seeded_network_evolution_method_tasks,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.pinch_design_method import (
    build_pinch_design_method_tasks,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.thermal_derivative_method import (
    build_seeded_thermal_derivative_method_tasks,
    build_thermal_derivative_method_tasks,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.application.workspace import PinchWorkspace
from OpenPinch.contracts.synthesis.common import HeatExchangerNetworkSynthesisManifest
from OpenPinch.contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from OpenPinch.domain.enums import HeatExchangerKind, HeatExchangerNetworkDesignMethod
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.support.paths import REPOSITORY_ROOT

REPO_ROOT = REPOSITORY_ROOT
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "openhens"
FORBIDDEN_SOLVER_MODULES = [
    "pyomo",
    "pyomo.environ",
    "pyomo.opt",
    "plotly",
    "plotly.graph_objects",
    "kaleido",
    "openpyxl",
    "wakepy",
]


def test_workflow_default_uses_local_synthesis_executor(monkeypatch) -> None:
    used_default = False

    class SpyLocalExecutor(FakeSynthesisExecutor):
        def __init__(self) -> None:
            nonlocal used_default
            used_default = True
            super().__init__()

    monkeypatch.setattr(workflow_module, "LocalSynthesisExecutor", SpyLocalExecutor)
    problem = _small_problem()

    result = workflow_module.execute_open_hens_method(
        problem,
        workflow_settings_from_problem(problem),
    )

    assert used_default
    assert result.accepted_result.network.exchangers


def test_fake_executor_task_graph_uses_successful_topology_only() -> None:
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    pdm_tasks = build_pinch_design_method_tasks(settings)
    pdm_executor = FakeSynthesisExecutor(failures={pdm_tasks[0].task_id})
    pdm_outcomes = pdm_executor.execute(
        pdm_tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=settings.max_parallel,
    )

    tdm_tasks = build_thermal_derivative_method_tasks(settings, pdm_outcomes)
    tdm_executor = FakeSynthesisExecutor(failures={tdm_tasks[0].task_id})
    tdm_outcomes = tdm_executor.execute(
        tdm_tasks,
        problem=problem,
        parent_outcomes=_outcome_map(pdm_outcomes),
        max_parallel=settings.max_parallel,
    )
    esm_tasks = build_network_evolution_method_tasks(settings, tdm_outcomes)

    assert len(pdm_tasks) == 2
    assert len(tdm_tasks) == 2
    assert {task.parent_task_id for task in tdm_tasks} == {pdm_tasks[1].task_id}
    assert all(task.topology_restrictions for task in tdm_tasks)
    assert len(esm_tasks) == 1
    assert esm_tasks[0].parent_task_id == tdm_tasks[1].task_id
    assert esm_tasks[0].topology_restrictions
    assert pdm_executor.stage_order == ["pinch_design_method"]
    assert tdm_executor.stage_order == ["thermal_derivative_method"]


def test_seeded_method_task_builders_use_consistent_method_inputs() -> None:
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    seed = (
        FakeSynthesisExecutor()
        .execute(
            build_pinch_design_method_tasks(settings)[:1],
            problem=problem,
            parent_outcomes={},
            max_parallel=settings.max_parallel,
        )[0]
        .network
    )
    assert seed is not None

    tdm_tasks = build_seeded_thermal_derivative_method_tasks(settings, (seed, seed))
    evolution_tasks = build_seeded_network_evolution_method_tasks(settings, (seed,))

    # A repeated seed topology gives no further tasks: one per threshold.
    assert len(tdm_tasks) == 2
    assert len({task.task_id for task in tdm_tasks}) == 2
    assert {task.method for task in tdm_tasks} == {"thermal_derivative_method"}
    assert {task.seed_network_index for task in tdm_tasks} == {0}
    assert all(task.parent_task_id is None for task in tdm_tasks)
    assert all(task.topology_restrictions for task in tdm_tasks)
    assert len(evolution_tasks) == 1
    assert evolution_tasks[0].method == "network_evolution_method"
    assert evolution_tasks[0].seed_network_index == 0
    assert evolution_tasks[0].topology_restrictions


def test_missing_couenne_skips_derivative_stage_and_runs_evolution() -> None:
    class MissingCouenneForTopologyExecutor(FakeSynthesisExecutor):
        def execute(self, tasks, *, problem, parent_outcomes, max_parallel):
            if tasks and tasks[0].method == "thermal_derivative_method":
                return tuple(
                    HeatExchangerNetworkSynthesisTaskOutcome(
                        task=task,
                        status="failed",
                        solver_status="failed",
                        error=(
                            "The 'couenne' solver executable is required for "
                            "couenne heat exchanger network synthesis solves, but it was not found "
                            "on PATH."
                        ),
                    )
                    for task in tasks
                )
            return super().execute(
                tasks,
                problem=problem,
                parent_outcomes=parent_outcomes,
                max_parallel=max_parallel,
            )

    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)

    with pytest.warns(RuntimeWarning, match="Couenne is unavailable"):
        result = workflow_module.execute_open_hens_method(
            problem,
            settings,
            executor=MissingCouenneForTopologyExecutor(),
        )

    outcomes = result.outcomes
    pdm_success = next(
        outcome
        for outcome in outcomes
        if outcome.status == "success" and outcome.task.method == "pinch_design_method"
    )
    tdm_failures = [
        outcome
        for outcome in outcomes
        if outcome.task.method == "thermal_derivative_method"
    ]
    esm_success = next(
        outcome
        for outcome in outcomes
        if outcome.status == "success"
        and outcome.task.method == "network_evolution_method"
    )

    # The cheapest successful network is accepted, whichever method made it.
    cheapest = min(
        (outcome for outcome in outcomes if outcome.status == "success"),
        key=lambda outcome: outcome.objective_value,
    )
    assert result.accepted_result.task_id == cheapest.task.task_id
    assert all(outcome.status == "failed" for outcome in tdm_failures)
    assert esm_success.task.parent_task_id == pdm_success.task.task_id
    assert esm_success.task.derivative_threshold is None


def test_missing_couenne_before_pdm_runs_direct_evolution() -> None:
    class MissingCouenneForPdmExecutor(FakeSynthesisExecutor):
        def execute(self, tasks, *, problem, parent_outcomes, max_parallel):
            if tasks and tasks[0].method == "pinch_design_method":
                return tuple(
                    HeatExchangerNetworkSynthesisTaskOutcome(
                        task=task,
                        status="failed",
                        solver_status="failed",
                        error=(
                            "The 'couenne' solver executable is required for "
                            "couenne heat exchanger network synthesis solves, but it was not found "
                            "on PATH."
                        ),
                    )
                    for task in tasks
                )
            return super().execute(
                tasks,
                problem=problem,
                parent_outcomes=parent_outcomes,
                max_parallel=max_parallel,
            )

    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)

    with pytest.warns(RuntimeWarning, match="pinch-design-method"):
        result = workflow_module.execute_open_hens_method(
            problem,
            settings,
            executor=MissingCouenneForPdmExecutor(),
        )

    outcomes = result.outcomes
    pdm_failures = [
        outcome for outcome in outcomes if outcome.task.method == "pinch_design_method"
    ]
    tdm_outcomes = [
        outcome
        for outcome in outcomes
        if outcome.task.method == "thermal_derivative_method"
    ]
    esm_successes = [
        outcome
        for outcome in outcomes
        if outcome.status == "success"
        and outcome.task.method == "network_evolution_method"
    ]

    assert all(outcome.status == "failed" for outcome in pdm_failures)
    assert tdm_outcomes == []
    assert esm_successes
    assert all(outcome.task.parent_task_id is None for outcome in esm_successes)
    assert result.accepted_result.method == "network_evolution_method"


class _MissingCouenneExecutor(FakeSynthesisExecutor):
    def __init__(self, missing_for: str) -> None:
        super().__init__()
        self.missing_for = missing_for

    def execute(self, tasks, *, problem, parent_outcomes, max_parallel):
        if tasks and tasks[0].method == self.missing_for:
            return tuple(
                HeatExchangerNetworkSynthesisTaskOutcome(
                    task=task,
                    status="failed",
                    solver_status="failed",
                    error=(
                        "The 'couenne' solver executable is required for "
                        "couenne heat exchanger network synthesis solves, but it was not found "
                        "on PATH."
                    ),
                )
                for task in tasks
            )
        return super().execute(
            tasks,
            problem=problem,
            parent_outcomes=parent_outcomes,
            max_parallel=max_parallel,
        )


def test_failed_workflow_reports_task_errors() -> None:
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    pdm_tasks = build_pinch_design_method_tasks(settings)
    executor = FakeSynthesisExecutor(
        failures={task.task_id for task in pdm_tasks if task.task_id is not None},
    )

    with pytest.raises(WorkflowContractError, match="configured fake executor failure"):
        workflow_module.execute_open_hens_method(
            problem,
            settings,
            executor=executor,
        )


def test_successful_infeasible_outcome_is_demoted_with_its_reason() -> None:
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    pdm_tasks = build_pinch_design_method_tasks(settings)
    outcomes = list(
        FakeSynthesisExecutor().execute(
            pdm_tasks,
            problem=problem,
            parent_outcomes={},
            max_parallel=settings.max_parallel,
        )
    )
    first = outcomes[0]
    assert first.network is not None
    outcomes[0] = first.model_copy(
        update={
            "network": first.network.model_copy(update={"total_annual_cost": 1.0}),
            "objective_value": 1.0,
        }
    )

    # The infeasible outcome is demoted to failed rather than aborting the run;
    # with no feasible outcome left, selection reports why.
    others = [
        outcome.model_copy(update={"status": "failed", "error": "not solved"})
        for outcome in outcomes[1:]
    ]
    with pytest.raises(
        WorkflowContractError,
        match="post-solve feasibility check failed",
    ):
        build_synthesis_result(settings, pdm_tasks, [outcomes[0], *others])


def test_unconverged_outcome_is_demoted_before_ranking() -> None:
    # Its objective is not a converged optimum, so it can't win on cost.
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    pdm_tasks = build_pinch_design_method_tasks(settings)
    outcome = FakeSynthesisExecutor().execute(
        pdm_tasks[:1],
        problem=problem,
        parent_outcomes={},
        max_parallel=1,
    )[0]
    unconverged = outcome.model_copy(
        update={"diagnostics": {"unconverged_termination": "maxTimeLimit"}}
    )

    (demoted,) = demote_infeasible_outcomes([unconverged])

    assert demoted.status == "failed"
    assert demoted.error == "solver stopped before converging (maxTimeLimit)"
    assert demoted.network is not None


def test_synthesis_task_ids_are_deterministic_and_serializable() -> None:
    settings = workflow_settings_from_problem(_small_problem())

    first = build_pinch_design_method_tasks(settings)
    second = build_pinch_design_method_tasks(settings)
    roundtrip = HeatExchangerNetworkSynthesisTask.model_validate_json(
        first[0].model_dump_json()
    )

    assert [task.task_id for task in first] == [task.task_id for task in second]
    assert roundtrip == first[0]


def test_downstream_topology_restrictions_are_required() -> None:
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    pdm_task = build_pinch_design_method_tasks(settings)[0]
    missing_network = HeatExchangerNetworkSynthesisTaskOutcome(
        task=pdm_task,
        status="success",
    )
    empty_network = HeatExchangerNetworkSynthesisTaskOutcome(
        task=pdm_task,
        status="success",
        network=HeatExchangerNetwork(
            run_id=pdm_task.run_id,
            task_id=pdm_task.task_id,
            method=pdm_task.method,
            stage_count=2,
        ),
    )

    with pytest.raises(WorkflowContractError, match="without a HeatExchangerNetwork"):
        build_thermal_derivative_method_tasks(settings, [missing_network])
    # A network without recovery matches is utility-only: a final candidate
    # with nothing for TDM to refine, not a contract error.
    assert build_thermal_derivative_method_tasks(settings, [empty_network]) == ()


def test_fake_outcomes_serialize_without_live_solver_objects() -> None:
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    pdm_tasks = build_pinch_design_method_tasks(settings)
    outcomes = FakeSynthesisExecutor().execute(
        pdm_tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=settings.max_parallel,
    )

    payload = outcomes[0].model_dump(mode="json")
    json.dumps(payload)

    assert payload["status"] == "success"
    assert payload["network"]["exchangers"]
    assert "problem" not in payload
    assert "solver_model" not in payload


def test_topology_restriction_handoff_uses_source_shaped_duty_values() -> None:
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    pdm_tasks = build_pinch_design_method_tasks(settings)
    pdm_outcomes = FakeSynthesisExecutor().execute(
        pdm_tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=settings.max_parallel,
    )
    tdm_task = build_thermal_derivative_method_tasks(settings, pdm_outcomes)[0]
    arrays = problem_to_solver_arrays(problem, dTmin=0.1)

    z_restriction, zhu_restriction, zcu_restriction = _topology_restriction_values(
        tdm_task, arrays
    )
    flat_restriction_cells = [
        cell
        for hot_matches in z_restriction
        for cold_matches in hot_matches
        for cell in cold_matches
    ]
    flat_restriction_values = [cell[0] for cell in flat_restriction_cells]

    assert zhu_restriction is None
    assert zcu_restriction is None
    assert all(
        isinstance(cell, list) and len(cell) == 1 for cell in flat_restriction_cells
    )
    assert 0.0 in flat_restriction_values
    assert max(flat_restriction_values) == max(
        restriction.duty for restriction in tdm_task.topology_restrictions
    )


def test_direct_design_run_populates_results_cache_and_preserves_targets(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()
    target_output = problem.target.all_heat_integration()

    design = problem.design.heat_exchanger_network()

    assert problem.results is not None
    assert problem._results is not problem.results
    assert problem._results.model_dump(mode="json") == problem.results.model_dump(
        mode="json"
    )
    assert problem.results.design == design.result
    assert problem.results.targets == target_output.targets
    assert design.selected_network.exchangers
    assert design.result.objective_values["total_annual_cost"] > 0
    assert design.result.design_method == HeatExchangerNetworkDesignMethod.OpenHENS
    assert design.result.manifest is not None
    assert (
        design.result.manifest.design_method
        == HeatExchangerNetworkDesignMethod.OpenHENS
    )
    manifest = design.result.manifest
    assert (manifest.search_width, manifest.search_depth) == (1, 2)


def _resolved_depth(design) -> int:
    assert design.result.manifest is not None
    return design.result.manifest.search_depth


def _reload_from_problem_json(problem: PinchProblem) -> PinchProblem:
    return PinchProblem(
        source=json.loads(json.dumps(problem.to_problem_json())),
        project_name=problem.project_name,
    )


def _reload_from_bundle(problem: PinchProblem) -> PinchProblem:
    workspace = PinchWorkspace()
    workspace.load(problem, case_name="case")
    bundle = workspace.to_bundle()
    reloaded = PinchWorkspace.from_bundle(
        type(bundle).model_validate(json.loads(bundle.model_dump_json()))
    )
    return reloaded.case("case")


def _two_period_problem_with_options(options: dict) -> PinchProblem:
    payload = _duplicate_two_state_no_sweep_payload()
    payload["options"].update(options)
    return PinchProblem(source=payload, project_name="two-period")


@pytest.mark.synthesis
@pytest.mark.solver
# A string condition is evaluated when the test runs, so collecting this module
# does not import the solver packages.
@pytest.mark.skipif(
    "not _live_solver_environment_available()",
    reason="needs the couenne and ipopt binaries and the pyomo package",
)
def test_duplicate_two_state_case_matches_single_state_without_sweeps() -> None:
    single_period = _single_state_no_sweep_problem()
    duplicate_period = _duplicate_two_state_no_sweep_problem()

    single_result = _run_single_candidate_open_hens(
        single_period,
        dt_cont_multiplier=1.0,
    )
    duplicate_result = _run_single_candidate_open_hens(
        duplicate_period,
        dt_cont_multiplier=1.0,
    )

    assert _task_counts_by_method(single_result) == {
        "network_evolution_method": 1,
        "pinch_design_method": 1,
        "thermal_derivative_method": 1,
    }
    assert _task_counts_by_method(duplicate_result) == _task_counts_by_method(
        single_result
    )
    assert _successful_counts_by_method(duplicate_result) == (
        _successful_counts_by_method(single_result)
    )

    single_design = single_result.accepted_result
    duplicate_design = duplicate_result.accepted_result
    assert single_design.method == "network_evolution_method"
    assert duplicate_design.method == "network_evolution_method"
    assert duplicate_design.objective_values["total_annual_cost"] == pytest.approx(
        single_design.objective_values["total_annual_cost"],
        abs=1e-5,
        rel=1e-9,
    )
    assert duplicate_design.objective_values["capital_cost"] == pytest.approx(
        single_design.objective_values["capital_cost"],
        abs=1e-5,
        rel=1e-9,
    )
    assert duplicate_design.objective_values["utility_cost"] == pytest.approx(
        single_design.objective_values["utility_cost"],
        abs=1e-5,
        rel=1e-9,
    )

    single_network = single_design.network
    duplicate_network = duplicate_design.network
    assert single_network.stage_count == duplicate_network.stage_count
    assert (
        single_network.summary_metrics["recovery_units"]
        == (duplicate_network.summary_metrics["recovery_units"])
    )
    assert (
        single_network.summary_metrics["hot_utility_units"]
        == (duplicate_network.summary_metrics["hot_utility_units"])
    )
    assert (
        single_network.summary_metrics["cold_utility_units"]
        == (duplicate_network.summary_metrics["cold_utility_units"])
    )
    assert network_structure_signature(duplicate_network) == (
        network_structure_signature(single_network)
    )
    single_period_id = single_network.period_ids[0]
    for kind in HeatExchangerKind:
        for duplicate_period_id in duplicate_network.period_ids:
            assert duplicate_network.total_duty(
                kind=kind,
                period_id=duplicate_period_id,
            ) == pytest.approx(
                single_network.total_duty(
                    kind=kind,
                    period_id=single_period_id,
                ),
                abs=1e-5,
                rel=1e-9,
            )


def test_direct_design_run_computes_targets_when_cache_is_empty(monkeypatch) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()

    design = problem.design.heat_exchanger_network()

    assert problem.results is not None
    assert problem.results.targets
    assert problem.results.design == design.result


def test_explicit_pdm_tdm_and_evolution_design_methods_share_result_shape(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()

    pdm = problem.design.pinch_design()
    tdm = problem.design.thermal_derivative()
    evolution = problem.design.network_evolution((tdm.selected_network,))

    assert pdm.result.method == "pinch_design_method"
    assert tdm.result.method == "thermal_derivative_method"
    assert evolution.result.method == "network_evolution_method"
    assert problem.results is not None
    assert problem.results.design == evolution.result
    assert all(outcome.network is not None for outcome in pdm.result.ranked_networks)
    assert all(outcome.network is not None for outcome in tdm.result.ranked_networks)
    assert all(
        outcome.network is not None for outcome in evolution.result.ranked_networks
    )
    assert {outcome.task.method for outcome in tdm.result.ranked_networks} == {
        "thermal_derivative_method",
    }
    assert {outcome.task.method for outcome in evolution.result.ranked_networks} == {
        "network_evolution_method",
    }


def test_named_design_accessor_dispatches_each_hen_design_method(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()

    pdm = problem.design.pinch_design()
    tdm = problem.design.thermal_derivative()
    evolution = problem.design.network_evolution((tdm.selected_network,))

    assert pdm.result.design_method == HeatExchangerNetworkDesignMethod.PinchDesign
    assert pdm.result.method == "pinch_design_method"
    assert (
        tdm.result.design_method == HeatExchangerNetworkDesignMethod.ThermalDerivative
    )
    assert tdm.result.method == "thermal_derivative_method"
    assert (
        evolution.result.design_method
        == HeatExchangerNetworkDesignMethod.NetworkEvolution
    )
    assert evolution.result.method == "network_evolution_method"


def test_design_accessor_rejects_retired_selector_and_invalid_seed_shapes(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()
    seed = problem.design.pinch_design().selected_network

    with pytest.raises(TypeError, match="unexpected keyword argument 'method'"):
        problem.design.heat_exchanger_network(method="not_a_method")

    # The default workflow takes no seed networks; the argument always raised.
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        problem.design.heat_exchanger_network(initial_networks=(seed,))

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        problem.design.pinch_design(initial_networks=(seed,))


def test_seeded_design_methods_require_seed_or_cached_design(monkeypatch) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()

    with pytest.raises(ValueError, match="initial_networks"):
        problem.design.thermal_derivative()

    with pytest.raises(ValueError, match="initial_networks"):
        problem.design.network_evolution()


def test_direct_design_run_reports_absolute_temperatures_in_kelvin(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    payload = _small_payload()
    for record in payload["streams"] + payload["utilities"]:
        for field in ("t_supply", "t_target"):
            record[field]["value"] -= 273.15
            record[field]["unit"] = "degC"
    problem = PinchProblem(source=payload, project_name="HENS Celsius Demo")

    design = problem.design.heat_exchanger_network()
    recovery = next(
        exchanger
        for exchanger in design.selected_network.exchangers
        if exchanger.kind is HeatExchangerKind.RECOVERY
    )

    assert recovery.state().source_inlet_temperature == pytest.approx(650.0)
    assert recovery.state().source_outlet_temperature > 273.15
    assert recovery.state().sink_inlet_temperature == pytest.approx(410.0)
    assert recovery.state().sink_outlet_temperature > 273.15
    assert design.selected_network.summary_metrics[
        "dt_cont_multiplier"
    ] == pytest.approx(0.2)


def test_workspace_design_workflow_dispatches_to_live_problem_design_path(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    workspace = PinchWorkspace(_small_payload(), project_name="HENS Demo")

    design = workspace.design.heat_exchanger_network()
    problem = workspace.case("baseline")

    assert design.selected_network.exchangers
    assert problem.results is not None
    assert problem.results.design is not None
    assert problem.results.design.problem_id == problem.project_name
    assert not hasattr(problem.target, "heat_exchanger_network_synthesis")


def test_optional_exports_round_trip_from_problem_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem(project_name="Export Demo")
    problem.design.heat_exchanger_network()

    manifest = export_heat_exchanger_network_synthesis_results(
        problem,
        tmp_path,
        workspace_variant="variant-a",
    )

    manifest_path = tmp_path / "manifest.json"
    summary_path = tmp_path / "metrics" / "run_summary.csv"
    solution_metrics_path = tmp_path / "metrics" / "solution_metrics.csv"
    loaded_manifest = HeatExchangerNetworkSynthesisManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    first_task_record = next(
        record
        for record in manifest.export_records
        if record.record_id.startswith("task:")
    )
    task_outcome = HeatExchangerNetworkSynthesisTaskOutcome.model_validate_json(
        (tmp_path / first_task_record.path).read_text(encoding="utf-8")
    )

    assert loaded_manifest == manifest
    assert manifest.problem_id == "Export Demo"
    assert manifest.workspace_variant == "variant-a"
    assert task_outcome.network is not None
    assert summary_path.exists()
    assert solution_metrics_path.exists()
    with summary_path.open(encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert summary_rows[0]["problem_id"] == "Export Demo"
    assert summary_rows[0]["workspace_variant"] == "variant-a"


def test_hen_synthesis_has_no_public_target_or_root_service_bypass() -> None:
    problem = PinchProblem()

    assert not hasattr(problem.target, "heat_exchanger_network_synthesis")
    assert not hasattr(OpenPinch, "heat_exchanger_network_synthesis_service")
    assert OpenPinch.__all__ == ["PinchProblem", "PinchWorkspace"]
    assert not hasattr(hens_package, "heat_exchanger_network_synthesis_service")
    assert not hasattr(hens_package, "run_synthesis_workflow")
    assert not hasattr(workflow_module, "run_synthesis_workflow")

    payload: Any = _small_payload()
    with pytest.raises(TypeError, match="PinchProblem"):
        heat_exchanger_network_synthesis_service(payload)


def test_synthesis_imports_do_not_load_live_solver_dependencies() -> None:
    script = f"""
import sys

import OpenPinch
from OpenPinch.application.problem import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import FakeSynthesisExecutor

problem = PinchProblem()
_ = problem.design
_ = FakeSynthesisExecutor

forbidden = {FORBIDDEN_SOLVER_MODULES!r}
loaded = [name for name in forbidden if name in sys.modules]
if loaded:
    raise SystemExit(f"loaded optional solver dependencies: {{loaded}}")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def _small_problem(*, project_name: str = "HENS Demo") -> PinchProblem:
    return PinchProblem(source=_small_payload(), project_name=project_name)


def _small_problem_with_options(
    options: dict,
    *,
    project_name: str = "HENS Demo",
) -> PinchProblem:
    payload = _small_payload()
    payload["options"] = {**payload["options"], **options}
    return PinchProblem(source=payload, project_name=project_name)


def _single_state_no_sweep_problem() -> PinchProblem:
    return PinchProblem(
        source=_single_state_no_sweep_payload(),
        project_name="single-state-no-sweep",
    )


def _duplicate_two_state_no_sweep_problem() -> PinchProblem:
    return PinchProblem(
        source=_duplicate_two_state_no_sweep_payload(),
        project_name="duplicate-two-state-no-sweep",
    )


def _duplicate_two_state_no_sweep_payload() -> dict:
    payload = _single_state_no_sweep_payload()
    for record in payload["streams"]:
        for key, value in list(record.items()):
            if _is_scalar_quantity_payload(value):
                record[key] = {
                    **value,
                    "values": [value["value"], value["value"]],
                }
                del record[key]["value"]
    payload["options"] = {
        **payload["options"],
        "PROBLEM_PERIOD_IDS": ["base", "duplicate"],
        "PROBLEM_PERIOD_WEIGHTS": [0.1, 0.9],
        "HENS_RUN_ID": "duplicate-two-state-no-sweep",
    }
    return payload


def _single_state_no_sweep_payload() -> dict:
    return {
        "streams": [
            {
                "heat_capacity_flowrate": {
                    "unit": "kW/delta_degC",
                    "value": 10.0,
                },
                "heat_flow": {"unit": "kW", "value": 2000.0},
                "htc": {"unit": "kW/m^2/K", "value": 1.0},
                "name": "H1",
                "t_supply": {"unit": "K", "value": 500.0},
                "t_target": {"unit": "K", "value": 300.0},
                "zone": "Site/Process A",
            },
            {
                "heat_capacity_flowrate": {
                    "unit": "kW/delta_degC",
                    "value": 5.0,
                },
                "heat_flow": {"unit": "kW", "value": 1000.0},
                "htc": {"unit": "kW/m^2/K", "value": 1.0},
                "name": "C1",
                "t_supply": {"unit": "K", "value": 450.0},
                "t_target": {"unit": "K", "value": 650.0},
                "zone": "Site/Process A",
            },
        ],
        "utilities": [
            {
                "heat_flow": None,
                "htc": {"unit": "kW/m^2/K", "value": 5.0},
                "name": "HU",
                "price": {"unit": "$/MWh", "value": 80.0},
                "t_supply": {"unit": "K", "value": 700.0},
                "t_target": {"unit": "K", "value": 700.0},
                "type": "Hot",
            },
            {
                "heat_flow": None,
                "htc": {"unit": "kW/m^2/K", "value": 1.0},
                "name": "CU",
                "price": {"unit": "$/MWh", "value": 15.0},
                "t_supply": {"unit": "K", "value": 290.0},
                "t_target": {"unit": "K", "value": 310.0},
                "type": "Cold",
            },
        ],
        "options": {
            # Yee and Grossmann (1990) costs: $/kW/y prices and $/y exchangers.
            "COSTING_ANNUAL_OP_TIME": 1000.0,
            "COSTING_DISCOUNT_RATE": 0.0,
            "COSTING_SERVICE_LIFE": 1.0,
            "COSTING_HX_AREA_COEFF": 150.0,
            "COSTING_HX_AREA_EXP": 1.0,
            "COSTING_HX_UNIT_COST": 5500.0,
            "HENS_DT_CONT_MULTIPLIERS": [1.0],
            "HENS_BEST_SOLUTIONS_TO_SAVE": 1,
            "HENS_DERIVATIVE_THRESHOLDS": [0.5],
            "HENS_LOG_LEVEL": "WARNING",
            "HENS_MAX_PARALLEL": 1,
            "HENS_OUTPUT_FORMATS": [],
            "HENS_RUN_ID": "single-state-no-sweep",
            "HENS_SOLVER_EVM": "ipopt-pyomo",
            "HENS_SOLVER_OPTIONS_EVM": {},
            "HENS_SOLVER_OPTIONS_PDM": {},
            "HENS_SOLVER_OPTIONS_TDM": {},
            "HENS_SOLVER_PDM": "couenne",
            "HENS_SOLVER_TDM": "couenne",
            "HENS_SOLVE_TOLERANCE": 0.001,
            "HENS_STAGE_SELECTION": [1],
            "HENS_SEARCH_DEPTH": 2,
            # The MILP pinch design -> TDM -> EVM pipeline, without the
            # heuristic.
            "HENS_HEURISTIC_PDM_NETWORKS": 0,
        },
    }


def _is_scalar_quantity_payload(value: object) -> bool:
    return (
        isinstance(value, dict)
        and "value" in value
        and isinstance(value["value"], int | float)
    )


def _run_single_candidate_open_hens(
    problem: PinchProblem,
    *,
    dt_cont_multiplier: float = 1.4,
):
    problem.target.direct_heat_integration()
    settings = workflow_settings_from_problem(problem)
    assert settings.dt_cont_multipliers == (dt_cont_multiplier,)
    assert settings.derivative_thresholds == (0.5,)
    result = workflow_module.execute_open_hens_method(
        problem,
        settings,
        executor=LocalSynthesisExecutor(print_output=False),
    )
    assert result.accepted_result.network is not None
    return result


def _task_counts_by_method(result) -> dict[str, int]:
    return dict(Counter(task.method for task in result.tasks))


def _successful_counts_by_method(result) -> dict[str, int]:
    return dict(
        Counter(
            outcome.task.method
            for outcome in result.outcomes
            if outcome.status == "success"
        )
    )


def _live_solver_environment_available() -> bool:
    try:
        require_solver_binary(
            "couenne",
            purpose="single-period/multiperiod HeatExchangerNetworkLabel equivalence test",
        )
        require_solver_binary(
            "ipopt",
            purpose="single-period/multiperiod HeatExchangerNetworkLabel equivalence test",
        )
        require_synthesis_dependency(
            "pyomo.environ",
            package="pyomo",
            purpose="single-period/multiperiod HeatExchangerNetworkLabel equivalence test",
        )
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


def _use_fake_default_executor(monkeypatch) -> None:
    from OpenPinch.analysis.heat_exchanger_networks.targeting import (
        network_evolution_method,
        pinch_design_method,
        thermal_derivative_method,
    )

    for module in (
        workflow_module,
        network_evolution_method,
        pinch_design_method,
        thermal_derivative_method,
    ):
        monkeypatch.setattr(module, "LocalSynthesisExecutor", FakeSynthesisExecutor)


def _small_payload() -> dict:
    payload = json.loads(
        (FIXTURE_ROOT / "Four-stream-Yee-and-Grossmann-1990-1.json").read_text(
            encoding="utf-8"
        )
    )
    payload = deepcopy(payload)
    payload["options"] = {
        **payload["options"],
        "HENS_DT_CONT_MULTIPLIERS": [0.2, 0.4],
        "HENS_DERIVATIVE_THRESHOLDS": [0.5, 1.0],
        "HENS_STAGE_SELECTION": [2],
        "HENS_BEST_SOLUTIONS_TO_SAVE": 2,
        "HENS_MAX_PARALLEL": 1,
        "HENS_OUTPUT_FORMATS": ["json", "csv"],
        "HENS_RUN_ID": "hens-05-test",
        # These tests cover the MILP pinch design -> TDM -> EVM pipeline, so
        # the heuristic, the default pinch design method, is off.
        "HENS_HEURISTIC_PDM_NETWORKS": 0,
    }
    return payload


def _outcome_map(outcomes):
    return {
        outcome.task.task_id: outcome
        for outcome in outcomes
        if outcome.task.task_id is not None
    }


def test_default_design_runs_every_dt_cont_multiplier(monkeypatch) -> None:
    # Audit H8: the default entry built tasks only from the first sweep
    # value, while the manifest listed every one.
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem_with_options({"HENS_DT_CONT_MULTIPLIERS": [0.2, 0.4, 0.8]})

    design = problem.design.heat_exchanger_network()

    manifest = design.result.manifest
    assert manifest.task_count_by_method["pinch_design_method"] == 3
    assert manifest.dt_cont_multipliers == (0.2, 0.4, 0.8)


def test_manifest_lists_the_dt_cont_multipliers_that_ran(monkeypatch) -> None:
    # Unset, the search width sets the multipliers.
    _use_fake_default_executor(monkeypatch)
    payload = _small_payload()
    del payload["options"]["HENS_DT_CONT_MULTIPLIERS"]
    problem = PinchProblem(source=payload, project_name="HENS Demo")

    design = problem.design.heat_exchanger_network(width=2)

    assert design.result.manifest.dt_cont_multipliers == (0.5, 0.75, 1.0, 1.5, 2.0)


def test_call_argument_beats_call_options_beats_the_depth_the_user_set(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem_with_options({"HENS_SEARCH_DEPTH": 4})

    from_argument = problem.design.heat_exchanger_network(
        depth=3, options={"HENS_SEARCH_DEPTH": 1}
    )
    from_options = problem.design.heat_exchanger_network(
        options={"HENS_SEARCH_DEPTH": 1}
    )
    stored = problem.design.heat_exchanger_network()

    assert _resolved_depth(from_argument) == 3
    assert _resolved_depth(from_options) == 1
    assert _resolved_depth(stored) == 4
    assert problem.to_problem_json()["options"]["HENS_SEARCH_DEPTH"] == 4


@pytest.mark.parametrize("reload", [_reload_from_problem_json, _reload_from_bundle])
def test_a_set_depth_survives_a_reload(monkeypatch, reload) -> None:
    _use_fake_default_executor(monkeypatch)
    configured = reload(_small_problem_with_options({"HENS_SEARCH_DEPTH": 1}))
    unset = reload(_small_problem())

    assert _resolved_depth(configured.design.heat_exchanger_network()) == 1
    assert _resolved_depth(unset.design.heat_exchanger_network()) == 2


@pytest.mark.parametrize(
    ("options", "expected_depth"), [({}, 2), ({"HENS_SEARCH_DEPTH": 3}, 3)]
)
def test_period_zone_and_multiperiod_designs_resolve_the_same_depth(
    monkeypatch,
    options: dict,
    expected_depth: int,
) -> None:
    # The fake executor has no multi-period networks, so stop at the dispatch.
    class DepthResolved(Exception):
        pass

    def stop_at_dispatch(problem, settings, **kwargs):
        raise DepthResolved(settings.search_depth)

    monkeypatch.setattr(service_module, "execute_open_hens_method", stop_at_dispatch)
    problem = _two_period_problem_with_options(options)

    with pytest.raises(DepthResolved) as for_period:
        problem.design.heat_exchanger_network(period_id="base")
    with pytest.raises(DepthResolved) as for_all_periods:
        problem.design.multiperiod_heat_exchanger_network(prerequisites=True)

    assert for_period.value.args == (expected_depth,)
    assert for_all_periods.value.args == (expected_depth,)


@pytest.mark.parametrize(("width", "depth"), [(-1, 2), (4, 2), (1, -1), (1, 5)])
def test_out_of_range_width_or_depth_is_rejected(monkeypatch, width, depth) -> None:
    _use_fake_default_executor(monkeypatch)

    with pytest.raises(ValueError, match="HENS_SEARCH_"):
        _small_problem().design.heat_exchanger_network(width=width, depth=depth)
