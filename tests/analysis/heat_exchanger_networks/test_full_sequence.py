"""HENS-05 workflow, result-cache, export, and API-boundary tests."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

import OpenPinch
import OpenPinch.analysis.heat_exchanger_networks as hens_package
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
from OpenPinch.domain.enums import HeatExchangerKind, HENDesignMethod
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.support.paths import REPOSITORY_ROOT

REPO_ROOT = REPOSITORY_ROOT
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "openhens"
FORBIDDEN_SOLVER_MODULES = [
    "gekko",
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


def test_quality_tier_zero_skips_thermal_derivative_stage() -> None:
    problem = _small_problem_with_options({"HENS_SYNTHESIS_QUALITY_TIER": 0})
    settings = workflow_settings_from_problem(problem)
    executor = FakeSynthesisExecutor()

    result = workflow_module.execute_open_hens_method(
        problem,
        settings,
        executor=executor,
    )

    assert executor.stage_order == [
        "pinch_design_method",
        "network_evolution_method",
    ]
    assert not any(task.method == "thermal_derivative_method" for task in result.tasks)
    assert all(
        outcome.task.method != "thermal_derivative_method"
        for outcome in result.outcomes
    )


def test_quality_tier_one_keeps_standard_pdm_task_grid() -> None:
    settings = workflow_settings_from_problem(_small_problem())

    tasks = build_pinch_design_method_tasks(settings)

    assert settings.synthesis_quality_tier == 1
    assert len(tasks) == 2
    assert all(
        task.metadata.get("pathway_ids") == ["tier1-open-hens"] for task in tasks
    )
    assert all(not task.settings for task in tasks)


def test_quality_tier_three_expands_pdm_candidates() -> None:
    problem = _small_problem_with_options({"HENS_SYNTHESIS_QUALITY_TIER": 3})
    settings = workflow_settings_from_problem(problem)

    tasks = build_pinch_design_method_tasks(settings)

    assert settings.quality_dt_cont_multipliers == (1.0, 2.0)
    assert settings.quality_pdm_stage_pair_count == 0
    assert settings.effective_evm_n_ad_branches == 1
    assert settings.effective_evm_n_rm_branches == 1
    assert len(tasks) == 6
    assert {task.settings.get("pdm_mode") for task in tasks if task.settings} == {
        "compact",
        "raw",
    }
    assert any(
        "tier1-open-hens" in task.metadata.get("pathway_ids", ()) for task in tasks
    )
    assert any(
        "tier3-compact-1" in task.metadata.get("pathway_ids", ()) for task in tasks
    )
    assert any(
        task.approach_temperature == 4.0
        and "tier3-compact-2" in task.metadata.get("pathway_ids", ())
        for task in tasks
    )
    assert any(
        task.approach_temperature == 4.0
        and "tier3-raw-2" in task.metadata.get("pathway_ids", ())
        for task in tasks
    )


def test_quality_tier_four_adds_evm_branch_breadth_to_dtmin_sweep() -> None:
    problem = _small_problem_with_options({"HENS_SYNTHESIS_QUALITY_TIER": 4})
    settings = workflow_settings_from_problem(problem)

    tasks = build_pinch_design_method_tasks(settings)

    assert settings.quality_dt_cont_multipliers == (1.0, 2.0)
    assert settings.effective_evm_n_ad_branches == 2
    assert settings.effective_evm_n_rm_branches == 2
    assert len(tasks) == 6
    assert any(
        task.approach_temperature == 2.0
        and "tier4-compact-1" in task.metadata.get("pathway_ids", ())
        for task in tasks
    )
    assert any(
        task.approach_temperature == 4.0
        and "tier4-raw-2" in task.metadata.get("pathway_ids", ())
        for task in tasks
    )


def test_quality_tier_five_adds_experimental_reduced_dtmin_without_4x() -> None:
    problem = _small_problem_with_options({"HENS_SYNTHESIS_QUALITY_TIER": 5})
    settings = workflow_settings_from_problem(problem)

    tasks = build_pinch_design_method_tasks(problem=problem, settings=settings)

    assert settings.quality_dt_cont_multipliers == (0.5, 1.0, 2.0)
    assert settings.quality_pdm_stage_pair_count == 0
    assert settings.quality_derivative_thresholds == (0.5, 1.0)
    assert settings.effective_evm_n_ad_branches == 2
    assert settings.effective_evm_n_rm_branches == 2
    assert any(
        task.approach_temperature == 1.0
        and "tier5-compact-0p5" in task.metadata.get("pathway_ids", ())
        for task in tasks
    )
    assert not any(
        "tier5-compact-4" in task.metadata.get("pathway_ids", ())
        or "tier5-raw-4" in task.metadata.get("pathway_ids", ())
        for task in tasks
    )


def test_synthesis_quality_tier_five_retains_experimental_limits() -> None:
    settings = workflow_settings_from_problem(
        _small_problem_with_options({"HENS_SYNTHESIS_QUALITY_TIER": 5})
    )

    assert settings.quality_dt_cont_multipliers == (0.5, 1.0, 2.0)
    assert settings.quality_pdm_stage_pair_count == 0
    assert settings.quality_derivative_thresholds == (0.5, 1.0)
    assert settings.effective_evm_n_ad_branches == 2
    assert settings.effective_evm_n_rm_branches == 2


def test_user_dt_cont_multipliers_are_applied_as_multipliers() -> None:
    problem = _small_problem_with_options(
        {
            "HENS_SYNTHESIS_QUALITY_TIER": 3,
            "HENS_APPROACH_TEMPERATURES": [10.0, 20.0],
            "HENS_DT_CONT_MULTIPLIERS": [1.0, 1.5],
        }
    )
    settings = workflow_settings_from_problem(problem)

    tasks = build_pinch_design_method_tasks(settings)

    assert settings.quality_pdm_approach_temperatures == (10.0, 15.0)
    assert any(
        task.approach_temperature == 15.0
        and "tier3-compact-1p5" in task.metadata.get("pathway_ids", ())
        for task in tasks
    )
    assert not any(task.approach_temperature in {100.0, 200.0} for task in tasks)


def test_quality_tier_two_contains_tier_zero_and_tier_one_paths() -> None:
    problem = _small_problem_with_options({"HENS_SYNTHESIS_QUALITY_TIER": 2})
    settings = workflow_settings_from_problem(problem)
    executor = FakeSynthesisExecutor()

    result = workflow_module.execute_open_hens_method(
        problem,
        settings,
        executor=executor,
    )

    pdm_tasks = [task for task in result.tasks if task.method == "pinch_design_method"]
    tdm_tasks = [
        task for task in result.tasks if task.method == "thermal_derivative_method"
    ]
    evm_tasks = [
        task for task in result.tasks if task.method == "network_evolution_method"
    ]

    assert any(task.settings.get("pdm_mode") == "compact" for task in pdm_tasks)
    assert any(
        not task.settings and task.metadata.get("pathway_ids") == ["tier1-open-hens"]
        for task in pdm_tasks
    )
    assert any(
        "tier0-compact-1" in task.metadata.get("pathway_ids", ()) for task in evm_tasks
    )
    assert any(
        task.metadata.get("pathway_ids") == ["tier1-open-hens"] for task in evm_tasks
    )
    assert not any(
        "tier2-compact-1" in task.metadata.get("pathway_ids", ()) for task in tdm_tasks
    )
    assert any(
        "tier2-compact-1" in task.metadata.get("pathway_ids", ()) for task in evm_tasks
    )
    assert result.accepted_result.manifest is not None
    assert result.accepted_result.manifest.task_count_by_method[
        "network_evolution_method"
    ] == len(evm_tasks)


def test_higher_tier_can_select_protected_lower_tier_evm_result() -> None:
    class ProtectedTierWinsExecutor(FakeSynthesisExecutor):
        def execute(self, tasks, *, problem, parent_outcomes, max_parallel):
            outcomes = super().execute(
                tasks,
                problem=problem,
                parent_outcomes=parent_outcomes,
                max_parallel=max_parallel,
            )
            return tuple(
                _with_pathway_tac(
                    outcome,
                    tier1_tac=100.0,
                    other_tac=500.0,
                )
                for outcome in outcomes
            )

    problem = _small_problem_with_options({"HENS_SYNTHESIS_QUALITY_TIER": 2})
    settings = workflow_settings_from_problem(problem)

    result = workflow_module.execute_open_hens_method(
        problem,
        settings,
        executor=ProtectedTierWinsExecutor(),
    )

    manifest = result.accepted_result.manifest
    assert manifest is not None
    assert result.accepted_result.objective_values["total_annual_cost"] == 100.0
    assert manifest.selected_pathway_id == "tier1-open-hens"
    assert manifest.selected_protected_pathway is True


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

    assert len(tdm_tasks) == 4
    assert len({task.task_id for task in tdm_tasks}) == 4
    assert {task.method for task in tdm_tasks} == {"thermal_derivative_method"}
    assert {task.seed_network_index for task in tdm_tasks} == {0, 1}
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

    assert result.accepted_result.method == "network_evolution_method"
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


def test_successful_infeasible_outcome_reports_model_contract_issue() -> None:
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

    with pytest.raises(
        WorkflowContractError,
        match="solver-success heat exchanger network task failed post-solve",
    ):
        build_synthesis_result(settings, pdm_tasks, outcomes)


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
    with pytest.raises(WorkflowContractError, match="topology restrictions"):
        build_thermal_derivative_method_tasks(settings, [empty_network])


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
    target_output = problem.target()

    design = problem.design.heat_exchanger_network_synthesis()

    assert problem.results is not None
    assert problem._results is problem.results
    assert problem.results.design == design
    assert problem.results.targets == target_output.targets
    assert design.network.exchangers
    assert design.objective_values["total_annual_cost"] > 0
    assert design.design_method == HENDesignMethod.OpenHENS
    assert design.manifest is not None
    assert design.manifest.design_method == HENDesignMethod.OpenHENS
    assert design.manifest.synthesis_quality_tier == 0


def test_open_hens_method_accessor_runs_original_tier_one(monkeypatch) -> None:
    _use_fake_default_executor(monkeypatch)
    configured_problem = _small_problem_with_options({"HENS_SYNTHESIS_QUALITY_TIER": 4})
    original_config_tier = (
        configured_problem.master_zone.config.hens.synthesis_quality_tier
    )

    design = configured_problem.design.open_hens_method()

    assert design.design_method == HENDesignMethod.OpenHENS
    assert design.manifest is not None
    assert design.manifest.synthesis_quality_tier == 1
    assert configured_problem.master_zone.config.hens.synthesis_quality_tier == (
        original_config_tier
    )


def test_default_dispatch_is_fast_tier_zero_and_explicit_openhens_is_tier_one(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    default_problem = _small_problem_with_options({"HENS_SYNTHESIS_QUALITY_TIER": 4})
    explicit_problem = _small_problem()

    default_design = default_problem.design.heat_exchanger_network_synthesis()
    explicit_design = explicit_problem.design.heat_exchanger_network_synthesis(
        method=HENDesignMethod.OpenHENS,
    )

    assert default_design.design_method == HENDesignMethod.OpenHENS
    assert explicit_design.design_method == HENDesignMethod.OpenHENS
    assert default_design.manifest is not None
    assert explicit_design.manifest is not None
    assert default_design.manifest.synthesis_quality_tier == 0
    assert explicit_design.manifest.synthesis_quality_tier == 1
    assert default_problem.master_zone.config.hens.synthesis_quality_tier == 4


def test_enhanced_synthesis_method_runs_requested_quality_tier(monkeypatch) -> None:
    _use_fake_default_executor(monkeypatch)
    default_problem = _small_problem()
    explicit_problem = _small_problem_with_options({"HENS_SYNTHESIS_QUALITY_TIER": 5})
    original_config_tier = (
        explicit_problem.master_zone.config.hens.synthesis_quality_tier
    )

    default_design = default_problem.design.enhanced_synthesis_method()
    explicit_design = explicit_problem.design.enhanced_synthesis_method(
        quality_tier=3,
    )

    assert default_design.design_method == HENDesignMethod.OpenHENS
    assert explicit_design.design_method == HENDesignMethod.OpenHENS
    assert default_design.manifest is not None
    assert explicit_design.manifest is not None
    assert default_design.manifest.synthesis_quality_tier == 2
    assert explicit_design.manifest.synthesis_quality_tier == 3
    assert explicit_problem.master_zone.config.hens.synthesis_quality_tier == (
        original_config_tier
    )


@pytest.mark.parametrize("quality_tier", [-1, 6])
def test_enhanced_synthesis_method_rejects_out_of_range_quality_tier(
    monkeypatch,
    quality_tier: int,
) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()

    with pytest.raises(ValueError, match="quality_tier"):
        problem.design.enhanced_synthesis_method(quality_tier=quality_tier)


@pytest.mark.parametrize("quality_tier", [1.5, "2", True])
def test_enhanced_synthesis_method_rejects_non_integer_quality_tier(
    monkeypatch,
    quality_tier: object,
) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()

    with pytest.raises(TypeError, match="quality_tier"):
        problem.design.enhanced_synthesis_method(quality_tier=quality_tier)  # type: ignore[arg-type]


@pytest.mark.synthesis
@pytest.mark.solver
def test_duplicate_two_state_case_matches_single_state_without_sweeps() -> None:
    _skip_if_live_solver_environment_missing()
    single_period = _single_state_no_sweep_problem()
    duplicate_period = _duplicate_two_state_no_sweep_problem()

    single_result = _run_single_candidate_open_hens(
        single_period,
        approach_temperature=10.0,
    )
    duplicate_result = _run_single_candidate_open_hens(
        duplicate_period,
        approach_temperature=10.0,
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

    design = problem.design.heat_exchanger_network_synthesis()

    assert problem.results is not None
    assert problem.results.targets
    assert problem.results.design == design


def test_explicit_pdm_tdm_and_evolution_design_methods_share_result_shape(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()

    pdm = problem.design.pinch_design_method()
    tdm = problem.design.thermal_derivative_method()
    evolution = problem.design.network_evolution_method((tdm.network,))

    assert pdm.method == "pinch_design_method"
    assert tdm.method == "thermal_derivative_method"
    assert evolution.method == "network_evolution_method"
    assert problem.results is not None
    assert problem.results.design == evolution
    assert all(outcome.network is not None for outcome in pdm.ranked_networks)
    assert all(outcome.network is not None for outcome in tdm.ranked_networks)
    assert all(outcome.network is not None for outcome in evolution.ranked_networks)
    assert {outcome.task.method for outcome in tdm.ranked_networks} == {
        "thermal_derivative_method",
    }
    assert {outcome.task.method for outcome in evolution.ranked_networks} == {
        "network_evolution_method",
    }


def test_umbrella_design_accessor_dispatches_each_hen_design_method(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()

    pdm = problem.design.heat_exchanger_network_synthesis(
        method=HENDesignMethod.PinchDesign,
    )
    tdm = problem.design.heat_exchanger_network_synthesis(
        method=HENDesignMethod.ThermalDerivative,
    )
    evolution = problem.design.heat_exchanger_network_synthesis(
        method=HENDesignMethod.NetworkEvolution,
        initial_networks=(tdm.network,),
    )

    assert pdm.design_method == HENDesignMethod.PinchDesign
    assert pdm.method == "pinch_design_method"
    assert tdm.design_method == HENDesignMethod.ThermalDerivative
    assert tdm.method == "thermal_derivative_method"
    assert evolution.design_method == HENDesignMethod.NetworkEvolution
    assert evolution.method == "network_evolution_method"


def test_umbrella_design_accessor_rejects_invalid_method_and_invalid_seed(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()
    seed = problem.design.pinch_design_method().network

    with pytest.raises(
        ValueError, match="Unknown heat exchanger network design method"
    ):
        problem.design.heat_exchanger_network_synthesis(method="not_a_method")

    with pytest.raises(ValueError, match="open_hens_method does not accept"):
        problem.design.heat_exchanger_network_synthesis(
            method=HENDesignMethod.OpenHENS,
            initial_networks=(seed,),
        )

    with pytest.raises(ValueError, match="pinch_design_method does not accept"):
        problem.design.heat_exchanger_network_synthesis(
            method=HENDesignMethod.PinchDesign,
            initial_networks=(seed,),
        )


def test_seeded_design_methods_require_seed_or_cached_design(monkeypatch) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()

    with pytest.raises(ValueError, match="initial_networks"):
        problem.design.thermal_derivative_method()

    with pytest.raises(ValueError, match="initial_networks"):
        problem.design.network_evolution_method()


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

    design = problem.design.heat_exchanger_network_synthesis()
    recovery = next(
        exchanger
        for exchanger in design.network.exchangers
        if exchanger.kind is HeatExchangerKind.RECOVERY
    )

    assert recovery.state().source_inlet_temperature == pytest.approx(650.0)
    assert recovery.state().source_outlet_temperature > 273.15
    assert recovery.state().sink_inlet_temperature == pytest.approx(410.0)
    assert recovery.state().sink_outlet_temperature > 273.15
    assert design.network.summary_metrics["approach_temperature"] == pytest.approx(2.0)


def test_workspace_design_workflow_dispatches_to_live_problem_design_path(
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    workspace = PinchWorkspace(_small_payload(), project_name="HENS Demo")

    view = workspace.solve_variant(
        "baseline",
        workflow="heat_exchanger_network_synthesis",
    )
    problem = workspace.case("baseline")

    assert view.status == "solved"
    assert view.support_level == "advanced"
    assert problem.results is not None
    assert problem.results.design is not None
    assert problem.results.design.workspace_variant == "baseline"
    assert problem.results.design.problem_id == problem.project_name
    assert not hasattr(problem.target, "heat_exchanger_network_synthesis")


def test_optional_exports_round_trip_from_problem_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem(project_name="Export Demo")
    problem.design.heat_exchanger_network_synthesis(workspace_variant="variant-a")

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
    assert not hasattr(OpenPinch, "__all__")
    assert not hasattr(hens_package, "heat_exchanger_network_synthesis_service")
    assert not hasattr(hens_package, "run_synthesis_workflow")
    assert not hasattr(workflow_module, "run_synthesis_workflow")

    with pytest.raises(TypeError, match="PinchProblem"):
        heat_exchanger_network_synthesis_service(_small_payload())  # type: ignore[arg-type]


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
    return PinchProblem(
        source=payload,
        project_name="duplicate-two-state-no-sweep",
    )


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
            "COSTING_HX_AREA_COEFF": 150.0,
            "COSTING_HX_AREA_EXP": 1.0,
            "COSTING_HX_UNIT_COST": 5500.0,
            "HENS_APPROACH_TEMPERATURES": [10.0],
            "HENS_BEST_SOLUTIONS_TO_SAVE": 1,
            "HENS_DERIVATIVE_THRESHOLDS": [0.5],
            "HENS_LOG_LEVEL": "WARNING",
            "HENS_MAX_PARALLEL": 1,
            "HENS_METHOD_SEQUENCE": [
                "pinch_design_method",
                "thermal_derivative_method",
                "network_evolution_method",
            ],
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
            "HENS_SYNTHESIS_QUALITY_TIER": 1,
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
    approach_temperature: float = 14.0,
):
    problem.target()
    settings = workflow_settings_from_problem(problem)
    assert settings.approach_temperatures == (approach_temperature,)
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


def _skip_if_live_solver_environment_missing() -> None:
    try:
        require_solver_binary(
            "couenne",
            purpose="single-period/multiperiod HEN equivalence test",
        )
        require_solver_binary(
            "ipopt",
            purpose="single-period/multiperiod HEN equivalence test",
        )
        require_synthesis_dependency(
            "gekko",
            purpose="single-period/multiperiod HEN equivalence test",
        )
        require_synthesis_dependency(
            "pyomo.environ",
            package="pyomo",
            purpose="single-period/multiperiod HEN equivalence test",
        )
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError) as exc:
        pytest.skip(str(exc))


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
        "HENS_APPROACH_TEMPERATURES": [2.0, 4.0],
        "HENS_DERIVATIVE_THRESHOLDS": [0.5, 1.0],
        "HENS_STAGE_SELECTION": [2],
        "HENS_BEST_SOLUTIONS_TO_SAVE": 2,
        "HENS_MAX_PARALLEL": 1,
        "HENS_OUTPUT_FORMATS": ["json", "csv"],
        "HENS_RUN_ID": "hens-05-test",
    }
    return payload


def _outcome_map(outcomes):
    return {
        outcome.task.task_id: outcome
        for outcome in outcomes
        if outcome.task.task_id is not None
    }


def _with_pathway_tac(
    outcome: HeatExchangerNetworkSynthesisTaskOutcome,
    *,
    tier1_tac: float,
    other_tac: float,
) -> HeatExchangerNetworkSynthesisTaskOutcome:
    if outcome.task.method != "network_evolution_method" or outcome.network is None:
        return outcome
    pathway_ids = outcome.task.metadata.get("pathway_ids", ())
    tac = tier1_tac if pathway_ids == ["tier1-open-hens"] else other_tac
    network = outcome.network.model_copy(
        update={
            "objective_value": tac,
            "total_annual_cost": tac,
            "utility_cost": tac / 2.0,
            "capital_cost": tac / 2.0,
        }
    )
    return outcome.model_copy(
        update={
            "network": network,
            "objective_value": tac,
        }
    )
