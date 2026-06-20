"""HENS-05 workflow, result-cache, export, and API-boundary tests."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

import OpenPinch
import OpenPinch.services
import OpenPinch.services.heat_exchanger_network_synthesis.workflow as workflow_module
from OpenPinch import PinchProblem, PinchWorkspace
from OpenPinch.classes.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.lib import HeatExchangerKind
from OpenPinch.lib.schemas.synthesis import (
    HeatExchangerNetworkSynthesisManifest,
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from OpenPinch.services.heat_exchanger_network_synthesis import __all__ as hens_all
from OpenPinch.services.heat_exchanger_network_synthesis.array_adapter import (
    problem_to_solver_arrays,
)
from OpenPinch.services.heat_exchanger_network_synthesis.exports import (
    export_heat_exchanger_network_synthesis_results,
)
from OpenPinch.services.heat_exchanger_network_synthesis.service import (
    heat_exchanger_network_synthesis_service,
)
from OpenPinch.services.heat_exchanger_network_synthesis.workflow import (
    FakeSynthesisExecutor,
    WorkflowContractError,
    build_energy_stage_refinement_tasks,
    build_pinch_decomposition_tasks,
    build_topology_design_tasks,
    workflow_settings_from_problem,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
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

    result = workflow_module._execute_synthesis_workflow(
        problem,
        workflow_settings_from_problem(problem),
    )

    assert used_default
    assert result.accepted_result.network.exchangers


def test_fake_executor_task_graph_uses_successful_topology_only() -> None:
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    pdm_tasks = build_pinch_decomposition_tasks(settings)
    pdm_executor = FakeSynthesisExecutor(failures={pdm_tasks[0].task_id})
    pdm_outcomes = pdm_executor.execute(
        pdm_tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=settings.max_parallel,
    )

    tdm_tasks = build_topology_design_tasks(settings, pdm_outcomes)
    tdm_executor = FakeSynthesisExecutor(failures={tdm_tasks[0].task_id})
    tdm_outcomes = tdm_executor.execute(
        tdm_tasks,
        problem=problem,
        parent_outcomes=_outcome_map(pdm_outcomes),
        max_parallel=settings.max_parallel,
    )
    esm_tasks = build_energy_stage_refinement_tasks(settings, tdm_outcomes)

    assert len(pdm_tasks) == 2
    assert len(tdm_tasks) == 2
    assert {task.parent_task_id for task in tdm_tasks} == {pdm_tasks[1].task_id}
    assert all(task.topology_restrictions for task in tdm_tasks)
    assert len(esm_tasks) == 1
    assert esm_tasks[0].parent_task_id == tdm_tasks[1].task_id
    assert esm_tasks[0].topology_restrictions
    assert pdm_executor.stage_order == ["pinch_decomposition"]
    assert tdm_executor.stage_order == ["topology_design"]


def test_missing_couenne_skips_derivative_stage_and_runs_evolution() -> None:
    class MissingCouenneForTopologyExecutor(FakeSynthesisExecutor):
        def execute(self, tasks, *, problem, parent_outcomes, max_parallel):
            if tasks and tasks[0].method == "topology_design":
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
        result = workflow_module._execute_synthesis_workflow(
            problem,
            settings,
            executor=MissingCouenneForTopologyExecutor(),
        )

    outcomes = result.outcomes
    pdm_success = next(
        outcome
        for outcome in outcomes
        if outcome.status == "success" and outcome.task.method == "pinch_decomposition"
    )
    tdm_failures = [
        outcome for outcome in outcomes if outcome.task.method == "topology_design"
    ]
    esm_success = next(
        outcome
        for outcome in outcomes
        if outcome.status == "success"
        and outcome.task.method == "energy_stage_refinement"
    )

    assert result.accepted_result.method == "energy_stage_refinement"
    assert all(outcome.status == "failed" for outcome in tdm_failures)
    assert esm_success.task.parent_task_id == pdm_success.task.task_id
    assert esm_success.task.derivative_threshold is None


def test_missing_couenne_before_pdm_runs_direct_evolution() -> None:
    class MissingCouenneForPdmExecutor(FakeSynthesisExecutor):
        def execute(self, tasks, *, problem, parent_outcomes, max_parallel):
            if tasks and tasks[0].method == "pinch_decomposition":
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

    with pytest.warns(RuntimeWarning, match="pinch-decomposition"):
        result = workflow_module._execute_synthesis_workflow(
            problem,
            settings,
            executor=MissingCouenneForPdmExecutor(),
        )

    outcomes = result.outcomes
    pdm_failures = [
        outcome for outcome in outcomes if outcome.task.method == "pinch_decomposition"
    ]
    tdm_outcomes = [
        outcome for outcome in outcomes if outcome.task.method == "topology_design"
    ]
    esm_successes = [
        outcome
        for outcome in outcomes
        if outcome.status == "success"
        and outcome.task.method == "energy_stage_refinement"
    ]

    assert all(outcome.status == "failed" for outcome in pdm_failures)
    assert tdm_outcomes == []
    assert esm_successes
    assert all(outcome.task.parent_task_id is None for outcome in esm_successes)
    assert result.accepted_result.method == "energy_stage_refinement"


def test_failed_workflow_reports_task_errors() -> None:
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    pdm_tasks = build_pinch_decomposition_tasks(settings)
    executor = FakeSynthesisExecutor(
        failures={task.task_id for task in pdm_tasks if task.task_id is not None},
    )

    with pytest.raises(WorkflowContractError, match="configured fake executor failure"):
        workflow_module._execute_synthesis_workflow(
            problem,
            settings,
            executor=executor,
        )


def test_synthesis_task_ids_are_deterministic_and_serializable() -> None:
    settings = workflow_settings_from_problem(_small_problem())

    first = build_pinch_decomposition_tasks(settings)
    second = build_pinch_decomposition_tasks(settings)
    roundtrip = HeatExchangerNetworkSynthesisTask.model_validate_json(
        first[0].model_dump_json()
    )

    assert [task.task_id for task in first] == [task.task_id for task in second]
    assert roundtrip == first[0]


def test_downstream_topology_restrictions_are_required() -> None:
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    pdm_task = build_pinch_decomposition_tasks(settings)[0]
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
        build_topology_design_tasks(settings, [missing_network])
    with pytest.raises(WorkflowContractError, match="topology restrictions"):
        build_topology_design_tasks(settings, [empty_network])


def test_fake_outcomes_serialize_without_live_solver_objects() -> None:
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    pdm_tasks = build_pinch_decomposition_tasks(settings)
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


def test_legacy_z_restriction_handoff_uses_source_shaped_duty_values() -> None:
    problem = _small_problem()
    settings = workflow_settings_from_problem(problem)
    pdm_tasks = build_pinch_decomposition_tasks(settings)
    pdm_outcomes = FakeSynthesisExecutor().execute(
        pdm_tasks,
        problem=problem,
        parent_outcomes={},
        max_parallel=settings.max_parallel,
    )
    tdm_task = build_topology_design_tasks(settings, pdm_outcomes)[0]
    arrays = problem_to_solver_arrays(problem, dTmin=0.1)

    z_restriction, zhu_restriction, zcu_restriction = (
        workflow_module._legacy_z_restriction(tdm_task, arrays)
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


def test_direct_design_run_computes_targets_when_cache_is_empty(monkeypatch) -> None:
    _use_fake_default_executor(monkeypatch)
    problem = _small_problem()

    design = problem.design.heat_exchanger_network_synthesis()

    assert problem.results is not None
    assert problem.results.targets
    assert problem.results.design == design


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

    assert recovery.source_inlet_temperature == pytest.approx(650.0)
    assert recovery.source_outlet_temperature == pytest.approx(370.0)
    assert recovery.sink_inlet_temperature == pytest.approx(410.0)
    assert recovery.sink_outlet_temperature == pytest.approx(650.0)
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
    assert "heat_exchanger_network_synthesis_service" not in OpenPinch.__all__
    assert not hasattr(OpenPinch.services, "heat_exchanger_network_synthesis_service")
    assert "heat_exchanger_network_synthesis_service" not in OpenPinch.services.__all__
    assert "run_synthesis_workflow" not in hens_all
    assert not hasattr(workflow_module, "run_synthesis_workflow")

    with pytest.raises(TypeError, match="PinchProblem"):
        heat_exchanger_network_synthesis_service(_small_payload())  # type: ignore[arg-type]


def test_synthesis_imports_do_not_load_live_solver_dependencies() -> None:
    script = f"""
import sys

import OpenPinch
from OpenPinch.classes import PinchProblem
from OpenPinch.services.heat_exchanger_network_synthesis.workflow import FakeSynthesisExecutor

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


def _use_fake_default_executor(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow_module,
        "LocalSynthesisExecutor",
        FakeSynthesisExecutor,
    )


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
