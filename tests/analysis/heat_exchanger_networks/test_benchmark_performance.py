"""Benchmark diagnostics tests for HeatExchangerNetworkLabel quality-tier evaluation."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from OpenPinch.analysis.heat_exchanger_networks.context import (
    ensure_target_results,
    finalise_design_result,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.fake_executor import (
    FakeSynthesisExecutor,
)
from OpenPinch.analysis.heat_exchanger_networks.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.analysis.heat_exchanger_networks.targeting.open_hens_method import (
    execute_open_hens_method,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.contracts.synthesis.task import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.enums import (
    HeatExchangerKind,
    HeatExchangerNetworkDesignMethod,
    StreamID,
)
from OpenPinch.domain.heat_exchanger import HeatExchanger
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from tests.support.paths import FIXTURES_ROOT, REPOSITORY_ROOT

REPO_ROOT = REPOSITORY_ROOT
FIXTURE_ROOT = FIXTURES_ROOT / "openhens"
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark_performance.py"


def test_benchmark_failure_taxonomy_groups_solver_errors() -> None:
    benchmark = _load_benchmark_module()

    assert (
        benchmark._failure_category(
            "failed",
            "warning",
            "Ipopt converged to a locally infeasible point",
        )
        == "locally_infeasible"
    )
    assert benchmark._failure_category("timeout", None, "exceeded 10 seconds") == (
        "timeout"
    )
    assert benchmark._failure_category("failed", None, "verification failed: duty") == (
        "verification_failure"
    )


def test_benchmark_diagnostics_capture_tier_two_pathway_statistics() -> None:
    benchmark = _load_benchmark_module()
    problem = _small_problem_with_options(
        {"HENS_SYNTHESIS_QUALITY_TIER": 2, "HENS_DT_CONT_MULTIPLIERS": [1.0]}
    )
    target_output = ensure_target_results(problem, {}, None)
    settings = workflow_settings_from_problem(problem)
    trace_executor = benchmark.BenchmarkTraceExecutor(delegate=FakeSynthesisExecutor())

    workflow_result = execute_open_hens_method(
        problem,
        settings,
        executor=trace_executor,
    )
    design = finalise_design_result(problem, target_output, workflow_result)

    diagnostics = benchmark._benchmark_diagnostics(
        tier=2,
        trace_executor=trace_executor,
        selected_task_id=design.task_id,
    )

    assert diagnostics["stage_runtime_seconds"].keys() >= {
        "pdm",
        "direct_evm_from_pdm",
        "tdm",
        "evm_from_tdm",
    }
    assert diagnostics["task_counts"]["planned_by_pathway"]["tier0-compact-1"] > 0
    assert diagnostics["task_counts"]["planned_by_pathway"]["tier1-open-hens"] > 0
    assert diagnostics["task_counts"]["planned_by_pathway"]["tier2-compact-1"] > 0
    assert diagnostics["robustness"]["funnel"]["planned"] == len(
        trace_executor.task_records
    )
    assert diagnostics["robustness"]["funnel"]["accepted"] == 1


def test_benchmark_defaults_cover_small_unique_cases_and_tiers_zero_to_four() -> None:
    benchmark = _load_benchmark_module()

    assert benchmark.HENS_QUALITY_TIERS == (0, 1, 2, 3, 4)
    assert benchmark.HENS_MAX_PARALLEL == 10
    assert benchmark.HENS_BENCHMARK_OPTIONS["HENS_MAX_PARALLEL"] == 10
    assert benchmark.DEFAULT_HENS_BENCHMARK_JSON == REPO_ROOT / (
        "results/hens_t0_t4_upto9_fixture_defaults.json"
    )
    assert "Nine-stream-Linnhoff-and-Ahmad-1999-1.json" in (
        benchmark.HENS_BENCHMARK_CASES
    )
    assert not any(".reordered" in name for name in benchmark.HENS_BENCHMARK_CASES)
    assert set(benchmark.HENS_BENCHMARK_CASES) == set(
        benchmark.small_hens_benchmark_cases(max_streams=9)
    )


def test_benchmark_fixture_default_mode_preserves_fixture_hens_settings() -> None:
    benchmark = _load_benchmark_module()
    payload = json.loads(
        (FIXTURE_ROOT / "Four-stream-Yee-and-Grossmann-1990-1.json").read_text(
            encoding="utf-8"
        )
    )
    fixture_options = deepcopy(payload["options"])

    options = benchmark._hens_case_options(
        payload,
        options={
            "HENS_MAX_PARALLEL": 6,
            "HENS_DT_CONT_MULTIPLIERS": [99.9],
        },
        tier=3,
        run_id="fixture-default-test",
        use_fixture_defaults=True,
    )

    assert (
        options["HENS_DT_CONT_MULTIPLIERS"]
        == fixture_options["HENS_DT_CONT_MULTIPLIERS"]
    )
    assert (
        options["HENS_DERIVATIVE_THRESHOLDS"]
        == fixture_options["HENS_DERIVATIVE_THRESHOLDS"]
    )
    assert options["HENS_STAGE_SELECTION"] == fixture_options["HENS_STAGE_SELECTION"]
    assert options["HENS_MAX_PARALLEL"] == 6
    assert options["HENS_SYNTHESIS_QUALITY_TIER"] == 3
    assert options["HENS_RUN_ID"] == "fixture-default-test"
    assert options["HENS_OUTPUT_FORMATS"] == []


def test_benchmark_standard_mode_applies_benchmark_hens_overrides() -> None:
    benchmark = _load_benchmark_module()
    payload = json.loads(
        (FIXTURE_ROOT / "Four-stream-Yee-and-Grossmann-1990-1.json").read_text(
            encoding="utf-8"
        )
    )

    options = benchmark._hens_case_options(
        payload,
        options=benchmark.HENS_BENCHMARK_OPTIONS,
        tier=4,
        run_id="standard-mode-test",
        use_fixture_defaults=False,
    )

    assert options["HENS_DT_CONT_MULTIPLIERS"] == [1.0]
    assert options["HENS_DERIVATIVE_THRESHOLDS"] == [0.5]
    assert options["HENS_STAGE_SELECTION"] == [3]
    assert options["HENS_MAX_PARALLEL"] == 10
    assert options["HENS_SYNTHESIS_QUALITY_TIER"] == 4
    assert options["HENS_RUN_ID"] == "standard-mode-test"


def test_benchmark_trace_executor_forwards_max_parallel_to_delegate() -> None:
    benchmark = _load_benchmark_module()
    captured_parallel: list[int] = []

    class CapturingExecutor:
        def execute(self, tasks, *, problem, parent_outcomes, max_parallel):
            captured_parallel.append(max_parallel)
            task = tasks[0]
            return (
                HeatExchangerNetworkSynthesisTaskOutcome(
                    task=task,
                    status="failed",
                    solver_status="failed",
                    error="synthetic failure",
                ),
            )

    task = HeatExchangerNetworkSynthesisTask(
        run_id="run-1",
        method="network_evolution_method",
        dt_cont_multiplier=1.0,
        stage_count=2,
    )
    trace_executor = benchmark.BenchmarkTraceExecutor(delegate=CapturingExecutor())

    trace_executor.execute((task,), problem=None, parent_outcomes={}, max_parallel=4)

    assert captured_parallel == [4]
    assert trace_executor.task_records[0]["max_parallel"] == 4


def test_benchmark_solver_tracer_records_solve_boundary_calls(monkeypatch) -> None:
    benchmark = _load_benchmark_module()

    class FakeModel:
        _path = "/tmp/openhens/fake-model"

    def fake_solve(model, *, solver_name, disp=False):
        return benchmark.solver_backend.SolverRun(
            name=solver_name,
            status=1,
            objective_value=123.4,
            solve_time=0.25,
        )

    monkeypatch.setattr(benchmark.solver_backend, "solve_pyomo_model", fake_solve)
    tracer = benchmark.BenchmarkSolverCallTracer(case_name="case.json", tier=2)
    task = HeatExchangerNetworkSynthesisTask(
        run_id="run-1",
        method="pinch_design_method",
        dt_cont_multiplier=1.0,
        stage_count=2,
        task_id="task-1",
    )
    start_record = benchmark._task_start_record(
        task,
        {},
        stage_label="pdm",
        stage_index=0,
        max_parallel=1,
    )

    with tracer.patched_backend():
        with tracer.task_context(start_record):
            run = benchmark.solver_backend.solve_pyomo_model(
                FakeModel(),
                solver_name="highs",
            )

    assert run.status == 1
    assert len(tracer.records) == 1
    record = tracer.records[0]
    assert record["case"] == "case.json"
    assert record["tier"] == 2
    assert record["task_id"] == "task-1"
    assert record["stage_label"] == "pdm"
    assert record["solver_name"] == "highs"
    assert record["solver_status"] == 1
    assert record["objective_value"] == 123.4
    assert record["solver_call_role"] == "pdm_above"


def test_benchmark_solver_tracer_records_effective_forced_solver(monkeypatch) -> None:
    benchmark = _load_benchmark_module()

    def fake_solve(model, *, solver_name, disp=False):
        del model, solver_name, disp
        return benchmark.solver_backend.SolverRun(name="couenne", status=1)

    monkeypatch.setattr(benchmark.solver_backend, "solve_pyomo_model", fake_solve)
    tracer = benchmark.BenchmarkSolverCallTracer(case_name="case.json", tier=0)

    with tracer.patched_backend():
        benchmark.solver_backend.solve_pyomo_model(
            object(),
            solver_name="ipopt-pyomo",
        )

    assert tracer.records[0]["solver_name"] == "couenne"


def test_benchmark_solver_call_summaries_aggregate_by_layer() -> None:
    benchmark = _load_benchmark_module()

    diagnostics = benchmark._diagnostics_from_records(
        tier=4,
        task_records=[],
        stage_records=[],
        solver_call_records=[
            {"stage_label": "pdm", "duration_seconds": 1.5},
            {"stage_label": "tdm", "duration_seconds": 2.0},
            {"stage_label": "evm_from_tdm", "duration_seconds": 3.5},
        ],
        selected_task_id=None,
        in_progress_tasks=[],
    )

    assert diagnostics["solver_call_runtime_seconds"]["pdm"] == 1.5
    assert diagnostics["solver_call_runtime_seconds"]["tdm"] == 2.0
    assert diagnostics["solver_call_runtime_seconds"]["evm_from_tdm"] == 3.5
    assert diagnostics["solver_call_counts"]["by_stage"] == {
        "evm_from_tdm": 1,
        "pdm": 1,
        "tdm": 1,
    }


def test_benchmark_incremental_json_includes_partial_comparisons(tmp_path) -> None:
    benchmark = _load_benchmark_module()
    output_path = tmp_path / "benchmark.json"
    partial_trace = benchmark._empty_partial_trace()
    partial_trace["running_tasks"]["task-1"] = {
        "task_id": "task-1",
        "method": "network_evolution_method",
        "stage_label": "evm_from_tdm",
        "pathway_ids": ["tier2-compact-1"],
        "status": "running",
    }
    partial_trace["solver_call_records"].append(
        {
            "stage_label": "pdm",
            "solver_call_role": "pdm_above",
            "solver_name": "highs",
            "duration_seconds": 0.5,
        }
    )
    results = [
        benchmark._failed_hens_result(
            "case.json",
            tier=0,
            status="timeout",
            total_seconds=1.0,
            error="exceeded 1 seconds",
            diagnostics=benchmark._timeout_diagnostics(
                tier=2,
                partial_trace=partial_trace,
            ),
        )
    ]

    benchmark._write_incremental_results(output_path, results)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["status"] == "timeout"
    assert payload[0]["diagnostics"]["robustness"]["failure_categories"] == {
        "timeout": 1
    }
    assert payload[0]["diagnostics"]["robustness"]["funnel"]["in_progress"] == 1
    assert payload[0]["diagnostics"]["in_progress_tasks"][0]["task_id"] == "task-1"
    assert payload[0]["diagnostics"]["solver_call_runtime_seconds"]["pdm"] == 0.5
    assert len(payload[0]["diagnostics"]["solver_calls"]) == 1
    assert payload[1]["benchmark"] == "hens_quality_tier_comparison"
    assert payload[1]["status"] == "missing_baseline"


def test_benchmark_resolves_fixture_sample_case_and_path_names(tmp_path) -> None:
    benchmark = _load_benchmark_module()
    case_file = tmp_path / "own_case.json"
    case_file.write_text("{}", encoding="utf-8")

    assert benchmark.resolve_hens_case("Four-stream-Yee-and-Grossmann-1990-1.json") == (
        FIXTURE_ROOT / "Four-stream-Yee-and-Grossmann-1990-1.json"
    )
    assert benchmark.resolve_hens_case("pulp_mill.json") == (
        REPO_ROOT / "OpenPinch" / "tutorials" / "sample_cases" / "pulp_mill.json"
    )
    assert benchmark.resolve_hens_case(str(case_file)) == case_file
    with pytest.raises(FileNotFoundError, match="no-such-case.json"):
        benchmark.resolve_hens_case("no-such-case.json")


def test_network_meets_pinch_targets_per_period_within_tolerance() -> None:
    benchmark = _load_benchmark_module()
    targets = (
        SimpleNamespace(
            period_idx=0, hot_utility_target=50.0, cold_utility_target=30.0
        ),
        SimpleNamespace(period_idx=1, hot_utility_target=80.0, cold_utility_target=0.0),
    )
    on_target = _benchmark_network(
        ("hot_utility", "HU", "C1", None, (50.0, 80.0)),
        ("cold_utility", "H1", "CU", None, (30.0, 0.0)),
    )
    # 1e-4 kW above target is solver noise, inside the 1e-3 kW allowance.
    within_tolerance = _benchmark_network(
        ("hot_utility", "HU", "C1", None, (50.0001, 80.0)),
        ("cold_utility", "H1", "CU", None, (30.0, 0.0)),
    )
    over_in_second_period = _benchmark_network(
        ("hot_utility", "HU", "C1", None, (50.0, 81.0)),
        ("cold_utility", "H1", "CU", None, (30.0, 0.0)),
    )

    assert benchmark.network_meets_pinch_targets(on_target, targets)
    assert benchmark.network_meets_pinch_targets(within_tolerance, targets)
    assert not benchmark.network_meets_pinch_targets(over_in_second_period, targets)
    # Utility below the target breaks the energy balance, so it is not on target.
    under_target = _benchmark_network(
        ("hot_utility", "HU", "C1", None, (49.0, 80.0)),
        ("cold_utility", "H1", "CU", None, (30.0, 0.0)),
    )
    assert not benchmark.network_meets_pinch_targets(under_target, targets)


def test_pdm_topology_summary_counts_distinct_target_meeting_topologies() -> None:
    benchmark = _load_benchmark_module()
    two_units = _benchmark_network(
        ("recovery", "H1", "C1", 1, (60.0,)),
        ("hot_utility", "HU", "C1", None, (10.0,)),
    )
    # Same matches with other duties, as another approach temperature gives.
    two_units_again = _benchmark_network(
        ("recovery", "H1", "C1", 1, (55.0,)),
        ("hot_utility", "HU", "C1", None, (15.0,)),
    )
    three_units = _benchmark_network(
        ("recovery", "H1", "C1", 1, (40.0,)),
        ("recovery", "H1", "C1", 2, (20.0,)),
        ("hot_utility", "HU", "C1", None, (10.0,)),
    )
    off_target = _benchmark_network(("hot_utility", "HU", "C1", None, (70.0,)))

    summary = benchmark.pdm_topology_summary(
        [
            (two_units, True),
            (two_units_again, True),
            (three_units, True),
            (off_target, False),
        ]
    )

    assert summary == {
        "pdm_network_count": 4,
        "target_meeting_network_count": 3,
        "distinct_target_meeting_topologies": 2,
        "topologies_by_unit_count": {"2": 1, "3": 1},
        "min_units": 2,
    }


def test_benchmark_acceptance_metrics_cover_pdm_outcomes_and_top_tacs() -> None:
    benchmark = _load_benchmark_module()
    problem = _small_problem_with_options({"HENS_SYNTHESIS_QUALITY_TIER": 0})
    target_output = ensure_target_results(problem, {}, None)
    settings = workflow_settings_from_problem(problem)
    workflow_result = execute_open_hens_method(
        problem, settings, executor=FakeSynthesisExecutor()
    )
    design = finalise_design_result(problem, target_output, workflow_result)

    metrics = benchmark.acceptance_metrics(
        problem, workflow_result.outcomes, design.ranked_networks, top_n=2
    )

    pdm_outcomes = [
        outcome
        for outcome in workflow_result.outcomes
        if outcome.task.method == HeatExchangerNetworkDesignMethod.PinchDesign
        and outcome.status == "success"
    ]
    assert metrics["pdm_network_count"] == len(pdm_outcomes) > 0
    assert 0 <= metrics["target_meeting_network_count"] <= len(pdm_outcomes)
    assert (
        metrics["top_total_annual_costs"]
        == sorted(outcome.objective_value for outcome in design.ranked_networks)[:2]
    )


def test_benchmark_acceptance_comparison_flags_regressions() -> None:
    benchmark = _load_benchmark_module()
    baseline = {
        "name": "case.json",
        "tier": 0,
        "status": "success",
        "total_seconds": 12.0,
        "total_annual_cost": 100.0,
        "acceptance": {
            "distinct_target_meeting_topologies": 4,
            "topologies_by_unit_count": {"5": 1, "6": 3},
            "min_units": 5,
            "top_total_annual_costs": [100.0, 110.0, 120.0],
        },
    }
    variant = deepcopy(baseline)
    variant["total_annual_cost"] = 101.0
    variant["acceptance"] |= {
        "distinct_target_meeting_topologies": 9,
        "topologies_by_unit_count": {"5": 2, "6": 7},
        "top_total_annual_costs": [101.0, 105.0, 125.0],
    }

    comparison = benchmark.compare_acceptance(baseline, variant)

    assert comparison == {
        "name": "case.json",
        "tier": 0,
        "status": "success",
        "best_total_annual_cost_no_worse": False,
        "best_total_annual_cost_delta": 1.0,
        "distinct_topologies": (4, 9),
        "min_units": (5, 5),
        "topologies_by_unit_count": ({"5": 1, "6": 3}, {"5": 2, "6": 7}),
        "top_total_annual_cost_deltas": [1.0, -5.0, 5.0],
    }


def test_benchmark_acceptance_comparison_reports_a_baseline_without_metrics() -> None:
    benchmark = _load_benchmark_module()
    record = {
        "name": "case.json",
        "tier": 0,
        "status": "success",
        "total_seconds": 12.0,
        "total_annual_cost": 100.0,
    }
    # A baseline written before acceptance metrics were recorded.
    variant = record | {"acceptance": {}}

    assert benchmark.compare_acceptance(record, variant) == {
        "name": "case.json",
        "tier": 0,
        "status": "baseline_without_acceptance_metrics",
    }


def _benchmark_network(*units) -> HeatExchangerNetwork:
    kinds = {
        "recovery": (HeatExchangerKind.RECOVERY, StreamID.Process, StreamID.Process),
        "hot_utility": (
            HeatExchangerKind.HOT_UTILITY,
            StreamID.Utility,
            StreamID.Process,
        ),
        "cold_utility": (
            HeatExchangerKind.COLD_UTILITY,
            StreamID.Process,
            StreamID.Utility,
        ),
    }
    exchangers = []
    for kind, source, sink, stage, duties in units:
        exchanger_kind, source_role, sink_role = kinds[kind]
        exchangers.append(
            HeatExchanger(
                kind=exchanger_kind,
                source_stream=source,
                sink_stream=sink,
                source_stream_role=source_role,
                sink_stream_role=sink_role,
                stage=stage,
                period_states=tuple(
                    HeatExchangerPeriodState(
                        period_id=str(index),
                        period_idx=index,
                        duty=duty,
                        active=duty > 0.0,
                    )
                    for index, duty in enumerate(duties)
                ),
            )
        )
    return HeatExchangerNetwork(exchangers=tuple(exchangers), stage_count=2)


def _small_problem_with_options(options: dict) -> PinchProblem:
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
        "HENS_OUTPUT_FORMATS": [],
        "HENS_RUN_ID": "benchmark-test",
        **options,
    }
    return PinchProblem(source=payload, project_name="benchmark-test")


def _load_benchmark_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_performance_under_test",
        BENCHMARK_SCRIPT,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_nine_stream_script():
    spec = importlib.util.spec_from_file_location(
        "benchmark_nine_stream_under_test",
        REPO_ROOT / "scripts" / "benchmark_nine_stream.py",
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _nine_stream_outcome(method: str, status: str = "success"):
    return SimpleNamespace(task=SimpleNamespace(method=method), status=status)


def test_nine_stream_script_keeps_the_live_expectations_and_the_fixture() -> None:
    script = _load_nine_stream_script()
    expected = script.EXPECTED

    assert expected["best_solution"] == 2905508.6335796523
    assert expected["total_cases_attempted"] == 1133
    assert (FIXTURE_ROOT / f"{script.CASE_ID}.json").is_file()


def test_nine_stream_script_reports_each_mismatch_with_the_expected_value() -> None:
    script = _load_nine_stream_script()
    expected = script.EXPECTED
    outcomes = [_nine_stream_outcome("pinch_design_method")]
    metrics = {
        "recovery_units": expected["best_recovery_units"],
        "hot_utility_units": expected["best_hu_units"],
        "cold_utility_units": expected["best_cu_units"],
    }

    failures = script.compare_to_expected(
        total_annual_cost=expected["best_solution"] * 1.01,
        stage_count=expected["best_stages"],
        summary_metrics=metrics,
        outcomes=outcomes,
    )

    assert any("total_annual_cost" in failure for failure in failures)
    assert any("total_cases_attempted" in failure for failure in failures)
    assert any("solved_esm_count" in failure for failure in failures)
    assert not any("stage_count" in failure for failure in failures)


def test_nine_stream_script_accepts_the_expected_result() -> None:
    script = _load_nine_stream_script()
    expected = script.EXPECTED
    # 1133 weighted attempts = 1133 - 11 * 67 = 396 plain jobs plus 67 ESM runs.
    outcomes = [_nine_stream_outcome("pinch_design_method") for _ in range(396)] + [
        _nine_stream_outcome("network_evolution_method") for _ in range(67)
    ]
    metrics = {
        "recovery_units": expected["best_recovery_units"],
        "hot_utility_units": expected["best_hu_units"],
        "cold_utility_units": expected["best_cu_units"],
    }

    failures = script.compare_to_expected(
        total_annual_cost=expected["best_solution"],
        stage_count=expected["best_stages"],
        summary_metrics=metrics,
        outcomes=outcomes,
    )

    assert failures == []


def test_benchmark_solver_tracer_records_native_pyomo_solves(monkeypatch) -> None:
    # The pinch-design model solves through solve_pyomo_model with Couenne
    # and Ipopt, so the tracer must see those calls too.
    benchmark = _load_benchmark_module()

    def fake_solve(model, *, solver_name, disp=False):
        del model, disp
        return benchmark.solver_backend.SolverRun(name=solver_name, status=1)

    monkeypatch.setattr(benchmark.solver_backend, "solve_pyomo_model", fake_solve)
    tracer = benchmark.BenchmarkSolverCallTracer(case_name="case.json", tier=0)

    with tracer.patched_backend():
        run = benchmark.solver_backend.solve_pyomo_model(
            object(), solver_name="couenne"
        )

    assert run.status == 1
    assert [record["solver_name"] for record in tracer.records] == ["couenne"]
    assert benchmark.solver_backend.solve_pyomo_model is fake_solve
