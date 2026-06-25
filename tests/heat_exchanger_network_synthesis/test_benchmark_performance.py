"""Benchmark diagnostics tests for HEN quality-tier evaluation."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

from OpenPinch import PinchProblem
from OpenPinch.lib.schemas.synthesis import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution.fake_executor import (
    FakeSynthesisExecutor,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution.settings import (
    workflow_settings_from_problem,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.service_context import (
    ensure_target_results,
    finalise_design_result,
)
from OpenPinch.services.heat_exchanger_network_synthesis.targeting_services.open_hens_method import (
    execute_open_hens_method,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "openhens"
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
    problem = _small_problem_with_options({"HENS_SYNTHESIS_QUALITY_TIER": 2})
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
    assert benchmark.HENS_BENCHMARK_OPTIONS["HENS_MAX_PARALLEL"] == 4
    assert "Nine-stream-Linnhoff-and-Ahmad-1999-1.json" in (
        benchmark.HENS_BENCHMARK_CASES
    )
    assert not any(".reordered" in name for name in benchmark.HENS_BENCHMARK_CASES)
    assert set(benchmark.HENS_BENCHMARK_CASES) == set(
        benchmark.small_hens_benchmark_cases(max_streams=9)
    )


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
        approach_temperature=10.0,
        stage_count=2,
    )
    trace_executor = benchmark.BenchmarkTraceExecutor(delegate=CapturingExecutor())

    trace_executor.execute((task,), problem=None, parent_outcomes={}, max_parallel=4)

    assert captured_parallel == [4]
    assert trace_executor.task_records[0]["max_parallel"] == 4


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
    assert payload[1]["benchmark"] == "hens_quality_tier_comparison"
    assert payload[1]["status"] == "missing_baseline"


def _small_problem_with_options(options: dict) -> PinchProblem:
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
