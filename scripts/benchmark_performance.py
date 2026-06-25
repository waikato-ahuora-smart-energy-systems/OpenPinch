"""Manual HEN synthesis quality-tier benchmark with robustness diagnostics.

This script is intentionally non-gating. It runs the maintained OpenHENS public
method at selected quality tiers and emits incremental JSON records that can be
compared across changes.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import os
import queue as queue_module
import signal
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from OpenPinch.classes.pinch_problem import PinchProblem
from OpenPinch.lib.schemas.synthesis import (
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
    SynthesisMethod,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution import (
    executor as executor_module,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.execution import (
    settings as settings_module,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.service_context import (
    ensure_target_results,
    finalise_design_result,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.solver import (
    backend as solver_backend,
)
from OpenPinch.services.heat_exchanger_network_synthesis.targeting_services import (
    open_hens_method,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HENS_FIXTURE_ROOT = _REPO_ROOT / "tests" / "fixtures" / "openhens"
LocalSynthesisExecutor = executor_module.LocalSynthesisExecutor
workflow_settings_from_problem = settings_module.workflow_settings_from_problem
execute_open_hens_method = open_hens_method.execute_open_hens_method
HENS_MAX_STREAMS = 9
HENS_MAX_PARALLEL = 10
DEFAULT_HENS_BENCHMARK_JSON = (
    _REPO_ROOT / "results" / "hens_t0_t4_upto9_fixture_defaults.json"
)
HENS_BENCHMARK_CASES = ()
HENS_QUALITY_TIERS = (0, 1, 2, 3, 4)
HENS_BENCHMARK_OPTIONS = {
    "HENS_APPROACH_TEMPERATURES": [10.0],
    "HENS_DERIVATIVE_THRESHOLDS": [0.5],
    "HENS_STAGE_SELECTION": [3],
    "HENS_BEST_SOLUTIONS_TO_SAVE": 3,
    "HENS_MAX_PARALLEL": HENS_MAX_PARALLEL,
    "HENS_OUTPUT_FORMATS": [],
}
SOLVER_STAGE_LABELS = (
    "pdm",
    "tdm",
    "direct_evm_from_pdm",
    "evm_from_tdm",
    "direct_evm",
)


class BenchmarkSolverCallTracer:
    """Benchmark-local wrapper around the concrete GEKKO/Pyomo solve boundary."""

    def __init__(self, *, case_name: str, tier: int, progress_queue=None) -> None:
        self.case_name = case_name
        self.tier = int(tier)
        self.progress_queue = progress_queue
        self.records: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._current_context: dict[str, Any] | None = None
        self._solver_call_index = 0
        self._task_solver_counts: Counter[str] = Counter()

    @contextmanager
    def patched_backend(self):
        original_solve = solver_backend.solve_gekko_model

        def traced_solve(
            model, *, solver_name: str, disp: bool = False, debug: int = 0
        ):
            return self._trace_solve(
                original_solve,
                model,
                solver_name=solver_name,
                disp=disp,
                debug=debug,
            )

        solver_backend.solve_gekko_model = traced_solve
        try:
            yield self
        finally:
            solver_backend.solve_gekko_model = original_solve

    @contextmanager
    def task_context(self, record: dict[str, Any]):
        with self._lock:
            previous_context = self._current_context
            self._current_context = dict(record)
        try:
            yield
        finally:
            with self._lock:
                self._current_context = previous_context

    def _trace_solve(
        self,
        solve_function,
        model,
        *,
        solver_name: str,
        disp: bool,
        debug: int,
    ):
        start = time.perf_counter()
        solver_run = None
        error = None
        try:
            solver_run = solve_function(
                model,
                solver_name=solver_name,
                disp=disp,
                debug=debug,
            )
            return solver_run
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._record_solver_call(
                model,
                solver_name=solver_name,
                duration_seconds=time.perf_counter() - start,
                solver_run=solver_run,
                error=error,
            )

    def _record_solver_call(
        self,
        model,
        *,
        solver_name: str,
        duration_seconds: float,
        solver_run,
        error: BaseException | None,
    ) -> None:
        with self._lock:
            context = dict(self._current_context or {})
            self._solver_call_index += 1
            task_id = str(context.get("task_id") or "untagged")
            self._task_solver_counts[task_id] += 1
            call_index_within_task = self._task_solver_counts[task_id]
            record = {
                "case": self.case_name,
                "tier": self.tier,
                "solver_call_index": self._solver_call_index,
                "task_solver_call_index": call_index_within_task,
                "task_id": context.get("task_id"),
                "stage_label": context.get("stage_label", "untagged"),
                "method": context.get("method"),
                "model_name": _solver_model_name(model),
                "model_path": _solver_model_path(model),
                "solver_name": solver_name,
                "duration_seconds": duration_seconds,
                "solver_reported_seconds": _solver_run_attr(solver_run, "solve_time"),
                "solver_status": _solver_run_attr(solver_run, "status"),
                "objective_value": _solver_run_attr(solver_run, "objective_value"),
                "failure_reason": _solver_failure_reason(solver_run, error),
                "solver_call_role": _solver_call_role(
                    context.get("stage_label"),
                    context.get("method"),
                    call_index_within_task,
                ),
                "parent_task_id": context.get("parent_task_id"),
                "parent_method": context.get("parent_method"),
                "approach_temperature": context.get("approach_temperature"),
                "derivative_threshold": context.get("derivative_threshold"),
                "stage_count": context.get("stage_count"),
                "settings": context.get("settings") or {},
                "pathway_ids": context.get("pathway_ids") or [],
                "pathway_id": context.get("pathway_id"),
                "tier_origins": context.get("tier_origins") or [],
                "tier_origin": context.get("tier_origin"),
                "pathway_kind": context.get("pathway_kind"),
                "pdm_mode": context.get("pdm_mode"),
                "pdm_multiplier": context.get("pdm_multiplier"),
                "protected_pathway": context.get("protected_pathway"),
                "exact_open_hens": context.get("exact_open_hens"),
                "evm_n_ad_branches": context.get("evm_n_ad_branches"),
                "evm_n_rm_branches": context.get("evm_n_rm_branches"),
                "evm_no_improvement_patience": context.get(
                    "evm_no_improvement_patience"
                ),
            }
            self.records.append(record)
        if self.progress_queue is not None:
            self.progress_queue.put({"event": "solver_call_record", "record": record})


def small_hens_benchmark_cases(
    *,
    max_streams: int = HENS_MAX_STREAMS,
    include_reordered: bool = False,
) -> tuple[str, ...]:
    """Return benchmark fixture names for unique small OpenHENS problems."""

    cases: list[str] = []
    for fixture_path in sorted(_HENS_FIXTURE_ROOT.glob("*.json")):
        if not include_reordered and fixture_path.stem.endswith(".reordered"):
            continue
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        if len(payload.get("streams", ())) <= int(max_streams):
            cases.append(fixture_path.name)
    return tuple(cases)


HENS_BENCHMARK_CASES = small_hens_benchmark_cases()


def _hens_case_options(
    payload: dict[str, Any],
    *,
    options: dict[str, Any],
    tier: int,
    run_id: str,
    use_fixture_defaults: bool,
) -> dict[str, Any]:
    fixture_options = copy.deepcopy(payload.get("options", {}))
    benchmark_overrides = {
        "HENS_SYNTHESIS_QUALITY_TIER": int(tier),
        "HENS_RUN_ID": run_id,
    }
    if use_fixture_defaults:
        parallel_override = {}
        if "HENS_MAX_PARALLEL" in options:
            parallel_override["HENS_MAX_PARALLEL"] = int(options["HENS_MAX_PARALLEL"])
        return (
            fixture_options
            | parallel_override
            | benchmark_overrides
            | {"HENS_OUTPUT_FORMATS": []}
        )
    return fixture_options | options | benchmark_overrides


class BenchmarkTraceExecutor:
    """Sequential task executor that records benchmark-only task statistics."""

    def __init__(self, delegate=None, progress_queue=None, solver_tracer=None) -> None:
        self.delegate = delegate or LocalSynthesisExecutor()
        self.progress_queue = progress_queue
        self.solver_tracer = solver_tracer
        self.task_records: list[dict[str, Any]] = []
        self.stage_records: list[dict[str, Any]] = []
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
        if not tasks:
            return ()

        method = tasks[0].method
        self.stage_order.append(method)
        stage_label = _stage_label(tasks, parent_outcomes)
        stage_index = len(self.stage_records)
        stage_start = time.perf_counter()
        outcomes: list[HeatExchangerNetworkSynthesisTaskOutcome] = []

        for task in tasks:
            task_start = time.perf_counter()
            start_record = _task_start_record(
                task,
                parent_outcomes,
                stage_label=stage_label,
                stage_index=stage_index,
                max_parallel=max_parallel,
            )
            self._emit_progress(
                {
                    "event": "task_started",
                    "record": start_record,
                }
            )
            try:
                context = (
                    self.solver_tracer.task_context(start_record)
                    if self.solver_tracer is not None
                    else nullcontext()
                )
                with context:
                    task_outcomes = self.delegate.execute(
                        (task,),
                        problem=problem,
                        parent_outcomes=parent_outcomes,
                        max_parallel=max_parallel,
                    )
            except Exception as exc:
                duration = time.perf_counter() - task_start
                self._record_task(
                    _task_record(
                        task,
                        parent_outcomes,
                        stage_label=stage_label,
                        stage_index=stage_index,
                        max_parallel=max_parallel,
                        duration=duration,
                        outcome=None,
                        exception=exc,
                    )
                )
                raise

            if not task_outcomes:
                duration = time.perf_counter() - task_start
                self._record_task(
                    _task_record(
                        task,
                        parent_outcomes,
                        stage_label=stage_label,
                        stage_index=stage_index,
                        max_parallel=max_parallel,
                        duration=duration,
                        outcome=None,
                        exception=RuntimeError("executor returned no outcome"),
                    )
                )
                continue

            outcome = task_outcomes[0]
            outcomes.append(outcome)
            self.executed_tasks.append(task)
            self._record_task(
                _task_record(
                    task,
                    parent_outcomes,
                    stage_label=stage_label,
                    stage_index=stage_index,
                    max_parallel=max_parallel,
                    duration=time.perf_counter() - task_start,
                    outcome=outcome,
                    exception=None,
                )
            )

        self._record_stage(
            {
                "stage_index": stage_index,
                "stage_label": stage_label,
                "method": str(method),
                "planned_task_count": len(tasks),
                "outcome_count": len(outcomes),
                "duration_seconds": time.perf_counter() - stage_start,
            }
        )
        return tuple(outcomes)

    def _record_task(self, record: dict[str, Any]) -> None:
        self.task_records.append(record)
        self._emit_progress({"event": "task_record", "record": record})

    def _record_stage(self, record: dict[str, Any]) -> None:
        self.stage_records.append(record)
        self._emit_progress({"event": "stage_record", "record": record})

    def _emit_progress(self, payload: dict[str, Any]) -> None:
        if self.progress_queue is not None:
            self.progress_queue.put(payload)


def time_hens_tier_case(
    case_name: str,
    *,
    tier: int,
    options: dict[str, Any],
    use_fixture_defaults: bool = False,
    progress_queue=None,
) -> dict[str, Any]:
    fixture_path = _HENS_FIXTURE_ROOT / case_name
    payload = copy.deepcopy(json.loads(fixture_path.read_text(encoding="utf-8")))
    run_id = f"benchmark-{fixture_path.stem}-tier-{tier}"
    payload["options"] = _hens_case_options(
        payload,
        options=options,
        tier=tier,
        run_id=run_id,
        use_fixture_defaults=use_fixture_defaults,
    )
    problem = PinchProblem(source=payload, project_name=case_name)
    solver_tracer = BenchmarkSolverCallTracer(
        case_name=case_name,
        tier=tier,
        progress_queue=progress_queue,
    )
    executor = BenchmarkTraceExecutor(
        progress_queue=progress_queue,
        solver_tracer=solver_tracer,
    )

    start = time.perf_counter()
    try:
        with solver_tracer.patched_backend():
            target_output = ensure_target_results(problem, {}, None)
            settings = workflow_settings_from_problem(problem)
            workflow_result = execute_open_hens_method(
                problem,
                settings,
                executor=executor,
            )
            design = finalise_design_result(problem, target_output, workflow_result)
    except BaseException as exc:
        elapsed = time.perf_counter() - start
        return _failed_hens_result(
            case_name,
            tier=tier,
            status="failed",
            total_seconds=elapsed,
            error=f"{type(exc).__name__}: {exc}",
            diagnostics=_benchmark_diagnostics(
                tier=tier,
                trace_executor=executor,
                solver_call_records=solver_tracer.records,
                selected_task_id=None,
            ),
        )

    elapsed = time.perf_counter() - start
    network = design.network
    manifest = design.manifest
    ranked = tuple(design.ranked_networks)
    diagnostics = _benchmark_diagnostics(
        tier=tier,
        trace_executor=executor,
        solver_call_records=solver_tracer.records,
        selected_task_id=design.task_id,
    )
    return {
        "name": case_name,
        "benchmark": "hens_quality_tier",
        "variant": f"tier_{tier}",
        "tier": int(tier),
        "status": "success",
        "total_seconds": elapsed,
        "method": design.method,
        "stage_count": design.stage_count,
        "task_count": len(manifest.task_ids) if manifest is not None else None,
        "ranked_network_count": len(ranked),
        "successful_candidate_count": sum(
            1 for outcome in ranked if outcome.status == "success"
        ),
        "total_annual_cost": network.total_annual_cost,
        "utility_cost": network.utility_cost,
        "capital_cost": network.capital_cost,
        "task_id": design.task_id,
        "selected_settings": _manifest_settings(manifest),
        "fallback_usage": _fallback_usage(network.summary_metrics),
        "diagnostics": diagnostics,
    }


def time_hens_tier_case_with_timeout(
    case_name: str,
    *,
    tier: int,
    options: dict[str, Any],
    timeout_seconds: float | None,
    use_fixture_defaults: bool = False,
) -> dict[str, Any]:
    if timeout_seconds is None:
        return time_hens_tier_case(
            case_name,
            tier=tier,
            options=options,
            use_fixture_defaults=use_fixture_defaults,
        )

    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    partial_trace = _empty_partial_trace()
    start = time.perf_counter()
    process = ctx.Process(
        target=_time_hens_tier_case_worker,
        args=(queue, case_name, tier, options, use_fixture_defaults),
    )
    process.start()
    result = _wait_for_worker_result(
        process,
        queue,
        partial_trace,
        timeout_seconds=timeout_seconds,
    )
    elapsed = time.perf_counter() - start
    if result is not None:
        if result["status"] == "success":
            return result["payload"]
        return _failed_hens_result(
            case_name,
            tier=tier,
            status=result["status"],
            total_seconds=elapsed,
            error=result["error"],
            diagnostics=result.get("diagnostics"),
        )
    if process.is_alive():
        _terminate_worker(process)
        return _failed_hens_result(
            case_name,
            tier=tier,
            status="timeout",
            total_seconds=elapsed,
            error=f"exceeded {timeout_seconds:g} seconds",
            diagnostics=_timeout_diagnostics(tier=tier, partial_trace=partial_trace),
        )
    return _failed_hens_result(
        case_name,
        tier=tier,
        status="failed",
        total_seconds=elapsed,
        error=f"worker exited with code {process.exitcode}",
        diagnostics=_timeout_diagnostics(tier=tier, partial_trace=partial_trace),
    )


def summarize_hens_tier_comparisons(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for result in results:
        if result.get("benchmark") != "hens_quality_tier":
            continue
        grouped.setdefault(str(result["name"]), {})[int(result["tier"])] = result

    comparisons = []
    for case_name, tier_results in sorted(grouped.items()):
        baseline = tier_results.get(1)
        for tier in sorted(tier_results):
            if tier == 1:
                continue
            comparisons.append(_summarize_tier(case_name, baseline, tier_results[tier]))
    return comparisons


def _time_hens_tier_case_worker(
    queue,
    case_name: str,
    tier: int,
    options: dict[str, Any],
    use_fixture_defaults: bool,
) -> None:
    try:
        os.setsid()
    except OSError:
        pass
    try:
        payload = time_hens_tier_case(
            case_name,
            tier=tier,
            options=options,
            use_fixture_defaults=use_fixture_defaults,
            progress_queue=queue,
        )
    except BaseException as exc:
        queue.put(
            {
                "event": "result",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    else:
        queue.put({"event": "result", "status": "success", "payload": payload})


def _empty_partial_trace() -> dict[str, Any]:
    return {
        "task_records": [],
        "stage_records": [],
        "solver_call_records": [],
        "running_tasks": {},
    }


def _wait_for_worker_result(
    process,
    queue,
    partial_trace: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any] | None:
    deadline = time.perf_counter() + float(timeout_seconds)
    while process.is_alive():
        result = _drain_worker_queue(queue, partial_trace)
        if result is not None:
            process.join(0.1)
            return result
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            break
        process.join(min(1.0, remaining))
    return _drain_worker_queue(queue, partial_trace)


def _drain_worker_queue(queue, partial_trace: dict[str, Any]) -> dict[str, Any] | None:
    result = None
    while True:
        try:
            message = queue.get_nowait()
        except queue_module.Empty:
            break
        event = message.get("event")
        if event == "result":
            result = message
        elif event == "task_started":
            record = message["record"]
            partial_trace["running_tasks"][record["task_id"]] = record
        elif event == "task_record":
            record = message["record"]
            partial_trace["task_records"].append(record)
            partial_trace["running_tasks"].pop(record["task_id"], None)
        elif event == "stage_record":
            partial_trace["stage_records"].append(message["record"])
        elif event == "solver_call_record":
            partial_trace["solver_call_records"].append(message["record"])
    return result


def _terminate_worker(process) -> None:
    try:
        if process.pid is not None and os.getpgid(process.pid) == process.pid:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    except OSError:
        process.terminate()
    process.join(5.0)
    if process.is_alive():
        try:
            if process.pid is not None and os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        except OSError:
            process.kill()
        process.join()


def _failed_hens_result(
    case_name: str,
    *,
    tier: int,
    status: str,
    total_seconds: float,
    error: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "name": case_name,
        "benchmark": "hens_quality_tier",
        "variant": f"tier_{tier}",
        "tier": int(tier),
        "status": status,
        "total_seconds": total_seconds,
        "error": error,
    }
    if diagnostics is not None:
        payload["diagnostics"] = diagnostics
    return payload


def _benchmark_diagnostics(
    *,
    tier: int,
    trace_executor: BenchmarkTraceExecutor,
    solver_call_records: list[dict[str, Any]] | None = None,
    selected_task_id: str | None,
) -> dict[str, Any]:
    return _diagnostics_from_records(
        tier=tier,
        task_records=trace_executor.task_records,
        stage_records=trace_executor.stage_records,
        solver_call_records=list(solver_call_records or []),
        selected_task_id=selected_task_id,
        in_progress_tasks=[],
    )


def _diagnostics_from_records(
    *,
    tier: int,
    task_records: list[dict[str, Any]],
    stage_records: list[dict[str, Any]],
    solver_call_records: list[dict[str, Any]],
    selected_task_id: str | None,
    in_progress_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    task_records = [
        record | {"accepted": bool(record["task_id"] == selected_task_id)}
        for record in task_records
    ]
    return {
        "stage_runtime_seconds": _sum_stage_runtime(stage_records),
        "stage_records": stage_records,
        "task_counts": _task_count_statistics(task_records),
        "robustness": _robustness_statistics(
            task_records,
            in_progress_tasks=in_progress_tasks,
        ),
        "containment": _containment_statistics(tier, task_records),
        "slowest_tasks": _slowest_tasks(task_records, limit=10),
        "in_progress_tasks": in_progress_tasks,
        "solver_calls": solver_call_records,
        "solver_call_runtime_seconds": _sum_solver_call_runtime(solver_call_records),
        "solver_call_counts": _solver_call_count_statistics(solver_call_records),
        "slowest_solver_calls": _slowest_solver_calls(
            solver_call_records,
            limit=10,
        ),
    }


def _timeout_diagnostics(
    *,
    tier: int = -1,
    partial_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    partial_trace = partial_trace or _empty_partial_trace()
    diagnostics = _diagnostics_from_records(
        tier=tier,
        task_records=list(partial_trace["task_records"]),
        stage_records=list(partial_trace["stage_records"]),
        solver_call_records=list(partial_trace["solver_call_records"]),
        selected_task_id=None,
        in_progress_tasks=list(partial_trace["running_tasks"].values()),
    )
    robustness = diagnostics["robustness"]
    robustness["funnel"]["timeout"] = 1
    failure_categories = dict(robustness.get("failure_categories") or {})
    failure_categories["timeout"] = failure_categories.get("timeout", 0) + 1
    robustness["failure_categories"] = failure_categories
    return diagnostics


def _task_start_record(
    task: HeatExchangerNetworkSynthesisTask,
    parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
    *,
    stage_label: str,
    stage_index: int,
    max_parallel: int,
) -> dict[str, Any]:
    pathway = _pathway_summary(task)
    return {
        "task_id": task.task_id,
        "method": str(task.method),
        "stage_label": stage_label,
        "stage_index": stage_index,
        "parent_task_id": task.parent_task_id,
        "parent_method": _parent_method(task, parent_outcomes),
        "approach_temperature": task.approach_temperature,
        "derivative_threshold": task.derivative_threshold,
        "stage_count": task.stage_count,
        "settings": dict(task.settings),
        "max_parallel": int(max_parallel),
        "pathway_ids": pathway["pathway_ids"],
        "pathway_id": pathway["pathway_id"],
        "tier_origins": pathway["tier_origins"],
        "tier_origin": pathway["tier_origin"],
        "pathway_kind": pathway["pathway_kind"],
        "pdm_mode": pathway["pdm_mode"],
        "pdm_multiplier": pathway["pdm_multiplier"],
        "protected_pathway": pathway["protected_pathway"],
        "exact_open_hens": pathway["exact_open_hens"],
        "evm_n_ad_branches": pathway["evm_n_ad_branches"],
        "evm_n_rm_branches": pathway["evm_n_rm_branches"],
        "evm_no_improvement_patience": pathway["evm_no_improvement_patience"],
        "status": "running",
    }


def _task_record(
    task: HeatExchangerNetworkSynthesisTask,
    parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
    *,
    stage_label: str,
    stage_index: int,
    max_parallel: int,
    duration: float,
    outcome: HeatExchangerNetworkSynthesisTaskOutcome | None,
    exception: BaseException | None,
) -> dict[str, Any]:
    pathway = _pathway_summary(task)
    status = "failed"
    solver_status = None
    objective_value = None
    error = None
    network_present = False
    if outcome is not None:
        status = str(outcome.status)
        solver_status = outcome.solver_status
        objective_value = outcome.objective_value
        error = outcome.error
        network_present = outcome.network is not None
    if exception is not None:
        error = f"{type(exception).__name__}: {exception}"

    return {
        "task_id": task.task_id,
        "method": str(task.method),
        "stage_label": stage_label,
        "stage_index": stage_index,
        "parent_task_id": task.parent_task_id,
        "parent_method": _parent_method(task, parent_outcomes),
        "approach_temperature": task.approach_temperature,
        "derivative_threshold": task.derivative_threshold,
        "stage_count": task.stage_count,
        "settings": dict(task.settings),
        "max_parallel": int(max_parallel),
        "pathway_ids": pathway["pathway_ids"],
        "pathway_id": pathway["pathway_id"],
        "tier_origins": pathway["tier_origins"],
        "tier_origin": pathway["tier_origin"],
        "pathway_kind": pathway["pathway_kind"],
        "pdm_mode": pathway["pdm_mode"],
        "pdm_multiplier": pathway["pdm_multiplier"],
        "protected_pathway": pathway["protected_pathway"],
        "exact_open_hens": pathway["exact_open_hens"],
        "evm_n_ad_branches": pathway["evm_n_ad_branches"],
        "evm_n_rm_branches": pathway["evm_n_rm_branches"],
        "evm_no_improvement_patience": pathway["evm_no_improvement_patience"],
        "duration_seconds": duration,
        "status": status,
        "solver_status": solver_status,
        "objective_value": objective_value,
        "network_present": network_present,
        "error": error,
        "failure_category": _failure_category(status, solver_status, error),
    }


def _pathway_summary(task: HeatExchangerNetworkSynthesisTask) -> dict[str, Any]:
    raw_pathways = task.metadata.get("pathways", ())
    pathways = [item for item in raw_pathways if isinstance(item, dict)]
    first = pathways[0] if pathways else {}
    pathway_ids = [
        str(item.get("pathway_id"))
        for item in pathways
        if item.get("pathway_id") is not None
    ]
    tier_origins = sorted(
        {
            int(item["tier_origin"])
            for item in pathways
            if item.get("tier_origin") is not None
        }
    )
    return {
        "pathway_ids": pathway_ids,
        "pathway_id": pathway_ids[0] if pathway_ids else None,
        "tier_origins": tier_origins,
        "tier_origin": (
            int(first["tier_origin"]) if first.get("tier_origin") is not None else None
        ),
        "pathway_kind": first.get("pathway_kind"),
        "pdm_mode": task.settings.get("pdm_mode") or first.get("pdm_mode"),
        "pdm_multiplier": first.get("pdm_multiplier"),
        "protected_pathway": any(bool(item.get("protected")) for item in pathways),
        "exact_open_hens": any(bool(item.get("exact_open_hens")) for item in pathways),
        "evm_n_ad_branches": _setting_or_pathway(
            task,
            "evolution_n_ad_branches",
            first,
            "evm_n_ad_branches",
        ),
        "evm_n_rm_branches": _setting_or_pathway(
            task,
            "evolution_n_rm_branches",
            first,
            "evm_n_rm_branches",
        ),
        "evm_no_improvement_patience": _setting_or_pathway(
            task,
            "evolution_no_improvement_patience",
            first,
            "evm_no_improvement_patience",
        ),
    }


def _setting_or_pathway(
    task: HeatExchangerNetworkSynthesisTask,
    setting_key: str,
    pathway: dict[str, Any],
    pathway_key: str,
):
    if setting_key in task.settings:
        return task.settings[setting_key]
    return pathway.get(pathway_key)


def _stage_label(
    tasks: Sequence[HeatExchangerNetworkSynthesisTask],
    parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
) -> str:
    method = str(tasks[0].method)
    if method == "pinch_design_method":
        return "pdm"
    if method == "thermal_derivative_method":
        return "tdm"
    if method != "network_evolution_method":
        return method

    parent_methods = {
        _parent_method(task, parent_outcomes)
        for task in tasks
        if task.parent_task_id is not None
    }
    if parent_methods == {"pinch_design_method"}:
        return "direct_evm_from_pdm"
    if parent_methods == {"thermal_derivative_method"}:
        return "evm_from_tdm"
    if not parent_methods:
        return "direct_evm"
    return "mixed_evm"


def _parent_method(
    task: HeatExchangerNetworkSynthesisTask,
    parent_outcomes: dict[str, HeatExchangerNetworkSynthesisTaskOutcome],
) -> str | None:
    if task.parent_task_id is None:
        return None
    parent = parent_outcomes.get(task.parent_task_id)
    if parent is None:
        return None
    return str(parent.task.method)


def _sum_stage_runtime(stage_records: list[dict[str, Any]]) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for record in stage_records:
        totals[str(record["stage_label"])] += float(record["duration_seconds"])
    return dict(sorted(totals.items()))


def _sum_solver_call_runtime(
    solver_call_records: list[dict[str, Any]],
) -> dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for label in SOLVER_STAGE_LABELS:
        totals[label] += 0.0
    for record in solver_call_records:
        totals[str(record.get("stage_label") or "untagged")] += float(
            record.get("duration_seconds") or 0.0
        )
    return dict(sorted(totals.items()))


def _solver_call_count_statistics(
    solver_call_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "by_stage": _count_by(solver_call_records, "stage_label"),
        "by_role": _count_by(solver_call_records, "solver_call_role"),
        "by_solver": _count_by(solver_call_records, "solver_name"),
        "by_status": _count_by(solver_call_records, "solver_status"),
    }


def _task_count_statistics(task_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "planned_by_method": _count_by(task_records, "method"),
        "planned_by_stage": _count_by(task_records, "stage_label"),
        "planned_by_pathway": _count_by_pathway(task_records),
        "planned_by_tier_origin": _count_by_tier_origin(task_records),
        "planned_by_pdm_mode": _count_by(task_records, "pdm_mode"),
        "planned_by_multiplier": _count_by(task_records, "pdm_multiplier"),
        "planned_by_evm_setting": _count_by_evm_setting(task_records),
        "success_by_method": _count_by(
            [record for record in task_records if record["status"] == "success"],
            "method",
        ),
        "failure_by_method": _count_by(
            [record for record in task_records if record["status"] != "success"],
            "method",
        ),
    }


def _robustness_statistics(
    task_records: list[dict[str, Any]],
    *,
    in_progress_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    failures = [record for record in task_records if record["status"] != "success"]
    return {
        "funnel": {
            "planned": len(task_records),
            "executed": len(task_records),
            "built_or_returned": sum(
                1
                for record in task_records
                if record["failure_category"] not in {"exception", "missing_parent"}
            ),
            "solver_returned": sum(
                1
                for record in task_records
                if record["solver_status"] is not None or record["status"] == "success"
            ),
            "success": sum(
                1 for record in task_records if record["status"] == "success"
            ),
            "verified_network": sum(
                1
                for record in task_records
                if record["status"] == "success" and record["network_present"]
            ),
            "accepted": sum(1 for record in task_records if record.get("accepted")),
            "timeout": 0,
            "in_progress": len(in_progress_tasks),
        },
        "failure_categories": _counter_dict(
            record["failure_category"] for record in failures
        ),
        "failure_by_method": _count_by(failures, "method"),
        "failure_by_pathway": _count_by_pathway(failures),
        "status_by_method": _nested_status_count(task_records, "method"),
        "status_by_pathway": _nested_status_count_by_pathway(task_records),
        "evm_child_branch_failures_observed": None,
        "evm_child_branch_failure_note": (
            "Child branch failures are isolated inside EVM and are not surfaced "
            "as task outcomes by the production solver path."
        ),
        "in_progress_tasks": in_progress_tasks,
    }


def _containment_statistics(
    tier: int,
    task_records: list[dict[str, Any]],
) -> dict[str, Any]:
    evm_success = [
        record
        for record in task_records
        if record["method"] == "network_evolution_method"
        and record["status"] == "success"
        and record["objective_value"] is not None
    ]
    protected_tier1 = [
        record for record in evm_success if "tier1-open-hens" in record["pathway_ids"]
    ]
    protected_tier0 = [
        record
        for record in evm_success
        if any(pathway_id.startswith("tier0-") for pathway_id in record["pathway_ids"])
    ]
    current_expanded = [
        record
        for record in evm_success
        if not record["protected_pathway"]
        and any(
            pathway_id.startswith(f"tier{tier}-")
            for pathway_id in record["pathway_ids"]
        )
    ]
    protected_tier1_best = _best_objective(protected_tier1)
    current_expanded_best = _best_objective(current_expanded)
    return {
        "protected_tier0_best_tac": _best_objective(protected_tier0),
        "protected_tier1_best_tac": protected_tier1_best,
        "expanded_current_tier_best_tac": current_expanded_best,
        "expanded_current_tier_beats_protected_tier1": (
            None
            if protected_tier1_best is None or current_expanded_best is None
            else current_expanded_best < protected_tier1_best
        ),
        "successful_evm_pathway_count": len(
            {
                pathway_id
                for record in evm_success
                for pathway_id in record["pathway_ids"]
            }
        ),
    }


def _slowest_tasks(
    task_records: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    fields = (
        "task_id",
        "method",
        "stage_label",
        "pathway_ids",
        "parent_task_id",
        "duration_seconds",
        "status",
        "solver_status",
        "objective_value",
        "failure_category",
        "error",
    )
    return [
        {field: record.get(field) for field in fields}
        for record in sorted(
            task_records,
            key=lambda item: float(item["duration_seconds"]),
            reverse=True,
        )[:limit]
    ]


def _slowest_solver_calls(
    solver_call_records: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    fields = (
        "solver_call_index",
        "task_id",
        "stage_label",
        "method",
        "solver_call_role",
        "model_name",
        "solver_name",
        "duration_seconds",
        "solver_reported_seconds",
        "solver_status",
        "objective_value",
        "failure_reason",
        "pathway_ids",
    )
    return [
        {field: record.get(field) for field in fields}
        for record in sorted(
            solver_call_records,
            key=lambda item: float(item.get("duration_seconds") or 0.0),
            reverse=True,
        )[:limit]
    ]


def _failure_category(
    status: str,
    solver_status: str | None,
    error: str | None,
) -> str:
    if status == "success":
        return "success"
    text = " ".join(
        str(value or "") for value in (status, solver_status, error)
    ).lower()
    if "timeout" in text or "exceeded" in text:
        return "timeout"
    if "locally infeasible" in text:
        return "locally_infeasible"
    if "infeasible" in text:
        return "infeasible"
    if "verification failed" in text:
        return "verification_failure"
    if "missing parent" in text or "parent outcome" in text:
        return "missing_parent"
    if "without recovery topology" in text or "without a heatexchangernetwork" in text:
        return "invalid_or_no_network"
    if "exception" in text or "traceback" in text:
        return "exception"
    if error:
        return "failed"
    return "failed"


def _best_objective(records: list[dict[str, Any]]) -> float | None:
    values = [
        float(record["objective_value"])
        for record in records
        if record["objective_value"] is not None
    ]
    return min(values) if values else None


def _count_by(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return _counter_dict(_normalise_key(record.get(field)) for record in records)


def _count_by_pathway(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        pathway_ids = record.get("pathway_ids") or ()
        if not pathway_ids:
            counter["untagged"] += 1
            continue
        counter.update(str(pathway_id) for pathway_id in pathway_ids)
    return dict(sorted(counter.items()))


def _count_by_tier_origin(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        tier_origins = record.get("tier_origins") or ()
        if not tier_origins:
            counter["untagged"] += 1
            continue
        counter.update(str(tier_origin) for tier_origin in tier_origins)
    return dict(sorted(counter.items()))


def _count_by_evm_setting(records: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        if record["method"] != "network_evolution_method":
            continue
        key = (
            f"ad={record.get('evm_n_ad_branches') or 1};"
            f"rm={record.get('evm_n_rm_branches') or 1};"
            f"patience={record.get('evm_no_improvement_patience') or 'none'}"
        )
        counter[key] += 1
    return dict(sorted(counter.items()))


def _nested_status_count(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, int]]:
    nested: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        nested[_normalise_key(record.get(field))][str(record["status"])] += 1
    return {
        key: dict(sorted(counter.items())) for key, counter in sorted(nested.items())
    }


def _nested_status_count_by_pathway(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    nested: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        pathway_ids = record.get("pathway_ids") or ("untagged",)
        for pathway_id in pathway_ids:
            nested[str(pathway_id)][str(record["status"])] += 1
    return {
        key: dict(sorted(counter.items())) for key, counter in sorted(nested.items())
    }


def _counter_dict(values) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _normalise_key(value) -> str:
    if value is None:
        return "untagged"
    return str(value)


def _solver_model_name(model) -> str | None:
    for attr in (
        "_openpinch_benchmark_model_name",
        "_openpinch_model_name",
        "name",
        "_model_name",
    ):
        value = getattr(model, attr, None)
        if value:
            return str(value)
    return _solver_model_path(model)


def _solver_model_path(model) -> str | None:
    value = getattr(model, "_path", None)
    return str(value) if value is not None else None


def _solver_run_attr(solver_run, attr: str):
    if solver_run is None:
        return None
    return getattr(solver_run, attr, None)


def _solver_failure_reason(solver_run, error: BaseException | None) -> str | None:
    if error is not None:
        return f"{type(error).__name__}: {error}"
    return _solver_run_attr(solver_run, "failure_reason")


def _solver_call_role(
    stage_label,
    method,
    call_index_within_task: int,
) -> str:
    stage = str(stage_label or "")
    method = str(method or "")
    if stage == "pdm":
        if call_index_within_task == 1:
            return "pdm_above"
        if call_index_within_task == 2:
            return "pdm_below"
        return "pdm_stage_reduction"
    if stage == "tdm":
        if call_index_within_task == 1:
            return "tdm_base"
        return "tdm_stage_reduction"
    if method == "network_evolution_method" or stage.startswith("evm"):
        if call_index_within_task == 1:
            return "evm_base"
        return "evm_branch_or_recovery"
    return "solver_call"


def _manifest_settings(manifest) -> dict[str, Any] | None:
    if manifest is None:
        return None
    return {
        "synthesis_quality_tier": manifest.synthesis_quality_tier,
        "pdm_stage_pair_limit": manifest.pdm_stage_pair_limit,
        "tdm_parent_limit": manifest.tdm_parent_limit,
        "stage_packing": manifest.stage_packing,
        "derivative_thresholds": list(manifest.derivative_thresholds),
        "evm_n_ad_branches": manifest.evm_n_ad_branches,
        "evm_n_rm_branches": manifest.evm_n_rm_branches,
        "selected_pathway_id": manifest.selected_pathway_id,
        "selected_pathway_kind": manifest.selected_pathway_kind,
        "selected_pdm_mode": manifest.selected_pdm_mode,
        "selected_tier_origin": manifest.selected_tier_origin,
        "selected_protected_pathway": manifest.selected_protected_pathway,
        "task_count_by_method": dict(manifest.task_count_by_method),
    }


def _fallback_usage(summary_metrics: dict[str, Any]) -> dict[str, bool]:
    return {
        "used_seed_fallback": bool(summary_metrics.get("used_seed_fallback", False)),
        "used_parent_fallback": bool(
            summary_metrics.get("used_parent_fallback", False)
        ),
    }


def _summarize_tier(
    case_name: str,
    baseline: dict[str, Any] | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "name": case_name,
        "benchmark": "hens_quality_tier_comparison",
        "baseline_variant": "tier_1",
        "variant": result["variant"],
        "tier": result["tier"],
    }
    if baseline is None:
        return summary | {
            "status": "missing_baseline",
            "variant_status": result.get("status", "success"),
        }
    if not _successful_hens_result(baseline):
        return summary | {
            "status": "baseline_not_successful",
            "baseline_status": baseline.get("status", "success"),
            "variant_status": result.get("status", "success"),
        }
    if not _successful_hens_result(result):
        return summary | {
            "status": "variant_not_successful",
            "baseline_status": baseline.get("status", "success"),
            "variant_status": result.get("status", "success"),
        }

    baseline_seconds = float(baseline["total_seconds"])
    variant_seconds = float(result["total_seconds"])
    baseline_tac = float(baseline["total_annual_cost"])
    variant_tac = float(result["total_annual_cost"])
    tac_delta = variant_tac - baseline_tac
    tac_delta_percent = (
        100.0 * tac_delta / abs(baseline_tac) if baseline_tac != 0.0 else None
    )
    return summary | {
        "status": "success",
        "baseline_seconds": baseline_seconds,
        "variant_seconds": variant_seconds,
        "speedup": (
            baseline_seconds / variant_seconds if variant_seconds > 0.0 else None
        ),
        "baseline_total_annual_cost": baseline_tac,
        "variant_total_annual_cost": variant_tac,
        "total_annual_cost_delta": tac_delta,
        "total_annual_cost_delta_percent": tac_delta_percent,
        "no_worse_than_tier_1": variant_tac <= baseline_tac,
        "selected_pathway_id": (result.get("selected_settings") or {}).get(
            "selected_pathway_id"
        ),
        "selected_protected_pathway": (result.get("selected_settings") or {}).get(
            "selected_protected_pathway"
        ),
    }


def _successful_hens_result(result: dict[str, Any]) -> bool:
    return (
        result.get("status", "success") == "success"
        and "total_seconds" in result
        and "total_annual_cost" in result
    )


def _results_payload(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [*results, *summarize_hens_tier_comparisons(results)]


def _write_incremental_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_results_payload(results), indent=2) + "\n",
        encoding="utf-8",
    )


def _format_result_row(result: dict[str, Any]) -> str:
    name = str(result["name"]).replace(".json", "")
    tier = result["tier"]
    status = result.get("status", "success")
    seconds = float(result.get("total_seconds", 0.0))
    diagnostics = result.get("diagnostics") or {}
    stage_seconds = diagnostics.get("stage_runtime_seconds") or {}
    call_counts = (diagnostics.get("solver_call_counts") or {}).get("by_stage") or {}
    evm_direct_seconds = _stage_total(
        stage_seconds,
        ("direct_evm_from_pdm", "direct_evm"),
    )
    evm_from_tdm_seconds = _stage_total(stage_seconds, ("evm_from_tdm",))
    evm_direct_calls = _stage_total(
        call_counts,
        ("direct_evm_from_pdm", "direct_evm"),
    )
    evm_from_tdm_calls = _stage_total(call_counts, ("evm_from_tdm",))
    slowest_solver = _slowest_solver_call_label(diagnostics)
    failure_category = _dominant_failure_category(diagnostics)
    if status != "success":
        error = result.get("error", "")
        return (
            f"{name:48s} tier {tier}: {status:8s} total={seconds:8.2f}s "
            f"pdm={_stage_total(stage_seconds, ('pdm',)):7.2f}s "
            f"tdm={_stage_total(stage_seconds, ('tdm',)):7.2f}s "
            f"evm_direct={evm_direct_seconds:7.2f}s "
            f"evm_tdm={evm_from_tdm_seconds:7.2f}s "
            f"calls[pdm={int(_stage_total(call_counts, ('pdm',)))},"
            f"tdm={int(_stage_total(call_counts, ('tdm',)))},"
            f"evmD={int(evm_direct_calls)},evmT={int(evm_from_tdm_calls)}] "
            f"slowest={slowest_solver} category={failure_category} error={error}"
        )
    settings = result.get("selected_settings") or {}
    return (
        f"{name:48s} tier {tier}: success  total={seconds:8.2f}s "
        f"pdm={_stage_total(stage_seconds, ('pdm',)):7.2f}s "
        f"tdm={_stage_total(stage_seconds, ('tdm',)):7.2f}s "
        f"evm_direct={evm_direct_seconds:7.2f}s "
        f"evm_tdm={evm_from_tdm_seconds:7.2f}s "
        f"calls[pdm={int(_stage_total(call_counts, ('pdm',)))},"
        f"tdm={int(_stage_total(call_counts, ('tdm',)))},"
        f"evmD={int(evm_direct_calls)},evmT={int(evm_from_tdm_calls)}] "
        f"slowest={slowest_solver} "
        f"TAC={float(result['total_annual_cost']):12.2f} "
        f"selected={settings.get('selected_pathway_id') or 'untagged'}"
    )


def _format_comparison_row(comparison: dict[str, Any]) -> str:
    name = str(comparison["name"]).replace(".json", "")
    tier = comparison["tier"]
    status = comparison.get("status", "success")
    if status != "success":
        return (
            f"{name:48s} tier {tier} vs tier 1: {status} "
            f"baseline={comparison.get('baseline_status', 'n/a')} "
            f"variant={comparison.get('variant_status', 'n/a')}"
        )
    speedup = comparison.get("speedup")
    delta = float(comparison.get("total_annual_cost_delta") or 0.0)
    percent = comparison.get("total_annual_cost_delta_percent")
    route = "protected" if comparison.get("selected_protected_pathway") else "expanded"
    return (
        f"{name:48s} tier {tier} vs tier 1: "
        f"speedup={_format_ratio(speedup)} "
        f"TAC_delta={delta:+12.2f} ({_format_percent(percent)}) "
        f"no_worse={comparison.get('no_worse_than_tier_1')} "
        f"winner={route} "
        f"selected={comparison.get('selected_pathway_id') or 'untagged'}"
    )


def _stage_total(values: dict[str, Any], keys: tuple[str, ...]) -> float:
    return sum(float(values.get(key) or 0.0) for key in keys)


def _slowest_solver_call_label(diagnostics: dict[str, Any]) -> str:
    slowest = (diagnostics.get("slowest_solver_calls") or [{}])[0]
    if not slowest:
        return "none"
    return (
        f"{slowest.get('stage_label') or 'untagged'}/"
        f"{slowest.get('solver_call_role') or 'solver_call'}:"
        f"{float(slowest.get('duration_seconds') or 0.0):.2f}s"
    )


def _dominant_failure_category(diagnostics: dict[str, Any]) -> str:
    categories = (diagnostics.get("robustness") or {}).get("failure_categories") or {}
    if not categories:
        return "none"
    return max(categories.items(), key=lambda item: int(item[1]))[0]


def _format_ratio(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}x"


def _format_percent(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_HENS_BENCHMARK_JSON)
    parser.add_argument("--hens-timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--hens-max-parallel", type=int, default=HENS_MAX_PARALLEL)
    parser.add_argument("--hens-max-streams", type=int, default=HENS_MAX_STREAMS)
    parser.add_argument(
        "--use-fixture-defaults",
        action="store_true",
        help=(
            "Use each OpenHENS fixture's native HENS options; only tier, run id, "
            "max parallelism, and output formats are overridden."
        ),
    )
    parser.add_argument(
        "--include-reordered",
        action="store_true",
        help="Include reordered duplicate OpenHENS fixture variants.",
    )
    parser.add_argument("--tier", type=int, action="append", dest="tiers")
    parser.add_argument(
        "--hens-case",
        action="append",
        dest="hens_cases",
        help="OpenHENS fixture filename to benchmark. May be passed multiple times.",
    )
    args = parser.parse_args()

    tiers = tuple(args.tiers or HENS_QUALITY_TIERS)
    cases = tuple(
        args.hens_cases
        or small_hens_benchmark_cases(
            max_streams=args.hens_max_streams,
            include_reordered=args.include_reordered,
        )
    )
    options = (
        {"HENS_MAX_PARALLEL": int(args.hens_max_parallel)}
        if args.use_fixture_defaults
        else {
            **HENS_BENCHMARK_OPTIONS,
            "HENS_MAX_PARALLEL": int(args.hens_max_parallel),
        }
    )
    results: list[dict[str, Any]] = []
    for case_name in cases:
        for tier in tiers:
            result = time_hens_tier_case_with_timeout(
                case_name,
                tier=tier,
                options=options,
                timeout_seconds=args.hens_timeout_seconds,
                use_fixture_defaults=bool(args.use_fixture_defaults),
            )
            results.append(result)
            if args.json is not None:
                _write_incremental_results(args.json, results)
            print(_format_result_row(result), flush=True)

    comparisons = summarize_hens_tier_comparisons(results)
    if comparisons:
        print("\nTier comparison vs tier 1", flush=True)
        for comparison in comparisons:
            print(_format_comparison_row(comparison), flush=True)

    if args.json is None:
        print(json.dumps(_results_payload(results), indent=2))
    else:
        _write_incremental_results(args.json, results)


if __name__ == "__main__":
    main()
