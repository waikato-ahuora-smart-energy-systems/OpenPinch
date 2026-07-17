"""Compare top-ranked OpenHENS and OpenPinch HEN designs for fixture cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing
import os
import queue as queue_module
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_exchanger_networks.results.selection import ranked_networks

_HENS_PACKAGE = "OpenPinch.analysis.heat_exchanger_networks"
_executor_module = import_module(f"{_HENS_PACKAGE}.common.execution.executor")
_settings_module = import_module(f"{_HENS_PACKAGE}.common.execution.settings")
_open_hens_module = import_module(f"{_HENS_PACKAGE}.targeting.open_hens_method")
LocalSynthesisExecutor = _executor_module.LocalSynthesisExecutor
workflow_settings_from_problem = _settings_module.workflow_settings_from_problem
execute_open_hens_method = _open_hens_module.execute_open_hens_method

CASE_IDS = (
    "Four-stream-Escobar-and-Trierweiler-2013-1",
    "Four-stream-Yee-and-Grossmann-1990-1",
    "Five-stream-Bogataj-and-Kravanja-2012-1",
    "Five-stream-Kim-et-al-2017-1",
    "Six-stream-Spray-Dryer-2025-1",
    "Six-stream-Yee-and-Grossmann-1990-1",
)
CASE_ORDER = {case_id: index for index, case_id in enumerate(CASE_IDS)}
OPENHENS_DT_GRID = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
OPENHENS_DQDA_GRID = (0.5, 0.9, 1.3, 1.7, 2.1, 2.4, 2.8, 3.2, 3.6, 4.0)
DUTY_TOLERANCE_KW = 1.0
TAC_COMPARE_TOLERANCE = 1e-6
_SOURCE_TIMEOUTS: list[dict[str, Any]] = []
_NO_RESULT = object()
_TIMEOUT_FIELDNAMES = (
    "case_id",
    "problem",
    "framework",
    "dTmin",
    "min_dqda",
    "timeout_seconds",
)
_RUN_SUMMARY_FIELDNAMES = (
    "case_id",
    "d_tmin_grid",
    "dqda_grid",
    "is_full_openhens_grid",
    "top_n",
    "source_solution_count",
    "source_esm_solution_count",
    "source_unique_count",
    "openpinch_unique_count",
    "source_timeout_count",
)


@dataclass(frozen=True)
class RankedNetwork:
    engine: str
    case_id: str
    rank: int
    tac: float
    d_tmin: float | None
    min_dq: float | None
    stages: int | None
    recovery_units: int | None
    hot_utility_units: int | None
    cold_utility_units: int | None
    signature: tuple[str, ...]

    @property
    def signature_text(self) -> str:
        return ";".join(self.signature)

    @property
    def signature_hash(self) -> str:
        return hashlib.sha256(self.signature_text.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class CaseRunSummary:
    case_id: str
    d_tmin_grid: tuple[float, ...]
    dqda_grid: tuple[float, ...]
    is_full_openhens_grid: bool
    top_n: int
    source_solution_count: int | None
    source_esm_solution_count: int | None
    source_unique_count: int
    openpinch_unique_count: int
    source_timeout_count: int


def main() -> None:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    openhens_root = args.openhens_root.resolve()
    _configure_solver_path()
    _install_openhens_compatibility(
        openhens_root,
        source_runner=args.source_runner,
        source_task_timeout=args.source_task_timeout,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_engine: list[RankedNetwork] = []
    rows_by_rank: list[dict[str, Any]] = []
    run_summaries: list[CaseRunSummary] = []
    if args.append_existing:
        rows_by_engine = _read_network_rows(
            args.output_dir / "top5_unique_networks.csv"
        )
        rows_by_rank = _read_rank_rows(args.output_dir / "top5_rank_comparison.csv")
        rows_by_timeout = _read_timeout_rows(
            args.output_dir / "source_task_timeouts.csv"
        )
        run_summaries = _read_run_summaries(args.output_dir / "case_run_summary.csv")
        rerun_cases = set(args.case_ids)
        if not (args.source_diagnostics_only or args.openpinch_diagnostics_only):
            rows_by_engine = [
                row for row in rows_by_engine if row.case_id not in rerun_cases
            ]
            rows_by_rank = [
                row for row in rows_by_rank if row.get("case_id") not in rerun_cases
            ]
            rows_by_timeout = [
                row for row in rows_by_timeout if row.get("case_id") not in rerun_cases
            ]
            run_summaries = [
                row for row in run_summaries if row.case_id not in rerun_cases
            ]
        elif args.source_diagnostics_only:
            pass
    else:
        rows_by_timeout: list[dict[str, Any]] = []
    if args.report_only:
        if not rows_by_engine and not rows_by_rank:
            rows_by_engine = _read_network_rows(
                args.output_dir / "top5_unique_networks.csv"
            )
            rows_by_rank = _read_rank_rows(args.output_dir / "top5_rank_comparison.csv")
        if not rows_by_timeout:
            rows_by_timeout = _read_timeout_rows(
                args.output_dir / "source_task_timeouts.csv"
            )
        if not run_summaries:
            run_summaries = _read_run_summaries(
                args.output_dir / "case_run_summary.csv"
            )
        _write_outputs(
            args.output_dir,
            rows_by_rank=rows_by_rank,
            network_rows=rows_by_engine,
            timeout_rows=rows_by_timeout,
            run_summaries=run_summaries,
            d_tmin_grid=args.approach_temperatures,
            dqda_grid=args.derivative_thresholds,
            top_n=args.top,
        )
        return
    for case_id in args.case_ids:
        print(f"[compare] {case_id}", flush=True)
        if args.openpinch_diagnostics_only:
            source_rows = [
                row
                for row in rows_by_engine
                if row.case_id == case_id and row.engine == "OpenHENS"
            ]
            existing_openpinch_rows = [
                row
                for row in rows_by_engine
                if row.case_id == case_id and row.engine == "OpenPinch"
            ]
            new_openpinch_rows = _run_openpinch(
                case_id,
                repo_root=repo_root,
                d_tmin_grid=args.approach_temperatures,
                dqda_grid=args.derivative_thresholds,
                top_n=args.top,
                max_parallel=args.openpinch_max_parallel,
                task_timeout=args.openpinch_task_timeout,
            )
            merged_openpinch_rows = _merge_ranked_networks(
                [*existing_openpinch_rows, *new_openpinch_rows],
                top_n=args.top,
            )
            rows_by_engine = [
                row
                for row in rows_by_engine
                if not (row.case_id == case_id and row.engine == "OpenPinch")
            ]
            rows_by_engine.extend(merged_openpinch_rows)
            rows_by_rank = [
                row for row in rows_by_rank if row.get("case_id") != case_id
            ]
            rows_by_rank.extend(
                _compare_rank_rows(case_id, source_rows, merged_openpinch_rows)
            )
            run_summaries = [
                _summary_with_openpinch_count(
                    summary,
                    case_id=case_id,
                    openpinch_unique_count=len(merged_openpinch_rows),
                )
                for summary in run_summaries
            ]
            _write_outputs(
                args.output_dir,
                rows_by_rank=rows_by_rank,
                network_rows=rows_by_engine,
                timeout_rows=rows_by_timeout,
                run_summaries=run_summaries,
                d_tmin_grid=args.approach_temperatures,
                dqda_grid=args.derivative_thresholds,
                top_n=args.top,
            )
            continue
        timeout_start = len(_SOURCE_TIMEOUTS)
        source_rows, source_stats = _run_source_openhens(
            case_id,
            openhens_root=openhens_root,
            d_tmin_grid=args.approach_temperatures,
            dqda_grid=args.derivative_thresholds,
            top_n=args.top,
            max_parallel=args.source_max_parallel,
        )
        rows_by_timeout.extend(
            {**record, "case_id": case_id}
            for record in _SOURCE_TIMEOUTS[timeout_start:]
        )
        source_timeout_count = len(_SOURCE_TIMEOUTS) - timeout_start
        if args.source_diagnostics_only:
            existing_summary = next(
                (row for row in run_summaries if row.case_id == case_id),
                None,
            )
            existing_source_rows = [
                row
                for row in rows_by_engine
                if row.case_id == case_id and row.engine == "OpenHENS"
            ]
            openpinch_rows = [
                row
                for row in rows_by_engine
                if row.case_id == case_id and row.engine == "OpenPinch"
            ]
            merged_source_rows = _merge_ranked_networks(
                [*existing_source_rows, *source_rows],
                top_n=args.top,
            )
            rows_by_engine = [
                row
                for row in rows_by_engine
                if not (row.case_id == case_id and row.engine == "OpenHENS")
            ]
            rows_by_engine.extend(merged_source_rows)
            rows_by_rank = [
                row for row in rows_by_rank if row.get("case_id") != case_id
            ]
            rows_by_rank.extend(
                _compare_rank_rows(case_id, merged_source_rows, openpinch_rows)
            )
            total_source_timeout_count = sum(
                1 for row in rows_by_timeout if row.get("case_id") == case_id
            )
            current_is_full_grid = (
                args.approach_temperatures == OPENHENS_DT_GRID
                and args.derivative_thresholds == OPENHENS_DQDA_GRID
            )
            if existing_summary is None:
                summary_d_tmin_grid = args.approach_temperatures
                summary_dqda_grid = args.derivative_thresholds
                summary_is_full_grid = current_is_full_grid
                summary_source_solution_count = source_stats["source_solution_count"]
                summary_source_esm_count = source_stats["source_esm_solution_count"]
            else:
                summary_d_tmin_grid = existing_summary.d_tmin_grid
                summary_dqda_grid = existing_summary.dqda_grid
                if source_rows or source_timeout_count:
                    summary_d_tmin_grid = _merge_grids(
                        summary_d_tmin_grid,
                        args.approach_temperatures,
                    )
                    summary_dqda_grid = _merge_grids(
                        summary_dqda_grid,
                        args.derivative_thresholds,
                    )
                summary_is_full_grid = (
                    existing_summary.is_full_openhens_grid or current_is_full_grid
                )
                summary_source_solution_count = _add_optional_ints(
                    existing_summary.source_solution_count,
                    source_stats["source_solution_count"],
                )
                summary_source_esm_count = _add_optional_ints(
                    existing_summary.source_esm_solution_count,
                    source_stats["source_esm_solution_count"],
                )
            run_summaries = [row for row in run_summaries if row.case_id != case_id]
            run_summaries.append(
                CaseRunSummary(
                    case_id=case_id,
                    d_tmin_grid=summary_d_tmin_grid,
                    dqda_grid=summary_dqda_grid,
                    is_full_openhens_grid=summary_is_full_grid,
                    top_n=args.top,
                    source_solution_count=summary_source_solution_count,
                    source_esm_solution_count=summary_source_esm_count,
                    source_unique_count=len(merged_source_rows),
                    openpinch_unique_count=len(openpinch_rows),
                    source_timeout_count=total_source_timeout_count,
                )
            )
            _write_outputs(
                args.output_dir,
                rows_by_rank=rows_by_rank,
                network_rows=rows_by_engine,
                timeout_rows=rows_by_timeout,
                run_summaries=run_summaries,
                d_tmin_grid=args.approach_temperatures,
                dqda_grid=args.derivative_thresholds,
                top_n=args.top,
            )
            continue
        openpinch_rows = _run_openpinch(
            case_id,
            repo_root=repo_root,
            d_tmin_grid=args.approach_temperatures,
            dqda_grid=args.derivative_thresholds,
            top_n=args.top,
            max_parallel=args.openpinch_max_parallel,
            task_timeout=args.openpinch_task_timeout,
        )
        rows_by_engine.extend(source_rows)
        rows_by_engine.extend(openpinch_rows)
        rows_by_rank.extend(_compare_rank_rows(case_id, source_rows, openpinch_rows))
        run_summaries.append(
            CaseRunSummary(
                case_id=case_id,
                d_tmin_grid=args.approach_temperatures,
                dqda_grid=args.derivative_thresholds,
                is_full_openhens_grid=(
                    args.approach_temperatures == OPENHENS_DT_GRID
                    and args.derivative_thresholds == OPENHENS_DQDA_GRID
                ),
                top_n=args.top,
                source_solution_count=source_stats["source_solution_count"],
                source_esm_solution_count=source_stats["source_esm_solution_count"],
                source_unique_count=len(source_rows),
                openpinch_unique_count=len(openpinch_rows),
                source_timeout_count=source_timeout_count,
            )
        )
        _write_outputs(
            args.output_dir,
            rows_by_rank=rows_by_rank,
            network_rows=rows_by_engine,
            timeout_rows=rows_by_timeout,
            run_summaries=run_summaries,
            d_tmin_grid=args.approach_temperatures,
            dqda_grid=args.derivative_thresholds,
            top_n=args.top,
        )


def _write_outputs(
    output_dir: Path,
    *,
    rows_by_rank: list[dict[str, Any]],
    network_rows: list[RankedNetwork],
    timeout_rows: list[dict[str, Any]],
    run_summaries: list[CaseRunSummary],
    d_tmin_grid: tuple[float, ...],
    dqda_grid: tuple[float, ...],
    top_n: int,
) -> None:
    rows_by_rank = _enrich_rank_rows_with_networks(rows_by_rank, network_rows)
    rows_by_rank = sorted(rows_by_rank, key=_rank_row_sort_key)
    network_rows = sorted(network_rows, key=_network_row_sort_key)
    timeout_rows = sorted(timeout_rows, key=_timeout_row_sort_key)
    run_summaries = sorted(run_summaries, key=_run_summary_sort_key)
    _write_network_rows(output_dir / "top5_unique_networks.csv", network_rows)
    _write_rank_rows(output_dir / "top5_rank_comparison.csv", rows_by_rank)
    _write_timeout_rows(output_dir / "source_task_timeouts.csv", timeout_rows)
    _write_run_summaries(output_dir / "case_run_summary.csv", run_summaries)
    _write_markdown_report(
        output_dir / "top5_rank_comparison.md",
        rows_by_rank=rows_by_rank,
        network_rows=network_rows,
        timeout_rows=timeout_rows,
        run_summaries=run_summaries,
        d_tmin_grid=d_tmin_grid,
        dqda_grid=dqda_grid,
        top_n=top_n,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Path to the OpenPinch checkout.",
    )
    parser.add_argument(
        "--openhens-root",
        type=Path,
        default=Path("../OpenHENS"),
        help="Path to the source OpenHENS checkout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/developer/openhens-openpinch-top5-comparison"),
        help="Directory for comparison CSV and Markdown artifacts.",
    )
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        choices=CASE_IDS,
        help="Case id to compare. Repeat to run a subset.",
    )
    parser.add_argument(
        "--approach-temperatures",
        type=_float_csv,
        default=OPENHENS_DT_GRID,
        help="Comma-separated dTmin grid. Defaults to the OpenHENS grid.",
    )
    parser.add_argument(
        "--derivative-thresholds",
        type=_float_csv,
        default=OPENHENS_DQDA_GRID,
        help="Comma-separated min dQ/dA grid. Defaults to the OpenHENS grid.",
    )
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument(
        "--source-runner",
        choices=("parallel", "bounded", "sequential"),
        default="bounded",
    )
    parser.add_argument("--source-max-parallel", type=int, default=10)
    parser.add_argument(
        "--source-task-timeout",
        type=float,
        default=600.0,
        help="Per-source-task wall-clock timeout in bounded source mode.",
    )
    parser.add_argument("--openpinch-max-parallel", type=int, default=10)
    parser.add_argument(
        "--openpinch-task-timeout",
        type=float,
        default=0.0,
        help=("Per OpenPinch grid wall-clock timeout. Disabled when 0 or lower."),
    )
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Load existing output CSVs and replace only the requested cases.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate Markdown/CSV artifacts from existing output CSVs.",
    )
    parser.add_argument(
        "--source-diagnostics-only",
        action="store_true",
        help=(
            "Run only source OpenHENS for coverage diagnostics, preserving "
            "existing paired comparison rows for the requested cases."
        ),
    )
    parser.add_argument(
        "--openpinch-diagnostics-only",
        action="store_true",
        help=(
            "Run only OpenPinch for selected grids, merging results into "
            "existing comparison rows for the requested cases."
        ),
    )
    args = parser.parse_args()
    args.case_ids = tuple(args.case_ids or CASE_IDS)
    args.approach_temperatures = tuple(args.approach_temperatures)
    args.derivative_thresholds = tuple(args.derivative_thresholds)
    if args.top < 1:
        parser.error("--top must be at least 1")
    if args.source_diagnostics_only and args.openpinch_diagnostics_only:
        parser.error(
            "--source-diagnostics-only and --openpinch-diagnostics-only "
            "are mutually exclusive"
        )
    return args


def _float_csv(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one numeric value")
    return values


def _configure_solver_path() -> None:
    idaes_bin = Path.home() / ".idaes" / "bin"
    os.environ["PATH"] = str(idaes_bin) + os.pathsep + os.environ.get("PATH", "")


def _install_openhens_compatibility(
    openhens_root: Path,
    *,
    source_runner: str,
    source_task_timeout: float,
) -> None:
    sys.path.insert(0, str(openhens_root))
    import openhens.main as source_main
    from openhens.classes.pinch_classes import process as process_module
    from openhens.classes.pinch_classes import publicOperations as public_ops

    def organise_array(input_array, num_dim=2, reverse=True):
        if num_dim == 2:
            input_array[0] = [item for item in input_array[0] if item is not None]
            public_ops.QuickSort_2D(input_array)
            public_ops.RemoveDuplicates_2D(input_array)
            if reverse:
                public_ops.ReverseArray_2D(input_array)
        else:
            input_array = [item for item in input_array if item is not None]
            public_ops.QuickSort_1D(input_array)
            input_array = public_ops.RemoveDuplicates_1D(input_array)
            if reverse:
                public_ops.ReverseArray_1D(input_array)
        return input_array

    public_ops.OrganiseArray = organise_array
    process_module.OrganiseArray = organise_array

    if source_runner in {"parallel", "bounded"}:
        multiprocessing.set_start_method("fork", force=True)
    if source_runner == "parallel":
        return

    from openhens.utils.branching import run_single_solution

    if source_runner == "bounded":
        source_main.run_parallel_solutions = _bounded_source_runner(
            run_single_solution,
            task_timeout=source_task_timeout,
        )
        return

    def run_sequential_solutions(
        problems,
        max_parallel=1,
        print_output=False,
        evolution=False,
    ):
        del max_parallel
        solved_cases = []
        for problem in problems:
            solved = run_single_solution(problem, print_output, evolution)
            if solved:
                solved_cases.extend(solved)
        return solved_cases

    source_main.run_parallel_solutions = run_sequential_solutions


def _bounded_source_runner(run_single_solution, *, task_timeout: float):
    def run_bounded_solutions(
        problems,
        max_parallel=1,
        print_output=False,
        evolution=False,
    ):
        solved_cases = []
        pending = list(problems)
        active: list[dict[str, Any]] = []

        def launch_next() -> None:
            if not pending:
                return
            problem = pending.pop(0)
            queue = multiprocessing.Queue(maxsize=1)
            process = multiprocessing.Process(
                target=_run_source_problem_child,
                args=(run_single_solution, problem, print_output, evolution, queue),
            )
            process.start()
            active.append(
                {
                    "problem": problem,
                    "process": process,
                    "queue": queue,
                    "start": time.monotonic(),
                }
            )

        for _ in range(max(1, int(max_parallel))):
            launch_next()

        while active:
            for item in list(active):
                process = item["process"]
                problem = item["problem"]
                elapsed = time.monotonic() - item["start"]
                solved = _queue_result(item["queue"], problem)
                if solved is not _NO_RESULT:
                    if solved:
                        solved_cases.extend(solved)
                    process.join(timeout=5)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                    item["queue"].close()
                    active.remove(item)
                    launch_next()
                    continue
                if process.is_alive() and elapsed <= task_timeout:
                    continue
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
                    d_tmin, min_dqda = _source_problem_grid_values(problem)
                    timeout_record = {
                        "problem": getattr(problem, "name", repr(problem)),
                        "framework": getattr(problem, "framework", None),
                        "dTmin": d_tmin,
                        "min_dqda": min_dqda,
                        "timeout_seconds": task_timeout,
                    }
                    _SOURCE_TIMEOUTS.append(timeout_record)
                    logging.getLogger("openhens").warning(
                        "[Timeout] skipped OpenHENS task %s after %.1fs",
                        timeout_record["problem"],
                        task_timeout,
                    )
                else:
                    process.join()
                    solved = _queue_result(item["queue"], problem)
                    if solved is not _NO_RESULT and solved:
                        solved_cases.extend(solved)
                item["queue"].close()
                active.remove(item)
                launch_next()
            time.sleep(0.2)
        return solved_cases

    return run_bounded_solutions


def _run_source_problem_child(
    run_single_solution,
    problem,
    print_output: bool,
    evolution: bool,
    queue,
) -> None:
    try:
        queue.put(("ok", run_single_solution(problem, print_output, evolution)))
    except BaseException as exc:
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _queue_result(queue, problem) -> Any:
    try:
        status, result = queue.get_nowait()
    except queue_module.Empty:
        return _NO_RESULT
    if status == "ok":
        return result
    error_message = result
    logging.getLogger("openhens").warning(
        "[Failed] source task %s raised %s",
        getattr(problem, "name", repr(problem)),
        error_message,
    )
    return None


def _source_problem_grid_values(problem: Any) -> tuple[Any, Any]:
    framework = getattr(problem, "framework", None)
    if framework == "ESM":
        tdm = getattr(problem, "parent", None)
        pdm = getattr(tdm, "parent", None)
        return (
            getattr(pdm, "dTmin", getattr(problem, "dTmin", None)),
            getattr(tdm, "min_dqda", getattr(problem, "min_dqda", None)),
        )
    if framework == "TDM":
        pdm = getattr(problem, "parent", None)
        return (
            getattr(pdm, "dTmin", getattr(problem, "dTmin", None)),
            getattr(problem, "min_dqda", None),
        )
    return (
        getattr(problem, "dTmin", None),
        getattr(problem, "min_dqda", None),
    )


def _run_source_openhens(
    case_id: str,
    *,
    openhens_root: Path,
    d_tmin_grid: tuple[float, ...],
    dqda_grid: tuple[float, ...],
    top_n: int,
    max_parallel: int,
) -> tuple[list[RankedNetwork], dict[str, int]]:
    from openhens import OpenHENS

    with tempfile.TemporaryDirectory(prefix=f"openhens-{case_id}-") as tmpdir:
        model = OpenHENS(
            input_folder=str(openhens_root / "examples" / "cases" / f"{case_id}.csv"),
            output_folder=tmpdir,
            min_dT_list=list(d_tmin_grid),
            min_dqda_list=list(dqda_grid),
            stage_selection="automated",
            tolerance=1e-3,
            max_parallel=max_parallel,
            best_solns_to_save=top_n,
            log_level=logging.WARNING,
        )
        model.solve()
        source_rows = _source_ranked_networks(
            case_id,
            model.solutions,
            top_n=top_n,
        )
        return source_rows, {
            "source_solution_count": len(model.solutions),
            "source_esm_solution_count": sum(
                1
                for item in model.solutions
                if getattr(item, "framework", None) == "ESM"
            ),
        }


def _source_ranked_networks(
    case_id: str,
    solutions: list[Any],
    *,
    top_n: int,
) -> list[RankedNetwork]:
    rows: list[RankedNetwork] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for problem in sorted(
        (item for item in solutions if getattr(item, "framework", None) == "ESM"),
        key=lambda item: item.case.TAC,
    ):
        signature = _source_signature(problem.case)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        rows.append(
            RankedNetwork(
                engine="OpenHENS",
                case_id=case_id,
                rank=len(rows) + 1,
                tac=float(problem.case.TAC),
                d_tmin=float(problem.parent.parent.dTmin),
                min_dq=float(problem.parent.min_dqda),
                stages=_optional_int(getattr(problem.case, "stages", None)),
                recovery_units=_optional_int(
                    getattr(problem.case, "n_recovery_units", None)
                ),
                hot_utility_units=_optional_int(
                    getattr(problem.case, "n_hu_units", None)
                ),
                cold_utility_units=_optional_int(
                    getattr(problem.case, "n_cu_units", None)
                ),
                signature=signature,
            )
        )
        if len(rows) >= top_n:
            break
    return rows


def _source_signature(case: Any) -> tuple[str, ...]:
    links: list[str] = []
    for i in range(case.I):
        for j in range(case.J):
            for k in range(case.S):
                if _nested_float(case.Q_r, i, j, k) > DUTY_TOLERANCE_KW:
                    links.append(f"R:H{i + 1}->C{j + 1}@S{k + 1}")
    for j in range(case.J):
        if _nested_float(case.Q_h, j) > DUTY_TOLERANCE_KW:
            links.append(f"HU->C{j + 1}")
    for i in range(case.I):
        if _nested_float(case.Q_c, i) > DUTY_TOLERANCE_KW:
            links.append(f"H{i + 1}->CU")
    return tuple(sorted(links))


def _run_openpinch(
    case_id: str,
    *,
    repo_root: Path,
    d_tmin_grid: tuple[float, ...],
    dqda_grid: tuple[float, ...],
    top_n: int,
    max_parallel: int,
    task_timeout: float,
) -> list[RankedNetwork]:
    if task_timeout > 0:
        return _run_openpinch_bounded(
            case_id,
            repo_root=repo_root,
            d_tmin_grid=d_tmin_grid,
            dqda_grid=dqda_grid,
            top_n=top_n,
            max_parallel=max_parallel,
            task_timeout=task_timeout,
        )
    return _run_openpinch_grid(
        case_id,
        repo_root=repo_root,
        d_tmin_grid=d_tmin_grid,
        dqda_grid=dqda_grid,
        top_n=top_n,
        max_parallel=max_parallel,
    )


def _run_openpinch_grid(
    case_id: str,
    *,
    repo_root: Path,
    d_tmin_grid: tuple[float, ...],
    dqda_grid: tuple[float, ...],
    top_n: int,
    max_parallel: int,
) -> list[RankedNetwork]:
    fixture_path = repo_root / "tests" / "fixtures" / "openhens" / f"{case_id}.json"
    problem = PinchProblem(source=fixture_path)
    problem.target.all_heat_integration()
    settings = replace(
        workflow_settings_from_problem(problem),
        approach_temperatures=d_tmin_grid,
        derivative_thresholds=dqda_grid,
        best_solutions_to_save=top_n,
        max_parallel=max_parallel,
    )
    workflow_result = execute_open_hens_method(
        problem,
        settings,
        executor=LocalSynthesisExecutor(print_output=False),
    )
    return _openpinch_ranked_networks(
        case_id,
        ranked_networks(workflow_result.accepted_result, top_n),
        fixture_path=fixture_path,
    )


def _run_openpinch_bounded(
    case_id: str,
    *,
    repo_root: Path,
    d_tmin_grid: tuple[float, ...],
    dqda_grid: tuple[float, ...],
    top_n: int,
    max_parallel: int,
    task_timeout: float,
) -> list[RankedNetwork]:
    pending = [
        (float(d_tmin), float(dqda)) for d_tmin in d_tmin_grid for dqda in dqda_grid
    ]
    active: list[dict[str, Any]] = []
    rows: list[RankedNetwork] = []

    def launch_next() -> None:
        if not pending:
            return
        d_tmin, dqda = pending.pop(0)
        queue = multiprocessing.Queue(maxsize=1)
        process = multiprocessing.Process(
            target=_run_openpinch_grid_child,
            args=(case_id, repo_root, d_tmin, dqda, top_n, queue),
        )
        process.start()
        active.append(
            {
                "process": process,
                "queue": queue,
                "start": time.monotonic(),
                "d_tmin": d_tmin,
                "dqda": dqda,
            }
        )

    for _ in range(max(1, int(max_parallel))):
        launch_next()

    try:
        while active:
            for item in list(active):
                process = item["process"]
                elapsed = time.monotonic() - item["start"]
                result = _openpinch_queue_result(
                    item["queue"],
                    d_tmin=item["d_tmin"],
                    dqda=item["dqda"],
                )
                if result is not _NO_RESULT:
                    if result:
                        rows.extend(result)
                    process.join(timeout=5)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)
                    item["queue"].close()
                    active.remove(item)
                    launch_next()
                    continue
                if process.is_alive() and elapsed <= task_timeout:
                    continue
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
                    logging.getLogger("openhens").warning(
                        "[Timeout] skipped OpenPinch task dTmin=%s min_dQ/dA=%s "
                        "after %.1fs",
                        _format_number(item["d_tmin"], digits=3),
                        _format_number(item["dqda"], digits=3),
                        task_timeout,
                    )
                else:
                    process.join()
                    result = _openpinch_queue_result(
                        item["queue"],
                        d_tmin=item["d_tmin"],
                        dqda=item["dqda"],
                    )
                    if result is not _NO_RESULT and result:
                        rows.extend(result)
                item["queue"].close()
                active.remove(item)
                launch_next()
            time.sleep(0.2)
    finally:
        for item in list(active):
            process = item["process"]
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            item["queue"].close()
            active.remove(item)
    return _merge_ranked_networks(rows, top_n=top_n)


def _run_openpinch_grid_child(
    case_id: str,
    repo_root: Path,
    d_tmin: float,
    dqda: float,
    top_n: int,
    queue,
) -> None:
    try:
        rows = _run_openpinch_grid(
            case_id,
            repo_root=repo_root,
            d_tmin_grid=(d_tmin,),
            dqda_grid=(dqda,),
            top_n=top_n,
            max_parallel=1,
        )
        queue.put(("ok", rows))
    except BaseException as exc:
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _openpinch_queue_result(
    queue,
    *,
    d_tmin: float,
    dqda: float,
) -> Any:
    try:
        status, result = queue.get_nowait()
    except queue_module.Empty:
        return _NO_RESULT
    if status == "ok":
        return result
    error_message = result
    logging.getLogger("openhens").warning(
        "[Failed] OpenPinch task dTmin=%s min_dQ/dA=%s raised %s",
        _format_number(d_tmin, digits=3),
        _format_number(dqda, digits=3),
        error_message,
    )
    return None


def _openpinch_ranked_networks(
    case_id: str,
    outcomes,
    *,
    fixture_path: Path,
) -> list[RankedNetwork]:
    stream_maps = _stream_index_maps(fixture_path)
    rows: list[RankedNetwork] = []
    for outcome in outcomes:
        network = outcome.network
        if network is None:
            continue
        rows.append(
            RankedNetwork(
                engine="OpenPinch",
                case_id=case_id,
                rank=len(rows) + 1,
                tac=float(outcome.objective_value),
                d_tmin=_task_parameter(outcome.task, "approach_temperature"),
                min_dq=_task_parameter(outcome.task, "derivative_threshold"),
                stages=network.stage_count,
                recovery_units=_optional_int(
                    network.summary_metrics.get("recovery_units")
                ),
                hot_utility_units=_optional_int(
                    network.summary_metrics.get("hot_utility_units")
                ),
                cold_utility_units=_optional_int(
                    network.summary_metrics.get("cold_utility_units")
                ),
                signature=_openpinch_signature(network, stream_maps),
            )
        )
    return rows


def _merge_ranked_networks(
    rows: list[RankedNetwork],
    *,
    top_n: int,
) -> list[RankedNetwork]:
    by_signature: dict[tuple[str, ...], RankedNetwork] = {}
    for row in rows:
        current = by_signature.get(row.signature)
        if current is None or row.tac < current.tac:
            by_signature[row.signature] = row
    ranked = sorted(
        by_signature.values(),
        key=lambda row: (row.tac, row.signature_text),
    )[:top_n]
    return [replace(row, rank=rank) for rank, row in enumerate(ranked, start=1)]


def _summary_with_openpinch_count(
    summary: CaseRunSummary,
    *,
    case_id: str,
    openpinch_unique_count: int,
) -> CaseRunSummary:
    if summary.case_id != case_id:
        return summary
    return replace(summary, openpinch_unique_count=openpinch_unique_count)


def _merge_grids(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(sorted({*left, *right}))


def _add_optional_ints(
    left: int | None,
    right: int | None,
) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _openpinch_signature(network: Any, stream_maps: dict[str, dict[str, int]]):
    links: list[str] = []
    hot_streams = stream_maps["hot"]
    cold_streams = stream_maps["cold"]
    for exchanger in network.exchangers:
        if (
            not exchanger.active
            or not exchanger.match_allowed
            or exchanger.duty <= DUTY_TOLERANCE_KW
        ):
            continue
        if exchanger.kind.value == "recovery":
            links.append(
                "R:"
                f"H{hot_streams[exchanger.source_stream]}"
                f"->C{cold_streams[exchanger.sink_stream]}"
                f"@S{exchanger.stage}"
            )
        elif exchanger.kind.value == "hot_utility":
            links.append(f"HU->C{cold_streams[exchanger.sink_stream]}")
        elif exchanger.kind.value == "cold_utility":
            links.append(f"H{hot_streams[exchanger.source_stream]}->CU")
    return tuple(sorted(links))


def _stream_index_maps(fixture_path: Path) -> dict[str, dict[str, int]]:
    case_input = json.loads(fixture_path.read_text())
    hot: dict[str, int] = {}
    cold: dict[str, int] = {}
    for stream in case_input["streams"]:
        name = stream["name"]
        zone_name = str(stream.get("zone", "")).split("/")[-1]
        supply = float(stream["t_supply"]["value"])
        target = float(stream["t_target"]["value"])
        if supply > target:
            index = max(hot.values(), default=0) + 1
            _add_stream_aliases(hot, name, zone_name, index)
        else:
            index = max(cold.values(), default=0) + 1
            _add_stream_aliases(cold, name, zone_name, index)
    return {"hot": hot, "cold": cold}


def _add_stream_aliases(
    mapping: dict[str, int],
    name: str,
    zone_name: str,
    index: int,
) -> None:
    names = {name, name.strip()}
    for alias in names:
        if not alias:
            continue
        mapping[alias] = index
        if zone_name:
            mapping[f"{zone_name}.{alias}"] = index


def _task_parameter(task: Any, key: str) -> float | None:
    value = getattr(task, key, None)
    return None if value is None else float(value)


def _compare_rank_rows(
    case_id: str,
    source_rows: list[RankedNetwork],
    openpinch_rows: list[RankedNetwork],
) -> list[dict[str, Any]]:
    by_source_signature = {row.signature: row for row in source_rows}
    by_openpinch_signature = {row.signature: row for row in openpinch_rows}
    rows: list[dict[str, Any]] = []
    for rank in range(1, max(len(source_rows), len(openpinch_rows)) + 1):
        source = _row_at_rank(source_rows, rank)
        openpinch = _row_at_rank(openpinch_rows, rank)
        source_match = (
            by_openpinch_signature.get(source.signature) if source is not None else None
        )
        openpinch_match = (
            by_source_signature.get(openpinch.signature)
            if openpinch is not None
            else None
        )
        rows.append(
            {
                "case_id": case_id,
                "rank": rank,
                "openhens_tac": _network_value(source, "tac"),
                "openpinch_tac": _network_value(openpinch, "tac"),
                "openhens_d_tmin": _network_value(source, "d_tmin"),
                "openpinch_d_tmin": _network_value(openpinch, "d_tmin"),
                "openhens_min_dq": _network_value(source, "min_dq"),
                "openpinch_min_dq": _network_value(openpinch, "min_dq"),
                "rank_tac_delta": _tac_delta(source, openpinch),
                "rank_tac_delta_percent": _tac_delta_percent(source, openpinch),
                "rank_signature_match": (
                    source is not None
                    and openpinch is not None
                    and source.signature == openpinch.signature
                ),
                "openhens_signature_hash": _network_value(source, "signature_hash"),
                "openpinch_signature_hash": _network_value(openpinch, "signature_hash"),
                "openhens_signature": _network_value(source, "signature_text"),
                "openpinch_signature": _network_value(openpinch, "signature_text"),
                "openhens_rank_for_openpinch_signature": (
                    openpinch_match.rank if openpinch_match is not None else None
                ),
                "openpinch_rank_for_openhens_signature": (
                    source_match.rank if source_match is not None else None
                ),
            }
        )
    return rows


def _row_at_rank(rows: list[RankedNetwork], rank: int) -> RankedNetwork | None:
    return rows[rank - 1] if rank <= len(rows) else None


def _network_value(row: RankedNetwork | None, attr: str) -> Any:
    if row is None:
        return None
    return getattr(row, attr)


def _tac_delta(
    source: RankedNetwork | None,
    openpinch: RankedNetwork | None,
) -> float | None:
    if source is None or openpinch is None:
        return None
    return openpinch.tac - source.tac


def _tac_delta_percent(
    source: RankedNetwork | None,
    openpinch: RankedNetwork | None,
) -> float | None:
    delta = _tac_delta(source, openpinch)
    if delta is None or source.tac == 0:
        return None
    return 100.0 * delta / source.tac


def _write_network_rows(path: Path, rows: list[RankedNetwork]) -> None:
    fieldnames = (
        "case_id",
        "engine",
        "rank",
        "tac",
        "d_tmin",
        "min_dq",
        "stages",
        "recovery_units",
        "hot_utility_units",
        "cold_utility_units",
        "signature_hash",
        "signature",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row.case_id,
                    "engine": row.engine,
                    "rank": row.rank,
                    "tac": row.tac,
                    "d_tmin": row.d_tmin,
                    "min_dq": row.min_dq,
                    "stages": row.stages,
                    "recovery_units": row.recovery_units,
                    "hot_utility_units": row.hot_utility_units,
                    "cold_utility_units": row.cold_utility_units,
                    "signature_hash": row.signature_hash,
                    "signature": row.signature_text,
                }
            )


def _read_network_rows(path: Path) -> list[RankedNetwork]:
    if not path.exists():
        return []
    rows: list[RankedNetwork] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                RankedNetwork(
                    engine=row["engine"],
                    case_id=row["case_id"],
                    rank=int(row["rank"]),
                    tac=float(row["tac"]),
                    d_tmin=_optional_float(row["d_tmin"]),
                    min_dq=_optional_float(row["min_dq"]),
                    stages=_optional_int_or_none(row["stages"]),
                    recovery_units=_optional_int_or_none(row["recovery_units"]),
                    hot_utility_units=_optional_int_or_none(row["hot_utility_units"]),
                    cold_utility_units=_optional_int_or_none(row["cold_utility_units"]),
                    signature=tuple(
                        item for item in row["signature"].split(";") if item
                    ),
                )
            )
    return rows


def _write_rank_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = (
        "case_id",
        "rank",
        "openhens_tac",
        "openpinch_tac",
        "openhens_d_tmin",
        "openpinch_d_tmin",
        "openhens_min_dq",
        "openpinch_min_dq",
        "rank_tac_delta",
        "rank_tac_delta_percent",
        "rank_signature_match",
        "openhens_signature_hash",
        "openpinch_signature_hash",
        "openhens_rank_for_openpinch_signature",
        "openpinch_rank_for_openhens_signature",
        "openhens_signature",
        "openpinch_signature",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_rank_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    numeric_fields = {
        "openhens_tac",
        "openpinch_tac",
        "openhens_d_tmin",
        "openpinch_d_tmin",
        "openhens_min_dq",
        "openpinch_min_dq",
        "rank_tac_delta",
        "rank_tac_delta_percent",
    }
    int_fields = {
        "rank",
        "openhens_rank_for_openpinch_signature",
        "openpinch_rank_for_openhens_signature",
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, Any] = dict(row)
            for field in numeric_fields:
                converted[field] = _optional_float(converted.get(field))
            for field in int_fields:
                converted[field] = _optional_int_or_none(converted.get(field))
            converted["rank_signature_match"] = (
                converted.get("rank_signature_match") == "True"
            )
            rows.append(converted)
    return rows


def _enrich_rank_rows_with_networks(
    rows_by_rank: list[dict[str, Any]],
    network_rows: list[RankedNetwork],
) -> list[dict[str, Any]]:
    by_engine_rank = {(row.case_id, row.engine, row.rank): row for row in network_rows}
    enriched_rows: list[dict[str, Any]] = []
    for row in rows_by_rank:
        enriched = dict(row)
        case_id = str(enriched["case_id"])
        rank = int(enriched["rank"])
        for prefix, engine in (
            ("openhens", "OpenHENS"),
            ("openpinch", "OpenPinch"),
        ):
            network = by_engine_rank.get((case_id, engine, rank))
            if network is None:
                continue
            enriched.setdefault(f"{prefix}_d_tmin", network.d_tmin)
            enriched.setdefault(f"{prefix}_min_dq", network.min_dq)
            if enriched[f"{prefix}_d_tmin"] is None:
                enriched[f"{prefix}_d_tmin"] = network.d_tmin
            if enriched[f"{prefix}_min_dq"] is None:
                enriched[f"{prefix}_min_dq"] = network.min_dq
        enriched_rows.append(enriched)
    return enriched_rows


def _write_timeout_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_TIMEOUT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in _TIMEOUT_FIELDNAMES})


def _read_timeout_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, Any] = dict(row)
            for field in ("dTmin", "min_dqda", "timeout_seconds"):
                converted[field] = _optional_float(converted.get(field))
            if (
                converted.get("framework") in {"TDM", "ESM"}
                and converted.get("dTmin") is not None
                and float(converted["dTmin"]) <= 0.1000001
            ):
                converted["dTmin"] = None
            rows.append(converted)
    return rows


def _write_run_summaries(path: Path, rows: list[CaseRunSummary]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_RUN_SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row.case_id,
                    "d_tmin_grid": _format_grid(row.d_tmin_grid),
                    "dqda_grid": _format_grid(row.dqda_grid),
                    "is_full_openhens_grid": row.is_full_openhens_grid,
                    "top_n": row.top_n,
                    "source_solution_count": row.source_solution_count,
                    "source_esm_solution_count": row.source_esm_solution_count,
                    "source_unique_count": row.source_unique_count,
                    "openpinch_unique_count": row.openpinch_unique_count,
                    "source_timeout_count": row.source_timeout_count,
                }
            )


def _read_run_summaries(path: Path) -> list[CaseRunSummary]:
    if not path.exists():
        return []
    rows: list[CaseRunSummary] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                CaseRunSummary(
                    case_id=row["case_id"],
                    d_tmin_grid=_parse_grid(row.get("d_tmin_grid", "")),
                    dqda_grid=_parse_grid(row.get("dqda_grid", "")),
                    is_full_openhens_grid=(row.get("is_full_openhens_grid") == "True"),
                    top_n=int(row["top_n"]),
                    source_solution_count=_optional_int_or_none(
                        row.get("source_solution_count")
                    ),
                    source_esm_solution_count=_optional_int_or_none(
                        row.get("source_esm_solution_count")
                    ),
                    source_unique_count=int(row["source_unique_count"]),
                    openpinch_unique_count=int(row["openpinch_unique_count"]),
                    source_timeout_count=int(row["source_timeout_count"]),
                )
            )
    return rows


def _rank_row_sort_key(row: dict[str, Any]) -> tuple[int, int]:
    case_id = str(row.get("case_id", ""))
    return (CASE_ORDER.get(case_id, len(CASE_ORDER)), int(row.get("rank") or 0))


def _network_row_sort_key(row: RankedNetwork) -> tuple[int, int, int]:
    engine_order = {"OpenHENS": 0, "OpenPinch": 1}
    return (
        CASE_ORDER.get(row.case_id, len(CASE_ORDER)),
        engine_order.get(row.engine, len(engine_order)),
        row.rank,
    )


def _timeout_row_sort_key(row: dict[str, Any]) -> tuple[int, float, float]:
    case_id = str(row.get("case_id", ""))
    return (
        CASE_ORDER.get(case_id, len(CASE_ORDER)),
        float(row.get("dTmin") or 0.0),
        float(row.get("min_dqda") or 0.0),
    )


def _run_summary_sort_key(row: CaseRunSummary) -> int:
    return CASE_ORDER.get(row.case_id, len(CASE_ORDER))


def _write_markdown_report(
    path: Path,
    *,
    rows_by_rank: list[dict[str, Any]],
    network_rows: list[RankedNetwork],
    timeout_rows: list[dict[str, Any]],
    run_summaries: list[CaseRunSummary],
    d_tmin_grid: tuple[float, ...],
    dqda_grid: tuple[float, ...],
    top_n: int,
) -> None:
    lines = [
        "# OpenHENS/OpenPinch Top-5 HEN Comparison",
        "",
        "- dTmin and min dQ/dA are listed per ranked network below.",
        "- Appended reports may include cases run with different comparison grids.",
        f"- structural signature duty threshold: {DUTY_TOLERANCE_KW:g} kW",
        (
            "- Cases with fewer than five comparable ranks are complete only when "
            "the coverage summary shows a full-grid run exhausted the available "
            "unique source networks."
        ),
        "",
        "## Coverage Summary",
        "",
        (
            "| Case | Grid | Source ESM | OpenHENS Unique | OpenPinch Unique | "
            "Exact Paired Ranks | Source Timeouts | Coverage |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in _coverage_rows(
        rows_by_rank=rows_by_rank,
        network_rows=network_rows,
        timeout_rows=timeout_rows,
        run_summaries=run_summaries,
        top_n=top_n,
    ):
        lines.append(
            "| {case_id} | {grid} | {source_esm_solutions} | "
            "{openhens_unique} | {openpinch_unique} | {exact_paired_ranks} | "
            "{source_timeouts} | {coverage} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Source Task Timeouts",
            "",
        ]
    )
    if timeout_rows:
        lines.extend(
            [
                "| Case | Framework | dTmin | min dQ/dA | Timeout s | Problem |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for row in timeout_rows:
            lines.append(
                "| {case_id} | {framework} | {dTmin} | {min_dqda} | "
                "{timeout_seconds} | `{problem}` |".format(
                    case_id=row.get("case_id") or "",
                    framework=row.get("framework") or "",
                    dTmin=_format_number(row.get("dTmin"), digits=3),
                    min_dqda=_format_number(row.get("min_dqda"), digits=3),
                    timeout_seconds=_format_number(
                        row.get("timeout_seconds"),
                        digits=1,
                    ),
                    problem=row.get("problem") or "",
                )
            )
    else:
        lines.append("No source task timeouts are recorded in this artifact.")
    lines.extend(
        [
            "",
            "## Rank Comparison",
            "",
            (
                "| Case | Rank | OpenHENS TAC | OpenPinch TAC | "
                "OpenHENS dTmin/min dQ | OpenPinch dTmin/min dQ | "
                "Delta | Delta % | Same Signature |"
            ),
            "|---|---:|---:|---:|---|---|---:|---:|---|",
        ]
    )
    for row in rows_by_rank:
        lines.append(
            "| {case_id} | {rank} | {openhens_tac} | {openpinch_tac} | "
            "{openhens_grid} | {openpinch_grid} | "
            "{rank_tac_delta} | {rank_tac_delta_percent} | "
            "{rank_signature_match} |".format(
                case_id=row["case_id"],
                rank=row["rank"],
                openhens_tac=_format_number(row["openhens_tac"]),
                openpinch_tac=_format_number(row["openpinch_tac"]),
                openhens_grid=_grid_pair(
                    row.get("openhens_d_tmin"),
                    row.get("openhens_min_dq"),
                ),
                openpinch_grid=_grid_pair(
                    row.get("openpinch_d_tmin"),
                    row.get("openpinch_min_dq"),
                ),
                rank_tac_delta=_format_number(row["rank_tac_delta"]),
                rank_tac_delta_percent=_format_number(
                    row["rank_tac_delta_percent"],
                    digits=6,
                ),
                rank_signature_match=row["rank_signature_match"],
            )
        )
    lines.extend(
        [
            "",
            "## Structural Signatures",
            "",
            (
                "| Case | Engine | Rank | TAC | dTmin | min dQ/dA | Stages | "
                "R/HU/CU Units | Signature Hash | Signature |"
            ),
            "|---|---|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in network_rows:
        unit_text = (
            f"{row.recovery_units}/{row.hot_utility_units}/{row.cold_utility_units}"
        )
        lines.append(
            f"| {row.case_id} | {row.engine} | {row.rank} | "
            f"{_format_number(row.tac)} | "
            f"{_format_number(row.d_tmin, digits=3)} | "
            f"{_format_number(row.min_dq, digits=3)} | "
            f"{row.stages} | {unit_text} | `{row.signature_hash}` | "
            f"`{row.signature_text}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _coverage_rows(
    *,
    rows_by_rank: list[dict[str, Any]],
    network_rows: list[RankedNetwork],
    timeout_rows: list[dict[str, Any]],
    run_summaries: list[CaseRunSummary],
    top_n: int,
) -> list[dict[str, Any]]:
    summaries_by_case = {row.case_id: row for row in run_summaries}
    discovered_case_ids = {
        *CASE_IDS,
        *(row.case_id for row in network_rows),
        *(str(row.get("case_id")) for row in rows_by_rank if row.get("case_id")),
        *(str(row.get("case_id")) for row in timeout_rows if row.get("case_id")),
        *(row.case_id for row in run_summaries),
    }
    case_ids = [
        *CASE_IDS,
        *sorted(discovered_case_ids.difference(CASE_IDS)),
    ]
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        openhens_unique = sum(
            1
            for row in network_rows
            if row.case_id == case_id and row.engine == "OpenHENS"
        )
        openpinch_unique = sum(
            1
            for row in network_rows
            if row.case_id == case_id and row.engine == "OpenPinch"
        )
        exact_paired_ranks = sum(
            1
            for row in rows_by_rank
            if row.get("case_id") == case_id
            and row.get("rank_signature_match") is True
            and _is_zero(row.get("rank_tac_delta"), tolerance=TAC_COMPARE_TOLERANCE)
        )
        source_timeouts = sum(
            1 for row in timeout_rows if row.get("case_id") == case_id
        )
        summary = summaries_by_case.get(case_id)
        if summary is not None:
            source_timeouts = summary.source_timeout_count
            openhens_unique = summary.source_unique_count
            openpinch_unique = summary.openpinch_unique_count
        source_esm_solutions = (
            ""
            if summary is None or summary.source_esm_solution_count is None
            else summary.source_esm_solution_count
        )
        grid = _coverage_grid_text(summary)
        if (
            openhens_unique >= top_n
            and openpinch_unique >= top_n
            and exact_paired_ranks >= top_n
        ):
            coverage = "complete top-5 exact"
        elif (
            summary is not None
            and summary.is_full_openhens_grid
            and summary.source_timeout_count == 0
            and exact_paired_ranks == openhens_unique == openpinch_unique
            and exact_paired_ranks > 0
        ):
            coverage = "complete available unique exact"
        elif exact_paired_ranks and exact_paired_ranks == min(
            openhens_unique,
            openpinch_unique,
        ):
            coverage = "partial exact"
        elif openhens_unique or openpinch_unique:
            coverage = "mismatch or incomplete"
        else:
            coverage = "not run"
        rows.append(
            {
                "case_id": case_id,
                "grid": grid,
                "source_esm_solutions": source_esm_solutions,
                "openhens_unique": openhens_unique,
                "openpinch_unique": openpinch_unique,
                "exact_paired_ranks": exact_paired_ranks,
                "source_timeouts": source_timeouts,
                "coverage": coverage,
            }
        )
    return rows


def _coverage_grid_text(summary: CaseRunSummary | None) -> str:
    if summary is None:
        return "not recorded"
    if summary.is_full_openhens_grid:
        return "full default"
    return (
        f"appended subset: dTmin={_format_grid(summary.d_tmin_grid)}; "
        f"min dQ/dA={_format_grid(summary.dqda_grid)}"
    )


def _grid_pair(d_tmin: Any, min_dq: Any) -> str:
    if d_tmin is None and min_dq is None:
        return ""
    return f"{_format_number(d_tmin, digits=3)} / {_format_number(min_dq, digits=3)}"


def _is_zero(value: Any, *, tolerance: float) -> bool:
    if value is None:
        return False
    return abs(float(value)) <= tolerance


def _format_grid(values: tuple[float, ...]) -> str:
    return ", ".join(f"{value:g}" for value in values)


def _parse_grid(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _format_number(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _nested_float(values: Any, *indices: int) -> float:
    value = values
    for index in indices:
        value = value[index]
    return _solver_float(value)


def _solver_float(value: Any) -> float:
    if isinstance(value, list) and len(value) == 1:
        return _solver_float(value[0])
    if hasattr(value, "VALUE"):
        return _solver_float(value.VALUE)
    if hasattr(value, "value"):
        return _solver_float(value.value)
    try:
        return float(value)
    except TypeError:
        return _solver_float(value[0])


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_int_or_none(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


if __name__ == "__main__":
    main()
