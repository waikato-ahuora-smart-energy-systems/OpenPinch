"""Optional HEN synthesis exports generated from ``problem.results``."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ...classes.pinch_problem import PinchProblem
from ...lib.schemas.synthesis import (
    HeatExchangerNetworkSynthesisExportRecord,
    HeatExchangerNetworkSynthesisManifest,
)


def export_heat_exchanger_network_synthesis_results(
    problem: PinchProblem,
    output_dir: str | Path,
    *,
    workspace_variant: str | None = None,
) -> HeatExchangerNetworkSynthesisManifest:
    """Write optional JSON/CSV views from one problem-owned design result."""
    if not isinstance(problem, PinchProblem):
        raise TypeError("HEN synthesis exports require a live PinchProblem.")
    if problem.results is None or problem.results.design is None:
        raise RuntimeError(
            "Run problem.design.heat_exchanger_network_synthesis(...) before export."
        )

    design = problem.results.design
    root = Path(output_dir)
    results_dir = root / "results"
    metrics_dir = root / "metrics"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    records: list[HeatExchangerNetworkSynthesisExportRecord] = []
    for outcome in design.task_outcomes:
        task_id = outcome.task.task_id or "unknown-task"
        outcome_path = results_dir / f"{task_id}.json"
        outcome_path.write_text(outcome.model_dump_json(indent=2), encoding="utf-8")
        records.append(
            _record(
                design.run_id,
                "json",
                outcome_path,
                root,
                record_id=f"task:{task_id}",
                content_type="application/json",
            )
        )

    solution_metrics_path = metrics_dir / "solution_metrics.csv"
    _write_csv(solution_metrics_path, _solution_metric_rows(design))
    records.append(
        _record(
            design.run_id,
            "csv",
            solution_metrics_path,
            root,
            record_id="metrics:solution",
            content_type="text/csv",
        )
    )

    run_summary_path = metrics_dir / "run_summary.csv"
    _write_csv(
        run_summary_path,
        [
            _run_summary_row(
                design,
                problem_id=design.problem_id or problem.project_name,
                workspace_variant=workspace_variant or design.workspace_variant,
            )
        ],
    )
    records.append(
        _record(
            design.run_id,
            "csv",
            run_summary_path,
            root,
            record_id="metrics:run-summary",
            content_type="text/csv",
        )
    )

    manifest_path = root / "manifest.json"
    records.append(
        _record(
            design.run_id,
            "json",
            manifest_path,
            root,
            record_id="manifest",
            content_type="application/json",
        )
    )
    manifest = _manifest_with_export_records(
        design,
        records=tuple(records),
        problem_id=design.problem_id or problem.project_name,
        workspace_variant=workspace_variant or design.workspace_variant,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def _manifest_with_export_records(
    design,
    *,
    records: tuple[HeatExchangerNetworkSynthesisExportRecord, ...],
    problem_id: str | None,
    workspace_variant: str | None,
) -> HeatExchangerNetworkSynthesisManifest:
    base = design.manifest
    if base is None:
        return HeatExchangerNetworkSynthesisManifest(
            run_id=design.run_id,
            approach_temperatures=(1.0,),
            derivative_thresholds=(1.0,),
            stage_selection=(design.stage_count or 1,),
            task_ids=tuple(
                outcome.task.task_id
                for outcome in design.task_outcomes
                if outcome.task.task_id is not None
            ),
            problem_id=problem_id,
            workspace_variant=workspace_variant,
            state_id=design.state_id,
            export_records=records,
        )
    return base.model_copy(
        update={
            "problem_id": problem_id,
            "workspace_variant": workspace_variant,
            "state_id": design.state_id,
            "export_records": records,
        }
    )


def _record(
    run_id: str,
    format_name: str,
    path: Path,
    root: Path,
    *,
    record_id: str,
    content_type: str,
) -> HeatExchangerNetworkSynthesisExportRecord:
    return HeatExchangerNetworkSynthesisExportRecord(
        run_id=run_id,
        format=format_name,
        path=path.relative_to(root).as_posix(),
        record_id=record_id,
        content_type=content_type,
    )


def _solution_metric_rows(design) -> list[dict[str, Any]]:
    rows = []
    for outcome in design.task_outcomes:
        rows.append(
            {
                "run_id": design.run_id,
                "problem_id": design.problem_id,
                "workspace_variant": design.workspace_variant,
                "task_id": outcome.task.task_id,
                "method": outcome.task.method,
                "status": outcome.status,
                "solver_status": outcome.solver_status,
                "objective_value": outcome.objective_value,
            }
        )
    return rows


def _run_summary_row(
    design,
    *,
    problem_id: str | None,
    workspace_variant: str | None,
) -> dict[str, Any]:
    solved = [
        outcome for outcome in design.task_outcomes if outcome.status == "success"
    ]
    return {
        "run_id": design.run_id,
        "problem_id": problem_id,
        "workspace_variant": workspace_variant,
        "accepted_task_id": design.task_id,
        "task_count": len(design.task_outcomes),
        "solved_task_count": len(solved),
        "total_annual_cost": design.objective_values.get("total_annual_cost"),
        "utility_cost": design.objective_values.get("utility_cost"),
        "capital_cost": design.objective_values.get("capital_cost"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
