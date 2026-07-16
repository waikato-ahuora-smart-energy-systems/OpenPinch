"""Solved-variant views and comparison shaping."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from ....lib.schemas.workspace import (
    ProblemTableDiffView,
    ProblemTableView,
    ScenarioVariantView,
    SummaryCard,
    TableView,
    ValidationReport,
    VariantMetricDelta,
)
from ...pinch_problem import PinchProblem
from .common import (
    _unit_column,
    dataframe_to_table_view,
    json_safe,
    maybe_string,
    numeric_delta,
)
from .graphs import graph_catalog_entries, graph_data_entries
from .problem_tables import problem_table_views

_SUMMARY_METRIC_COLUMNS = [
    "Hot Utility Target",
    "Cold Utility Target",
    "Heat Recovery",
    "Hot Pinch",
    "Cold Pinch",
]


def invalid_variant_view(
    *,
    variant_name: str,
    workflow: str,
    workflow_options: dict[str, Any],
    validation: ValidationReport,
    support_level: str,
    warnings_list: list[str],
) -> ScenarioVariantView:
    """Build a deterministic invalid-variant response view."""
    return ScenarioVariantView(
        variant_name=variant_name,
        period_id=workflow_options.get("period_id"),
        workflow=workflow,
        workflow_options=workflow_options,
        status="invalid",
        support_level=support_level,
        validation=validation,
        warnings=warnings_list,
    )


def error_variant_view(
    *,
    variant_name: str,
    workflow: str,
    workflow_options: dict[str, Any],
    validation: ValidationReport,
    support_level: str,
    warnings_list: list[str],
    error_message: str,
    error_category: str,
) -> ScenarioVariantView:
    """Build a deterministic workflow-error response view."""
    return ScenarioVariantView(
        variant_name=variant_name,
        period_id=workflow_options.get("period_id"),
        workflow=workflow,
        workflow_options=workflow_options,
        status="error",
        support_level=support_level,
        validation=validation,
        warnings=warnings_list,
        error_message=error_message,
        error_category=error_category,
    )


def problem_to_variant_view(
    problem: PinchProblem,
    *,
    variant_name: str,
    workflow: str,
    workflow_options: dict[str, Any],
    validation: ValidationReport,
    support_level: str,
    warnings_list: list[str],
) -> ScenarioVariantView:
    """Convert one solved problem into the workspace frontend view model."""
    summary_frame = problem.summary_frame()
    graph_data = problem.plot.get_graph_data()
    graph_catalog = graph_catalog_entries(graph_data)
    graph_entries = graph_data_entries(graph_data)
    problem_tables = problem_table_views(problem)

    return ScenarioVariantView(
        variant_name=variant_name,
        period_id=workflow_options.get("period_id")
        or getattr(problem._results, "period_id", None),
        workflow=workflow,
        workflow_options=workflow_options,
        status="solved",
        support_level=support_level,
        validation=validation,
        warnings=warnings_list,
        summary_cards=summary_cards(summary_frame),
        summary_table=dataframe_to_table_view(summary_frame),
        graph_catalog=graph_catalog,
        graph_data_entries=graph_entries,
        problem_tables=problem_tables,
    )


def summary_cards(frame: pd.DataFrame) -> list[SummaryCard]:
    """Build summary-card rows from a compact summary dataframe."""
    cards = []
    for _, row in frame.iterrows():
        target_name = str(row.get("Target"))
        target_id = target_name
        for metric in _SUMMARY_METRIC_COLUMNS:
            cards.append(
                SummaryCard(
                    card_id=f"{target_id}::{metric}",
                    target_id=target_id,
                    target_name=target_name,
                    label=metric,
                    value=json_safe(row.get(metric)),
                    unit=maybe_string(row.get(_unit_column(metric))),
                )
            )
    return cards


def summary_metric_deltas(
    base_name: str,
    base_view: ScenarioVariantView,
    variant_name: str,
    variant_view: ScenarioVariantView,
) -> list[VariantMetricDelta]:
    """Build deterministic summary metric deltas between two solved variants."""
    base_rows = summary_rows_by_target(base_view.summary_table)
    variant_rows = summary_rows_by_target(variant_view.summary_table)
    target_ids = sorted(set(base_rows) | set(variant_rows))
    deltas = []
    for target_id in target_ids:
        base_row = base_rows.get(target_id, {})
        variant_row = variant_rows.get(target_id, {})
        target_name = str(
            base_row.get("Target") or variant_row.get("Target") or target_id
        )
        for metric in _SUMMARY_METRIC_COLUMNS:
            base_value = json_safe(base_row.get(metric))
            variant_value = json_safe(variant_row.get(metric))
            unit = maybe_string(
                base_row.get(_unit_column(metric))
                or variant_row.get(_unit_column(metric))
            )
            units_match = base_row.get(_unit_column(metric)) == variant_row.get(
                _unit_column(metric)
            )
            deltas.append(
                VariantMetricDelta(
                    base_variant=base_name,
                    variant_name=variant_name,
                    target_id=target_id,
                    target_name=target_name,
                    metric=metric,
                    base_value=base_value,
                    variant_value=variant_value,
                    unit=unit,
                    delta=(
                        numeric_delta(base_value, variant_value)
                        if units_match
                        else None
                    ),
                )
            )
    return deltas


def summary_rows_by_target(table: Optional[TableView]) -> dict[str, dict[str, Any]]:
    """Index summary rows by their target label."""
    if table is None:
        return {}
    return {
        str(row.get("Target")): row
        for row in table.rows
        if row.get("Target") not in (None, "")
    }


def problem_table_diffs(
    base_name: str,
    base_view: ScenarioVariantView,
    variant_name: str,
    variant_view: ScenarioVariantView,
) -> list[ProblemTableDiffView]:
    """Build structural and cell-level diffs between problem-table views."""
    base_tables = {table.table_id: table for table in base_view.problem_tables}
    variant_tables = {table.table_id: table for table in variant_view.problem_tables}
    diffs = []
    for table_id in sorted(set(base_tables) | set(variant_tables)):
        base_table = base_tables.get(table_id)
        variant_table = variant_tables.get(table_id)
        target_name = (
            base_table.target_name
            if base_table is not None
            else variant_table.target_name
        )
        table_kind = (
            base_table.table_kind
            if base_table is not None
            else variant_table.table_kind
        )
        base_rows = len(base_table.table.rows) if base_table is not None else 0
        variant_rows = len(variant_table.table.rows) if variant_table is not None else 0
        shared_columns = sorted(
            set(base_table.table.columns if base_table is not None else [])
            & set(variant_table.table.columns if variant_table is not None else [])
        )
        diffs.append(
            ProblemTableDiffView(
                base_variant=base_name,
                variant_name=variant_name,
                target_id=target_name,
                target_name=target_name,
                table_kind=table_kind,
                base_rows=base_rows,
                variant_rows=variant_rows,
                shared_columns=shared_columns,
                changed_cells=count_changed_cells(
                    base_table,
                    variant_table,
                    shared_columns,
                ),
                shape_changed=base_rows != variant_rows
                or (
                    (base_table.table.columns if base_table is not None else [])
                    != (
                        variant_table.table.columns if variant_table is not None else []
                    )
                ),
            )
        )
    return diffs


def count_changed_cells(
    base_table: Optional[ProblemTableView],
    variant_table: Optional[ProblemTableView],
    shared_columns: list[str],
) -> Optional[int]:
    """Count changed cells across the shared shape of two problem tables."""
    if base_table is None or variant_table is None:
        return None
    changed = 0
    row_count = min(len(base_table.table.rows), len(variant_table.table.rows))
    for index in range(row_count):
        base_row = base_table.table.rows[index]
        variant_row = variant_table.table.rows[index]
        for column in shared_columns:
            if json_safe(base_row.get(column)) != json_safe(variant_row.get(column)):
                changed += 1
    return changed
