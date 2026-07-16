"""View shaping helpers for :class:`OpenPinch.classes.PinchWorkspace`."""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ...lib.config_metadata import (
    CONFIG_FIELD_SPECS,
    configuration_field_support_level,
)
from ...lib.schemas.workspace import (
    ConfigurationFieldMetadata,
    GraphCatalogEntry,
    GraphDataEntry,
    InputRecordView,
    ProblemTableDiffView,
    ProblemTableView,
    ScenarioVariantView,
    SummaryCard,
    TableView,
    ValidationReport,
    VariantMetricDelta,
    ZoneNodeView,
)
from ...streamlit_webviewer.web_graphing import (
    collect_targets,
    problem_table_to_dataframe,
)
from ..pinch_problem import PinchProblem

_SUMMARY_METRIC_COLUMNS = [
    "Hot Utility Target",
    "Cold Utility Target",
    "Heat Recovery",
    "Hot Pinch",
    "Cold Pinch",
]


def _unit_column(metric: str) -> str:
    return f"{metric} (unit)"


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


def configuration_field_metadata() -> list[ConfigurationFieldMetadata]:
    """Return declarative metadata for editable configuration fields."""
    fields = []
    for name, spec in CONFIG_FIELD_SPECS.items():
        enum_cls = spec.enum_cls
        field_type, multiple = annotation_metadata(spec.annotation, enum_cls=enum_cls)
        fields.append(
            ConfigurationFieldMetadata(
                name=name,
                label=name.replace("_", " ").title(),
                field_type=field_type,
                group=spec.group,
                config_path=list(spec.config_path),
                support_level=configuration_field_support_level(name),
                runtime_status=spec.runtime_status,
                enum_choices=[str(item.value) for item in enum_cls] if enum_cls else [],
                numeric_min=spec.numeric_min,
                numeric_max=spec.numeric_max,
                multiple=multiple,
            )
        )
    return fields


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


def graph_catalog_entries(graph_data: dict[str, Any]) -> list[GraphCatalogEntry]:
    """Flatten graph metadata into a catalog view."""
    entries = []
    for graph_set_id, graph_set in graph_data.items():
        target_name = str(graph_set.get("name", graph_set_id))
        target_id = target_name
        zone_name = maybe_string(graph_set.get("zone_name"))
        zone_address = maybe_string(graph_set.get("zone_address"))
        target_type = maybe_string(graph_set.get("target_type"))
        for index, graph in enumerate(graph_set.get("graphs", [])):
            graph_type = maybe_string(graph.get("type"))
            graph_name = str(graph.get("name") or graph_type or f"Graph {index + 1}")
            entries.append(
                GraphCatalogEntry(
                    graph_id=f"{graph_set_id}::{index}::{graph_type or 'graph'}",
                    graph_set_id=str(graph_set_id),
                    target_id=target_id,
                    target_name=target_name,
                    zone_name=zone_name,
                    zone_address=zone_address,
                    target_type=target_type,
                    graph_type=graph_type,
                    graph_name=graph_name,
                    index=index,
                )
            )
    return entries


def graph_data_entries(graph_data: dict[str, Any]) -> list[GraphDataEntry]:
    """Flatten raw graph data into stable graph entry records."""
    entries = []
    for graph_set_id, graph_set in graph_data.items():
        target_name = str(graph_set.get("name", graph_set_id))
        target_id = target_name
        for index, graph in enumerate(graph_set.get("graphs", [])):
            graph_type = maybe_string(graph.get("type"))
            graph_name = str(graph.get("name") or graph_type or f"Graph {index + 1}")
            entries.append(
                GraphDataEntry(
                    graph_id=f"{graph_set_id}::{index}::{graph_type or 'graph'}",
                    graph_set_id=str(graph_set_id),
                    target_id=target_id,
                    target_name=target_name,
                    graph_type=graph_type,
                    graph_name=graph_name,
                    graph_data=json_safe(graph),
                )
            )
    return entries


def problem_table_views(problem: PinchProblem) -> list[ProblemTableView]:
    """Build serializable problem-table views for one solved problem."""
    if problem.master_zone is None:
        return []

    tables = []
    targets = collect_targets(problem.master_zone)
    for target_name in sorted(targets):
        target = targets[target_name]
        for table_kind, attr_name in (("shifted", "pt"), ("real", "pt_real")):
            frame = problem_table_to_dataframe(
                getattr(target, attr_name, None),
                round_decimals=2,
            )
            if frame.empty:
                continue
            tables.append(
                ProblemTableView(
                    table_id=f"{target_name}::{table_kind}",
                    target_id=target_name,
                    target_name=target_name,
                    table_kind=table_kind,
                    table=dataframe_to_table_view(frame),
                )
            )
    return tables


def dataframe_to_table_view(frame: pd.DataFrame) -> TableView:
    """Convert a dataframe to a JSON-safe table view."""
    safe_frame = frame.copy()
    rows = [
        {column: json_safe(value) for column, value in row.items()}
        for row in safe_frame.to_dict(orient="records")
    ]
    return TableView(columns=[str(column) for column in safe_frame.columns], rows=rows)


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


def zone_tree_view(zone_tree: Any) -> list[ZoneNodeView]:
    """Flatten a nested zone tree into frontend-friendly node records."""
    if not isinstance(zone_tree, dict):
        return []

    nodes: list[ZoneNodeView] = []

    def walk(node: dict[str, Any], parent_path: Optional[str]) -> None:
        name = str(node.get("name", "Zone"))
        path = name if parent_path is None else f"{parent_path}/{name}"
        nodes.append(
            ZoneNodeView(
                zone_id=path,
                path=path,
                name=name,
                zone_type=maybe_string(node.get("type")),
                parent_id=parent_path,
                dt_cont_multiplier=maybe_float(node.get("dt_cont_multiplier")),
            )
        )
        children = node.get("children") or []
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    walk(child, path)

    walk(zone_tree, None)
    return nodes


def record_views(records: Any, *, section: str) -> list[InputRecordView]:
    """Convert stream/utility input rows into editable frontend records."""
    if not isinstance(records, list):
        return []
    views = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        path = f"{section}[{index}]"
        views.append(
            InputRecordView(
                record_id=path,
                path=path,
                section=section,
                index=index,
                name=maybe_string(record.get("name")),
                zone=maybe_string(record.get("zone")),
                data=json_safe(record),
            )
        )
    return views


def annotation_metadata(
    annotation: Any,
    *,
    enum_cls: Optional[type[Enum]],
) -> tuple[str, bool]:
    """Map Python annotations to simple frontend input field kinds."""
    if enum_cls is not None:
        return "enum", False
    if annotation in {bool}:
        return "boolean", False
    if annotation in {int, float}:
        return "number", False
    if annotation in {str}:
        return "string", False
    if annotation in {list[str], list}:
        return "string_list", True

    text = str(annotation)
    if "List" in text or "list[" in text:
        return "string_list", True
    if "bool" in text:
        return "boolean", False
    if "float" in text or "int" in text:
        return "number", False
    if "dict" in text:
        return "object", False
    return "string", False


def numeric_delta(base_value: Any, variant_value: Any) -> Optional[float]:
    """Return a numeric delta when both inputs are plain numbers."""
    if isinstance(base_value, bool) or isinstance(variant_value, bool):
        return None
    if is_number(base_value) and is_number(variant_value):
        return float(variant_value) - float(base_value)
    return None


def is_number(value: Any) -> bool:
    """Return whether a value is a plain numeric scalar."""
    if value is None or isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def maybe_string(value: Any) -> Optional[str]:
    """Return a non-empty string representation when present."""
    if value in (None, ""):
        return None
    return str(value)


def maybe_float(value: Any) -> Optional[float]:
    """Return a finite float when conversion is possible."""
    if value is None:
        return None
    try:
        value = float(value)
    except TypeError, ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def json_safe(value: Any) -> Any:
    """Convert nested values to JSON-safe plain data structures."""
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if value is None:
        return None
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump(mode="python"))
    if hasattr(value, "to_dict"):
        try:
            return json_safe(value.to_dict())
        except Exception:
            pass
    return str(value)
