"""Shared helpers for the PinchWorkspace multi-case surface."""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
import math
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from pydantic import ValidationError

from ..lib.enums import ST
from ..lib.schemas.io import TargetInput
from ..lib.schemas.workspace import (
    GraphCatalogEntry,
    GraphPayloadEntry,
    PayloadRecordView,
    ProblemTableDiffView,
    ProblemTableView,
    ScenarioVariantView,
    SummaryCard,
    TableView,
    ValidationIssue,
    ValidationReport,
    VariantMetricDelta,
    ZoneNodeView,
)
from ..streamlit_webviewer.web_graphing import (
    collect_targets,
    problem_table_to_dataframe,
)
from .pinch_problem import PinchProblem, _build_validation_context, _maybe_get_value

JsonDict = Dict[str, Any]

_SUMMARY_METRIC_COLUMNS = [
    "Hot Utility Target",
    "Cold Utility Target",
    "Heat Recovery",
    "Hot Pinch",
    "Cold Pinch",
]

_WORKFLOW_SUPPORT_LEVELS = {
    "target": "stable",
    "direct_heat_integration": "stable",
    "indirect_heat_integration": "stable",
    "direct_heat_pump": "advanced",
    "indirect_heat_pump": "advanced",
    "direct_refrigeration": "advanced",
    "indirect_refrigeration": "advanced",
    "cogeneration": "advanced",
    "area_cost": "advanced",
}

def normalise_payload(payload: TargetInput | JsonDict) -> JsonDict:
    if isinstance(payload, TargetInput):
        return payload.model_dump(mode="python")
    if not isinstance(payload, dict):
        raise TypeError("Workspace payloads must be a dict or TargetInput instance.")
    return deepcopy(payload)


def project_name_from_payload(payload: JsonDict) -> Optional[str]:
    zone_tree = payload.get("zone_tree")
    if isinstance(zone_tree, dict):
        name = zone_tree.get("name")
        if name not in (None, ""):
            return str(name)
    return None


def merge_payloads(base: JsonDict, overlay: JsonDict) -> JsonDict:
    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_payloads(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def build_validation_report(payload: JsonDict) -> ValidationReport:
    context = _build_validation_context(payload, source_kind="target_input")
    issues: list[ValidationIssue] = []

    try:
        validated = TargetInput.model_validate(payload)
    except ValidationError as exc:
        for error in exc.errors():
            issues.append(schema_issue_to_view(error, context=context))
        return ValidationReport(valid=False, issues=issues)

    issues.extend(semantic_issues(validated, context=context))
    return ValidationReport(
        valid=not any(issue.severity == "error" for issue in issues),
        issues=issues,
    )


def schema_issue_to_view(
    error: dict[str, Any],
    *,
    context: dict[str, list[dict[str, Any]]],
) -> ValidationIssue:
    loc = tuple(error.get("loc", ()))
    section = loc[0] if loc and isinstance(loc[0], str) else None
    record_index = loc[1] if len(loc) > 1 and isinstance(loc[1], int) else None
    field = ".".join(str(part) for part in loc[2:]) if len(loc) > 2 else None
    path = path_from_loc(loc)
    record_label = validation_record_label(section, record_index, context)
    return ValidationIssue(
        severity="error",
        path=path,
        message=error.get("msg", "Invalid value."),
        section=section,
        record_index=record_index,
        field=field,
        record_label=record_label,
    )


def semantic_issues(
    payload: TargetInput,
    *,
    context: dict[str, list[dict[str, Any]]],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if len(payload.streams) == 0:
        issues.append(
            ValidationIssue(
                severity="error",
                path="streams",
                message="At least one stream must be provided.",
            )
        )

    for index, stream in enumerate(payload.streams):
        label = validation_record_label("streams", index, context)
        t_supply = _maybe_get_value(stream.t_supply)
        t_target = _maybe_get_value(stream.t_target)
        if t_supply == t_target:
            issues.append(
                ValidationIssue(
                    severity="error",
                    path=f"streams[{index}].t_supply",
                    field="t_supply/t_target",
                    message="Supply and target temperatures must differ.",
                    section="streams",
                    record_index=index,
                    record_label=label,
                )
            )

        value = _maybe_get_value(getattr(stream, "heat_flow"))
        if value is not None and value < 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    path=f"streams[{index}].heat_flow",
                    field="heat_flow",
                    message="Value must be non-negative.",
                    section="streams",
                    record_index=index,
                    record_label=label,
                )
            )

        value = _maybe_get_value(getattr(stream, "dt_cont"))
        if value is not None and value < 0:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    path=f"streams[{index}].dt_cont",
                    field="dt_cont",
                    message="Value should be non-negative.",
                    section="streams",
                    record_index=index,
                    record_label=label,
                )
            )

        value = _maybe_get_value(getattr(stream, "htc"))
        if value is not None and value <= 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    path=f"streams[{index}].htc",
                    field="htc",
                    message="Value must be positive.",
                    section="streams",
                    record_index=index,
                    record_label=label,
                )
            )

    for index, utility in enumerate(payload.utilities):
        label = validation_record_label("utilities", index, context)
        utility_type = str(utility.type)
        t_supply = _maybe_get_value(utility.t_supply)
        t_target = _maybe_get_value(utility.t_target)
        if (
            utility_type == ST.Hot.value
            and t_supply is not None
            and t_target is not None
            and t_supply < t_target
        ):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    path=f"utilities[{index}].t_supply",
                    field="t_supply/t_target",
                    message="Hot utilities should have t_supply >= t_target.",
                    section="utilities",
                    record_index=index,
                    record_label=label,
                )
            )
        if (
            utility_type == ST.Cold.value
            and t_supply is not None
            and t_target is not None
            and t_supply > t_target
        ):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    path=f"utilities[{index}].t_supply",
                    field="t_supply/t_target",
                    message="Cold utilities should have t_supply <= t_target.",
                    section="utilities",
                    record_index=index,
                    record_label=label,
                )
            )

        for field_name in ("dt_cont", "price", "heat_flow"):
            value = _maybe_get_value(getattr(utility, field_name))
            if value is not None and value < 0:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        path=f"utilities[{index}].{field_name}",
                        field=field_name,
                        message="Value should be non-negative.",
                        section="utilities",
                        record_index=index,
                        record_label=label,
                    )
                )

        value = _maybe_get_value(getattr(utility, "htc"))
        if value is not None and value <= 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    path=f"utilities[{index}].htc",
                    field="htc",
                    message="Value must be positive.",
                    section="utilities",
                    record_index=index,
                    record_label=label,
                )
            )
    return issues


def path_from_loc(loc: tuple[Any, ...]) -> str:
    parts = []
    for part in loc:
        if isinstance(part, int):
            if not parts:
                parts.append(f"[{part}]")
            else:
                parts[-1] = f"{parts[-1]}[{part}]"
        else:
            parts.append(str(part))
    return ".".join(parts)


def validation_record_label(
    section: Optional[str],
    record_index: Optional[int],
    context: dict[str, list[dict[str, Any]]],
) -> Optional[str]:
    if section is None or record_index is None:
        return None
    records = context.get(section, [])
    if 0 <= record_index < len(records):
        record = records[record_index]
        label = "Stream" if section == "streams" else "Utility"
        name = record.get("name")
        if name in (None, ""):
            return f"{label} {record_index + 1}"
        return f"{label} {record_index + 1} '{name}'"
    return None


def workflow_support_level(workflow: str) -> str:
    normalized = normalise_workflow_name(workflow)
    return _WORKFLOW_SUPPORT_LEVELS.get(normalized, "unsupported")


def workflow_warnings(workflow: str, support_level: str) -> list[str]:
    if support_level == "advanced":
        return [
            f"Workflow '{workflow}' should be treated as an advanced PinchWorkspace workflow."
        ]
    if support_level == "unsupported":
        return [f"Workflow '{workflow}' is not a supported PinchWorkspace workflow."]
    return []


def run_problem_workflow(
    problem: PinchProblem,
    workflow: str,
    workflow_options: dict[str, Any],
) -> None:
    normalized = normalise_workflow_name(workflow)
    if normalized == "target":
        problem.target()
        return

    if not hasattr(problem.target, normalized):
        raise ValueError(
            f"Unknown workflow {workflow!r}. Supported workflows include: "
            f"target, {', '.join(sorted(_WORKFLOW_SUPPORT_LEVELS))}."
        )

    problem.target()
    method = getattr(problem.target, normalized)
    method(**workflow_options)


def normalise_workflow_name(workflow: str) -> str:
    return str(workflow).strip().lower().replace("-", "_").replace(" ", "_")


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
    summary_frame = problem.summary_frame()
    graph_payload = problem.plot.get_graph_data()
    graph_catalog = graph_catalog_entries(graph_payload)
    graph_payloads = graph_payload_entries(graph_payload)
    problem_tables = problem_table_views(problem)

    return ScenarioVariantView(
        variant_name=variant_name,
        workflow=workflow,
        workflow_options=workflow_options,
        status="solved",
        support_level=support_level,
        validation=validation,
        warnings=warnings_list,
        summary_cards=summary_cards(summary_frame),
        summary_table=dataframe_to_table_view(summary_frame),
        graph_catalog=graph_catalog,
        graph_payloads=graph_payloads,
        problem_tables=problem_tables,
    )


def summary_cards(frame: pd.DataFrame) -> list[SummaryCard]:
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
                )
            )
    return cards


def graph_catalog_entries(graph_payload: JsonDict) -> list[GraphCatalogEntry]:
    entries = []
    for graph_set_id, graph_set in graph_payload.items():
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


def graph_payload_entries(graph_payload: JsonDict) -> list[GraphPayloadEntry]:
    payloads = []
    for graph_set_id, graph_set in graph_payload.items():
        target_name = str(graph_set.get("name", graph_set_id))
        target_id = target_name
        for index, graph in enumerate(graph_set.get("graphs", [])):
            graph_type = maybe_string(graph.get("type"))
            graph_name = str(graph.get("name") or graph_type or f"Graph {index + 1}")
            payloads.append(
                GraphPayloadEntry(
                    graph_id=f"{graph_set_id}::{index}::{graph_type or 'graph'}",
                    graph_set_id=str(graph_set_id),
                    target_id=target_id,
                    target_name=target_name,
                    graph_type=graph_type,
                    graph_name=graph_name,
                    payload=json_safe(graph),
                )
            )
    return payloads


def problem_table_views(problem: PinchProblem) -> list[ProblemTableView]:
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
            deltas.append(
                VariantMetricDelta(
                    base_variant=base_name,
                    variant_name=variant_name,
                    target_id=target_id,
                    target_name=target_name,
                    metric=metric,
                    base_value=base_value,
                    variant_value=variant_value,
                    delta=numeric_delta(base_value, variant_value),
                )
            )
    return deltas


def summary_rows_by_target(table: Optional[TableView]) -> dict[str, dict[str, Any]]:
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
                    base_table, variant_table, shared_columns
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


def record_views(records: Any, *, section: str) -> list[PayloadRecordView]:
    if not isinstance(records, list):
        return []
    views = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        path = f"{section}[{index}]"
        views.append(
            PayloadRecordView(
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
    if isinstance(base_value, bool) or isinstance(variant_value, bool):
        return None
    if is_number(base_value) and is_number(variant_value):
        return float(variant_value) - float(base_value)
    return None


def is_number(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def maybe_string(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value)


def maybe_float(value: Any) -> Optional[float]:
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


__all__ = [
    "JsonDict",
    "annotation_metadata",
    "build_validation_report",
    "graph_catalog_entries",
    "graph_payload_entries",
    "json_safe",
    "merge_payloads",
    "normalise_payload",
    "problem_table_diffs",
    "problem_to_variant_view",
    "project_name_from_payload",
    "record_views",
    "run_problem_workflow",
    "summary_metric_deltas",
    "workflow_support_level",
    "workflow_warnings",
    "zone_tree_view",
]
