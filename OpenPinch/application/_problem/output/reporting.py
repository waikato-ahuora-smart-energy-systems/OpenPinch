"""Typed report request and comparison shaping for application problems."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from pint.errors import DimensionalityError

from ....contracts.report_units import split_report_value
from ....contracts.reporting import GraphAvailability, ProblemReport, ReportMetric
from ....contracts.workspace import ValidationReport
from ....domain.enums import IntegrationType, TargetMethod
from ....domain.value import Value
from ....presentation.reporting.workbook import build_summary_dataframe

_SUMMARY_METRICS = {
    "Hot Utility Target": "Qh",
    "Cold Utility Target": "Qc",
    "Heat Recovery": "Qr",
    "Hot Pinch": "pinch_temp.hot_temp",
    "Cold Pinch": "pinch_temp.cold_temp",
}


def build_problem_summary_frame(
    results: Any,
    *,
    detailed: bool = False,
) -> pd.DataFrame:
    """Build either the compact or detailed public problem summary table."""
    if detailed:
        return build_summary_dataframe(results.targets)

    rows = []
    for target in results.targets:
        period_idx = getattr(target, "period_idx", None)
        qh_value, qh_unit = split_report_value(target.Qh, period_idx=period_idx)
        qc_value, qc_unit = split_report_value(target.Qc, period_idx=period_idx)
        qr_value, qr_unit = split_report_value(target.Qr, period_idx=period_idx)
        hot_pinch_value, hot_pinch_unit = split_report_value(
            target.pinch_temp.hot_temp,
            period_idx=period_idx,
        )
        cold_pinch_value, cold_pinch_unit = split_report_value(
            target.pinch_temp.cold_temp,
            period_idx=period_idx,
        )
        rows.append(
            {
                "Scope": target.scope,
                "Zone Type": target.zone_type,
                "Integration Type": target.integration_type,
                "Target Method": target.target_method,
                "Period ID": getattr(target, "period_id", None),
                "Hot Utility Target": format_res(
                    value=qh_value,
                    unit=qh_unit,
                ),
                "Cold Utility Target": format_res(
                    value=qc_value,
                    unit=qc_unit,
                ),
                "Heat Recovery": format_res(
                    value=qr_value,
                    unit=qr_unit,
                ),
                "Hot Pinch": format_res(
                    value=hot_pinch_value,
                    unit=hot_pinch_unit,
                ),
                "Cold Pinch": format_res(
                    value=cold_pinch_value,
                    unit=cold_pinch_unit,
                ),
                "Hot Utilities": ", ".join(
                    format_res(
                        name=utility.name,
                        value=utility.heat_flow,
                        period_idx=period_idx,
                    )
                    for utility in target.hot_utilities
                ),
                "Cold Utilities": ", ".join(
                    format_res(
                        name=utility.name,
                        value=utility.heat_flow,
                        period_idx=period_idx,
                    )
                    for utility in target.cold_utilities
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_numeric_summary_frame(results: Any) -> pd.DataFrame:
    """Build the private numeric summary used for problem comparisons."""
    rows = []
    for target in results.targets:
        row = {
            "Scope": target.scope,
            "Zone Type": target.zone_type,
            "Integration Type": target.integration_type,
            "Target Method": target.target_method,
            "Period ID": getattr(target, "period_id", None),
        }
        for label, attr_path in _SUMMARY_METRICS.items():
            value, unit = split_report_value(
                _target_attr(target, attr_path),
                period_idx=getattr(target, "period_idx", None),
            )
            row[label] = value
            row[f"{label} (unit)"] = unit
        row["Hot Utilities"] = ", ".join(
            format_res(
                name=utility.name,
                value=utility.heat_flow,
                period_idx=getattr(target, "period_idx", None),
            )
            for utility in target.hot_utilities
        )
        row["Cold Utilities"] = ", ".join(
            format_res(
                name=utility.name,
                value=utility.heat_flow,
                period_idx=getattr(target, "period_idx", None),
            )
            for utility in target.cold_utilities
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_report_metrics(results: Any) -> list[ReportMetric]:
    """Return stable typed metrics for all summary target rows."""
    metrics = []
    for target in results.targets:
        period_id = getattr(target, "period_id", None)
        for label, attr_path in _SUMMARY_METRICS.items():
            value, unit = split_report_value(
                _target_attr(target, attr_path),
                period_idx=getattr(target, "period_idx", None),
            )
            metrics.append(
                ReportMetric(
                    scope=target.scope,
                    zone_type=target.zone_type,
                    integration_type=target.integration_type,
                    target_method=target.target_method,
                    metric=label,
                    label=label,
                    value=value,
                    unit=unit,
                    period_id=period_id,
                )
            )
    return metrics


def build_graph_availability(
    graph_data: dict[str, Any] | None,
) -> list[GraphAvailability]:
    """Flatten graph data into typed graph availability records."""
    if not graph_data:
        return []
    entries = []
    for graph_set_id, graph_set in graph_data.items():
        target_name = str(graph_set.get("name", graph_set_id))
        for index, graph in enumerate(graph_set.get("graphs", [])):
            graph_type = graph.get("type")
            graph_name = graph.get("name", f"Graph {index + 1}")
            entries.append(
                GraphAvailability(
                    graph_id=f"{graph_set_id}::{graph_type or 'graph'}::{index}",
                    graph_set_id=str(graph_set_id),
                    target_name=target_name,
                    zone_name=graph_set.get("zone_name"),
                    zone_address=graph_set.get("zone_address"),
                    target_type=graph_set.get("target_type"),
                    graph_type=graph_type,
                    graph_name=str(graph_name),
                    index=index,
                )
            )
    return entries


def build_problem_report(
    *,
    project_name: str,
    validation: ValidationReport,
    results: Any | None,
    graph_data: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> ProblemReport:
    """Build a typed report from current validation and optional solved results."""
    return ProblemReport(
        project_name=project_name,
        solved=results is not None,
        validation=validation,
        targets=list(getattr(results, "targets", []) or []),
        metrics=build_report_metrics(results) if results is not None else [],
        graph_catalog=build_graph_availability(graph_data),
        warnings=list(warnings or []),
    )


def locate_summary_row(
    frame: pd.DataFrame,
    *,
    scope: Optional[str] = None,
    zone_type: Optional[str] = None,
    integration_type: Optional[str] = None,
    target_method: Optional[str] = None,
) -> pd.Series:
    """Locate one summary row using explicit public target metadata."""
    identity_columns = {
        "Scope": scope,
        "Zone Type": zone_type,
        "Integration Type": integration_type,
        "Target Method": target_method,
    }
    missing = [column for column in identity_columns if column not in frame.columns]
    if frame.empty or missing:
        detail = f"; missing {missing}" if missing else ""
        raise ValueError(f"Summary frame is empty or missing identity columns{detail}.")

    candidates = frame
    explicit = any(value is not None for value in identity_columns.values())
    defaults = {
        "Integration Type": IntegrationType.Process.value,
        "Target Method": TargetMethod.HeatExchange.value,
    }
    for column, value in identity_columns.items():
        selected = value if value is not None else defaults.get(column)
        if selected is not None:
            candidates = candidates.loc[candidates[column].astype(str) == str(selected)]
    if scope is None and not candidates.empty:
        depth = candidates["Scope"].astype(str).str.count("/")
        candidates = candidates.loc[depth == depth.min()]
    if candidates.empty:
        requested = {
            column: value
            for column, value in identity_columns.items()
            if value is not None
        }
        if not explicit:
            requested = defaults
        raise KeyError(
            f"Target metadata {requested!r} was not found in the summary output."
        )
    return candidates.iloc[0]


def compare_problem_summaries(
    base_frame: pd.DataFrame,
    other_frame: pd.DataFrame,
    *,
    scope: str | None,
    zone_type: str | None,
    integration_type: str | None,
    target_method: str | None,
    base_label: str,
    other_label: str,
) -> pd.DataFrame:
    """Compare unit-aware numeric metrics from two solved summary frames."""
    selectors = {
        "scope": scope,
        "zone_type": zone_type,
        "integration_type": integration_type,
        "target_method": target_method,
    }
    base_row = locate_summary_row(base_frame, **selectors)
    other_row = locate_summary_row(
        other_frame,
        scope=scope or str(base_row["Scope"]),
        zone_type=zone_type or str(base_row["Zone Type"]),
        integration_type=integration_type or str(base_row["Integration Type"]),
        target_method=target_method or str(base_row["Target Method"]),
    )
    columns = [
        "Hot Utility Target",
        "Cold Utility Target",
        "Heat Recovery",
        "Hot Pinch",
        "Cold Pinch",
    ]
    unit_columns = {column: f"{column} (unit)" for column in columns}
    row_columns = [*columns, *unit_columns.values()]
    identity_columns = ["Scope", "Zone Type", "Integration Type", "Target Method"]
    base_row_data = {
        **{column: base_row.get(column) for column in identity_columns},
        **{column: base_row.get(column) for column in row_columns},
    }
    other_row_data = {
        **{column: other_row.get(column) for column in identity_columns},
        **{column: other_row.get(column) for column in row_columns},
    }
    change_row: dict[str, object] = {
        column: base_row.get(column) for column in identity_columns
    }
    for column in columns:
        unit_column = unit_columns[column]
        base_unit = base_row.get(unit_column)
        other_unit = other_row.get(unit_column)
        try:
            base_value = Value(base_row.get(column), base_unit)
            other_value = Value(other_row.get(column), other_unit).to(base_unit)
            change_row[column] = float(other_value) - float(base_value)
            change_row[unit_column] = base_unit
        except DimensionalityError, TypeError, ValueError:
            change_row[column] = None
            change_row[unit_column] = None

    return pd.DataFrame.from_dict(
        {
            base_label: base_row_data,
            other_label: other_row_data,
            "Change": change_row,
        },
        orient="index",
        columns=[*identity_columns, *row_columns],
    )


def build_graph_data(results: Any) -> Optional[dict[str, Any]]:
    """Extract JSON-safe graph data from solved results when available."""
    graphs = getattr(results, "graphs", None)
    if not graphs:
        return None
    return {
        key: value.model_dump() if hasattr(value, "model_dump") else dict(value)
        for key, value in graphs.items()
    }


def format_res(
    name: str | None = None,
    value: Any | None = None,
    period_idx: int | None = None,
    unit: str | None = None,
) -> str:
    """Render one utility summary item."""
    val, unt = split_report_value(value, period_idx=period_idx)
    if unt is None:
        unt = unit
    if val is None:
        if name:
            return f"{name}: n/a"
        else:
            return "n/a"
    if unt is None:
        if name:
            return f"{name}: {float(val):.2f}"
        else:
            return f"{float(val):.2f}"
    if name:
        return f"{name}: {float(val):.2f} {unt}"
    else:
        return f"{float(val):.2f} {unt}"


def _target_attr(target: Any, attr_path: str) -> Any:
    value = target
    for part in attr_path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value
