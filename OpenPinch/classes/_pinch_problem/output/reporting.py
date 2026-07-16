"""Presentation helpers for :class:`OpenPinch.classes.PinchProblem`."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from ....lib.schemas.report_units import split_report_value
from ....lib.schemas.reporting import GraphAvailability, ProblemReport, ReportMetric
from ....lib.schemas.workspace import ValidationReport
from ....utils.export import build_summary_dataframe

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
    format: str = "compact",
) -> pd.DataFrame:
    """Build either the compact or detailed problem summary table."""
    if detailed or format == "detailed":
        return build_summary_dataframe(results.targets)
    if format == "plain":
        return build_plain_summary_frame(results)
    if format != "compact":
        raise ValueError("summary format must be 'compact', 'plain', or 'detailed'.")

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
                "Target": target.name,
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


def build_plain_summary_frame(results: Any) -> pd.DataFrame:
    """Build a user-readable summary with numeric value and unit columns."""
    rows = []
    for target in results.targets:
        row = {
            "Target": target.name,
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
                    target_name=target.name,
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
    target_name: Optional[str] = None,
) -> pd.Series:
    """Locate one summary row using explicit name or common defaults."""
    if "Target" not in frame.columns or frame.empty:
        raise ValueError("Summary frame is empty or missing the 'Target' column.")

    targets = frame["Target"].astype(str)
    if target_name is not None:
        exact_match = frame.loc[targets == str(target_name)]
        if not exact_match.empty:
            return exact_match.iloc[0]

        suffix = str(target_name).split("/", 1)[-1]
        suffix_match = frame.loc[targets.str.endswith(suffix)]
        if not suffix_match.empty:
            return suffix_match.iloc[0]

        raise KeyError(f"Target {target_name!r} was not found in the summary output.")

    preferred_match = frame.loc[targets == "Plant/Direct Integration"]
    if not preferred_match.empty:
        return preferred_match.iloc[0]

    direct_match = frame.loc[targets.str.endswith("/Direct Integration")]
    if not direct_match.empty:
        return direct_match.iloc[0]
    return frame.iloc[0]


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
