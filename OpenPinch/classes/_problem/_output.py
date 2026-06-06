"""Presentation helpers for :class:`OpenPinch.classes.PinchProblem`."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from ...lib.schemas.report_units import split_report_value
from ...utils.export import build_summary_dataframe


def build_problem_summary_frame(
    results: Any,
    *,
    detailed: bool = False,
) -> pd.DataFrame:
    """Build either the compact or detailed problem summary table."""
    if detailed:
        return build_summary_dataframe(results.targets)

    rows = []
    for target in results.targets:
        idx = getattr(target, "idx", None)
        qh_value, qh_unit = split_report_value(target.Qh, idx=idx)
        qc_value, qc_unit = split_report_value(target.Qc, idx=idx)
        qr_value, qr_unit = split_report_value(target.Qr, idx=idx)
        hot_pinch_value, hot_pinch_unit = split_report_value(
            target.pinch_temp.hot_temp,
            idx=idx,
        )
        cold_pinch_value, cold_pinch_unit = split_report_value(
            target.pinch_temp.cold_temp,
            idx=idx,
        )
        rows.append(
            {
                "Target": target.name,
                "State ID": getattr(target, "state_id", None),
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
                        idx=idx,
                    )
                    for utility in target.hot_utilities
                ),
                "Cold Utilities": ", ".join(
                    format_res(
                        name=utility.name,
                        value=utility.heat_flow,
                        idx=idx,
                    )
                    for utility in target.cold_utilities
                ),
            }
        )
    return pd.DataFrame(rows)


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


def build_graph_payload(results: Any) -> Optional[dict[str, Any]]:
    """Extract a JSON-safe graph payload from solved results when available."""
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
    idx: int | None = None,
    unit: str | None = None,
) -> str:
    """Render one utility summary item."""
    val, unt = split_report_value(value, idx=idx)
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
