"""Presentation helpers for :class:`OpenPinch.classes.PinchProblem`."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from ...utils.export import build_summary_dataframe
from ...utils.value_resolution import get_scalar_value


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
        rows.append(
            {
                "Target": target.name,
                "State ID": getattr(target, "state_id", None),
                "Hot Utility Target": get_scalar_value(
                    target.Qh,
                    idx=idx,
                ),
                "Cold Utility Target": get_scalar_value(
                    target.Qc,
                    idx=idx,
                ),
                "Heat Recovery": get_scalar_value(
                    target.Qr,
                    idx=idx,
                ),
                "Hot Pinch": get_scalar_value(
                    target.temp_pinch.hot_temp,
                    idx=idx,
                ),
                "Cold Pinch": get_scalar_value(
                    target.temp_pinch.cold_temp,
                    idx=idx,
                ),
                "Hot Utilities": ", ".join(
                    format_utility(
                        utility.name,
                        utility.heat_flow,
                        idx=idx,
                    )
                    for utility in target.hot_utilities
                ),
                "Cold Utilities": ", ".join(
                    format_utility(
                        utility.name,
                        utility.heat_flow,
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


def format_utility(
    name: str,
    heat_flow: Any,
    idx: int | None = None,
) -> str:
    """Render one utility summary item."""
    value = get_scalar_value(heat_flow, idx=idx)
    if value is None:
        return f"{name}: n/a"
    return f"{name}: {value:.2f}"
