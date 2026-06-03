"""Presentation helpers for :class:`OpenPinch.classes.PinchProblem`."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from ...utils.export import build_summary_dataframe
from ...utils.miscellaneous import get_value


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
        rows.append(
            {
                "Target": target.name,
                "State ID": getattr(target, "state_id", None),
                "Hot Utility Target": maybe_get_value(
                    target.Qh, state_id=getattr(target, "state_id", None)
                ),
                "Cold Utility Target": maybe_get_value(
                    target.Qc, state_id=getattr(target, "state_id", None)
                ),
                "Heat Recovery": maybe_get_value(
                    target.Qr, state_id=getattr(target, "state_id", None)
                ),
                "Hot Pinch": maybe_get_value(
                    target.temp_pinch.hot_temp,
                    state_id=getattr(target, "state_id", None),
                ),
                "Cold Pinch": maybe_get_value(
                    target.temp_pinch.cold_temp,
                    state_id=getattr(target, "state_id", None),
                ),
                "Hot Utilities": ", ".join(
                    format_utility(
                        utility.name,
                        utility.heat_flow,
                        state_id=getattr(target, "state_id", None),
                    )
                    for utility in target.hot_utilities
                ),
                "Cold Utilities": ", ".join(
                    format_utility(
                        utility.name,
                        utility.heat_flow,
                        state_id=getattr(target, "state_id", None),
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


def maybe_get_value(value: Any, state_id: str | None = None) -> Any:
    """Return the scalar value for OpenPinch value objects."""
    if value is None:
        return None
    return get_value(value, state_id=state_id)


def format_utility(
    name: str,
    heat_flow: Any,
    state_id: str | None = None,
) -> str:
    """Render one utility summary item."""
    value = maybe_get_value(heat_flow, state_id=state_id)
    if value is None:
        return f"{name}: n/a"
    return f"{name}: {value:.2f}"
