
"""Streamlit helpers for visualising OpenPinch outputs.

The functions in this module provide a lightweight dashboard scaffold that
renders the composite-curve style graphs emitted by :mod:`OpenPinch.analysis`
alongside the corresponding problem tables.  The dashboard is intentionally
minimal so user projects can layer additional controls as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..classes import EnergyTarget, ProblemTable, Zone
from ..lib.enums import ArrowHead, LineColour
from .graph_data import get_output_graph_data

__all__ = [
    "StreamlitGraphSet",
    "collect_targets",
    "problem_table_to_dataframe",
    "render_streamlit_dashboard",
]


# Matplotlib-friendly colours keyed by the internal ``LineColour`` palette.
_SEGMENT_COLOUR_MAP: Dict[int, str] = {
    LineColour.Hot.value: "#d62728",  # warm red
    LineColour.Cold.value: "#1f77b4",  # cool blue
    LineColour.Other.value: "#7f7f7f",  # neutral grey
    LineColour.Black.value: "#111111",
}


@dataclass(slots=True)
class StreamlitGraphSet:
    """Convenience wrapper storing graphs grouped by target name."""

    name: str
    graphs: List[MutableMapping]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "StreamlitGraphSet":
        return cls(
            name=str(payload.get("name", "Graph Set")),
            graphs=list(payload.get("graphs", [])),
        )


def collect_targets(zone: Zone) -> Dict[str, EnergyTarget]:
    """Flattens all energy targets beneath ``zone`` keyed by their display name."""

    def _iter(current: Zone) -> Iterator[tuple[str, EnergyTarget]]:
        for name, target in current.targets.items():
            yield name, target
        for subzone in current.subzones.values():
            yield from _iter(subzone)

    return dict(_iter(zone))


def problem_table_to_dataframe(
    table: Optional[ProblemTable], *, round_decimals: int = 2
) -> pd.DataFrame:
    """Convert a :class:`ProblemTable` into a :class:`pandas.DataFrame`."""
    if table is None or getattr(table, "data", None) is None:
        return pd.DataFrame()

    data = table.data
    columns = getattr(table, "columns", [])
    if data.size == 0 or len(columns) == 0:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(data=data, columns=columns).copy()
    if round_decimals is not None:
        numeric_cols = frame.select_dtypes(include="number").columns
        frame.loc[:, numeric_cols] = frame.loc[:, numeric_cols].round(round_decimals)
    return frame


def render_streamlit_dashboard(
    zone: Zone,
    *,
    graph_payload: Optional[Mapping[str, Mapping[str, object]]] = None,
    page_title: Optional[str] = None,
    value_rounding: int = 2,
) -> None:
    """Render a basic Streamlit dashboard for ``zone``."""
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - streamlit dependency guard
        raise ImportError(
            "Streamlit is required for 'render_streamlit_dashboard'. "
            "Install it with 'pip install streamlit'."
        ) from exc

    st.set_page_config(
        page_title=page_title or f"{zone.name} Pinch Dashboard",
        layout="wide",
    )

    st.title(page_title or f"{zone.name} Pinch Dashboard")

    targets = collect_targets(zone)
    if not targets:
        st.warning("No targets available for the selected zone.")
        return

    graph_payload = graph_payload or get_output_graph_data(zone)
    graph_sets = {
        name: StreamlitGraphSet.from_payload(payload)
        for name, payload in graph_payload.items()
    }

    base_key = f"{zone.name}_{id(zone)}"

    target_names = sorted(targets.keys())
    selected_target_name = st.sidebar.selectbox(
        "Target",
        target_names,
        index=0 if target_names else None,
        key=f"target_select_{base_key}",
    )
    target = targets[selected_target_name]

    st.sidebar.write(f"**Cold Pinch:** {target.cold_pinch:.2f}")
    st.sidebar.write(f"**Hot Pinch:** {target.hot_pinch:.2f}")
    st.sidebar.write(f"**Heat Recovery Target:** {target.heat_recovery_target:.2f}")

    tabs = st.tabs(
        [
            "Graphs",
            "Problem Table (Shifted)",
            "Problem Table (Real Temperatures)",
        ]
    )

    with tabs[0]:
        graph_set = graph_sets.get(selected_target_name)
        if graph_set is None or not graph_set.graphs:
            st.info("No graphs available for this target.")
        else:
            graph_names = [
                str(graph.get("name") or graph.get("type") or f"Graph {idx + 1}")
                for idx, graph in enumerate(graph_set.graphs)
            ]
            graph_choice = st.selectbox(
                "Graph type",
                graph_names,
                key=f"graph_select_{base_key}_{selected_target_name}",
            )
            graph = graph_set.graphs[graph_names.index(graph_choice)]
            figure = _build_matplotlib_graph(graph)
            st.pyplot(figure, clear_figure=True)
            plt.close(figure)

    with tabs[1]:
        pt_df = problem_table_to_dataframe(target.pt, round_decimals=value_rounding)
        if pt_df.empty:
            st.info("No shifted problem table data available.")
        else:
            st.dataframe(pt_df, width="stretch")

    with tabs[2]:
        pt_real_df = problem_table_to_dataframe(
            target.pt_real, round_decimals=value_rounding
        )
        if pt_real_df.empty:
            st.info("No real-temperature problem table data available.")
        else:
            st.dataframe(pt_real_df, width="stretch")


def _build_matplotlib_graph(graph: Mapping[str, object]) -> plt.Figure:
    """Create a matplotlib figure for the provided graph payload."""
    fig, ax = plt.subplots()

    segments: Iterable[Mapping[str, object]] = graph.get("segments", [])
    show_legend = False

    for segment in segments:
        x_vals, y_vals = _extract_segment_xy(segment)
        if not x_vals or not y_vals:
            continue

        colour_idx = segment.get("colour")
        colour = _SEGMENT_COLOUR_MAP.get(colour_idx, "#333333")
        label = segment.get("title")
        ax.plot(x_vals, y_vals, color=colour, linewidth=2.0, label=label)
        _maybe_draw_arrow(ax, x_vals, y_vals, segment.get("arrow"), colour)
        if label:
            show_legend = True

    title = graph.get("name") or graph.get("type") or "Graph"
    ax.set_title(title)
    ax.set_xlabel("Enthalpy")
    ax.set_ylabel("Temperature")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if show_legend:
        ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    return fig


def _extract_segment_xy(segment: Mapping[str, object]) -> tuple[List[float], List[float]]:
    """Return x/y coordinate lists for a graph segment payload."""
    points = segment.get("data_points", []) or []
    x_vals = [point["x"] for point in points if "x" in point and "y" in point]
    y_vals = [point["y"] for point in points if "x" in point and "y" in point]
    return x_vals, y_vals


def _maybe_draw_arrow(
    ax: plt.Axes,
    x_vals: List[float],
    y_vals: List[float],
    arrow: Optional[str],
    colour: str,
) -> None:
    """Annotate a curve with directional arrows when requested."""
    if arrow is None or len(x_vals) < 2 or len(y_vals) < 2:
        return

    if arrow == ArrowHead.END.value:
        start_idx, end_idx = -2, -1
    elif arrow == ArrowHead.START.value:
        start_idx, end_idx = 1, 0
    else:
        return

    ax.annotate(
        "",
        xy=(x_vals[end_idx], y_vals[end_idx]),
        xytext=(x_vals[start_idx], y_vals[start_idx]),
        arrowprops=dict(arrowstyle="->", color=colour, shrinkA=0, shrinkB=0),
    )
