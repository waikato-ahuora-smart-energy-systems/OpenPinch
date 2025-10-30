
"""Streamlit helpers for visualising OpenPinch outputs.

The functions in this module provide a lightweight dashboard scaffold that
renders the composite-curve style graphs emitted by :mod:`OpenPinch.analysis`
alongside the corresponding problem tables.  The dashboard is intentionally
minimal so user projects can layer additional controls as needed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from io import BytesIO
from typing import Dict, Iterator, List, Mapping, MutableMapping, Optional, Tuple

import plotly.graph_objects as go
import pandas as pd
import openpyxl as xl_writer

from ..classes import EnergyTarget, ProblemTable, Zone, Stream
from ..lib.enums import ArrowHead, LineColour
from ..analysis.graph_data import get_output_graph_data

__all__ = [
    "StreamlitGraphSet",
    "collect_targets",
    "problem_table_to_dataframe",
    "render_streamlit_dashboard",
]


# Plotly-friendly colours keyed by the internal ``LineColour`` palette.
_SEGMENT_COLOUR_MAP: Dict[int, str] = {
    LineColour.HotS.value: "#e66e6e",  # warm red
    LineColour.ColdS.value: "#5ca5d9",  # cool blue
    LineColour.HotU.value: "#C22323",  # warm red
    LineColour.ColdU.value: "#244abd",  # cool blue
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
        "**Select zone**",
        target_names,
        index=0 if target_names else None,
        key=f"target_select_{base_key}",
    )
    target = targets[selected_target_name]

    st.sidebar.divider()
    st.sidebar.write(f"**Targets**")
    st.sidebar.write(f"Cold pinch: {target.cold_pinch:.1f} \N{DEGREE SIGN}C")
    st.sidebar.write(f"Hot pinch: {target.hot_pinch:.1f} \N{DEGREE SIGN}C")
    st.sidebar.write(f"Hot utility: {target.hot_utility_target:,.0f} kW")
    st.sidebar.write(f"Cold Utility: {target.cold_utility_target:,.0f} kW")
    st.sidebar.write(f"Heat Recovery: {target.heat_recovery_target:,.0f} kW")
    st.sidebar.write(f"Degree of Integration: {target.degree_of_int:.0%}")

    ut_dict ={
        "Hot utilities" : target.hot_utilities, 
        "Cold utilities" : target.cold_utilities,
    }
    for entry in ut_dict.keys():
        st.sidebar.divider()
        st.sidebar.write(f"**{entry}**")
        if len(ut_dict[entry]):
            u: Stream
            for u in ut_dict[entry]:
                st.sidebar.write(f"{u.name}: {u.heat_flow:,.0f} kW")
        else:
            st.sidebar.write(f"{entry} not required.")


    tabs = st.tabs(
        [
            "Graphs",
            "Problem Table (Shifted)",
            "Problem Table (Real)",
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
            columns = st.columns(2)
            for idx, graph in enumerate(graph_set.graphs):
                column = columns[idx % 2]
                with column:
                    st.markdown(f"**{graph_names[idx]}**")
                    figure = _build_plotly_graph(graph)
                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                        config={"displaylogo": False},
                    )

    with tabs[1]:
        pt_df = problem_table_to_dataframe(
            target.pt, round_decimals=value_rounding
        )
        # problem_table_to_dataframe(target.pt, round_decimals=value_rounding)
        if pt_df.empty:
            st.info("No shifted problem table data available.")
        else:
            st.badge("Extended problem table based on shifted process temperatures. Note: Interval delta values shown in line with zeros at the top of the coloumns.")
            st.dataframe(pt_df, width="stretch")
            default_loc = f"results/{selected_target_name.replace('/', '-')}_shifted.xlsx"

            _build_download(
                st=st,
                default=default_loc,
                base_key=base_key,
                selected_target_name=selected_target_name,
                df=pt_df,
                key_suffix="shifted",
            )

    with tabs[2]:
        pt_real_df = problem_table_to_dataframe(
            target.pt_real, round_decimals=value_rounding
        )
        if pt_real_df.empty:
            st.info("No real-temperature problem table data available.")
        else:
            st.badge("Extended problem table based on real process temperatures. Note: Interval delta values shown in line with zeros at the top of the coloumns.")
            st.dataframe(pt_real_df, width="stretch")
            default_loc = f"results/{selected_target_name.replace('/', '-')}_real.xlsx"

            _build_download(
                st=st,
                default=default_loc,
                base_key=base_key,
                selected_target_name=selected_target_name,
                df=pt_real_df,
                key_suffix="real",
            )


def _build_download(
    st,
    default: str,
    *,
    base_key: str,
    selected_target_name: str,
    df: pd.DataFrame,
    key_suffix: str,
) -> None:
    save_path = st.text_input(
        "Save location",
        default,
        key=f"save_path_{base_key}_{selected_target_name}_{key_suffix}",
    )
    if st.button(
        "Save table as Excel",
        key=f"save_button_{base_key}_{selected_target_name}_{key_suffix}",
    ):
        destination = save_path.strip()
        if not destination:
            st.error("Please provide a file path to save the table.")
        else:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine=xl_writer.__name__) as writer:
                df.to_excel(writer, index=False, sheet_name="Problem Table")
            try:
                with open(destination, "wb") as out_file:
                    out_file.write(buffer.getvalue())
                st.success(f"Saved table to {destination}")
            except OSError as exc:
                st.error(f"Failed to save file: {exc}")                   


def _build_plotly_graph(graph: Mapping[str, object]) -> go.Figure:
    """Create a Plotly figure for the provided graph payload."""
    fig = go.Figure()
    legend_seen: Dict[str, bool] = {}
    for segment in graph.get("segments", []):
        trace = _segment_trace(segment, graph, legend_seen)
        if trace is not None:
            fig.add_trace(trace)
    _apply_default_layout(fig)
    return fig


def _segment_trace(
    segment: Mapping[str, object],
    graph: Mapping[str, object],
    legend_seen: Dict[str, bool],
) -> Optional[go.Scatter]:
    x_vals, y_vals = _extract_segment_xy(segment)
    if not x_vals or not y_vals:
        return None
    title = segment.get("title") or graph.get("type") or "Segment"
    colour = _segment_colour(segment)
    legend_label, series_id, show = _legend_details(segment, title, legend_seen)
    mode, marker = _segment_mode_and_markers(segment, x_vals, y_vals, colour)
    return go.Scatter(
        x=x_vals,
        y=y_vals,
        mode=mode,
        name=legend_label,
        line=_line_style(segment, colour),
        marker=marker,
        hovertemplate=_hover_template(segment, title, legend_label),
        legendgroup=series_id,
        showlegend=show,
    )


def _segment_colour(segment: Mapping[str, object]) -> str:
    colour_idx = segment.get("colour")
    return _SEGMENT_COLOUR_MAP.get(colour_idx, "#333333")


def _legend_details(
    segment: Mapping[str, object],
    title: str,
    legend_seen: Dict[str, bool],
) -> Tuple[str, str, bool]:
    series_label = segment.get("series")
    legend_label = str(series_label).strip() if series_label else _legend_group_name(title)
    series_id = str(segment.get("series_id") or legend_label)
    show = not legend_seen.get(series_id, False)
    legend_seen[series_id] = True
    return legend_label, series_id, show


def _segment_mode_and_markers(
    segment: Mapping[str, object],
    x_vals: List[float],
    y_vals: List[float],
    colour: str,
) -> Tuple[str, Optional[dict]]:
    arrow = segment.get("arrow")
    if arrow not in {ArrowHead.END.value, ArrowHead.START.value} or len(x_vals) < 2:
        return "lines", None
    tip_idx, ref_idx = _arrow_indices(arrow, len(x_vals))
    dx = x_vals[tip_idx] - x_vals[ref_idx]
    dy = y_vals[tip_idx] - y_vals[ref_idx]
    angle = math.degrees(math.atan2(dy, dx))
    marker = {
        "symbol": ["triangle-right"] * len(x_vals),
        "size": [12 if i == tip_idx else 0.0 for i in range(len(x_vals))],
        "angle": [angle if i == tip_idx else 0.0 for i in range(len(x_vals))],
        "color": colour,
        "line": {"width": 0},
    }
    return "lines+markers", marker


def _arrow_indices(arrow: str, length: int) -> Tuple[int, int]:
    if arrow == ArrowHead.START.value:
        return 0, 1
    return length - 1, length - 2


def _line_style(segment: Mapping[str, object], colour: str) -> dict:
    style = {"color": colour, "width": 2}
    if segment.get("is_vertical") and segment.get("is_utility_stream"):
        style["dash"] = "dash"
    return style


def _hover_template(segment: Mapping[str, object], title: str, legend_label: str) -> str:
    descriptor = segment.get("series_description") or legend_label or title
    return (
        f"{descriptor}<br>"
        "Heat Flow / kW: %{x}<br>"
        "Temperature / °C: %{y}<extra></extra>"
    )


def _apply_default_layout(fig: go.Figure) -> None:
    fig.update_layout(
        xaxis_title="Heat Flow / kW",
        yaxis_title="Temperature / \N{DEGREE SIGN}C",
        template="plotly_white",
        hovermode="closest",
        legend={"title": "", "orientation": "h", "yanchor": "bottom", "y": 1.02},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    fig.update_yaxes(rangemode="tozero")


def _extract_segment_xy(segment: Mapping[str, object]) -> tuple[List[float], List[float]]:
    """Return x/y coordinate lists for a graph segment payload."""
    points = segment.get("data_points", []) or []
    x_vals = [point["x"] for point in points if "x" in point and "y" in point]
    y_vals = [point["y"] for point in points if "x" in point and "y" in point]
    return x_vals, y_vals


def _legend_group_name(title: str) -> str:
    """Return a legend label grouping sequential segments with incremented suffixes."""
    if not title:
        return "Segment"
    base, _, suffix = title.rpartition(" ")
    if suffix.isdigit() and base:
        return base
    return title
