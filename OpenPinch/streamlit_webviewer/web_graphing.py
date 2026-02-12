
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
        initial_sidebar_state="expanded",
    )

    _apply_dashboard_theme(st)

    st.markdown(
        f"""
        <div class="op-header">
            <div>
                <div class="op-title">{page_title or f"{zone.name} Pinch Dashboard"}</div>
                <div class="op-subtitle">Energy targeting summary with composite curve visualisation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        "Select zone",
        target_names,
        index=0 if target_names else None,
        key=f"target_select_{base_key}",
    )
    target = targets[selected_target_name]

    st.sidebar.divider()
    st.sidebar.write("Targets")
    st.sidebar.markdown(
        f"""
        <div class="op-metric-grid">
            <div class="op-metric">
                <div class="op-metric-label">Cold pinch</div>
                <div class="op-metric-value">{target.cold_pinch:.1f}&nbsp;\N{DEGREE SIGN}C</div>
            </div>
            <div class="op-metric">
                <div class="op-metric-label">Hot pinch</div>
                <div class="op-metric-value">{target.hot_pinch:.1f}&nbsp;\N{DEGREE SIGN}C</div>
            </div>
            <div class="op-metric">
                <div class="op-metric-label">Hot utility</div>
                <div class="op-metric-value">{target.hot_utility_target:,.0f}&nbsp;kW</div>
            </div>
            <div class="op-metric">
                <div class="op-metric-label">Cold utility</div>
                <div class="op-metric-value">{target.cold_utility_target:,.0f}&nbsp;kW</div>
            </div>
            <div class="op-metric">
                <div class="op-metric-label">Heat recovery</div>
                <div class="op-metric-value">{target.heat_recovery_target:,.0f}&nbsp;kW</div>
            </div>
            <div class="op-metric">
                <div class="op-metric-label">Degree of integration</div>
                <div class="op-metric-value">{target.degree_of_int:.0%}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    ut_dict = {
        "Hot utilities" : target.hot_utilities, 
        "Cold utilities" : target.cold_utilities,
    }
    for entry, utilities in ut_dict.items():
        st.sidebar.divider()
        st.sidebar.markdown(
            f"<div class='op-utility-title'>{entry}</div>",
            unsafe_allow_html=True,
        )
        if utilities:
            cards = "".join(
                f"<div class=\"op-utility-card\">"
                f"<div class=\"op-utility-name\">{u.name}</div>"
                f"<div class=\"op-utility-value\">{u.heat_flow:,.0f}&nbsp;kW</div>"
                f"</div>"
                for u in utilities
            )
            st.sidebar.markdown(
                f"<div class='op-utility-grid'>{cards}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                "<div class=\"op-utility-grid\">"
                "<div class=\"op-utility-card op-utility-empty\">Not required</div>"
                "</div>",
                unsafe_allow_html=True,
            )


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
                    st.markdown(f"<div class='op-card-title'>{graph_names[idx]}</div>", unsafe_allow_html=True)
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
        traces = _segment_trace(segment, graph, legend_seen)
        for trace in traces:
            fig.add_trace(trace)
    _apply_default_layout(fig)
    return fig


def _segment_trace(
    segment: Mapping[str, object],
    graph: Mapping[str, object],
    legend_seen: Dict[str, bool],
) -> List[go.Scatter]:
    x_vals, y_vals = _extract_segment_xy(segment)
    if not x_vals or not y_vals:
        return []
    title = segment.get("title") or graph.get("type") or "Segment"
    colour = _segment_colour(segment)
    legend_label, series_id, show = _legend_details(segment, title, legend_seen)
    arrow = segment.get("arrow")
    line_trace = go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="lines",
        name=legend_label,
        line=_line_style(segment, colour),
        hovertemplate=_hover_template(segment, title, legend_label),
        legendgroup=series_id,
        showlegend=show,
    )
    if arrow not in {ArrowHead.END.value, ArrowHead.START.value} or len(x_vals) < 2:
        return [line_trace]

    tip_idx, ref_idx = _arrow_indices(arrow, len(x_vals))
    dx = x_vals[tip_idx] - x_vals[ref_idx]
    dy = y_vals[tip_idx] - y_vals[ref_idx]
    length = math.hypot(dx, dy)
    if length == 0:
        return [line_trace]
    backoff = min(length * 0.02, length * 0.25)
    ux, uy = dx / length, dy / length
    if arrow == ArrowHead.END.value:
        mx = x_vals[tip_idx] - ux * backoff
        my = y_vals[tip_idx] - uy * backoff
    else:
        mx = x_vals[tip_idx] + ux * backoff
        my = y_vals[tip_idx] + uy * backoff
    angle = math.degrees(math.atan2(dy, dx))
    marker_trace = go.Scatter(
        x=[mx],
        y=[my],
        mode="markers",
        marker={
            "symbol": "triangle-right",
            "size": 12,
            "angle": angle,
            "color": colour,
            "line": {"width": 1.5, "color": colour},
        },
        hoverinfo="skip",
        legendgroup=series_id,
        showlegend=False,
    )
    return [line_trace, marker_trace]


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
        legend={
            "title": "Click to toggle",
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.06,
            "title_font": {"color": "#000000", "size": 13},
            "font": {"color": "#000000", "size": 12},
        },
        margin={"l": 50, "r": 28, "t": 64, "b": 48},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": "IBM Plex Sans, Inter, system-ui, sans-serif", "size": 13, "color": "#000000"},
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": "#000000"}},
    )
    fig.update_xaxes(
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.25)",
        zerolinecolor="rgba(148, 163, 184, 0.25)",
        zerolinewidth=1,
        ticks="outside",
        tickcolor="#000000",
        showline=True,
        linecolor="#000000",
        tickfont={"color": "#000000"},
        title_font={"color": "#000000"},
    )
    fig.update_yaxes(
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.2)",
        zerolinecolor="rgba(148, 163, 184, 0.2)",
        zerolinewidth=1,
        ticks="outside",
        tickcolor="#000000",
        showline=True,
        linecolor="#000000",
        tickfont={"color": "#000000"},
        title_font={"color": "#000000"},
    )


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


def _apply_dashboard_theme(st) -> None:
    st.markdown(
        """
        <style>
            :root {
                --op-bg: #f5f7fb;
                --op-card: #ffffff;
                --op-ink: #0f172a;
                --op-muted: #64748b;
                --op-border: rgba(148, 163, 184, 0.35);
                --op-accent: #0ea5a4;
                --op-accent-soft: rgba(14, 165, 164, 0.12);
                --op-select-text: #262730;
            }

            .stApp {
                background: linear-gradient(180deg, #f5f7fb 0%, #eef2f7 60%, #f8fafc 100%);
                color: var(--op-ink);
                font-family: "IBM Plex Sans", "Inter", system-ui, sans-serif;
            }

            section[data-testid="stSidebar"] {
                background-color: #0f172a;
                color: #f8fafc;
                border-right: 1px solid rgba(148, 163, 184, 0.2);
            }

            section[data-testid="stSidebar"] * {
                color: #e2e8f0;
            }

            section[data-testid="stSidebar"] label {
                color: #94a3b8 !important;
            }

            section[data-testid="stSidebar"] div[data-baseweb="select"] span {
                color: var(--op-select-text) !important;
            }

            section[data-testid="stSidebar"] div[data-baseweb="select"] input {
                color: var(--op-select-text) !important;
            }

            section[data-testid="stSidebar"] div[data-baseweb="select"] * {
                color: var(--op-select-text) !important;
            }

            div[data-baseweb="menu"] span {
                color: var(--op-select-text) !important;
            }

            .op-header {
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                padding: 0.5rem 0 1rem;
            }

            .op-title {
                font-size: 2rem;
                font-weight: 600;
                letter-spacing: -0.02em;
                color: var(--op-ink);
            }

            .op-subtitle {
                color: var(--op-muted);
                font-size: 0.95rem;
                margin-top: 0.2rem;
            }

            .op-metric-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.75rem;
                margin-top: 0.6rem;
            }

            .op-metric {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 12px;
                padding: 0.65rem 0.75rem;
            }

            .op-metric-label {
                font-size: 0.72rem;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: #94a3b8;
                margin-bottom: 0.3rem;
            }

            .op-metric-value {
                font-size: 1.1rem;
                font-weight: 600;
            }

            .op-card-title {
                font-size: 1rem;
                font-weight: 600;
                color: var(--op-ink);
                margin-bottom: 0.3rem;
                padding-left: 0.1rem;
            }

            .op-utility-title {
                font-size: 0.72rem;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: #94a3b8;
                margin-bottom: 0.45rem;
            }

            .op-utility-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.6rem;
            }

            .op-utility-card {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 12px;
                padding: 0.55rem 0.75rem;
            }

            .op-utility-name {
                font-size: 0.9rem;
                font-weight: 600;
                color: #e2e8f0;
            }

            .op-utility-value {
                font-size: 0.92rem;
                color: #cbd5f5;
            }

            .op-utility-empty {
                color: #94a3b8;
                text-align: center;
                font-size: 0.88rem;
            }

            div[data-testid="stPlotlyChart"] {
                background: var(--op-card);
                border: 1px solid var(--op-border);
                border-radius: 14px;
                padding: 0.75rem;
                box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
                overflow: hidden;
            }

            div[data-testid="stPlotlyChart"] > div {
                width: 100% !important;
            }

            .stTabs [role="tab"] {
                font-weight: 600;
                letter-spacing: 0.01em;
                color: var(--op-muted);
            }

            .stTabs [role="tab"][aria-selected="true"] {
                color: var(--op-ink);
                border-bottom: 2px solid var(--op-accent);
            }

            .stBadge {
                background-color: var(--op-accent-soft) !important;
                color: var(--op-ink) !important;
                border: 1px solid rgba(14, 165, 164, 0.3);
            }

            div[data-testid="stDataFrame"] {
                background: var(--op-card);
                border: 1px solid var(--op-border);
                border-radius: 12px;
                padding: 0.4rem;
            }

            input, textarea {
                border-radius: 10px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
