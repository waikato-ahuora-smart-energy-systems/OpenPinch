"""Streamlit dashboard orchestration for solved OpenPinch zones."""

from __future__ import annotations

import html
from collections.abc import Iterator, Mapping
from importlib.resources import files

from ...analysis.graphs.service import get_output_graph_data
from ...domain.targets import BaseTargetModel
from ...domain.zone import Zone
from ..graphs.plotly import build_plotly_figure
from ..reporting.problem_table import problem_table_frame
from .dependencies import _require_streamlit
from .exports import render_table_export
from .state import _DashboardGraphSet


def _collect_targets(zone: Zone) -> dict[str, BaseTargetModel]:
    """Flattens all energy targets beneath ``zone`` keyed by their display name."""

    def _iter(current: Zone) -> Iterator[tuple[str, BaseTargetModel]]:
        for _, target in current.targets.items():
            if not target.reportable:
                continue
            yield target.name, target
        for subzone in current.subzones.values():
            yield from _iter(subzone)

    return dict(_iter(zone))


def render_streamlit_dashboard(
    zone: Zone,
    *,
    graph_data: Mapping[str, Mapping[str, object]] | None = None,
    page_title: str | None = None,
    value_rounding: int = 2,
) -> None:
    """Render a basic Streamlit dashboard for ``zone``."""
    st = _require_streamlit()

    st.set_page_config(
        page_title=page_title or f"{zone.name} Pinch Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _apply_dashboard_theme(st)

    resolved_title = page_title or f"{zone.name} Pinch Dashboard"

    st.markdown(
        f"""
        <div class="op-header">
            <div>
                <div class="op-title">{html.escape(resolved_title)}</div>
                <div class="op-subtitle">
                    Energy targeting summary with composite curve visualisation
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    targets = _collect_targets(zone)
    if not targets:
        st.warning("No targets available for the selected zone.")
        return

    if graph_data is None:
        graph_data = get_output_graph_data(zone)
    graph_sets = {
        name: _DashboardGraphSet.from_graph_data(graph_set_data)
        for name, graph_set_data in graph_data.items()
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
        "<div class='op-utility-title'>Overview</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"""
        <div class="op-metric-grid">
            <div class="op-metric">
                <div class="op-metric-label">Cold pinch</div>
                <div class="op-metric-value">
                    {target.cold_pinch:.1f}&nbsp;\N{DEGREE SIGN}C
                </div>
            </div>
            <div class="op-metric">
                <div class="op-metric-label">Hot pinch</div>
                <div class="op-metric-value">
                    {target.hot_pinch:.1f}&nbsp;\N{DEGREE SIGN}C
                </div>
            </div>
            <div class="op-metric">
                <div class="op-metric-label">Hot utility</div>
                <div class="op-metric-value">
                    {target.hot_utility_target:,.0f}&nbsp;kW
                </div>
            </div>
            <div class="op-metric">
                <div class="op-metric-label">Cold utility</div>
                <div class="op-metric-value">
                    {target.cold_utility_target:,.0f}&nbsp;kW
                </div>
            </div>
            <div class="op-metric">
                <div class="op-metric-label">Heat recovery</div>
                <div class="op-metric-value">
                    {target.heat_recovery_target:,.0f}&nbsp;kW
                </div>
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
        "Hot utilities": target.hot_utilities,
        "Cold utilities": target.cold_utilities,
    }
    for entry, utilities in ut_dict.items():
        st.sidebar.divider()
        st.sidebar.markdown(
            f"<div class='op-utility-title'>{html.escape(entry)}</div>",
            unsafe_allow_html=True,
        )
        if utilities:
            cards = "".join(
                f'<div class="op-utility-card">'
                f'<div class="op-utility-name">{html.escape(str(u.name))}</div>'
                f'<div class="op-utility-value">{u.heat_flow:,.0f}&nbsp;kW</div>'
                f"</div>"
                for u in utilities
            )
            st.sidebar.markdown(
                f"<div class='op-utility-grid'>{cards}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.markdown(
                '<div class="op-utility-grid">'
                '<div class="op-utility-card op-utility-empty">Not required</div>'
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
                    title = html.escape(graph_names[idx])
                    st.markdown(
                        f"<div class='op-card-title'>{title}</div>",
                        unsafe_allow_html=True,
                    )
                    figure = build_plotly_figure(graph)
                    st.plotly_chart(
                        figure,
                        use_container_width=True,
                        config={"displaylogo": False},
                    )

    with tabs[1]:
        pt_df = problem_table_frame(target.pt, round_decimals=value_rounding)
        if pt_df.empty:
            st.info("No shifted problem table data available.")
        else:
            st.badge(
                "Extended problem table based on shifted process temperatures. "
                "Note: interval delta values are shown with zeros at the top "
                "of the columns."
            )
            st.dataframe(pt_df, width="stretch")
            default_loc = (
                f"results/{selected_target_name.replace('/', '-')}_shifted.xlsx"
            )

            render_table_export(
                st,
                default_loc,
                dashboard_key=base_key,
                target_name=selected_target_name,
                frame=pt_df,
                table_kind="shifted",
            )

    with tabs[2]:
        pt_real_df = problem_table_frame(
            getattr(target, "pt_real", None),
            round_decimals=value_rounding,
        )
        if pt_real_df.empty:
            st.info("No real temperature Problem Table data available.")
        else:
            st.badge(
                "Extended problem table based on real process temperatures. "
                "Note: interval delta values are shown with zeros at the top "
                "of the columns."
            )
            st.dataframe(pt_real_df, width="stretch")
            default_loc = f"results/{selected_target_name.replace('/', '-')}_real.xlsx"

            render_table_export(
                st,
                default_loc,
                dashboard_key=base_key,
                target_name=selected_target_name,
                frame=pt_real_df,
                table_kind="real",
            )


def _apply_dashboard_theme(st) -> None:
    st.markdown(f"<style>\n{_dashboard_css()}</style>", unsafe_allow_html=True)


def _dashboard_css() -> str:
    """Return the packaged dashboard stylesheet shipped beside this module."""
    return files(__package__).joinpath("dashboard.css").read_text(encoding="utf-8")
