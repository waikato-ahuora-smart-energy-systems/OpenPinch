"""Streamlit dashboard orchestration for solved OpenPinch zones."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

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
                <div class="op-title">{resolved_title}</div>
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
            f"<div class='op-utility-title'>{entry}</div>",
            unsafe_allow_html=True,
        )
        if utilities:
            cards = "".join(
                f'<div class="op-utility-card">'
                f'<div class="op-utility-name">{u.name}</div>'
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
                    st.markdown(
                        f"<div class='op-card-title'>{graph_names[idx]}</div>",
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
                background: linear-gradient(
                    180deg,
                    #f5f7fb 0%,
                    #eef2f7 60%,
                    #f8fafc 100%
                );
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

            section[data-testid="stSidebar"] hr {
                margin: 0.8rem 0;
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
                gap: 0.45rem;
                margin-top: 0.35rem;
            }

            .op-metric {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(148, 163, 184, 0.2);
                border-radius: 12px;
                padding: 0.45rem 0.6rem;
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
