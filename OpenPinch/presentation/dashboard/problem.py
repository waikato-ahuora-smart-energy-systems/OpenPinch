"""Dashboard presentation for one solved application problem."""

from __future__ import annotations

from typing import Any

from ...streamlit_webviewer.web_graphing import render_streamlit_dashboard


def render_problem_dashboard(
    zone,
    *,
    graph_data: dict[str, Any] | None,
    page_title: str | None,
    value_rounding: int,
) -> None:
    """Render one prepared zone through the optional dashboard adapter."""
    render_streamlit_dashboard(
        zone,
        graph_data=graph_data,
        page_title=page_title,
        value_rounding=value_rounding,
    )


__all__ = ["render_problem_dashboard"]
