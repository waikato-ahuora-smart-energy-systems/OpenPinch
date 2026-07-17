"""Convert deterministic graph mappings into Plotly figures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...adapters.optional_dependencies import optional_dependency_error
from ...domain.enums import LineColour

_SEGMENT_COLOURS: dict[int, str] = {
    LineColour.HotS.value: "#e66e6e",
    LineColour.ColdS.value: "#5ca5d9",
    LineColour.HotU.value: "#C22323",
    LineColour.ColdU.value: "#244abd",
    LineColour.Other.value: "#7f7f7f",
    LineColour.Black.value: "#111111",
}


def build_plotly_figure(graph: Mapping[str, object]) -> Any:
    """Build a Plotly figure from one serialized OpenPinch graph."""
    plotly_go = _require_plotly()
    figure = plotly_go.Figure()
    legend_seen: dict[str, bool] = {}
    for segment in graph.get("segments", []):
        for trace in _segment_traces(segment, graph, legend_seen):
            figure.add_trace(trace)
    _apply_default_layout(figure)
    return figure


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            optional_dependency_error(
                package="Plotly",
                purpose="graph rendering",
                extras=("notebook", "dashboard"),
                docs="the graphing and exporting results guides",
            )
        ) from exc
    return go


def _segment_traces(
    segment: Mapping[str, object],
    graph: Mapping[str, object],
    legend_seen: dict[str, bool],
) -> list[Any]:
    x_values, y_values = _extract_segment_coordinates(segment)
    if not x_values or not y_values:
        return []

    title = segment.get("title") or graph.get("type") or "Segment"
    graph_type = graph.get("type")
    colour = _segment_colour(segment)
    legend_label, series_id, show_legend = _legend_details(
        segment,
        str(title),
        legend_seen,
    )

    if graph_type == "Site Utility Grand Composite Curve" and _is_vertical_segment(
        x_values
    ):
        colour = _SEGMENT_COLOURS[LineColour.Other.value]

    plotly_go = _require_plotly()
    return [
        plotly_go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name=legend_label,
            line=_line_style(segment, colour),
            hovertemplate=_hover_template(segment, str(title), legend_label),
            legendgroup=series_id,
            showlegend=show_legend,
        )
    ]


def _segment_colour(segment: Mapping[str, object]) -> str:
    if segment.get("is_vertical"):
        return _SEGMENT_COLOURS[LineColour.Black.value]
    return _SEGMENT_COLOURS.get(segment.get("colour"), "#333333")


def _is_vertical_segment(x_values: list[float], *, atol: float = 1e-9) -> bool:
    if len(x_values) < 2:
        return False
    first = x_values[0]
    return all(abs(value - first) <= atol for value in x_values[1:])


def _legend_details(
    segment: Mapping[str, object],
    title: str,
    legend_seen: dict[str, bool],
) -> tuple[str, str, bool]:
    series_label = segment.get("series")
    legend_label = (
        str(series_label).strip() if series_label else _legend_group_name(title)
    )
    series_id = str(segment.get("series_id") or legend_label)
    show_legend = not legend_seen.get(series_id, False)
    legend_seen[series_id] = True
    return legend_label, series_id, show_legend


def _line_style(segment: Mapping[str, object], colour: str) -> dict[str, object]:
    style: dict[str, object] = {"color": colour, "width": 2}
    if segment.get("is_vertical") and segment.get("is_utility_stream"):
        style["dash"] = "dash"
    return style


def _hover_template(
    segment: Mapping[str, object],
    title: str,
    legend_label: str,
) -> str:
    descriptor = segment.get("series_description") or legend_label or title
    return (
        f"{descriptor}<br>"
        "Heat Flow / kW: %{x}<br>"
        "Temperature / °C: %{y}<extra></extra>"
    )


def _apply_default_layout(figure: Any) -> None:
    figure.update_layout(
        width=720,
        height=540,
        autosize=False,
        xaxis_title="Heat Flow / kW",
        yaxis_title="Temperature / °C",
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
        font={
            "family": "IBM Plex Sans, Inter, system-ui, sans-serif",
            "size": 13,
            "color": "#000000",
        },
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": "#000000"}},
    )
    axis_style = {
        "rangemode": "tozero",
        "showgrid": True,
        "zeroline": True,
        "zerolinecolor": "rgba(15, 23, 42, 0.8)",
        "zerolinewidth": 1.25,
        "ticks": "outside",
        "tickcolor": "#000000",
        "showline": True,
        "linecolor": "#000000",
        "tickfont": {"color": "#000000"},
        "title_font": {"color": "#000000"},
    }
    figure.update_xaxes(
        **axis_style,
        gridcolor="rgba(148, 163, 184, 0.25)",
    )
    figure.update_yaxes(
        **axis_style,
        gridcolor="rgba(148, 163, 184, 0.2)",
    )


def _extract_segment_coordinates(
    segment: Mapping[str, object],
) -> tuple[list[float], list[float]]:
    points = segment.get("data_points", []) or []
    x_values = [point["x"] for point in points if "x" in point and "y" in point]
    y_values = [point["y"] for point in points if "x" in point and "y" in point]
    return x_values, y_values


def _legend_group_name(title: str) -> str:
    if not title:
        return "Segment"
    base, _, suffix = title.rpartition(" ")
    if suffix.isdigit() and base:
        return base
    return title


__all__ = ["build_plotly_figure"]
