"""Label placement and tooltip construction for network-grid diagrams."""

from __future__ import annotations

from typing import Any

from ._constants import _LABEL_BACKGROUND_COLOR
from .plotly import _plotly_marker_size
from .state import GridDiagramMatch


def add_label(renderer: Any, x: float, y: float, text: str, **kwargs: Any) -> None:
    """Place one label while separating duplicate annotation slots."""
    xytext = kwargs.pop("xytext", None)
    offset_key = xytext if xytext is not None else (0.0, 0.0)
    key = (
        round(x, 2),
        round(y, 2),
        round(offset_key[0], 2),
        round(offset_key[1], 2),
    )
    duplicate_count = renderer._label_slot_counts.get(key, 0)
    renderer._label_slot_counts[key] = duplicate_count + 1
    if duplicate_count:
        if xytext is None:
            y -= duplicate_count * 0.35
        else:
            xytext = (
                xytext[0],
                xytext[1] - duplicate_count * renderer.size_of_font,
            )
    kwargs.setdefault("zorder", 10)
    if xytext is None:
        renderer.ax.text(x, y, text, **kwargs)
        return
    renderer.ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        annotation_clip=False,
        clip_on=False,
        **kwargs,
    )


def add_duty_labels(renderer: Any) -> None:
    """Add recovery and utility duties to a rendered grid."""
    from .temperatures import (
        cold_utility_x,
        hot_utility_x,
        recovery_match_x_pair,
    )

    duty_y_offset = -duty_label_offset(renderer)
    for stage_index in range(renderer.S):
        for hot_index in range(renderer.I):
            for cold_index in range(renderer.J):
                if renderer.recovery_match[hot_index][cold_index][stage_index] != 1:
                    continue
                match = renderer.recovery_match_by_position[
                    (hot_index, cold_index, stage_index)
                ]
                if renderer.temperature_scaled:
                    _, duty_x = recovery_match_x_pair(renderer, match)
                else:
                    duty_x = renderer._staged_match_x(
                        hot_index,
                        cold_index,
                        stage_index,
                    )
                add_label(
                    renderer,
                    duty_x,
                    renderer.match_coords[(hot_index, cold_index, stage_index)][1],
                    f"{match.duty / 1000:.2f} MW",
                    fontsize=renderer.duty_font_size,
                    color="purple",
                    ha="center",
                    va="top",
                    xytext=(0, duty_y_offset),
                    bgcolor=_LABEL_BACKGROUND_COLOR,
                    borderpad=1,
                )

    for cold_index in range(renderer.J):
        match = renderer.hot_utility_by_cold_index.get(cold_index)
        if match is not None:
            add_label(
                renderer,
                hot_utility_x(renderer, match),
                renderer.cold_y_coords[-1 - cold_index],
                f"{match.duty / 1000:.2f} MW",
                fontsize=renderer.duty_font_size,
                color="purple",
                ha="center",
                va="top",
                xytext=(0, duty_y_offset),
                bgcolor=_LABEL_BACKGROUND_COLOR,
                borderpad=1,
            )

    for hot_index in range(renderer.I):
        match = renderer.cold_utility_by_hot_index.get(hot_index)
        if match is not None:
            add_label(
                renderer,
                cold_utility_x(renderer, match),
                renderer.hot_y_coords[-1 - hot_index],
                f"{match.duty / 1000:.2f} MW",
                fontsize=renderer.duty_font_size,
                color="purple",
                ha="center",
                va="top",
                xytext=(0, duty_y_offset),
                bgcolor=_LABEL_BACKGROUND_COLOR,
                borderpad=1,
            )


def configure_figure(renderer: Any) -> None:
    """Configure axes, stream names, dimensions, and margins."""
    from .temperatures import stream_endpoint_temperatures

    cold_stream_names = [f"C{index + 1}" for index in reversed(range(renderer.J))]
    hot_stream_names = [f"H{index + 1}" for index in reversed(range(renderer.I))]
    renderer.ax.set_yticks(
        renderer.hot_y_coords + renderer.cold_y_coords,
        hot_stream_names + cold_stream_names,
    )
    stream_label_offset = -font_points(renderer, renderer.size_of_font, 3.7)
    for name, y in zip(hot_stream_names, renderer.hot_y_coords, strict=True):
        add_label(
            renderer,
            renderer.x_start,
            y,
            name,
            fontsize=renderer.size_of_font,
            ha="right",
            va="center_baseline",
            xytext=(stream_label_offset, 0),
        )
    for name, y in zip(cold_stream_names, renderer.cold_y_coords, strict=True):
        add_label(
            renderer,
            renderer.x_start,
            y,
            name,
            fontsize=renderer.size_of_font,
            ha="right",
            va="center_baseline",
            xytext=(stream_label_offset, 0),
        )
    renderer.ax.set_xticks([])
    renderer.ax.set_xlim(renderer.x_start - 0.02, renderer.x_finish + 0.02)
    left_margin, right_margin = side_margins(
        renderer,
        stream_endpoint_temperatures(renderer.grid_model),
    )
    renderer.fig.update_layout(
        width=renderer.figure_width,
        height=renderer.figure_height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin={"l": left_margin, "r": right_margin, "t": 25, "b": 25},
    )
    renderer.fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    renderer.fig.update_yaxes(showgrid=False, zeroline=False, visible=False)


def side_margins(
    renderer: Any,
    temperatures: dict[str, dict[str, float]],
) -> tuple[int, int]:
    """Compute margins from visible endpoint and stream-label lengths."""
    endpoint_labels = [
        f"{temperature - 273.15:.0f} °C"
        for values in temperatures.values()
        for temperature in values.values()
    ]
    longest_endpoint = max((len(label) for label in endpoint_labels), default=0)
    stream_label_chars = max(
        (len(f"H{index + 1}") for index in range(renderer.I)),
        default=0,
    )
    stream_label_chars = max(
        stream_label_chars,
        max((len(f"C{index + 1}") for index in range(renderer.J)), default=0),
    )
    character_width = renderer.size_of_font * 0.55
    left = font_points(renderer, renderer.size_of_font, 3.7) + character_width * (
        stream_label_chars + longest_endpoint
    )
    right = font_points(renderer, renderer.temp_font_size, 1.2) + (
        renderer.temp_font_size * 0.55 * longest_endpoint
    )
    return max(70, int(left)), max(70, int(right))


def duty_label_offset(renderer: Any) -> float:
    """Return a marker-aware vertical offset for duty labels."""
    font_offset = font_points(renderer, renderer.duty_font_size, 0.5)
    marker_clearance = _plotly_marker_size(renderer.match_radius) * 0.5
    return max(font_offset, marker_clearance)


def temperature_label_offset(renderer: Any) -> float:
    """Return a line-aware vertical offset for temperature labels."""
    font_offset = font_points(renderer, renderer.temp_font_size, 0.1875)
    stream_clearance = renderer.stream_line_width * 0.5
    return max(font_offset, stream_clearance)


def font_points(renderer: Any, fontsize: float, multiplier: float = 1.0) -> float:
    """Scale a label offset against both font and stream widths."""
    return max(float(fontsize), renderer.stream_line_width) * multiplier


def register_stream_match_position(
    renderer: Any,
    stream: str,
    x: float,
    y: float,
    match: GridDiagramMatch,
) -> None:
    """Record a stream/exchanger intersection for later labels."""
    renderer._stream_match_positions.setdefault(stream, []).append((x, y, match))


def register_exchanger_artist(
    renderer: Any,
    artist: Any,
    match: GridDiagramMatch,
) -> None:
    """Attach a domain tooltip to one rendered exchanger trace."""
    tooltip = exchanger_tooltip(match)
    artist.openpinch_tooltip = tooltip
    if hasattr(renderer.ax, "update_line_hover"):
        renderer.ax.update_line_hover(artist, tooltip)
    renderer.exchanger_tooltips[artist] = tooltip


def short_stream_name(stream: str) -> str:
    """Return the final user-facing portion of a qualified stream name."""
    for separator in (".", "/", ":"):
        if separator in stream:
            stream = stream.rsplit(separator, maxsplit=1)[-1]
    return stream.strip() or stream


def exchanger_tooltip(match: GridDiagramMatch) -> str:
    """Build the hover label for one exchanger match."""
    exchanger = match.exchanger
    area = "n/a" if exchanger.area is None else f"{exchanger.area:.2f} m^2"
    stage = "" if match.stage is None else f" | stage {match.stage}"
    return (
        f"{short_stream_name(match.source_stream)} -> "
        f"{short_stream_name(match.sink_stream)}{stage}\n"
        f"Duty: {match.duty / 1000:.2f} MW\n"
        f"Area: {area}"
    )


__all__: list[str] = []
