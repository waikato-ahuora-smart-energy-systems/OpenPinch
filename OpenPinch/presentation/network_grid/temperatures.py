"""Temperature extraction, scaling, and labels for network-grid diagrams."""

from __future__ import annotations

from typing import Any

from ...domain.enums import HeatExchangerKind
from ._constants import _LABEL_BACKGROUND_COLOR
from .labels import add_label, font_points, temperature_label_offset
from .plotly import _SplitGroup
from .state import GridDiagramMatch, HeatExchangerNetworkGridModel


def add_temperature_labels(renderer: Any) -> None:
    """Add endpoint and intermediate stream temperatures to a grid."""
    format_string = r"{:.0f} $^\circ$C"
    temperatures = stream_endpoint_temperatures(renderer.grid_model)
    endpoint_x_offset = font_points(renderer, renderer.temp_font_size, 0.7)
    intermediate_y_offset = temperature_label_offset(renderer)

    for cold_index, stream in enumerate(renderer.grid_model.cold_streams):
        y = renderer.cold_y_coords[-1 - cold_index]
        supply_x, target_x = cold_stream_x_bounds(renderer, stream)
        cold_in = temperatures["cold_in"].get(stream)
        if cold_in is not None:
            add_label(
                renderer,
                supply_x,
                y,
                format_string.format(cold_in - 273.15),
                fontsize=renderer.temp_font_size,
                ha="left",
                va="center_baseline",
                xytext=(endpoint_x_offset, 0),
                bgcolor=_LABEL_BACKGROUND_COLOR,
                borderpad=1,
            )
        cold_out = temperatures["cold_out"].get(stream)
        if cold_out is not None:
            add_label(
                renderer,
                target_x,
                y,
                format_string.format(cold_out - 273.15),
                fontsize=renderer.temp_font_size,
                ha="right",
                va="center_baseline",
                xytext=(-endpoint_x_offset, 0),
                bgcolor=_LABEL_BACKGROUND_COLOR,
                borderpad=1,
            )
        add_intermediate_temperature_labels(
            renderer,
            stream,
            y,
            role="cold",
            xytext=(0, intermediate_y_offset),
            format_string=format_string,
        )

    for hot_index, stream in enumerate(renderer.grid_model.hot_streams):
        y = renderer.hot_y_coords[-1 - hot_index]
        supply_x, target_x = hot_stream_x_bounds(renderer, stream)
        hot_in = temperatures["hot_in"].get(stream)
        if hot_in is not None:
            add_label(
                renderer,
                supply_x,
                y,
                format_string.format(hot_in - 273.15),
                fontsize=renderer.temp_font_size,
                ha="right",
                va="center_baseline",
                xytext=(-endpoint_x_offset, 0),
                bgcolor=_LABEL_BACKGROUND_COLOR,
                borderpad=1,
            )
        hot_out = temperatures["hot_out"].get(stream)
        if hot_out is not None:
            add_label(
                renderer,
                target_x,
                y,
                format_string.format(hot_out - 273.15),
                fontsize=renderer.temp_font_size,
                ha="left",
                va="center_baseline",
                xytext=(endpoint_x_offset, 0),
                bgcolor=_LABEL_BACKGROUND_COLOR,
                borderpad=1,
            )
        add_intermediate_temperature_labels(
            renderer,
            stream,
            y,
            role="hot",
            xytext=(0, intermediate_y_offset),
            format_string=format_string,
        )


def add_intermediate_temperature_labels(
    renderer: Any,
    stream: str,
    y: float,
    *,
    role: str,
    xytext: tuple[float, float],
    format_string: str,
) -> None:
    """Label temperatures between adjacent exchanger matches."""
    positions = renderer._stream_match_positions.get(stream, [])
    if role == "hot":
        candidates = [
            (x, node_y, match, match.state.source_inlet_temperature)
            for x, node_y, match in positions
            if match.source_stream == stream
            and match.exchanger.kind
            in (HeatExchangerKind.RECOVERY, HeatExchangerKind.COLD_UTILITY)
            and match.state.source_inlet_temperature is not None
        ]
        candidates.sort(key=lambda item: item[0])
    else:
        candidates = [
            (x, node_y, match, match.state.sink_inlet_temperature)
            for x, node_y, match in positions
            if match.sink_stream == stream
            and match.exchanger.kind
            in (HeatExchangerKind.RECOVERY, HeatExchangerKind.HOT_UTILITY)
            and match.state.sink_inlet_temperature is not None
        ]
        candidates.sort(key=lambda item: item[0], reverse=True)

    segments: list[tuple[float, float, float, str]] = []
    for current, next_match in zip(candidates, candidates[1:], strict=False):
        if same_split_group(renderer, stream, current[0], next_match[0]):
            continue
        temperature = next_match[3]
        node_y = current[1] if current[1] == next_match[1] else y
        x_start, x_end = split_adjusted_temperature_segment(
            renderer,
            stream,
            current[0],
            next_match[0],
        )
        segments.append(
            (
                x_start,
                x_end,
                node_y,
                format_string.format(temperature - 273.15),
            )
        )

    for x_start, x_end, node_y, label in merge_temperature_label_segments(
        segments,
        stream_y=y,
    ):
        add_label(
            renderer,
            (x_start + x_end) / 2,
            node_y,
            label,
            fontsize=renderer.temp_font_size,
            ha="center",
            va="bottom",
            xytext=xytext,
            bgcolor=_LABEL_BACKGROUND_COLOR,
            borderpad=1,
        )


def hot_stream_x_bounds(renderer: Any, stream: str) -> tuple[float, float]:
    """Map one hot stream's endpoints onto the horizontal axis."""
    if not renderer.temperature_scaled:
        return renderer.x_start, renderer.x_finish
    temperatures = stream_endpoint_temperatures(renderer.grid_model)
    return (
        x_from_temperature(renderer, temperatures["hot_in"].get(stream)),
        x_from_temperature(renderer, temperatures["hot_out"].get(stream)),
    )


def cold_stream_x_bounds(renderer: Any, stream: str) -> tuple[float, float]:
    """Map one cold stream's endpoints onto the horizontal axis."""
    if not renderer.temperature_scaled:
        return renderer.x_finish, renderer.x_start
    temperatures = stream_endpoint_temperatures(renderer.grid_model)
    return (
        x_from_temperature(renderer, temperatures["cold_in"].get(stream)),
        x_from_temperature(renderer, temperatures["cold_out"].get(stream)),
    )


def recovery_match_x_pair(
    renderer: Any,
    match: GridDiagramMatch,
) -> tuple[float, float]:
    """Map source and sink midpoint temperatures for one recovery match."""
    if not renderer.temperature_scaled:
        return 0.0, 0.0
    return (
        x_from_temperature(
            renderer,
            midpoint_temperature(
                match.state.source_inlet_temperature,
                match.state.source_outlet_temperature,
            ),
        ),
        x_from_temperature(
            renderer,
            midpoint_temperature(
                match.state.sink_inlet_temperature,
                match.state.sink_outlet_temperature,
            ),
        ),
    )


def hot_utility_x(renderer: Any, match: GridDiagramMatch) -> float:
    """Return the x-position for a hot-utility match."""
    if renderer.temperature_scaled:
        return x_from_temperature(
            renderer,
            midpoint_temperature(
                match.state.sink_inlet_temperature,
                match.state.sink_outlet_temperature,
            ),
        )
    return renderer.x_start + (renderer.stage_start - renderer.x_start) / 2


def cold_utility_x(renderer: Any, match: GridDiagramMatch) -> float:
    """Return the x-position for a cold-utility match."""
    if renderer.temperature_scaled:
        return x_from_temperature(
            renderer,
            midpoint_temperature(
                match.state.source_inlet_temperature,
                match.state.source_outlet_temperature,
            ),
        )
    return (
        renderer.stage_boundaries[-1]
        + (renderer.x_finish - renderer.stage_boundaries[-1]) / 2
    )


def x_from_temperature(renderer: Any, temperature: float | None) -> float:
    """Map an absolute temperature to the grid's horizontal coordinates."""
    if not renderer.temperature_scaled or temperature is None:
        return renderer.x_start
    minimum, maximum = renderer.temperature_scale
    if maximum <= minimum:
        return (renderer.x_start + renderer.x_finish) / 2
    return renderer.x_start + ((maximum - temperature) / (maximum - minimum)) * (
        renderer.x_finish - renderer.x_start
    )


def stream_endpoint_temperatures(
    grid_model: HeatExchangerNetworkGridModel,
) -> dict[str, dict[str, float]]:
    """Collect process-stream endpoint extrema from active matches."""
    hot_in: dict[str, float] = {}
    hot_out: dict[str, float] = {}
    cold_in: dict[str, float] = {}
    cold_out: dict[str, float] = {}

    for match in grid_model.recovery_matches:
        state = match.state
        if state.source_inlet_temperature is not None:
            hot_in[match.source_stream] = max(
                hot_in.get(match.source_stream, state.source_inlet_temperature),
                state.source_inlet_temperature,
            )
        if state.source_outlet_temperature is not None:
            hot_out[match.source_stream] = min(
                hot_out.get(match.source_stream, state.source_outlet_temperature),
                state.source_outlet_temperature,
            )
        if state.sink_inlet_temperature is not None:
            cold_in[match.sink_stream] = min(
                cold_in.get(match.sink_stream, state.sink_inlet_temperature),
                state.sink_inlet_temperature,
            )
        if state.sink_outlet_temperature is not None:
            cold_out[match.sink_stream] = max(
                cold_out.get(match.sink_stream, state.sink_outlet_temperature),
                state.sink_outlet_temperature,
            )
    for match in grid_model.hot_utility_matches:
        state = match.state
        if state.sink_inlet_temperature is not None:
            cold_in[match.sink_stream] = min(
                cold_in.get(match.sink_stream, state.sink_inlet_temperature),
                state.sink_inlet_temperature,
            )
        if state.sink_outlet_temperature is not None:
            cold_out[match.sink_stream] = max(
                cold_out.get(match.sink_stream, state.sink_outlet_temperature),
                state.sink_outlet_temperature,
            )
    for match in grid_model.cold_utility_matches:
        state = match.state
        if state.source_inlet_temperature is not None:
            hot_in[match.source_stream] = max(
                hot_in.get(match.source_stream, state.source_inlet_temperature),
                state.source_inlet_temperature,
            )
        if state.source_outlet_temperature is not None:
            hot_out[match.source_stream] = min(
                hot_out.get(match.source_stream, state.source_outlet_temperature),
                state.source_outlet_temperature,
            )
    return {
        "hot_in": hot_in,
        "hot_out": hot_out,
        "cold_in": cold_in,
        "cold_out": cold_out,
    }


def temperature_scale(
    grid_model: HeatExchangerNetworkGridModel,
) -> tuple[float, float]:
    """Return the minimum and maximum active exchanger temperatures."""
    temperatures: list[float] = []
    for match in grid_model.recovery_matches:
        state = match.state
        temperatures.extend(
            temperature
            for temperature in (
                state.source_inlet_temperature,
                state.source_outlet_temperature,
                state.sink_inlet_temperature,
                state.sink_outlet_temperature,
            )
            if temperature is not None
        )
    for match in grid_model.hot_utility_matches:
        state = match.state
        temperatures.extend(
            temperature
            for temperature in (
                state.sink_inlet_temperature,
                state.sink_outlet_temperature,
            )
            if temperature is not None
        )
    for match in grid_model.cold_utility_matches:
        state = match.state
        temperatures.extend(
            temperature
            for temperature in (
                state.source_inlet_temperature,
                state.source_outlet_temperature,
            )
            if temperature is not None
        )
    if not temperatures:
        return 0.0, 1.0
    return min(temperatures), max(temperatures)


def midpoint_temperature(
    inlet_temperature: float | None,
    outlet_temperature: float | None,
) -> float | None:
    """Return the midpoint of the temperatures that are available."""
    temperatures = [
        temperature
        for temperature in (inlet_temperature, outlet_temperature)
        if temperature is not None
    ]
    if not temperatures:
        return None
    return sum(temperatures) / len(temperatures)


def same_split_group(
    renderer: Any,
    stream: str,
    x_start: float,
    x_end: float,
) -> bool:
    """Return whether two exchanger positions share one split group."""
    tolerance = 1e-9
    return any(
        group.start_x - tolerance <= x_start <= group.end_x + tolerance
        and group.start_x - tolerance <= x_end <= group.end_x + tolerance
        for group in renderer._split_groups_by_stream.get(stream, [])
    )


def split_adjusted_temperature_segment(
    renderer: Any,
    stream: str,
    x_start: float,
    x_end: float,
) -> tuple[float, float]:
    """Adjust a label span to split/reconnect connector positions."""
    adjusted_start = x_start
    adjusted_end = x_end
    for group in renderer._split_groups_by_stream.get(stream, []):
        adjusted_start = split_adjusted_temperature_x(adjusted_start, x_end, group)
        adjusted_end = split_adjusted_temperature_x(adjusted_end, x_start, group)
    return adjusted_start, adjusted_end


def merge_temperature_label_segments(
    segments: list[tuple[float, float, float, str]],
    *,
    stream_y: float,
) -> list[tuple[float, float, float, str]]:
    """Merge adjacent spans carrying the same temperature label."""
    if not segments:
        return []
    merged: list[tuple[float, float, float, str]] = []
    run_start, run_end, run_y, run_label = segments[0]
    run_y_values = [run_y]
    for segment_start, segment_end, segment_y, segment_label in segments[1:]:
        if segment_label == run_label:
            run_end = segment_end
            run_y_values.append(segment_y)
            continue
        merged.append(
            (
                run_start,
                run_end,
                run_y if all(y == run_y for y in run_y_values) else stream_y,
                run_label,
            )
        )
        run_start, run_end, run_y, run_label = (
            segment_start,
            segment_end,
            segment_y,
            segment_label,
        )
        run_y_values = [run_y]
    merged.append(
        (
            run_start,
            run_end,
            run_y if all(y == run_y for y in run_y_values) else stream_y,
            run_label,
        )
    )
    return merged


def split_adjusted_temperature_x(
    x_value: float,
    other_x: float,
    group: _SplitGroup,
) -> float:
    """Map a label endpoint from a split lane to its connector."""
    tolerance = 1e-9
    if not group.start_x - tolerance <= x_value <= group.end_x + tolerance:
        return x_value
    if other_x < group.start_x - tolerance:
        return group.left_connector_x
    if other_x > group.end_x + tolerance:
        return group.right_connector_x
    return x_value


__all__: list[str] = []
