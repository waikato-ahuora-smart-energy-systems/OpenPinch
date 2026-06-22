"""Plotly renderer for heat exchanger network grid diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...lib.enums import HeatExchangerKind
from .constants import (
    _COLD_STREAM_COLOR,
    _COLD_UTILITY_COLOR,
    _DEFAULT_STREAM_LINE_WIDTH,
    _GROUP_GAP_LANES,
    _HOT_STREAM_COLOR,
    _HOT_UTILITY_COLOR,
    _LABEL_BACKGROUND_COLOR,
    _LANE_HEIGHT_PX,
    _LAYOUT_X_MAX,
    _LAYOUT_X_MIN,
    _MARKER_ROW_PITCH_RATIO,
    _MATCH_LINE_TO_STREAM_LINE_WIDTH_RATIO,
    _MAX_MARKER_RADIUS,
    _MIN_FIGURE_HEIGHT_PX,
    _MIN_FIGURE_WIDTH_PX,
    _MIN_MARKER_RADIUS,
    _RECOVERY_MATCH_COLOR,
    _STREAM_LINE_TO_MARKER_RADIUS_RATIO,
)
from .models import GridDiagramMatch, HeatExchangerNetworkGridModel


@dataclass(frozen=True)
class _SplitGroup:
    start_x: float
    end_x: float
    left_connector_x: float
    right_connector_x: float


class _PlotlyLine:
    def __init__(self, xdata, ydata, **kwargs):
        self.xdata = tuple(xdata)
        self.ydata = tuple(ydata)
        self.kwargs = kwargs
        self.openpinch_tooltip: str | None = None
        self._plotly_trace_indices: list[int] = []

    def get_xdata(self):
        return self.xdata

    def get_ydata(self):
        return self.ydata


class _PlotlyText:
    def __init__(
        self,
        text: str,
        *,
        xy: tuple[float, float],
        position: tuple[float, float],
        ha: str,
        va: str,
    ):
        self.xy = xy
        self._position = position
        self._text = text
        self._ha = ha
        self._va = va

    def get_text(self) -> str:
        return self._text

    def get_position(self) -> tuple[float, float]:
        return self._position

    def get_ha(self) -> str:
        return self._ha

    def get_va(self) -> str:
        return self._va


class _PlotlyAxes:
    def __init__(self, fig: Any, graph_objects: Any):
        self.fig = fig
        self.go = graph_objects
        self.lines: list[_PlotlyLine] = []
        self.texts: list[_PlotlyText] = []
        self.stream_bounds: list[tuple[float, float]] = []
        self._yticks: list[float] = []

    def arrow(
        self,
        x: float,
        y: float,
        dx: float,
        dy: float,
        *,
        lw: float,
        color: str,
        **_kwargs,
    ) -> None:
        x_end = x + dx
        y_end = y + dy
        self.stream_bounds.append((min(x, x_end), max(x, x_end)))
        self.fig.add_trace(
            self.go.Scatter(
                x=[x, x_end],
                y=[y, y_end],
                mode="lines",
                line={"color": color, "width": lw},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        symbol = "triangle-right" if dx >= 0 else "triangle-left"
        self.fig.add_trace(
            self.go.Scatter(
                x=[x_end],
                y=[y_end],
                mode="markers",
                marker={"symbol": symbol, "size": max(16, lw * 3.2), "color": color},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    def add_line(self, line: _PlotlyLine) -> None:
        self.lines.append(line)
        mode = "lines+markers" if line.kwargs.get("marker") else "lines"
        marker = None
        if line.kwargs.get("marker"):
            marker = {
                "size": _plotly_marker_size(line.kwargs.get("markersize", 18)),
                "color": line.kwargs.get("markerfacecolor", line.kwargs.get("color")),
            }
        trace = self.go.Scatter(
            x=list(line.xdata),
            y=list(line.ydata),
            mode=mode,
            line={
                "color": line.kwargs.get("color", "black"),
                "width": line.kwargs.get("lw", 2),
                "dash": "dash" if line.kwargs.get("linestyle") == "dashed" else "solid",
            },
            marker=marker,
            hoverinfo="skip",
            showlegend=False,
        )
        self.fig.add_trace(trace)
        line._plotly_trace_indices.append(len(self.fig.data) - 1)

    def update_line_hover(self, line: _PlotlyLine, tooltip: str) -> None:
        for index in line._plotly_trace_indices:
            self.fig.data[index].update(
                hovertemplate=tooltip.replace("\n", "<br>") + "<extra></extra>",
                hoverinfo=None,
            )

    def text(self, x: float, y: float, text: str, **kwargs) -> _PlotlyText:
        return self.annotate(text, xy=(x, y), xytext=(0, 0), **kwargs)

    def annotate(
        self,
        text: str,
        *,
        xy: tuple[float, float],
        xytext: tuple[float, float] = (0, 0),
        **kwargs,
    ) -> _PlotlyText:
        ha = kwargs.get("ha", kwargs.get("horizontalalignment", "left"))
        va = kwargs.get("va", kwargs.get("verticalalignment", "baseline"))
        label = _PlotlyText(text, xy=xy, position=xytext, ha=ha, va=va)
        self.texts.append(label)
        font = {"color": kwargs.get("color", "black")}
        if kwargs.get("fontsize") is not None:
            font["size"] = kwargs["fontsize"]
        annotation = {
            "x": xy[0],
            "y": xy[1],
            "text": text.replace("$^\\circ$", "°"),
            "showarrow": False,
            "xshift": xytext[0],
            "yshift": xytext[1],
            "xanchor": _plotly_xanchor(ha),
            "yanchor": _plotly_yanchor(va),
            "font": font,
        }
        if kwargs.get("bgcolor") is not None:
            annotation["bgcolor"] = kwargs["bgcolor"]
        if kwargs.get("borderpad") is not None:
            annotation["borderpad"] = kwargs["borderpad"]
        self.fig.add_annotation(**annotation)
        return label

    def set_yticks(self, ticks, labels) -> None:
        self._yticks = list(ticks)
        self.fig.update_yaxes(tickmode="array", tickvals=self._yticks, ticktext=labels)

    def get_yticks(self):
        return self._yticks

    def set_xticks(self, _ticks) -> None:
        self.fig.update_xaxes(showticklabels=False, ticks="")

    def set_xlim(self, left: float, right: float) -> None:
        self.fig.update_xaxes(range=[left, right])


def _plotly_marker_size(markersize: float) -> float:
    return max(12.0, min(float(markersize) * 0.42, 30.0))


def _plotly_xanchor(ha: str) -> str:
    return {"left": "left", "right": "right", "center": "center"}.get(ha, "left")


def _plotly_yanchor(va: str) -> str:
    return {
        "top": "top",
        "bottom": "bottom",
        "center": "middle",
        "center_baseline": "middle",
        "baseline": "middle",
    }.get(va, "middle")


class _PlotlyGridRenderer:
    """Plotly renderer for heat exchanger network grid diagrams."""

    def __init__(
        self,
        grid_model: HeatExchangerNetworkGridModel,
        *,
        graph_objects: Any,
        stream_line_width: float,
        temperature_scaled: bool,
    ):
        self.grid_model = grid_model
        self.network = grid_model.network
        self.go = graph_objects
        self.I = len(grid_model.hot_streams)
        self.J = len(grid_model.cold_streams)
        self.S = len(grid_model.stages)
        _validate_stream_line_width(stream_line_width)
        # Keep the public argument for compatibility; the rendered width is derived
        # from marker radius so the diagram scales consistently with lane geometry.
        self.stream_line_width = _DEFAULT_STREAM_LINE_WIDTH
        self.temperature_scaled = bool(temperature_scaled)

        self.hot_index = {
            stream: index for index, stream in enumerate(grid_model.hot_streams)
        }
        self.cold_index = {
            stream: index for index, stream in enumerate(grid_model.cold_streams)
        }
        self.stage_index = {
            stage: index for index, stage in enumerate(grid_model.stages)
        }
        self.recovery_match_by_position = {
            (
                self.hot_index[match.source_stream],
                self.cold_index[match.sink_stream],
                self.stage_index[match.stage],
            ): match
            for match in grid_model.recovery_matches
            if match.stage is not None and match.stage in self.stage_index
        }
        self.hot_utility_by_cold_index = {
            self.cold_index[match.sink_stream]: match
            for match in grid_model.hot_utility_matches
        }
        self.cold_utility_by_hot_index = {
            self.hot_index[match.source_stream]: match
            for match in grid_model.cold_utility_matches
        }

        self.x_start = _LAYOUT_X_MIN
        self.size_of_font = 18
        self.duty_font_size = 14
        self.temp_font_size = 14
        self.draw_stages = False

        self.hot_y_coords: list[float] = []
        self.cold_y_coords: list[float] = []
        self.stage_boundaries: list[float] = []
        self.match_coords: dict[tuple[int, int, int], list[float]] = {}
        self.match_x_by_position: dict[tuple[int, int, int], float] = {}
        self._stream_match_positions: dict[
            str,
            list[tuple[float, float, GridDiagramMatch]],
        ] = {}
        self.exchanger_tooltips: dict[Any, str] = {}
        self._label_slot_counts: dict[tuple[float, float, float, float], int] = {}
        self._split_groups_by_stream: dict[str, list[_SplitGroup]] = {}
        self.stage_recovery_positions: dict[int, list[tuple[int, int]]] = {}

        self._process_match_existence()
        self.stage_recovery_positions = self._build_stage_recovery_positions()
        self._calculate_spacing()
        self.temperature_scale = _temperature_scale(self.grid_model)
        self.figure_width, self.figure_height = self._figure_size()
        self.match_radius = self._marker_radius()
        self.stream_line_width = self._stream_line_width()
        self.match_line_width = self._match_line_width()
        self.fig = self.go.Figure()
        self.ax = _PlotlyAxes(self.fig, self.go)
        self.draw_streams()
        self.draw_branches()
        self.draw_recovery_matches()
        self.draw_utility_match()
        self.add_duties()
        self.add_temps()
        self.plot_setup()

    def _process_match_existence(self) -> None:
        self.recovery_match = [
            [
                [
                    1 if (i, j, k) in self.recovery_match_by_position else 0
                    for k in range(self.S)
                ]
                for j in range(self.J)
            ]
            for i in range(self.I)
        ]
        self.CU_matches = [
            1 if i in self.cold_utility_by_hot_index else 0 for i in range(self.I)
        ]
        self.HU_matches = [
            1 if j in self.hot_utility_by_cold_index else 0 for j in range(self.J)
        ]

    def _calculate_spacing(self) -> None:
        self.stage_count = [
            sum(
                1 if self.recovery_match[i][j][k] > 0 else 0
                for j in range(self.J)
                for i in range(self.I)
            )
            for k in range(self.S)
        ]
        self.hot_branch_count = [
            [
                sum(1 if self.recovery_match[i][j][k] > 0 else 0 for j in range(self.J))
                for k in range(self.S)
            ]
            for i in range(self.I)
        ]
        self.cold_branch_count = [
            [
                sum(1 if self.recovery_match[i][j][k] > 0 else 0 for i in range(self.I))
                for k in range(self.S)
            ]
            for j in range(self.J)
        ]
        self.lane_count = self._lane_count()
        self.stream_spacing = 1.0 / (self.lane_count + 1 + _GROUP_GAP_LANES)
        self.branch_spacing = self.stream_spacing
        self.x_finish = _LAYOUT_X_MAX
        self.utility_length = 1.0 / (2 * (self.S + 1))
        self.stage_start = self.x_start + self.utility_length
        self.stage_finish = self.x_finish - self.utility_length
        self.recovery_length = self.stage_finish - self.stage_start
        total_matches = max(1, sum(self.stage_count))
        self.match_spacing = self.recovery_length / total_matches
        self.match_start = self.match_spacing

    def _lane_count(self) -> int:
        lane_count = 0
        for j in range(self.J):
            lane_count += max(
                [self.cold_branch_count[-1 - j][k] for k in range(self.S)] or [1]
            )
        for i in range(self.I):
            lane_count += max(
                [self.hot_branch_count[-1 - i][k] for k in range(self.S)] or [1]
            )
        return max(1, lane_count)

    def _figure_size(self) -> tuple[int, int]:
        recovery_count = len(self.grid_model.recovery_matches)
        width = max(
            _MIN_FIGURE_WIDTH_PX,
            int(520 + 80 * max(1, self.S) + 52 * max(1, recovery_count)),
        )
        height = max(
            _MIN_FIGURE_HEIGHT_PX,
            int(180 + _LANE_HEIGHT_PX * self.lane_count),
        )
        return width, height

    def _marker_radius(self) -> float:
        row_pitch = self.figure_height * self.stream_spacing
        return max(
            _MIN_MARKER_RADIUS,
            min(row_pitch * _MARKER_ROW_PITCH_RATIO, _MAX_MARKER_RADIUS),
        )

    def _stream_line_width(self) -> float:
        return max(
            1.0,
            self.match_radius * _STREAM_LINE_TO_MARKER_RADIUS_RATIO,
        )

    def _match_line_width(self) -> float:
        return max(
            1.0,
            self.stream_line_width * _MATCH_LINE_TO_STREAM_LINE_WIDTH_RATIO,
        )

    def draw_streams(self) -> None:
        y_cursor = self.stream_spacing
        for j in range(self.J):
            stream = self.grid_model.cold_streams[-1 - j]
            branches = max(
                [self.cold_branch_count[-1 - j][k] for k in range(self.S)] or [1]
            )
            if branches == 0:
                branches = 1
            y_cursor += self.stream_spacing * branches
            x_tail, x_head = self._cold_stream_x_bounds(stream)
            self.ax.arrow(
                x_tail,
                y_cursor,
                x_head - x_tail,
                0,
                head_width=0.25,
                head_length=self.recovery_length * 0.02,
                lw=self.stream_line_width,
                length_includes_head=True,
                color=_COLD_STREAM_COLOR,
            )
            self.cold_y_coords.append(y_cursor)

        y_cursor = (
            self.cold_y_coords[-1] + self.stream_spacing * _GROUP_GAP_LANES
            if self.cold_y_coords
            else self.stream_spacing
        )

        for i in range(self.I):
            stream = self.grid_model.hot_streams[-1 - i]
            branches = max(
                max([self.hot_branch_count[-1 - i][k] for k in range(self.S)] or [1]),
                1,
            )
            if branches == 0:
                branches = 1
            y_cursor += self.stream_spacing * branches
            x_tail, x_head = self._hot_stream_x_bounds(stream)
            self.ax.arrow(
                x_tail,
                y_cursor,
                x_head - x_tail,
                0,
                head_width=0.25,
                head_length=self.recovery_length * 0.02,
                lw=self.stream_line_width,
                length_includes_head=True,
                color=_HOT_STREAM_COLOR,
            )
            self.hot_y_coords.append(y_cursor)

        boundary = self.stage_start
        top_y = (
            self.hot_y_coords[-1] + self.stream_spacing * 0.5
            if self.hot_y_coords
            else y_cursor + self.stream_spacing * 0.5
        )
        bottom_y = (
            self.cold_y_coords[0] - self.stream_spacing * 0.5
            if self.cold_y_coords
            else self.stream_spacing * 0.5
        )
        for k in range(self.S + 1):
            self.stage_boundaries.append(boundary)
            boundary_line = _PlotlyLine(
                (boundary, boundary),
                (top_y, bottom_y),
                lw=5.0,
                linestyle="dashed",
                color=_RECOVERY_MATCH_COLOR,
            )
            if self.draw_stages:
                self.ax.add_line(boundary_line)
            if k < self.S:
                boundary += self.stage_count[k] * self.match_spacing
            else:
                break

    def draw_branches(self) -> None:
        if self.temperature_scaled:
            return

        for j in range(self.J):
            for k in range(self.S):
                if self.cold_branch_count[j][k] > 1:
                    self._draw_split_branches(
                        stream=self.grid_model.cold_streams[j],
                        branch_matches=[
                            self._staged_match_x(i, j, k)
                            for i in range(self.I)
                            if self.recovery_match[i][j][k] > 0
                        ],
                        stream_unit_x=self._cold_stream_unit_x_positions(j),
                        y0=self.cold_y_coords[-1 - j],
                        color=_COLD_STREAM_COLOR,
                    )

        for i in range(self.I):
            for k in range(self.S):
                if self.hot_branch_count[i][k] > 1:
                    self._draw_split_branches(
                        stream=self.grid_model.hot_streams[i],
                        branch_matches=[
                            self._staged_match_x(i, j, k)
                            for j in range(self.J)
                            if self.recovery_match[i][j][k] > 0
                        ],
                        stream_unit_x=self._hot_stream_unit_x_positions(i),
                        y0=self.hot_y_coords[-1 - i],
                        color=_HOT_STREAM_COLOR,
                    )

    def _draw_split_branches(
        self,
        *,
        stream: str,
        branch_matches: list[float],
        stream_unit_x: list[float],
        y0: float,
        color: str,
    ) -> None:
        if len(branch_matches) <= 1:
            return

        branch_matches = sorted(branch_matches)
        split_group_start = branch_matches[0]
        split_group_end = branch_matches[-1]
        previous_x, _ = _neighboring_x_positions(stream_unit_x, split_group_start)
        _, next_x = _neighboring_x_positions(stream_unit_x, split_group_end)
        split_start, split_end = self._split_connector_x_pair(
            previous_x,
            split_group_start,
            leans_right=True,
        )
        reconnect_start, reconnect_end = self._split_connector_x_pair(
            split_group_end,
            next_x,
            leans_right=False,
        )
        self._split_groups_by_stream.setdefault(stream, []).append(
            _SplitGroup(
                start_x=split_group_start,
                end_x=split_group_end,
                left_connector_x=(split_start + split_end) / 2.0,
                right_connector_x=(reconnect_start + reconnect_end) / 2.0,
            )
        )

        for branch_index, _branch_x in enumerate(branch_matches[:-1]):
            y1 = y0 - self.branch_spacing * (len(branch_matches) - 1 - branch_index)
            self.ax.add_line(
                _PlotlyLine(
                    (split_end, reconnect_end),
                    (y1, y1),
                    lw=self.stream_line_width,
                    color=color,
                )
            )
            self.ax.add_line(
                _PlotlyLine(
                    (split_start, split_end),
                    (y0, y1),
                    lw=self.stream_line_width,
                    color=color,
                )
            )
            self.ax.add_line(
                _PlotlyLine(
                    (reconnect_start, reconnect_end),
                    (y0, y1),
                    lw=self.stream_line_width,
                    color=color,
                )
            )

    def _split_connector_x_pair(
        self,
        left_x: float | None,
        right_x: float | None,
        *,
        leans_right: bool,
    ) -> tuple[float, float]:
        if left_x is None or right_x is None:
            center = self.stage_start if left_x is None else left_x + self.match_spacing
            half_span = self.match_spacing * 0.15
        else:
            gap = abs(right_x - left_x)
            center = (left_x + right_x) / 2.0
            half_span = min(gap, self.match_spacing) * 0.15

        if leans_right:
            return center - half_span, center + half_span
        return center + half_span, center - half_span

    def _staged_match_x(
        self,
        hot_index: int,
        cold_index: int,
        stage_index: int,
    ) -> float:
        match_number = self.stage_recovery_positions[stage_index].index(
            (hot_index, cold_index)
        )
        return (
            self.stage_boundaries[stage_index]
            + self.match_start / 2
            + match_number * self.match_spacing
        )

    def _build_stage_recovery_positions(self) -> dict[int, list[tuple[int, int]]]:
        return {
            stage_index: self._sorted_stage_recovery_positions(stage_index)
            for stage_index in range(self.S)
        }

    def _sorted_stage_recovery_positions(
        self,
        stage_index: int,
    ) -> list[tuple[int, int]]:
        positions = [
            (i, j)
            for i in range(self.I)
            for j in range(self.J)
            if self.recovery_match[i][j][stage_index] > 0
        ]
        return sorted(
            positions,
            key=lambda position: self._stage_recovery_position_sort_key(
                position[0],
                position[1],
                stage_index,
            ),
        )

    def _stage_recovery_position_sort_key(
        self,
        hot_index: int,
        cold_index: int,
        stage_index: int,
    ) -> tuple[bool, float, bool, float, int, int]:
        match = self.recovery_match_by_position[(hot_index, cold_index, stage_index)]
        source_mid_temperature = match.exchanger.source_mid_temperature
        sink_mid_temperature = match.exchanger.sink_mid_temperature
        return (
            source_mid_temperature is None,
            0.0 if source_mid_temperature is None else -source_mid_temperature,
            sink_mid_temperature is None,
            0.0 if sink_mid_temperature is None else sink_mid_temperature,
            hot_index,
            cold_index,
        )

    def _cold_stream_unit_x_positions(self, cold_index: int) -> list[float]:
        positions = [
            self._staged_match_x(i, cold_index, k)
            for k in range(self.S)
            for i in range(self.I)
            if self.recovery_match[i][cold_index][k] > 0
        ]
        if cold_index in self.hot_utility_by_cold_index:
            positions.append(
                self._hot_utility_x(self.hot_utility_by_cold_index[cold_index])
            )
        return sorted(set(positions))

    def _hot_stream_unit_x_positions(self, hot_index: int) -> list[float]:
        positions = [
            self._staged_match_x(hot_index, j, k)
            for k in range(self.S)
            for j in range(self.J)
            if self.recovery_match[hot_index][j][k] > 0
        ]
        if hot_index in self.cold_utility_by_hot_index:
            positions.append(
                self._cold_utility_x(self.cold_utility_by_hot_index[hot_index])
            )
        return sorted(set(positions))

    def draw_recovery_matches(self) -> None:
        self.match_coords = {
            (i, j, k): [0.0, 1.0]
            for i in range(self.I)
            for j in range(self.J)
            for k in range(self.S)
        }

        for k in range(self.S):
            for j in range(self.J):
                branch_number = self.cold_branch_count[j][k] - 1
                for i in range(self.I):
                    if self.recovery_match[i][j][k] > 0:
                        if self.cold_branch_count[j][k] > 1:
                            self.match_coords[(i, j, k)][1] = (
                                self.cold_y_coords[-1 - j]
                                - self.branch_spacing * branch_number
                            )
                            branch_number -= 1
                        else:
                            self.match_coords[(i, j, k)][1] = self.cold_y_coords[-1 - j]

        for k in range(self.S):
            for i in range(self.I):
                branch_number = self.hot_branch_count[i][k] - 1
                for j in range(self.J):
                    if self.recovery_match[i][j][k] > 0:
                        if self.hot_branch_count[i][k] > 1:
                            self.match_coords[(i, j, k)][0] = (
                                self.hot_y_coords[-1 - i]
                                - self.branch_spacing * branch_number
                            )
                            branch_number -= 1
                        else:
                            self.match_coords[(i, j, k)][0] = self.hot_y_coords[-1 - i]

        for k in range(self.S):
            for i in range(self.I):
                for j in range(self.J):
                    if self.recovery_match[i][j][k] > 0:
                        match = self.recovery_match_by_position[(i, j, k)]
                        if self.temperature_scaled:
                            hot_match_x, cold_match_x = self._recovery_match_x_pair(
                                match
                            )
                        else:
                            hot_match_x = self._staged_match_x(i, j, k)
                            cold_match_x = hot_match_x
                        recovery_hx = _PlotlyLine(
                            (hot_match_x, cold_match_x),
                            (
                                self.match_coords[(i, j, k)][0],
                                self.match_coords[(i, j, k)][1],
                            ),
                            lw=self.match_line_width,
                            color=_RECOVERY_MATCH_COLOR,
                            marker=".",
                            markersize=self.match_radius,
                            markerfacecolor=_RECOVERY_MATCH_COLOR,
                            markeredgecolor=_RECOVERY_MATCH_COLOR,
                            markeredgewidth=1.0,
                        )
                        self.ax.add_line(recovery_hx)
                        self._register_exchanger_artist(recovery_hx, match)
                        self._register_stream_match_position(
                            match.source_stream,
                            hot_match_x,
                            self.match_coords[(i, j, k)][0],
                            match,
                        )
                        self._register_stream_match_position(
                            match.sink_stream,
                            cold_match_x,
                            self.match_coords[(i, j, k)][1],
                            match,
                        )
                        self.match_x_by_position[(i, j, k)] = (
                            hot_match_x + cold_match_x
                        ) / 2.0

    def draw_utility_match(self) -> None:
        self.match_HU_x = self.x_start + (self.stage_start - self.x_start) / 2
        for j in range(self.J):
            if self.HU_matches[j] > 0:
                match = self.hot_utility_by_cold_index[j]
                match_hu_x = self._hot_utility_x(match)
                utility_hx = _PlotlyLine(
                    (match_hu_x, match_hu_x),
                    (self.cold_y_coords[-1 - j], self.cold_y_coords[-1 - j]),
                    lw=self.stream_line_width,
                    color=_HOT_UTILITY_COLOR,
                    marker=".",
                    markersize=self.match_radius,
                    markerfacecolor=_HOT_UTILITY_COLOR,
                    markeredgecolor=_HOT_UTILITY_COLOR,
                    markeredgewidth=1.0,
                )
                self.ax.add_line(utility_hx)
                self._register_exchanger_artist(utility_hx, match)
                self._register_stream_match_position(
                    match.sink_stream,
                    match_hu_x,
                    self.cold_y_coords[-1 - j],
                    match,
                )
                self._add_label(
                    match_hu_x,
                    self.cold_y_coords[-1 - j],
                    _short_stream_name(match.source_stream),
                    color=_HOT_UTILITY_COLOR,
                    fontsize=self.duty_font_size,
                    ha="center",
                    va="bottom",
                    xytext=(0, self._font_points(self.duty_font_size, 0.95)),
                    bgcolor=_LABEL_BACKGROUND_COLOR,
                    borderpad=1,
                )

        self.match_CU_x = (
            self.stage_boundaries[-1] + (self.x_finish - self.stage_boundaries[-1]) / 2
        )
        for i in range(self.I):
            if self.CU_matches[i] > 0:
                match = self.cold_utility_by_hot_index[i]
                match_cu_x = self._cold_utility_x(match)
                utility_hx = _PlotlyLine(
                    (match_cu_x, match_cu_x),
                    (self.hot_y_coords[-1 - i], self.hot_y_coords[-1 - i]),
                    lw=self.stream_line_width,
                    color=_COLD_UTILITY_COLOR,
                    marker=".",
                    markersize=self.match_radius,
                    markerfacecolor=_COLD_UTILITY_COLOR,
                    markeredgecolor=_COLD_UTILITY_COLOR,
                    markeredgewidth=1.0,
                )
                self.ax.add_line(utility_hx)
                self._register_exchanger_artist(utility_hx, match)
                self._register_stream_match_position(
                    match.source_stream,
                    match_cu_x,
                    self.hot_y_coords[-1 - i],
                    match,
                )
                self._add_label(
                    match_cu_x,
                    self.hot_y_coords[-1 - i],
                    _short_stream_name(match.sink_stream),
                    color=_COLD_UTILITY_COLOR,
                    fontsize=self.duty_font_size,
                    ha="center",
                    va="bottom",
                    xytext=(0, self._font_points(self.duty_font_size, 0.95)),
                    bgcolor=_LABEL_BACKGROUND_COLOR,
                    borderpad=1,
                )

    def add_duties(self) -> None:
        duty_y_offset = -self._duty_label_offset()
        for k in range(self.S):
            for i in range(self.I):
                for j in range(self.J):
                    if self.recovery_match[i][j][k] == 1:
                        match = self.recovery_match_by_position[(i, j, k)]
                        if self.temperature_scaled:
                            _, duty_x = self._recovery_match_x_pair(match)
                        else:
                            duty_x = self._staged_match_x(i, j, k)
                        self._add_label(
                            duty_x,
                            self.match_coords[(i, j, k)][1],
                            f"{match.duty / 1000:.2f} MW",
                            fontsize=self.duty_font_size,
                            color="purple",
                            ha="center",
                            va="top",
                            xytext=(0, duty_y_offset),
                            bgcolor=_LABEL_BACKGROUND_COLOR,
                            borderpad=1,
                        )
        for j in range(self.J):
            match = self.hot_utility_by_cold_index.get(j)
            if match is not None:
                match_hu_x = self._hot_utility_x(match)
                self._add_label(
                    match_hu_x,
                    self.cold_y_coords[-1 - j],
                    f"{match.duty / 1000:.2f} MW",
                    fontsize=self.duty_font_size,
                    color="purple",
                    ha="center",
                    va="top",
                    xytext=(0, duty_y_offset),
                    bgcolor=_LABEL_BACKGROUND_COLOR,
                    borderpad=1,
                )

        for i in range(self.I):
            match = self.cold_utility_by_hot_index.get(i)
            if match is not None:
                match_cu_x = self._cold_utility_x(match)
                self._add_label(
                    match_cu_x,
                    self.hot_y_coords[-1 - i],
                    f"{match.duty / 1000:.2f} MW",
                    fontsize=self.duty_font_size,
                    color="purple",
                    ha="center",
                    va="top",
                    xytext=(0, duty_y_offset),
                    bgcolor=_LABEL_BACKGROUND_COLOR,
                    borderpad=1,
                )

    def add_temps(self) -> None:
        format_str = r"{:.0f} $^\circ$C"
        temperatures = _stream_endpoint_temperatures(self.grid_model)
        endpoint_x_offset = self._font_points(self.temp_font_size, 0.7)
        intermediate_y_offset = self._temperature_label_offset()

        for j, stream in enumerate(self.grid_model.cold_streams):
            y = self.cold_y_coords[-1 - j]
            supply_x, target_x = self._cold_stream_x_bounds(stream)
            cold_in = temperatures["cold_in"].get(stream)
            if cold_in is not None:
                self._add_label(
                    supply_x,
                    y,
                    format_str.format(cold_in - 273.15),
                    fontsize=self.temp_font_size,
                    ha="left",
                    va="center_baseline",
                    xytext=(endpoint_x_offset, 0),
                    bgcolor=_LABEL_BACKGROUND_COLOR,
                    borderpad=1,
                )
            cold_out = temperatures["cold_out"].get(stream)
            if cold_out is not None:
                self._add_label(
                    target_x,
                    y,
                    format_str.format(cold_out - 273.15),
                    fontsize=self.temp_font_size,
                    ha="right",
                    va="center_baseline",
                    xytext=(-endpoint_x_offset, 0),
                    bgcolor=_LABEL_BACKGROUND_COLOR,
                    borderpad=1,
                )
            self._add_intermediate_temperature_labels(
                stream,
                y,
                role="cold",
                xytext=(0, intermediate_y_offset),
                format_str=format_str,
            )

        for i, stream in enumerate(self.grid_model.hot_streams):
            y = self.hot_y_coords[-1 - i]
            supply_x, target_x = self._hot_stream_x_bounds(stream)
            hot_in = temperatures["hot_in"].get(stream)
            if hot_in is not None:
                self._add_label(
                    supply_x,
                    y,
                    format_str.format(hot_in - 273.15),
                    fontsize=self.temp_font_size,
                    ha="right",
                    va="center_baseline",
                    xytext=(-endpoint_x_offset, 0),
                    bgcolor=_LABEL_BACKGROUND_COLOR,
                    borderpad=1,
                )
            hot_out = temperatures["hot_out"].get(stream)
            if hot_out is not None:
                self._add_label(
                    target_x,
                    y,
                    format_str.format(hot_out - 273.15),
                    fontsize=self.temp_font_size,
                    ha="left",
                    va="center_baseline",
                    xytext=(endpoint_x_offset, 0),
                    bgcolor=_LABEL_BACKGROUND_COLOR,
                    borderpad=1,
                )
            self._add_intermediate_temperature_labels(
                stream,
                y,
                role="hot",
                xytext=(0, intermediate_y_offset),
                format_str=format_str,
            )

    def plot_setup(self) -> None:
        cold_stream_names = [f"C{j + 1}" for j in reversed(range(self.J))]
        hot_stream_names = [f"H{i + 1}" for i in reversed(range(self.I))]
        self.ax.set_yticks(
            self.hot_y_coords + self.cold_y_coords,
            hot_stream_names + cold_stream_names,
        )
        stream_label_offset = -self._font_points(self.size_of_font, 3.7)
        for name, y in zip(hot_stream_names, self.hot_y_coords, strict=True):
            self._add_label(
                self.x_start,
                y,
                name,
                fontsize=self.size_of_font,
                ha="right",
                va="center_baseline",
                xytext=(stream_label_offset, 0),
            )
        for name, y in zip(cold_stream_names, self.cold_y_coords, strict=True):
            self._add_label(
                self.x_start,
                y,
                name,
                fontsize=self.size_of_font,
                ha="right",
                va="center_baseline",
                xytext=(stream_label_offset, 0),
            )
        self.ax.set_xticks([])
        self.ax.set_xlim(self.x_start - 0.02, self.x_finish + 0.02)
        left_margin, right_margin = self._side_margins()
        self.fig.update_layout(
            width=self.figure_width,
            height=self.figure_height,
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            margin={"l": left_margin, "r": right_margin, "t": 25, "b": 25},
        )
        self.fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
        self.fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

    def _side_margins(self) -> tuple[int, int]:
        temperatures = _stream_endpoint_temperatures(self.grid_model)
        endpoint_labels = [
            f"{temperature - 273.15:.0f} °C"
            for values in temperatures.values()
            for temperature in values.values()
        ]
        longest_endpoint = max((len(label) for label in endpoint_labels), default=0)
        stream_label_chars = max(
            (len(f"H{index + 1}") for index in range(self.I)),
            default=0,
        )
        stream_label_chars = max(
            stream_label_chars,
            max((len(f"C{index + 1}") for index in range(self.J)), default=0),
        )
        character_width = self.size_of_font * 0.55
        left = self._font_points(self.size_of_font, 3.7) + character_width * (
            stream_label_chars + longest_endpoint
        )
        right = self._font_points(self.temp_font_size, 1.2) + (
            self.temp_font_size * 0.55 * longest_endpoint
        )
        return max(70, int(left)), max(70, int(right))

    def _add_label(self, x: float, y: float, text: str, **kwargs) -> None:
        xytext = kwargs.pop("xytext", None)
        offset_key = xytext if xytext is not None else (0.0, 0.0)
        key = (
            round(x, 2),
            round(y, 2),
            round(offset_key[0], 2),
            round(offset_key[1], 2),
        )
        duplicate_count = self._label_slot_counts.get(key, 0)
        self._label_slot_counts[key] = duplicate_count + 1
        if duplicate_count:
            if xytext is None:
                y -= duplicate_count * 0.35
            else:
                xytext = (xytext[0], xytext[1] - duplicate_count * self.size_of_font)
        kwargs.setdefault("zorder", 10)
        if xytext is None:
            self.ax.text(x, y, text, **kwargs)
            return
        self.ax.annotate(
            text,
            xy=(x, y),
            xytext=xytext,
            textcoords="offset points",
            annotation_clip=False,
            clip_on=False,
            **kwargs,
        )

    def _add_intermediate_temperature_labels(
        self,
        stream: str,
        y: float,
        *,
        role: str,
        xytext: tuple[float, float],
        format_str: str,
    ) -> None:
        positions = self._stream_match_positions.get(stream, [])
        if role == "hot":
            candidates = [
                (
                    x,
                    node_y,
                    match,
                    match.exchanger.source_inlet_temperature,
                )
                for x, node_y, match in positions
                if match.source_stream == stream
                and match.exchanger.kind
                in (HeatExchangerKind.RECOVERY, HeatExchangerKind.COLD_UTILITY)
                and match.exchanger.source_inlet_temperature is not None
            ]
            candidates.sort(key=lambda item: item[0])
        else:
            candidates = [
                (
                    x,
                    node_y,
                    match,
                    match.exchanger.sink_inlet_temperature,
                )
                for x, node_y, match in positions
                if match.sink_stream == stream
                and match.exchanger.kind
                in (HeatExchangerKind.RECOVERY, HeatExchangerKind.HOT_UTILITY)
                and match.exchanger.sink_inlet_temperature is not None
            ]
            candidates.sort(key=lambda item: item[0], reverse=True)

        segments: list[tuple[float, float, float, str]] = []
        for current, next_match in zip(candidates, candidates[1:], strict=False):
            if self._same_split_group(stream, current[0], next_match[0]):
                continue
            temperature = next_match[3]
            if temperature is None:
                continue
            node_y = current[1] if current[1] == next_match[1] else y
            x_start, x_end = self._split_adjusted_temperature_segment(
                stream,
                current[0],
                next_match[0],
            )
            segments.append(
                (
                    x_start,
                    x_end,
                    node_y,
                    format_str.format(temperature - 273.15),
                )
            )

        for x_start, x_end, node_y, label in _merge_temperature_label_segments(
            segments,
            stream_y=y,
        ):
            self._add_label(
                (x_start + x_end) / 2,
                node_y,
                label,
                fontsize=self.temp_font_size,
                ha="center",
                va="bottom",
                xytext=xytext,
                bgcolor=_LABEL_BACKGROUND_COLOR,
                borderpad=1,
            )

    def _same_split_group(self, stream: str, x_start: float, x_end: float) -> bool:
        tolerance = 1e-9
        return any(
            group.start_x - tolerance <= x_start <= group.end_x + tolerance
            and group.start_x - tolerance <= x_end <= group.end_x + tolerance
            for group in self._split_groups_by_stream.get(stream, [])
        )

    def _split_adjusted_temperature_segment(
        self,
        stream: str,
        x_start: float,
        x_end: float,
    ) -> tuple[float, float]:
        adjusted_start = x_start
        adjusted_end = x_end
        for group in self._split_groups_by_stream.get(stream, []):
            adjusted_start = _split_adjusted_temperature_x(adjusted_start, x_end, group)
            adjusted_end = _split_adjusted_temperature_x(adjusted_end, x_start, group)
        return adjusted_start, adjusted_end

    def _duty_label_offset(self) -> float:
        font_offset = self._font_points(self.duty_font_size, 0.5)
        marker_clearance = _plotly_marker_size(self.match_radius) * 0.5
        return max(font_offset, marker_clearance)

    def _temperature_label_offset(self) -> float:
        font_offset = self._font_points(self.temp_font_size, 0.1875)
        stream_clearance = self.stream_line_width * 0.5
        return max(font_offset, stream_clearance)

    def _font_points(self, fontsize: float, multiplier: float = 1.0) -> float:
        return max(float(fontsize), self.stream_line_width) * multiplier

    def _register_stream_match_position(
        self,
        stream: str,
        x: float,
        y: float,
        match: GridDiagramMatch,
    ) -> None:
        self._stream_match_positions.setdefault(stream, []).append((x, y, match))

    def _register_exchanger_artist(self, artist: Any, match: GridDiagramMatch) -> None:
        tooltip = _exchanger_tooltip(match)
        artist.openpinch_tooltip = tooltip
        if hasattr(self.ax, "update_line_hover"):
            self.ax.update_line_hover(artist, tooltip)
        self.exchanger_tooltips[artist] = tooltip

    def _hot_stream_x_bounds(self, stream: str) -> tuple[float, float]:
        if not self.temperature_scaled:
            return self.x_start, self.x_finish
        temperatures = _stream_endpoint_temperatures(self.grid_model)
        hot_in = temperatures["hot_in"].get(stream)
        hot_out = temperatures["hot_out"].get(stream)
        return self._x_from_temperature(hot_in), self._x_from_temperature(hot_out)

    def _cold_stream_x_bounds(self, stream: str) -> tuple[float, float]:
        if not self.temperature_scaled:
            return self.x_finish, self.x_start
        temperatures = _stream_endpoint_temperatures(self.grid_model)
        cold_in = temperatures["cold_in"].get(stream)
        cold_out = temperatures["cold_out"].get(stream)
        return self._x_from_temperature(cold_in), self._x_from_temperature(cold_out)

    def _recovery_match_x_pair(self, match: GridDiagramMatch) -> tuple[float, float]:
        if not self.temperature_scaled:
            return 0.0, 0.0
        return (
            self._x_from_temperature(
                _midpoint_temperature(
                    match.exchanger.source_inlet_temperature,
                    match.exchanger.source_outlet_temperature,
                )
            ),
            self._x_from_temperature(
                _midpoint_temperature(
                    match.exchanger.sink_inlet_temperature,
                    match.exchanger.sink_outlet_temperature,
                )
            ),
        )

    def _hot_utility_x(self, match: GridDiagramMatch) -> float:
        if self.temperature_scaled:
            return self._x_from_temperature(
                _midpoint_temperature(
                    match.exchanger.sink_inlet_temperature,
                    match.exchanger.sink_outlet_temperature,
                )
            )
        return self.x_start + (self.stage_start - self.x_start) / 2

    def _cold_utility_x(self, match: GridDiagramMatch) -> float:
        if self.temperature_scaled:
            return self._x_from_temperature(
                _midpoint_temperature(
                    match.exchanger.source_inlet_temperature,
                    match.exchanger.source_outlet_temperature,
                )
            )
        return (
            self.stage_boundaries[-1] + (self.x_finish - self.stage_boundaries[-1]) / 2
        )

    def _x_from_temperature(self, temperature: float | None) -> float:
        if not self.temperature_scaled or temperature is None:
            return self.x_start
        t_min, t_max = self.temperature_scale
        if t_max <= t_min:
            return (self.x_start + self.x_finish) / 2
        return self.x_start + ((t_max - temperature) / (t_max - t_min)) * (
            self.x_finish - self.x_start
        )


def _stream_endpoint_temperatures(
    grid_model: HeatExchangerNetworkGridModel,
) -> dict[str, dict[str, float]]:
    hot_in: dict[str, float] = {}
    hot_out: dict[str, float] = {}
    cold_in: dict[str, float] = {}
    cold_out: dict[str, float] = {}

    for match in grid_model.recovery_matches:
        exchanger = match.exchanger
        if exchanger.source_inlet_temperature is not None:
            hot_in[match.source_stream] = max(
                hot_in.get(match.source_stream, exchanger.source_inlet_temperature),
                exchanger.source_inlet_temperature,
            )
        if exchanger.source_outlet_temperature is not None:
            hot_out[match.source_stream] = min(
                hot_out.get(match.source_stream, exchanger.source_outlet_temperature),
                exchanger.source_outlet_temperature,
            )
        if exchanger.sink_inlet_temperature is not None:
            cold_in[match.sink_stream] = min(
                cold_in.get(match.sink_stream, exchanger.sink_inlet_temperature),
                exchanger.sink_inlet_temperature,
            )
        if exchanger.sink_outlet_temperature is not None:
            cold_out[match.sink_stream] = max(
                cold_out.get(match.sink_stream, exchanger.sink_outlet_temperature),
                exchanger.sink_outlet_temperature,
            )
    for match in grid_model.hot_utility_matches:
        exchanger = match.exchanger
        if exchanger.sink_inlet_temperature is not None:
            cold_in[match.sink_stream] = min(
                cold_in.get(match.sink_stream, exchanger.sink_inlet_temperature),
                exchanger.sink_inlet_temperature,
            )
        if exchanger.sink_outlet_temperature is not None:
            cold_out[match.sink_stream] = max(
                cold_out.get(match.sink_stream, exchanger.sink_outlet_temperature),
                exchanger.sink_outlet_temperature,
            )
    for match in grid_model.cold_utility_matches:
        exchanger = match.exchanger
        if exchanger.source_inlet_temperature is not None:
            hot_in[match.source_stream] = max(
                hot_in.get(match.source_stream, exchanger.source_inlet_temperature),
                exchanger.source_inlet_temperature,
            )
        if exchanger.source_outlet_temperature is not None:
            hot_out[match.source_stream] = min(
                hot_out.get(match.source_stream, exchanger.source_outlet_temperature),
                exchanger.source_outlet_temperature,
            )

    return {
        "hot_in": hot_in,
        "hot_out": hot_out,
        "cold_in": cold_in,
        "cold_out": cold_out,
    }


def _temperature_scale(
    grid_model: HeatExchangerNetworkGridModel,
) -> tuple[float, float]:
    temperatures: list[float] = []
    for match in grid_model.recovery_matches:
        exchanger = match.exchanger
        temperatures.extend(
            temperature
            for temperature in (
                exchanger.source_inlet_temperature,
                exchanger.source_outlet_temperature,
                exchanger.sink_inlet_temperature,
                exchanger.sink_outlet_temperature,
            )
            if temperature is not None
        )
    for match in grid_model.hot_utility_matches:
        exchanger = match.exchanger
        temperatures.extend(
            temperature
            for temperature in (
                exchanger.sink_inlet_temperature,
                exchanger.sink_outlet_temperature,
            )
            if temperature is not None
        )
    for match in grid_model.cold_utility_matches:
        exchanger = match.exchanger
        temperatures.extend(
            temperature
            for temperature in (
                exchanger.source_inlet_temperature,
                exchanger.source_outlet_temperature,
            )
            if temperature is not None
        )
    if not temperatures:
        return 0.0, 1.0
    return min(temperatures), max(temperatures)


def _midpoint_temperature(
    inlet_temperature: float | None,
    outlet_temperature: float | None,
) -> float | None:
    temperatures = [
        temperature
        for temperature in (inlet_temperature, outlet_temperature)
        if temperature is not None
    ]
    if not temperatures:
        return None
    return sum(temperatures) / len(temperatures)


def _validate_stream_line_width(value: float) -> float:
    value = float(value)
    if value <= 0:
        raise ValueError("stream_line_width must be positive")
    return value


def _short_stream_name(stream: str) -> str:
    for separator in (".", "/", ":"):
        if separator in stream:
            stream = stream.rsplit(separator, maxsplit=1)[-1]
    return stream.strip() or stream


def _exchanger_tooltip(match: GridDiagramMatch) -> str:
    exchanger = match.exchanger
    area = "n/a" if exchanger.area is None else f"{exchanger.area:.2f} m^2"
    stage = "" if match.stage is None else f" | stage {match.stage}"
    return (
        f"{_short_stream_name(match.source_stream)} -> "
        f"{_short_stream_name(match.sink_stream)}{stage}\n"
        f"Duty: {match.duty / 1000:.2f} MW\n"
        f"Area: {area}"
    )


def _merge_temperature_label_segments(
    segments: list[tuple[float, float, float, str]],
    *,
    stream_y: float,
) -> list[tuple[float, float, float, str]]:
    if not segments:
        return []

    merged: list[tuple[float, float, float, str]] = []
    run_start, run_end, run_y, run_label = segments[0]
    run_ys = [run_y]

    for segment_start, segment_end, segment_y, segment_label in segments[1:]:
        if segment_label == run_label:
            run_end = segment_end
            run_ys.append(segment_y)
            continue
        merged.append(
            (
                run_start,
                run_end,
                run_y if all(y == run_y for y in run_ys) else stream_y,
                run_label,
            )
        )
        run_start = segment_start
        run_end = segment_end
        run_y = segment_y
        run_label = segment_label
        run_ys = [run_y]

    merged.append(
        (
            run_start,
            run_end,
            run_y if all(y == run_y for y in run_ys) else stream_y,
            run_label,
        )
    )
    return merged


def _neighboring_x_positions(
    positions: list[float],
    x_value: float,
) -> tuple[float | None, float | None]:
    previous_positions = [x for x in positions if x < x_value]
    next_positions = [x for x in positions if x > x_value]
    previous_x = max(previous_positions) if previous_positions else None
    next_x = min(next_positions) if next_positions else None
    return previous_x, next_x


def _split_adjusted_temperature_x(
    x_value: float,
    other_x: float,
    group: _SplitGroup,
) -> float:
    tolerance = 1e-9
    if not group.start_x - tolerance <= x_value <= group.end_x + tolerance:
        return x_value
    if other_x < group.start_x - tolerance:
        return group.left_connector_x
    if other_x > group.end_x + tolerance:
        return group.right_connector_x
    return x_value


__all__ = ["_PlotlyGridRenderer"]
