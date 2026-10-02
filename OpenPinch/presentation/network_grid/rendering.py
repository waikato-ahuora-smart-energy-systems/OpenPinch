"""Plotly renderer for heat exchanger network grid diagrams."""

from __future__ import annotations

from typing import Any

from ._constants import (
    _COLD_STREAM_COLOR,
    _COLD_UTILITY_COLOR,
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
from .geometry import neighboring_x_positions
from .labels import (
    add_duty_labels,
    add_label,
    configure_figure,
    font_points,
    register_exchanger_artist,
    register_stream_match_position,
    short_stream_name,
)
from .plotly import (
    _PlotlyAxes,
    _PlotlyLine,
    _SplitGroup,
)
from .state import GridDiagramMatch, HeatExchangerNetworkGridModel
from .temperatures import (
    add_temperature_labels,
    cold_stream_x_bounds,
    cold_utility_x,
    hot_stream_x_bounds,
    hot_utility_x,
    recovery_match_x_pair,
    temperature_scale,
)


class _PlotlyGridRenderer:
    """Plotly renderer for heat exchanger network grid diagrams."""

    def __init__(
        self,
        grid_model: HeatExchangerNetworkGridModel,
        *,
        graph_objects: Any,
        temperature_scaled: bool,
    ):
        self.grid_model = grid_model
        self.network = grid_model.network
        self.go = graph_objects
        self.I = len(grid_model.hot_streams)
        self.J = len(grid_model.cold_streams)
        self.S = len(grid_model.stages)
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
        self.hot_utilities_by_cold_index: dict[int, list[GridDiagramMatch]] = {}
        for match in grid_model.hot_utility_matches:
            self.hot_utilities_by_cold_index.setdefault(
                self.cold_index[match.sink_stream], []
            ).append(match)
        self.cold_utilities_by_hot_index: dict[int, list[GridDiagramMatch]] = {}
        for match in grid_model.cold_utility_matches:
            self.cold_utilities_by_hot_index.setdefault(
                self.hot_index[match.source_stream], []
            ).append(match)
        self.utility_x_by_match: dict[int, float] = {}

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
        self.temperature_scale = temperature_scale(self.grid_model)
        self.figure_width, self.figure_height = self._figure_size()
        self.match_radius = self._marker_radius()
        self.stream_line_width = self._stream_line_width()
        self.match_line_width = self._match_line_width()
        self.fig = self.go.Figure()
        self.ax = _PlotlyAxes(self.fig, self.go)
        self.draw_streams()
        self._place_utility_matches()
        self.draw_branches()
        self.draw_recovery_matches()
        self.draw_utility_match()
        add_duty_labels(self)
        add_temperature_labels(self)
        configure_figure(self)

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
            len(self.cold_utilities_by_hot_index.get(i, ())) for i in range(self.I)
        ]
        self.HU_matches = [
            len(self.hot_utilities_by_cold_index.get(j, ())) for j in range(self.J)
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
        # Widen an end band when several utility exchangers sit there in series.
        heater_band = self.utility_length * self._end_utility_slots(heaters=True)
        cooler_band = self.utility_length * self._end_utility_slots(heaters=False)
        self.stage_start = self.x_start + heater_band
        self.stage_finish = self.x_finish - cooler_band
        self.recovery_length = self.stage_finish - self.stage_start
        total_matches = max(1, sum(self.stage_count))
        self.match_spacing = self.recovery_length / total_matches
        self.match_start = self.match_spacing

    def _end_utility_slots(self, *, heaters: bool) -> float:
        groups = (
            self.hot_utilities_by_cold_index
            if heaters
            else self.cold_utilities_by_hot_index
        )
        stages = self.grid_model.stages
        largest = 1
        for matches in groups.values():
            at_end = sum(
                1
                for match in matches
                if match.stage is None
                or (match.stage <= stages[0] if heaters else match.stage >= stages[-1])
            )
            largest = max(largest, at_end)
        return min(1.0 + 0.6 * (largest - 1), 2.5)

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
            x_tail, x_head = cold_stream_x_bounds(self, stream)
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
            y_cursor += self.stream_spacing * branches
            x_tail, x_head = hot_stream_x_bounds(self, stream)
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
        previous_x, _ = neighboring_x_positions(stream_unit_x, split_group_start)
        _, next_x = neighboring_x_positions(stream_unit_x, split_group_end)
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
        source_mid_temperature = match.state.source_mid_temperature
        sink_mid_temperature = match.state.sink_mid_temperature
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
        positions.extend(
            hot_utility_x(self, match)
            for match in self.hot_utilities_by_cold_index.get(cold_index, ())
        )
        return sorted(set(positions))

    def _hot_stream_unit_x_positions(self, hot_index: int) -> list[float]:
        positions = [
            self._staged_match_x(hot_index, j, k)
            for k in range(self.S)
            for j in range(self.J)
            if self.recovery_match[hot_index][j][k] > 0
        ]
        positions.extend(
            cold_utility_x(self, match)
            for match in self.cold_utilities_by_hot_index.get(hot_index, ())
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
                            hot_match_x, cold_match_x = recovery_match_x_pair(
                                self, match
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
                        register_exchanger_artist(self, recovery_hx, match)
                        register_stream_match_position(
                            self,
                            match.source_stream,
                            hot_match_x,
                            self.match_coords[(i, j, k)][0],
                            match,
                        )
                        register_stream_match_position(
                            self,
                            match.sink_stream,
                            cold_match_x,
                            self.match_coords[(i, j, k)][1],
                            match,
                        )
                        self.match_x_by_position[(i, j, k)] = (
                            hot_match_x + cold_match_x
                        ) / 2.0

    def _place_utility_matches(self) -> None:
        """Give every utility exchanger an x-position on its stream.

        End exchangers share the utility band at the stream end; one placed
        after a stage sits on that stage boundary. Exchangers at one position
        are spread in series order along the stream's flow direction (cold
        streams flow right to left, hot streams left to right).
        """

        first_stage, last_stage = self.grid_model.stages[0], self.grid_model.stages[-1]
        width = 0.8 * self.match_spacing
        for matches in self.hot_utilities_by_cold_index.values():
            groups: dict[int | None, list[GridDiagramMatch]] = {}
            for match in matches:
                stage = match.stage
                boundary = (
                    None
                    if stage is None
                    or stage <= first_stage
                    or stage not in self.stage_index
                    else self.stage_index[stage]
                )
                groups.setdefault(boundary, []).append(match)
            for boundary, members in groups.items():
                members.sort(key=_heater_series_key)
                count = len(members)
                for position, match in enumerate(members):
                    fraction = (position + 1) / (count + 1)
                    if boundary is None:
                        x = (
                            self.stage_start
                            - (self.stage_start - self.x_start) * fraction
                        )
                    else:
                        x = self.stage_boundaries[boundary] + width * (0.5 - fraction)
                    self.utility_x_by_match[id(match)] = x
        for matches in self.cold_utilities_by_hot_index.values():
            groups = {}
            for match in matches:
                stage = match.stage
                boundary = (
                    None
                    if stage is None
                    or stage >= last_stage
                    or stage not in self.stage_index
                    else self.stage_index[stage] + 1
                )
                groups.setdefault(boundary, []).append(match)
            for boundary, members in groups.items():
                members.sort(key=_cooler_series_key)
                count = len(members)
                for position, match in enumerate(members):
                    fraction = (position + 1) / (count + 1)
                    if boundary is None:
                        end = self.stage_boundaries[-1]
                        x = end + (self.x_finish - end) * fraction
                    else:
                        x = self.stage_boundaries[boundary] + width * (fraction - 0.5)
                    self.utility_x_by_match[id(match)] = x

    def draw_utility_match(self) -> None:
        self.match_HU_x = self.x_start + (self.stage_start - self.x_start) / 2
        for j in range(self.J):
            for match in self.hot_utilities_by_cold_index.get(j, ()):
                self._draw_utility(
                    match,
                    x=hot_utility_x(self, match),
                    y=self.cold_y_coords[-1 - j],
                    stream=match.sink_stream,
                    label=match.source_stream,
                    color=_HOT_UTILITY_COLOR,
                )

        self.match_CU_x = (
            self.stage_boundaries[-1] + (self.x_finish - self.stage_boundaries[-1]) / 2
        )
        for i in range(self.I):
            for match in self.cold_utilities_by_hot_index.get(i, ()):
                self._draw_utility(
                    match,
                    x=cold_utility_x(self, match),
                    y=self.hot_y_coords[-1 - i],
                    stream=match.source_stream,
                    label=match.sink_stream,
                    color=_COLD_UTILITY_COLOR,
                )

    def _draw_utility(
        self,
        match: GridDiagramMatch,
        *,
        x: float,
        y: float,
        stream: str,
        label: str,
        color: str,
    ) -> None:
        utility_hx = _PlotlyLine(
            (x, x),
            (y, y),
            lw=self.stream_line_width,
            color=color,
            marker=".",
            markersize=self.match_radius,
            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=1.0,
        )
        self.ax.add_line(utility_hx)
        register_exchanger_artist(self, utility_hx, match)
        register_stream_match_position(self, stream, x, y, match)
        add_label(
            self,
            x,
            y,
            short_stream_name(label),
            color=color,
            fontsize=self.duty_font_size,
            ha="center",
            va="bottom",
            xytext=(0, font_points(self, self.duty_font_size, 0.95)),
            bgcolor=_LABEL_BACKGROUND_COLOR,
            borderpad=1,
        )


def _heater_series_key(match: GridDiagramMatch) -> float:
    """First in series heats the coldest stream: lowest sink inlet first."""

    value = match.state.sink_inlet_temperature
    return float("inf") if value is None else float(value)


def _cooler_series_key(match: GridDiagramMatch) -> float:
    """First in series cools the hottest stream: highest source inlet first."""

    value = match.state.source_inlet_temperature
    return float("inf") if value is None else -float(value)


__all__ = ["_PlotlyGridRenderer"]
