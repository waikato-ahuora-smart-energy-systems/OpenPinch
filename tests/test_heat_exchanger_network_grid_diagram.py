"""Tests for heat exchanger network grid diagrams."""

from __future__ import annotations

from pathlib import Path

import pytest

from OpenPinch.classes import (
    HeatExchanger,
    HeatExchangerKind,
    HeatExchangerNetwork,
    HeatExchangerStreamRole,
)
from OpenPinch.lib.schemas.synthesis import (
    HeatExchangerNetworkSynthesisResult,
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from OpenPinch.services.heat_exchanger_network_synthesis.grid_diagram import (
    build_grid_model,
)


def test_grid_model_recovers_process_topology_and_branches() -> None:
    network = _network("candidate-a")

    grid_model = build_grid_model(network)

    assert grid_model.hot_streams == ("H1", "H2")
    assert grid_model.cold_streams == ("C1", "C2")
    assert grid_model.stages == (1, 2, 3)
    assert len(grid_model.recovery_matches) == 5
    assert len(grid_model.hot_utility_matches) == 1
    assert len(grid_model.cold_utility_matches) == 1
    assert grid_model.branch_counts[("H1", 3)] == 2
    assert grid_model.branch_counts[("C1", 3)] == 2


def test_grid_diagram_ranks_successful_networks_by_accepted_method() -> None:
    _require_plotly()
    result = _result()

    diagram = result.grid_diagram(solution_rank=1)

    assert diagram.solution_rank == 1
    assert diagram.network.run_id == "accepted-best"
    assert diagram.grid_model.network is diagram.network
    assert diagram.fig.__class__.__module__.startswith("plotly.")


def test_grid_diagram_uses_one_based_ranking_and_reports_missing_rank() -> None:
    _require_plotly()
    result = _result()

    diagram = result.grid_diagram(solution_rank=2)

    assert diagram.network.run_id == "accepted-second"
    with pytest.raises(IndexError, match="solution_rank 3 is unavailable"):
        result.grid_diagram(solution_rank=3)
    with pytest.raises(IndexError, match="1-based"):
        result.grid_diagram(solution_rank=0)


def test_grid_diagram_save_writes_image(tmp_path: Path) -> None:
    _require_plotly()
    result = _result()
    output_path = tmp_path / "grid.png"

    diagram = result.grid_diagram(solution_rank=1)
    diagram.save(output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_grid_diagram_labels_utility_markers_with_short_names() -> None:
    _require_plotly()
    result = _result()

    diagram = result.grid_diagram(solution_rank=1)

    labels = {text.get_text() for text in diagram.ax.texts}
    assert "HU1" in labels
    assert "CU1" in labels
    assert "Utilities.HU1" not in labels
    assert "Cold Utility/CU1" not in labels


def test_grid_diagram_places_endpoint_temperatures_outside_stream_lines() -> None:
    _require_plotly()
    result = _result()

    diagram = result.grid_diagram(solution_rank=1)
    hot_supply = _endpoint_labels(diagram.ax, "377 $^\\circ$C")
    hot_target = _endpoint_labels(diagram.ax, "277 $^\\circ$C")
    cold_supply = _endpoint_labels(diagram.ax, "27 $^\\circ$C")
    cold_target = _endpoint_labels(diagram.ax, "127 $^\\circ$C")

    assert all(label.xy[0] == 0 for label in hot_supply)
    assert all(label.get_position()[0] < 0 for label in hot_supply)
    assert all(label.get_position()[1] == 0 for label in hot_supply)
    assert all(label.get_va() == "center_baseline" for label in hot_supply)
    assert any(label.xy[0] == 1.0 for label in hot_target)
    assert all(label.get_position()[0] > 0 for label in hot_target)
    assert all(label.get_position()[1] == 0 for label in hot_target)
    assert all(label.get_va() == "center_baseline" for label in hot_target)
    assert all(label.xy[0] == 1.0 for label in cold_supply)
    assert all(label.get_position()[0] > 0 for label in cold_supply)
    assert all(label.get_position()[1] == 0 for label in cold_supply)
    assert all(label.get_va() == "center_baseline" for label in cold_supply)
    assert all(label.xy[0] == 0 for label in cold_target)
    assert all(label.get_position()[0] < 0 for label in cold_target)
    assert all(label.get_position()[1] == 0 for label in cold_target)
    assert all(label.get_va() == "center_baseline" for label in cold_target)


def test_grid_diagram_places_intermediate_temperatures_between_exchanger_nodes() -> (
    None
):
    _require_plotly()
    result = _result()

    diagram = result.grid_diagram(solution_rank=1)

    intermediate_hot = _intermediate_labels(diagram.ax, "347 $^\\circ$C")
    intermediate_cold = _intermediate_labels(diagram.ax, "57 $^\\circ$C")
    intermediate_hot_h2 = _intermediate_labels(diagram.ax, "307 $^\\circ$C")
    intermediate_hot_utility = _intermediate_labels(diagram.ax, "287 $^\\circ$C")
    intermediate_cold_c2 = _intermediate_labels(diagram.ax, "67 $^\\circ$C")
    intermediate_cold_utility = _intermediate_labels(diagram.ax, "92 $^\\circ$C")
    hot_midpoints = {round(label.xy[0], 2) for label in intermediate_hot}
    cold_midpoints = {round(label.xy[0], 2) for label in intermediate_cold}

    assert hot_midpoints == {0.28}
    assert {round(label.xy[0], 2) for label in intermediate_hot_h2} == {0.58}
    assert {round(label.xy[0], 2) for label in intermediate_hot_utility} == {0.87}
    assert cold_midpoints == {0.28}
    assert {round(label.xy[0], 2) for label in intermediate_cold_c2} == {0.5}
    assert {round(label.xy[0], 2) for label in intermediate_cold_utility} == {0.21}
    assert len({(label.get_text(), label.xy[1]) for label in intermediate_hot}) == len(
        intermediate_hot
    )
    assert len({(label.get_text(), label.xy[1]) for label in intermediate_cold}) == len(
        intermediate_cold
    )
    assert all(label.get_position()[1] > 0 for label in intermediate_hot)
    assert all(label.get_position()[1] == 2.625 for label in intermediate_hot)
    assert all(label.get_position()[1] > 0 for label in intermediate_hot_h2)
    assert all(label.get_position()[1] > 0 for label in intermediate_hot_utility)
    assert all(label.get_position()[1] > 0 for label in intermediate_cold)
    assert all(label.get_position()[1] > 0 for label in intermediate_cold_c2)
    assert all(label.get_position()[1] > 0 for label in intermediate_cold_utility)


def test_grid_diagram_split_connections_align_with_unit_gap_midpoints() -> None:
    _require_plotly()
    result = _result()

    diagram = result.grid_diagram(solution_rank=1)

    connector_centers_by_color = {
        color: {
            round(sum(line.get_xdata()) / 2, 3)
            for line in diagram.ax.lines
            if line.kwargs.get("color") == color and len(set(line.get_ydata())) > 1
        }
        for color in ("blue", "red")
    }
    assert connector_centers_by_color["blue"] == {0.35, 0.95}
    assert connector_centers_by_color["red"] == {0.35, 0.8}


def test_grid_diagram_right_side_split_temperature_labels_move_right() -> None:
    _require_plotly()

    diagram = _result_for_network(_right_side_split_network()).grid_diagram(
        solution_rank=1
    )

    right_label = _intermediate_labels(diagram.ax, "237 $^\\circ$C")
    assert {round(label.xy[0], 2) for label in right_label} == {0.73}


def test_grid_diagram_centres_duty_labels_below_heat_exchanger_dots() -> None:
    _require_plotly()
    result = _result()

    diagram = result.grid_diagram(solution_rank=1)

    duty_label = next(text for text in diagram.ax.texts if text.get_text() == "1.20 MW")
    assert duty_label.xy == pytest.approx((0.2, 0.56338))
    assert duty_label.get_ha() == "center"
    assert duty_label.get_position()[0] == 0
    assert duty_label.get_position()[1] < 0


def test_grid_diagram_uses_same_vertical_offset_for_utility_duty_labels() -> None:
    _require_plotly()
    result = _result()

    diagram = result.grid_diagram(solution_rank=1)

    recovery_duty = next(
        text for text in diagram.ax.texts if text.get_text() == "1.20 MW"
    )
    utility_duty = next(
        text for text in diagram.ax.texts if text.get_text() == "0.20 MW"
    )
    assert utility_duty.get_position()[1] == recovery_duty.get_position()[1]
    assert utility_duty.get_ha() == "center"
    assert utility_duty.get_va() == "top"


def test_grid_diagram_exchanger_artists_include_hover_tooltips() -> None:
    _require_plotly()
    result = _result()

    diagram = result.grid_diagram(solution_rank=1)

    tooltips = [
        line.openpinch_tooltip
        for line in diagram.ax.lines
        if getattr(line, "openpinch_tooltip", None)
    ]
    assert any("H1 -> C1 | stage 1" in tooltip for tooltip in tooltips)
    assert any("Duty: 1.20 MW" in tooltip for tooltip in tooltips)
    assert any("Area: 123.40 m^2" in tooltip for tooltip in tooltips)


def test_grid_diagram_auto_sizes_height_from_stream_line_width() -> None:
    _require_plotly()
    result = _result()

    default = result.grid_diagram(solution_rank=1)
    thick = result.grid_diagram(solution_rank=1, stream_line_width=20)

    assert thick.fig.layout.height > default.fig.layout.height


def test_grid_diagram_height_is_proportional_to_lane_count() -> None:
    _require_plotly()

    split_four_stream = _result().grid_diagram(solution_rank=1)
    plain_four_stream = _result_for_network(_plain_four_stream_network()).grid_diagram(
        solution_rank=1
    )
    two_stream = _result_for_network(_two_stream_network()).grid_diagram(
        solution_rank=1
    )

    assert split_four_stream.fig.layout.height > plain_four_stream.fig.layout.height
    assert split_four_stream.fig.layout.height > two_stream.fig.layout.height


def test_grid_diagram_marker_radius_scales_with_row_pitch() -> None:
    _require_plotly()

    default = _result().grid_diagram(solution_rank=1)
    thick = _result().grid_diagram(solution_rank=1, stream_line_width=20)

    default_markers = _marker_sizes(default)
    thick_markers = _marker_sizes(thick)

    assert default_markers
    assert min(default_markers) >= 35
    assert max(default_markers) <= 70
    assert max(thick_markers) > max(default_markers)
    assert max(thick_markers) <= 70


def test_grid_diagram_label_offsets_use_geometry_clearance() -> None:
    _require_plotly()

    diagram = _result().grid_diagram(solution_rank=1, stream_line_width=20)

    temperature_label = next(
        text for text in diagram.ax.texts if text.get_text() == "347 $^\\circ$C"
    )
    duty_label = next(text for text in diagram.ax.texts if text.get_text() == "1.20 MW")
    marker_size = max(_marker_sizes(diagram))
    displayed_marker_size = max(12.0, min(marker_size * 0.42, 30.0))

    assert temperature_label.get_position()[1] >= 10
    assert abs(duty_label.get_position()[1]) >= displayed_marker_size * 0.25


def test_grid_diagram_terminal_utility_zones_are_half_stage_width() -> None:
    _require_plotly()

    diagram = _result().grid_diagram(solution_rank=1)

    hot_utility = next(text for text in diagram.ax.texts if text.get_text() == "HU1")
    cold_utility = next(text for text in diagram.ax.texts if text.get_text() == "CU1")

    assert hot_utility.xy[0] == pytest.approx(0.0625)
    assert cold_utility.xy[0] == pytest.approx(0.9375)


def test_grid_diagram_can_temperature_scale_stream_x_positions() -> None:
    _require_plotly()
    result = _result()

    staged = result.grid_diagram(solution_rank=1)
    scaled = result.grid_diagram(solution_rank=1, temperature_scaled=True)

    staged_x = _arrow_x_positions(staged)
    scaled_x = _arrow_x_positions(scaled)

    assert len(staged_x) == len(scaled_x)
    assert len({tuple(round(value, 6) for value in pair) for pair in staged_x}) == 1
    assert len({tuple(round(value, 6) for value in pair) for pair in scaled_x}) >= 2
    assert scaled_x != staged_x


def test_grid_diagram_uses_normalized_stream_geometry() -> None:
    _require_plotly()
    result = _result()

    diagram = result.grid_diagram(solution_rank=1)

    assert diagram.ax.stream_bounds
    assert all(0.0 <= left <= right <= 1.0 for left, right in diagram.ax.stream_bounds)
    assert min(y for text in diagram.ax.texts for y in [text.xy[1]]) >= 0.0
    assert max(y for text in diagram.ax.texts for y in [text.xy[1]]) <= 1.0


def _result() -> HeatExchangerNetworkSynthesisResult:
    pdm_task = _task(
        method="pinch_decomposition",
        task_id="pdm",
        approach_temperature=10.0,
    )
    esm_best_task = _task(
        method="energy_stage_refinement",
        task_id="esm-best",
        approach_temperature=14.0,
    )
    esm_second_task = _task(
        method="energy_stage_refinement",
        task_id="esm-second",
        approach_temperature=18.0,
    )
    return HeatExchangerNetworkSynthesisResult(
        network=_network("accepted-best"),
        run_id="result",
        method="energy_stage_refinement",
        task_outcomes=(
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=pdm_task,
                status="success",
                network=_network("pdm-better-objective"),
                objective_value=1.0,
            ),
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=esm_second_task,
                status="success",
                network=_network("accepted-second"),
                objective_value=200.0,
            ),
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=esm_best_task,
                status="success",
                network=_network("accepted-best"),
                objective_value=100.0,
            ),
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=_task(
                    method="energy_stage_refinement",
                    task_id="failed",
                    approach_temperature=20.0,
                ),
                status="failed",
                objective_value=0.0,
            ),
        ),
    )


def _result_for_network(
    network: HeatExchangerNetwork,
) -> HeatExchangerNetworkSynthesisResult:
    return HeatExchangerNetworkSynthesisResult(
        network=network,
        run_id="result",
        method="energy_stage_refinement",
        task_outcomes=(),
    )


def _require_plotly() -> None:
    pytest.importorskip("plotly")


def _task(
    *,
    method,
    task_id: str,
    approach_temperature: float,
) -> HeatExchangerNetworkSynthesisTask:
    return HeatExchangerNetworkSynthesisTask(
        task_id=task_id,
        run_id="run-1",
        method=method,
        approach_temperature=approach_temperature,
        derivative_threshold=0.5,
        stage_count=3,
    )


def _network(run_id: str) -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(
            _recovery("E1", "H1", "C1", 1, 1200.0),
            _recovery("E2", "H2", "C2", 2, 900.0),
            _recovery("E3", "H1", "C1", 3, 700.0),
            _recovery("E4", "H1", "C2", 3, 300.0),
            _recovery("E5", "H2", "C1", 3, 250.0),
            _hot_utility("Utilities.HU1", "C2", 200.0),
            _cold_utility("H2", "Cold Utility/CU1", 250.0),
            _recovery("tiny", "H2", "C1", 2, 0.5),
        ),
        run_id=run_id,
        method="energy_stage_refinement",
        stage_count=3,
        solver_axis_metadata={
            "axis_maps": {
                "hot_process_streams": {"H1": 0, "H2": 1},
                "cold_process_streams": {"C1": 0, "C2": 1},
            },
        },
        source_metadata={
            "hot_stage_boundary_temperatures": (
                (650.0, 620.0, 590.0, 550.0),
                (650.0, 610.0, 580.0, 550.0),
            ),
            "cold_stage_boundary_temperatures": (
                (400.0, 360.0, 330.0, 300.0),
                (400.0, 370.0, 340.0, 300.0),
            ),
        },
    )


def _plain_four_stream_network() -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(
            _recovery("E1", "H1", "C1", 1, 1200.0),
            _recovery("E2", "H2", "C2", 2, 900.0),
        ),
        run_id="plain-four-stream",
        method="energy_stage_refinement",
        stage_count=2,
    )


def _two_stream_network() -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(_recovery("E1", "H1", "C1", 1, 1200.0),),
        run_id="two-stream",
        method="energy_stage_refinement",
        stage_count=1,
    )


def _right_side_split_network() -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(
            _recovery("E1", "H1", "C1", 1, 1200.0),
            _recovery("E2", "H1", "C2", 2, 900.0),
            _recovery(
                "E3",
                "H1",
                "C3",
                2,
                700.0,
                source_outlet_temperature=510.0,
            ),
            _recovery("E4", "H1", "C4", 3, 300.0),
        ),
        run_id="right-side-split",
        method="energy_stage_refinement",
        stage_count=3,
    )


def _recovery(
    exchanger_id: str,
    hot_stream: str,
    cold_stream: str,
    stage: int,
    duty: float,
    source_outlet_temperature: float = 550.0,
) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=exchanger_id,
        kind=HeatExchangerKind.RECOVERY,
        source_stream=hot_stream,
        sink_stream=cold_stream,
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        stage=stage,
        duty=duty,
        area=123.4,
        source_inlet_temperature=650.0,
        source_outlet_temperature=source_outlet_temperature,
        sink_inlet_temperature=300.0,
        sink_outlet_temperature=400.0,
    )


def _hot_utility(
    hot_utility: str,
    cold_stream: str,
    duty: float,
) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=f"{hot_utility}-{cold_stream}",
        kind=HeatExchangerKind.HOT_UTILITY,
        source_stream=hot_utility,
        sink_stream=cold_stream,
        source_stream_role=HeatExchangerStreamRole.UTILITY,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        duty=duty,
        area=45.6,
        sink_inlet_temperature=365.0,
        sink_outlet_temperature=400.0,
    )


def _cold_utility(
    hot_stream: str,
    cold_utility: str,
    duty: float,
) -> HeatExchanger:
    return HeatExchanger(
        exchanger_id=f"{hot_stream}-{cold_utility}",
        kind=HeatExchangerKind.COLD_UTILITY,
        source_stream=hot_stream,
        sink_stream=cold_utility,
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.UTILITY,
        duty=duty,
        area=67.8,
        source_inlet_temperature=560.0,
        source_outlet_temperature=550.0,
    )


def _arrow_x_positions(diagram) -> list[tuple[float, float]]:
    return sorted(diagram.ax.stream_bounds)


def _marker_sizes(diagram) -> list[float]:
    return [
        float(line.kwargs["markersize"])
        for line in diagram.ax.lines
        if line.kwargs.get("marker")
    ]


def _endpoint_labels(ax, text: str):
    return [
        item
        for item in ax.texts
        if item.get_text() == text and item.get_ha() in {"left", "right"}
    ]


def _intermediate_labels(ax, text: str):
    return [
        item
        for item in ax.texts
        if item.get_text() == text
        and item.get_ha() == "center"
        and item.get_va() == "bottom"
    ]
