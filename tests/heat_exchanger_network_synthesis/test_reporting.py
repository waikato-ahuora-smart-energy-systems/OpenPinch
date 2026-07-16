"""Tests for heat exchanger network grid diagrams."""

from __future__ import annotations

from pathlib import Path

import pytest

from OpenPinch.classes import (
    HeatExchanger,
    HeatExchangerKind,
    HeatExchangerNetwork,
    HeatExchangerPeriodState,
    HeatExchangerStreamRole,
)
from OpenPinch.lib.schemas.synthesis import (
    HeatExchangerNetworkSynthesisResult,
    HeatExchangerNetworkSynthesisTask,
    HeatExchangerNetworkSynthesisTaskOutcome,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.reporting.ranking import (
    network_structure_signature,
)
from OpenPinch.services.heat_exchanger_network_synthesis.common.reporting.verification import (
    _finite,
    _heat_added,
    _heat_removed,
    _optional_float,
    _overall_heat_transfer_coefficient,
    _terminal_lmtd,
    _uses_isothermal_stage_boundaries,
    is_network_feasible,
    verify_network_feasibility,
    verify_synthesis_result,
)
from OpenPinch.services.network_grid_diagram import (
    build_grid_diagram,
    build_grid_model,
)
from OpenPinch.services.network_grid_diagram._dependencies import (
    MissingNetworkGridDiagramDependencyError,
    require_network_grid_diagram_dependency,
)
from OpenPinch.services.network_grid_diagram.constants import (
    _COLD_STREAM_COLOR,
    _COLD_UTILITY_COLOR,
    _HOT_STREAM_COLOR,
    _HOT_UTILITY_COLOR,
    _LABEL_BACKGROUND_COLOR,
    _RECOVERY_MATCH_COLOR,
)
from OpenPinch.services.network_grid_diagram.models import (
    HeatExchangerNetworkGridDiagram,
    HeatExchangerNetworkGridModel,
)
from OpenPinch.services.network_grid_diagram.renderer import (
    _midpoint_temperature,
    _PlotlyAxes,
    _PlotlyGridRenderer,
    _PlotlyLine,
    _split_adjusted_temperature_x,
    _SplitGroup,
    _temperature_scale,
    _validate_stream_line_width,
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


def test_grid_model_skips_unstaged_recovery_and_accumulates_duplicate_utilities() -> (
    None
):
    unstaged_recovery = _recovery("unstaged", "H1", "C1", 1, 50.0).model_copy(
        update={"stage": None}
    )
    network = HeatExchangerNetwork.model_construct(
        exchangers=(
            unstaged_recovery,
            _hot_utility("HU1", "C1", 100.0),
            _hot_utility("HU2", "C1", 25.0),
        ),
        run_id="unstaged",
        method="network_evolution_method",
        stage_count=None,
    )

    grid_model = build_grid_model(network)

    assert grid_model.stages == (1,)
    assert grid_model.recovery_matches == ()
    assert grid_model.hot_utility_matches[0].duty == pytest.approx(125.0)


def test_grid_diagram_service_accepts_one_network() -> None:
    _require_plotly()
    network = _network("standalone")

    diagram = build_grid_diagram(network)

    assert diagram.network is network
    assert diagram.grid_model.network is network
    assert diagram.fig.__class__.__module__.startswith("plotly.")


def test_network_build_grid_diagram_uses_selected_network() -> None:
    _require_plotly()
    network = _network("selected")

    diagram = network.build_grid_diagram(stream_line_width=6.0)

    assert diagram.network is network
    assert diagram.grid_model.network is network
    assert diagram.fig.__class__.__module__.startswith("plotly.")


def test_grid_diagram_service_accepts_multiple_networks_without_index() -> None:
    _require_plotly()
    networks = (_network("first"), _distinct_network("second"))

    diagrams = build_grid_diagram(networks)

    assert isinstance(diagrams, tuple)
    assert [diagram.network.run_id for diagram in diagrams] == ["first", "second"]


def test_grid_diagram_service_uses_zero_based_index() -> None:
    _require_plotly()
    networks = [_network("first"), _distinct_network("second")]

    diagram = build_grid_diagram(networks, index=1)

    assert diagram.network.run_id == "second"


def test_grid_diagram_service_reports_invalid_network_inputs() -> None:
    _require_plotly()

    with pytest.raises(ValueError, match="at least one HeatExchangerNetwork"):
        build_grid_diagram(())
    with pytest.raises(IndexError, match="must be 0"):
        build_grid_diagram(_network("single"), index=1)
    with pytest.raises(IndexError, match="0-based"):
        build_grid_diagram((_network("first"),), index=-1)
    with pytest.raises(IndexError, match="index 2 is unavailable"):
        build_grid_diagram((_network("first"),), index=2)


def test_grid_diagram_service_reports_wrong_input_types() -> None:
    with pytest.raises(TypeError, match="networks must be a HeatExchangerNetwork"):
        build_grid_diagram("not-a-network")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="item 1 is str"):
        build_grid_diagram(  # type: ignore[list-item]
            [_network("first"), "not-a-network"],
        )


def test_grid_diagram_dependency_error_names_extra() -> None:
    with pytest.raises(
        MissingNetworkGridDiagramDependencyError, match="openpinch\\[synthesis\\]"
    ):
        require_network_grid_diagram_dependency(
            "definitely_missing_grid_dependency",
            package="Plotly",
            purpose="grid rendering",
        )


def test_grid_diagram_save_uses_html_writer_for_html_paths(tmp_path: Path) -> None:
    class FakeFigure:
        def __init__(self):
            self.html_path = None
            self.image_path = None

        def show(self):
            self.shown = True

        def write_html(self, path):
            self.html_path = path

        def write_image(self, path):
            self.image_path = path

    fig = FakeFigure()
    network = _network("save-html")
    diagram = HeatExchangerNetworkGridDiagram(
        fig=fig,
        ax=object(),
        network=network,
        grid_model=build_grid_model(network),
    )

    diagram.show()
    diagram.save(tmp_path / "grid.html")

    assert fig.shown is True
    assert fig.html_path == tmp_path / "grid.html"
    assert fig.image_path is None


def test_grid_diagram_ranks_successful_networks_by_accepted_method() -> None:
    _require_plotly()
    result = _result()

    diagram = result.grid_diagram(solution_rank=1)

    assert diagram.network.run_id == "accepted-best"
    assert diagram.grid_model.network is diagram.network
    assert diagram.fig.__class__.__module__.startswith("plotly.")


def test_result_ranks_only_unique_network_structures() -> None:
    result = _result()

    ranked = result.get_n_best_networks()

    assert [outcome.network.run_id for outcome in ranked] == [
        "accepted-best",
        "accepted-second",
    ]
    assert network_structure_signature(
        ranked[0].network
    ) != network_structure_signature(ranked[1].network)


def test_result_ranks_stage_shifted_networks_as_duplicate_structures() -> None:
    best_task = _task(
        method="network_evolution_method",
        task_id="esm-stage-best",
        approach_temperature=14.0,
    )
    shifted_task = _task(
        method="network_evolution_method",
        task_id="esm-stage-shifted",
        approach_temperature=16.0,
    )
    distinct_task = _task(
        method="network_evolution_method",
        task_id="esm-distinct",
        approach_temperature=18.0,
    )
    best_network = _stage_gap_network("stage-best")
    shifted_network = _stage_compressed_network("stage-shifted")
    distinct_network = _distinct_network("stage-distinct")
    result = HeatExchangerNetworkSynthesisResult(
        network=best_network,
        run_id="result",
        method="network_evolution_method",
        ranked_networks=(
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=shifted_task,
                status="success",
                network=shifted_network,
                objective_value=100.05,
            ),
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=distinct_task,
                status="success",
                network=distinct_network,
                objective_value=200.0,
            ),
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=best_task,
                status="success",
                network=best_network,
                objective_value=100.0,
            ),
        ),
    )

    ranked = result.get_n_best_networks()

    assert network_structure_signature(best_network) == network_structure_signature(
        shifted_network
    )
    assert [outcome.network.run_id for outcome in ranked] == [
        "stage-best",
        "stage-distinct",
    ]


def test_result_selects_ranked_network() -> None:
    result = _result()

    assert result.select_network() is result
    assert result.network.run_id == "accepted-best"
    assert result.task_id == "esm-best"
    assert result.objective_values == {}

    result.select_network(solution_rank=2)

    assert result.network.run_id == "accepted-second"
    assert result.task_id == "esm-second"
    assert result.method == "network_evolution_method"
    assert len(result.ranked_networks) == 2


def test_network_feasibility_checks_cost_and_heat_balance() -> None:
    feasible = _feasible_balance_network("feasible")

    assert verify_network_feasibility(feasible) == ()
    assert is_network_feasible(feasible) is True
    assert verify_network_feasibility(None) == ("network output is missing",)
    assert is_network_feasible(None) is False

    under_costed = _feasible_balance_network(
        "under-costed",
        total_annual_cost=75.0,
    )
    heat_balance_failure = _feasible_balance_network(
        "bad-balance",
        cold_utility_duty=400.0,
    )

    assert any(
        "total annual cost" in reason
        for reason in verify_network_feasibility(under_costed)
    )
    assert any(
        "hot stream H1 heat removed" in reason
        for reason in verify_network_feasibility(heat_balance_failure)
    )


def test_synthesis_result_verification_reports_contract_failures() -> None:
    accepted = HeatExchangerNetwork(
        exchangers=(),
        run_id="network-run",
        task_id="different-task",
    )
    infeasible_ranked = _feasible_balance_network(
        "ranked",
        total_annual_cost=75.0,
    )
    result = HeatExchangerNetworkSynthesisResult(
        network=accepted,
        run_id="result-run",
        task_id="accepted-task",
        method="network_evolution_method",
        ranked_networks=(
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=_task(
                    method="network_evolution_method",
                    task_id="missing-network",
                    approach_temperature=14.0,
                ),
                status="success",
                network=None,
                objective_value=1.0,
            ),
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=_task(
                    method="network_evolution_method",
                    task_id="bad-network",
                    approach_temperature=15.0,
                ),
                status="success",
                network=infeasible_ranked,
                objective_value=75.0,
            ),
        ),
    )

    failures = verify_synthesis_result(result)

    assert any("at least one exchanger" in failure for failure in failures)
    assert any("network run_id must match" in failure for failure in failures)
    assert any("network task_id must match" in failure for failure in failures)
    assert any("total annual cost metadata" in failure for failure in failures)
    assert any("missing network output" in failure for failure in failures)
    assert any("bad-network infeasible" in failure for failure in failures)


def test_network_feasibility_reports_summary_and_temperature_failures() -> None:
    summary_bad = _feasible_balance_network("summary").model_copy(
        update={
            "summary_metrics": {
                "total_units": 99,
                "recovery_load": 1.0,
            }
        }
    )
    temperature_bad = HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="bad-temperature",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                period_states=(
                    _period_state(
                        duty=10.0,
                        source_inlet_temperature=100.0,
                        source_outlet_temperature=120.0,
                        sink_inlet_temperature=80.0,
                        sink_outlet_temperature=70.0,
                    ),
                ),
                area=10.0,
            ),
            HeatExchanger(
                exchanger_id="crossed-terminal",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H2",
                sink_stream="C2",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                period_states=(
                    _period_state(
                        duty=10.0,
                        source_inlet_temperature=100.0,
                        source_outlet_temperature=70.0,
                        sink_inlet_temperature=80.0,
                        sink_outlet_temperature=120.0,
                    ),
                ),
                area=10.0,
            ),
        ),
        run_id="temperature",
    )
    inactive_bad = HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="inactive",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                period_states=(
                    _period_state(
                        duty=10.0,
                        active=False,
                        source_inlet_temperature=100.0,
                        source_outlet_temperature=120.0,
                        sink_inlet_temperature=80.0,
                        sink_outlet_temperature=70.0,
                    ),
                ),
                area=None,
            ),
        ),
        run_id="inactive",
        solver_axis_metadata={
            "axis_maps": {
                "hot_process_streams": {"H1": 0},
                "cold_process_streams": {"C1": 0},
            }
        },
        source_metadata={
            "hot_stream_heat_capacity_flowrates": [1.0],
            "cold_stream_heat_capacity_flowrates": [1.0],
            "hot_stream_heat_transfer_coefficients": [1.0],
            "cold_stream_heat_transfer_coefficients": [1.0],
        },
    )

    summary_failures = verify_network_feasibility(summary_bad)
    temperature_failures = verify_network_feasibility(temperature_bad)

    assert any("total_units summary" in reason for reason in summary_failures)
    assert any("recovery_load summary" in reason for reason in summary_failures)
    assert any(
        "source temperature increases" in reason for reason in temperature_failures
    )
    assert any(
        "sink temperature decreases" in reason for reason in temperature_failures
    )
    assert any(
        "hot inlet is colder than cold outlet" in reason
        for reason in temperature_failures
    )
    assert any(
        "hot outlet is colder than cold inlet" in reason
        for reason in temperature_failures
    )
    assert verify_network_feasibility(inactive_bad) == ()


def test_network_feasibility_checks_minimum_approach_temperature() -> None:
    network = _feasible_balance_network(
        "low-approach",
        source_metadata={
            "solver_dTmin": 175.0,
            "hot_stream_heat_capacity_flowrates": [10.0],
            "cold_stream_heat_capacity_flowrates": [10.0],
            "hot_stream_supply_temperatures": [500.0],
            "hot_stream_target_temperatures": [400.0],
            "cold_stream_supply_temperatures": [300.0],
            "cold_stream_target_temperatures": [350.0],
        },
    )

    assert any(
        "minimum approach" in reason for reason in verify_network_feasibility(network)
    )


def test_network_feasibility_checks_installed_area_against_lmtd() -> None:
    network = _feasible_balance_network("under-area", recovery_area=1.0)

    assert any(
        "below required area" in reason
        for reason in verify_network_feasibility(network)
    )


def test_network_feasibility_checks_area_presence_and_utility_area_branches() -> None:
    missing_area = HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="missing-area",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                period_states=(
                    _period_state(
                        duty=20.0,
                        source_inlet_temperature=150.0,
                        source_outlet_temperature=120.0,
                        sink_inlet_temperature=60.0,
                        sink_outlet_temperature=90.0,
                    ),
                ),
                area=None,
            ),
        ),
        run_id="missing-area",
    )
    hot_utility_under_area = HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="HU1-C1",
                kind=HeatExchangerKind.HOT_UTILITY,
                source_stream="HU1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.UTILITY,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                period_states=(
                    _period_state(
                        duty=200.0,
                        source_inlet_temperature=250.0,
                        source_outlet_temperature=240.0,
                        sink_inlet_temperature=100.0,
                        sink_outlet_temperature=150.0,
                    ),
                ),
                area=0.01,
            ),
        ),
        run_id="hot-utility-area",
        solver_axis_metadata={
            "axis_maps": {
                "hot_utilities": {"HU1": 0},
                "cold_process_streams": {"C1": 0},
            }
        },
        source_metadata={
            "hot_utility_heat_transfer_coefficients": [1.0],
            "cold_stream_heat_transfer_coefficients": [1.0],
        },
    )
    cold_utility_under_area = HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="H1-CU1",
                kind=HeatExchangerKind.COLD_UTILITY,
                source_stream="H1",
                sink_stream="CU1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.UTILITY,
                period_states=(
                    _period_state(
                        duty=200.0,
                        source_inlet_temperature=250.0,
                        source_outlet_temperature=200.0,
                        sink_inlet_temperature=80.0,
                        sink_outlet_temperature=90.0,
                    ),
                ),
                area=0.01,
            ),
        ),
        run_id="cold-utility-area",
        solver_axis_metadata={
            "axis_maps": {
                "hot_process_streams": {"H1": 0},
                "cold_utilities": {"CU1": 0},
            }
        },
        source_metadata={
            "hot_stream_heat_transfer_coefficients": [1.0],
            "cold_utility_heat_transfer_coefficients": [1.0],
        },
    )

    assert any(
        "no positive area" in reason
        for reason in verify_network_feasibility(missing_area)
    )
    assert any(
        "below required area" in reason
        for reason in verify_network_feasibility(hot_utility_under_area)
    )
    assert any(
        "below required area" in reason
        for reason in verify_network_feasibility(cold_utility_under_area)
    )


def test_network_feasibility_skips_match_level_checks_for_isothermal_stage_models() -> (
    None
):
    network = HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="stage-boundary-match",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                period_states=(
                    _period_state(
                        duty=999.0,
                        source_inlet_temperature=300.0,
                        source_outlet_temperature=250.0,
                        sink_inlet_temperature=100.0,
                        sink_outlet_temperature=150.0,
                    ),
                ),
                area=0.01,
            ),
        ),
        run_id="isothermal-stage",
        source_metadata={
            "solver_non_isothermal_model": False,
            "solver_framework": "PDM",
            "hot_stream_heat_transfer_coefficients": [1.0],
            "cold_stream_heat_transfer_coefficients": [1.0],
        },
        solver_axis_metadata={
            "axis_maps": {
                "hot_process_streams": {"H1": 0},
                "cold_process_streams": {"C1": 0},
            }
        },
    )

    failures = verify_network_feasibility(network)

    assert not any("heat balance" in reason for reason in failures)
    assert not any("below required area" in reason for reason in failures)
    assert _uses_isothermal_stage_boundaries(network.source_metadata) is True
    assert (
        _uses_isothermal_stage_boundaries(
            {"solver_non_isothermal_model": False, "solver_framework": "EVM"}
        )
        is False
    )


def test_network_feasibility_reports_cold_stream_balance_and_handles_missing_temperatures() -> (
    None
):
    cold_balance_failure = _feasible_balance_network("cold-balance").model_copy(
        update={
            "exchangers": (
                _feasible_balance_network("cold-balance").exchangers[0],
                HeatExchanger(
                    exchanger_id="HU1-C1",
                    kind=HeatExchangerKind.HOT_UTILITY,
                    source_stream="HU1",
                    sink_stream="C1",
                    source_stream_role=HeatExchangerStreamRole.UTILITY,
                    sink_stream_role=HeatExchangerStreamRole.PROCESS,
                    period_states=(
                        _period_state(
                            duty=50.0,
                            sink_inlet_temperature=350.0,
                            sink_outlet_temperature=360.0,
                        ),
                    ),
                    area=10.0,
                ),
            )
        }
    )
    missing_temperatures = HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="missing-temperature",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                period_states=(
                    _period_state(
                        duty=20.0,
                        source_inlet_temperature=150.0,
                        sink_inlet_temperature=60.0,
                    ),
                ),
                area=10.0,
            ),
        ),
        run_id="missing-temperature",
        solver_axis_metadata={
            "axis_maps": {
                "hot_process_streams": {"H1": 0},
                "cold_process_streams": {"C1": 0},
            }
        },
        source_metadata={
            "hot_stream_heat_capacity_flowrates": [1.0],
            "cold_stream_heat_capacity_flowrates": [1.0],
            "hot_stream_supply_temperatures": [150.0],
            "cold_stream_supply_temperatures": [60.0],
        },
    )

    assert any(
        "cold stream C1 heat added" in reason
        for reason in verify_network_feasibility(cold_balance_failure)
    )
    assert not any(
        "heat balance" in reason
        for reason in verify_network_feasibility(missing_temperatures)
    )


def test_verification_numeric_helpers_cover_invalid_inputs_and_defensive_branches() -> (
    None
):
    assert _finite("bad-number") is False
    assert _optional_float("bad-number") is None
    assert _heat_removed(2.0, None, 50.0) is None
    assert _heat_added(2.0, 50.0, None) is None

    missing_terminal = HeatExchanger(
        exchanger_id="missing-terminal",
        kind=HeatExchangerKind.RECOVERY,
        source_stream="H1",
        sink_stream="C1",
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        stage=1,
        period_states=(
            _period_state(
                duty=10.0,
                source_inlet_temperature=100.0,
                sink_inlet_temperature=80.0,
            ),
        ),
        area=1.0,
    )
    negative_terminal = HeatExchanger(
        exchanger_id="negative-terminal",
        kind=HeatExchangerKind.RECOVERY,
        source_stream="H1",
        sink_stream="C1",
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        stage=1,
        period_states=(
            _period_state(
                duty=10.0,
                source_inlet_temperature=100.0,
                source_outlet_temperature=80.0,
                sink_inlet_temperature=90.0,
                sink_outlet_temperature=120.0,
            ),
        ),
        area=1.0,
    )
    invalid_kind = HeatExchanger.model_construct(
        exchanger_id="invalid-kind",
        kind="custom-kind",
        source_stream="H1",
        sink_stream="C1",
        source_stream_role=HeatExchangerStreamRole.PROCESS,
        sink_stream_role=HeatExchangerStreamRole.PROCESS,
        stage=1,
        period_states=(_period_state(duty=10.0),),
        area=1.0,
    )

    assert _terminal_lmtd(missing_terminal.state()) is None
    assert _terminal_lmtd(negative_terminal.state()) is None
    assert (
        _overall_heat_transfer_coefficient(
            invalid_kind,
            hot_htc={},
            cold_htc={},
            hot_utility_htc={},
            cold_utility_htc={},
        )
        is None
    )


def test_ranked_network_selection_skips_infeasible_low_tac_candidate() -> None:
    low_tac_infeasible = _feasible_balance_network(
        "low-tac-infeasible",
        total_annual_cost=75.0,
    )
    feasible = _feasible_balance_network("feasible")
    low_task = _task(
        method="network_evolution_method",
        task_id="low",
        approach_temperature=14.0,
    )
    feasible_task = _task(
        method="network_evolution_method",
        task_id="feasible",
        approach_temperature=14.0,
    )
    result = HeatExchangerNetworkSynthesisResult(
        network=low_tac_infeasible,
        run_id="result",
        method="network_evolution_method",
        ranked_networks=(
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=low_task,
                status="success",
                network=low_tac_infeasible,
                objective_value=75.0,
            ),
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=feasible_task,
                status="success",
                network=feasible,
                objective_value=150.0,
            ),
        ),
    )

    result.select_network()

    assert result.network.run_id == "feasible"
    assert result.task_id == "feasible"
    assert [outcome.network.run_id for outcome in result.ranked_networks] == [
        "feasible"
    ]


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

    intermediate_hot = _intermediate_labels(diagram.ax, "377 $^\\circ$C")
    intermediate_cold = _intermediate_labels(diagram.ax, "27 $^\\circ$C")
    intermediate_hot_utility = _intermediate_labels(diagram.ax, "287 $^\\circ$C")
    hot_midpoints = {round(label.xy[0], 2) for label in intermediate_hot}
    cold_midpoints = {round(label.xy[0], 2) for label in intermediate_cold}

    assert hot_midpoints == {0.28, 0.58}
    assert {round(label.xy[0], 2) for label in intermediate_hot_utility} == {0.87}
    assert cold_midpoints == {0.28, 0.5}
    assert {
        round(label.xy[0], 2)
        for label in _intermediate_labels(diagram.ax, "92 $^\\circ$C")
    } == {0.21}
    assert len({(label.get_text(), label.xy[1]) for label in intermediate_hot}) == len(
        intermediate_hot
    )
    assert len({(label.get_text(), label.xy[1]) for label in intermediate_cold}) == len(
        intermediate_cold
    )
    assert all(label.get_position()[1] > 0 for label in intermediate_hot)
    assert all(label.get_position()[1] == 2.625 for label in intermediate_hot)
    assert all(label.get_position()[1] > 0 for label in intermediate_hot_utility)
    assert all(label.get_position()[1] > 0 for label in intermediate_cold)
    assert not _intermediate_labels(diagram.ax, "347 $^\\circ$C")
    assert not _intermediate_labels(diagram.ax, "307 $^\\circ$C")


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
        for color in (_COLD_STREAM_COLOR, _HOT_STREAM_COLOR)
    }
    assert connector_centers_by_color[_COLD_STREAM_COLOR] == {0.35, 0.95}
    assert connector_centers_by_color[_HOT_STREAM_COLOR] == {0.35, 0.8}


def test_grid_diagram_right_side_split_temperature_labels_move_right() -> None:
    _require_plotly()

    diagram = _result_for_network(_right_side_split_network()).grid_diagram(
        solution_rank=1
    )

    split_label = _intermediate_labels(diagram.ax, "377 $^\\circ$C")
    assert {round(label.xy[0], 2) for label in split_label} == {0.5}
    assert len(split_label) == 1


def test_grid_diagram_hot_utility_intermediate_temperature_uses_utility_inlet() -> None:
    _require_plotly()

    diagram = _result_for_network(_supply_side_hot_utility_network()).grid_diagram(
        solution_rank=1
    )

    hot_utility = next(text for text in diagram.ax.texts if text.get_text() == "HU1")
    recovery = next(
        line
        for line in diagram.ax.lines
        if line.kwargs.get("marker")
        and line.kwargs.get("color") == _RECOVERY_MATCH_COLOR
    )

    assert hot_utility.xy[0] < min(recovery.get_xdata())
    assert _intermediate_labels(diagram.ax, "-3 $^\\circ$C")
    assert not _intermediate_labels(diagram.ax, "27 $^\\circ$C")


def test_grid_diagram_orders_stage_matches_by_mid_temperature_attributes() -> None:
    _require_plotly()

    diagram = _result_for_network(_out_of_order_midpoint_network()).grid_diagram(
        solution_rank=1
    )
    recovery_lines = {
        line.openpinch_tooltip.split(" | ", maxsplit=1)[0]: line
        for line in diagram.ax.lines
        if line.kwargs.get("marker")
        and line.kwargs.get("color") == _RECOVERY_MATCH_COLOR
    }

    x_positions = {match: line.get_xdata()[0] for match, line in recovery_lines.items()}
    assert x_positions["H2 -> C2"] < x_positions["H3 -> C3"]
    assert x_positions["H3 -> C3"] < x_positions["H1 -> C1"]


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


def test_grid_diagram_ignores_requested_stream_line_width_for_scaled_geometry() -> None:
    _require_plotly()
    result = _result()

    default = result.grid_diagram(solution_rank=1)
    thick = result.grid_diagram(solution_rank=1, stream_line_width=20)

    assert thick.fig.layout.height == default.fig.layout.height
    assert max(_marker_sizes(thick)) == pytest.approx(max(_marker_sizes(default)))
    assert max(_line_widths(thick)) == pytest.approx(max(_line_widths(default)))


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


def test_grid_diagram_marker_radius_is_bounded_by_row_pitch() -> None:
    _require_plotly()

    diagrams = (
        _result().grid_diagram(solution_rank=1),
        _result_for_network(_plain_four_stream_network()).grid_diagram(solution_rank=1),
        _result_for_network(_two_stream_network()).grid_diagram(solution_rank=1),
    )

    for diagram in diagrams:
        markers = _marker_sizes(diagram)
        assert markers
        assert min(markers) >= 35
        assert max(markers) <= 70


def test_grid_diagram_label_offsets_use_geometry_clearance() -> None:
    _require_plotly()

    diagram = _result().grid_diagram(solution_rank=1, stream_line_width=20)

    temperature_label = _intermediate_labels(diagram.ax, "377 $^\\circ$C")[0]
    duty_label = next(text for text in diagram.ax.texts if text.get_text() == "1.20 MW")
    marker_size = max(_marker_sizes(diagram))
    displayed_marker_size = max(12.0, min(marker_size * 0.42, 30.0))
    line_width = max(_line_widths(diagram))

    assert temperature_label.get_position()[1] >= line_width * 0.5
    assert abs(duty_label.get_position()[1]) >= displayed_marker_size * 0.5


def test_grid_diagram_line_width_is_marker_radius_ratio() -> None:
    _require_plotly()

    diagram = _result().grid_diagram(solution_rank=1, stream_line_width=20)

    assert max(_line_widths(diagram)) == pytest.approx(
        max(_marker_sizes(diagram)) * 0.1
    )
    assert max(_recovery_line_widths(diagram)) == pytest.approx(
        max(_line_widths(diagram)) * 0.5
    )


def test_grid_diagram_uses_stream_and_utility_colours() -> None:
    _require_plotly()

    diagram = _result().grid_diagram(solution_rank=1)
    figure_line_colours = {
        trace.line.color for trace in diagram.fig.data if getattr(trace, "line", None)
    }
    utility_colours = {
        line.kwargs.get("color")
        for line in diagram.ax.lines
        if line.kwargs.get("marker")
        and line.kwargs.get("color") in {_HOT_UTILITY_COLOR, _COLD_UTILITY_COLOR}
    }

    assert _HOT_STREAM_COLOR in figure_line_colours
    assert _COLD_STREAM_COLOR in figure_line_colours
    assert _RECOVERY_MATCH_COLOR in figure_line_colours
    assert utility_colours == {_HOT_UTILITY_COLOR, _COLD_UTILITY_COLOR}


def test_grid_diagram_labels_have_background_to_read_over_connectors() -> None:
    _require_plotly()

    diagram = _result().grid_diagram(solution_rank=1)
    annotations_by_text = {
        annotation.text: annotation for annotation in diagram.fig.layout.annotations
    }

    assert annotations_by_text["1.20 MW"].bgcolor == _LABEL_BACKGROUND_COLOR
    assert annotations_by_text["377 °C"].bgcolor == _LABEL_BACKGROUND_COLOR


def test_grid_diagram_terminal_utility_positions_use_utility_sections() -> None:
    _require_plotly()

    diagram = _result().grid_diagram(solution_rank=1)

    hot_utility = next(text for text in diagram.ax.texts if text.get_text() == "HU1")
    cold_utility = next(text for text in diagram.ax.texts if text.get_text() == "CU1")

    assert hot_utility.xy[0] == pytest.approx(0.0625)
    assert cold_utility.xy[0] == pytest.approx(0.9375)


def test_plotly_axes_text_and_ticks_round_trip_to_figure() -> None:
    go = pytest.importorskip("plotly.graph_objects")
    fig = go.Figure()
    ax = _PlotlyAxes(fig, go)

    label = ax.text(
        1.0,
        2.0,
        "95 $^\\circ$C",
        fontsize=12,
        ha="center",
        va="top",
        bgcolor="#fff",
        borderpad=2,
    )
    ax.set_yticks([1.0, 2.0], ["C1", "H1"])
    line = _PlotlyLine([0.0, 1.0], [1.0, 1.0], marker="o", markersize=100.0)
    ax.add_line(line)
    ax.update_line_hover(line, "A\nB")

    assert label.get_text() == "95 $^\\circ$C"
    assert ax.get_yticks() == [1.0, 2.0]
    assert fig.layout.annotations[0].text == "95 °C"
    assert fig.layout.annotations[0].bgcolor == "#fff"
    assert fig.data[-1].marker.size == 30.0
    assert fig.data[-1].hovertemplate == "A<br>B<extra></extra>"


def test_grid_renderer_helper_edges_cover_temperature_and_split_fallbacks() -> None:
    go = pytest.importorskip("plotly.graph_objects")
    renderer = _PlotlyGridRenderer(
        build_grid_model(_network("helper-edges")),
        graph_objects=go,
        stream_line_width=4.0,
        temperature_scaled=False,
    )
    recovery_match = renderer.grid_model.recovery_matches[0]

    assert renderer._recovery_match_x_pair(recovery_match) == (0.0, 0.0)
    assert renderer._cold_stream_unit_x_positions(renderer.cold_index["C2"])
    assert renderer._hot_stream_unit_x_positions(renderer.hot_index["H2"])
    renderer._draw_split_branches(
        stream="H1",
        branch_matches=[0.5],
        stream_unit_x=[0.2, 0.5, 0.8],
        y0=0.4,
        color=_HOT_STREAM_COLOR,
    )

    fallback_renderer = object.__new__(_PlotlyGridRenderer)
    fallback_renderer.temperature_scaled = True
    fallback_renderer.temperature_scale = (100.0, 100.0)
    fallback_renderer.x_start = 0.0
    fallback_renderer.x_finish = 1.0
    assert fallback_renderer._x_from_temperature(None) == pytest.approx(0.0)
    assert fallback_renderer._x_from_temperature(100.0) == pytest.approx(0.5)

    empty_model = HeatExchangerNetworkGridModel(
        network=HeatExchangerNetwork(),
        period_id="0",
        hot_streams=(),
        cold_streams=(),
        stages=(),
        recovery_matches=(),
        hot_utility_matches=(),
        cold_utility_matches=(),
        branch_counts={},
    )
    assert _temperature_scale(empty_model) == (0.0, 1.0)
    assert _midpoint_temperature(None, None) is None
    with pytest.raises(ValueError, match="stream_line_width"):
        _validate_stream_line_width(0.0)

    split_group = _SplitGroup(
        start_x=0.2,
        end_x=0.8,
        left_connector_x=0.3,
        right_connector_x=0.7,
    )
    assert _split_adjusted_temperature_x(0.5, 0.6, split_group) == pytest.approx(0.5)


def test_grid_renderer_draw_streams_handles_empty_cold_branch_and_stage_lines() -> None:
    go = pytest.importorskip("plotly.graph_objects")
    cold_only_model = HeatExchangerNetworkGridModel(
        network=HeatExchangerNetwork(),
        period_id="0",
        hot_streams=(),
        cold_streams=("C1",),
        stages=(1,),
        recovery_matches=(),
        hot_utility_matches=(),
        cold_utility_matches=(),
        branch_counts={},
    )

    cold_only_renderer = _PlotlyGridRenderer(
        cold_only_model,
        graph_objects=go,
        stream_line_width=4.0,
        temperature_scaled=False,
    )
    assert cold_only_renderer.cold_y_coords

    staged_renderer = _PlotlyGridRenderer(
        build_grid_model(_network("stage-lines")),
        graph_objects=go,
        stream_line_width=4.0,
        temperature_scaled=False,
    )
    line_count = len(staged_renderer.ax.lines)
    staged_renderer.draw_stages = True
    staged_renderer.draw_streams()

    assert len(staged_renderer.ax.lines) > line_count
    assert any(
        line.kwargs.get("linestyle") == "dashed"
        for line in staged_renderer.ax.lines[line_count:]
    )


def test_grid_renderer_duplicate_labels_without_offsets_stack_downward() -> None:
    class FakeAxes:
        def __init__(self) -> None:
            self.calls: list[tuple[float, float, str]] = []

        def text(self, x, y, text, **_kwargs):
            self.calls.append((x, y, text))

    renderer = object.__new__(_PlotlyGridRenderer)
    renderer._label_slot_counts = {}
    renderer.size_of_font = 18
    renderer.ax = FakeAxes()

    renderer._add_label(0.25, 0.5, "A")
    renderer._add_label(0.25, 0.5, "B")

    assert renderer.ax.calls[0] == (0.25, 0.5, "A")
    assert renderer.ax.calls[1][0] == pytest.approx(0.25)
    assert renderer.ax.calls[1][1] == pytest.approx(0.15)
    assert renderer.ax.calls[1][2] == "B"


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


def test_grid_diagram_temperature_scaled_recovery_matches_are_diagonal() -> None:
    _require_plotly()
    result = _result()

    scaled = result.grid_diagram(solution_rank=1, temperature_scaled=True)

    recovery_lines = [
        line
        for line in scaled.ax.lines
        if line.kwargs.get("marker")
        and line.kwargs.get("color") == _RECOVERY_MATCH_COLOR
    ]
    assert any(line.get_xdata()[0] != line.get_xdata()[1] for line in recovery_lines)


def test_grid_diagram_temperature_scaled_uses_process_midpoints_for_recovery() -> None:
    _require_plotly()

    scaled = _result_for_network(_temperature_scaled_network()).grid_diagram(
        solution_rank=1,
        temperature_scaled=True,
    )

    recovery_line = next(
        line
        for line in scaled.ax.lines
        if line.kwargs.get("marker")
        and line.kwargs.get("color") == _RECOVERY_MATCH_COLOR
    )
    hot_utility_line = next(
        line
        for line in scaled.ax.lines
        if line.kwargs.get("marker") and line.kwargs.get("color") == _HOT_UTILITY_COLOR
    )
    cold_utility_line = next(
        line
        for line in scaled.ax.lines
        if line.kwargs.get("marker") and line.kwargs.get("color") == _COLD_UTILITY_COLOR
    )

    assert recovery_line.get_xdata() == pytest.approx((0.1, 0.7))
    assert hot_utility_line.get_xdata() == pytest.approx((0.9, 0.9))
    assert cold_utility_line.get_xdata() == pytest.approx((0.3, 0.3))


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
        method="pinch_design_method",
        task_id="pdm",
        approach_temperature=10.0,
    )
    esm_best_task = _task(
        method="network_evolution_method",
        task_id="esm-best",
        approach_temperature=14.0,
    )
    esm_second_task = _task(
        method="network_evolution_method",
        task_id="esm-second",
        approach_temperature=18.0,
    )
    return HeatExchangerNetworkSynthesisResult(
        network=_network("accepted-best"),
        run_id="result",
        method="network_evolution_method",
        ranked_networks=(
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=pdm_task,
                status="success",
                network=_network("pdm-better-objective"),
                objective_value=1.0,
            ),
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=esm_second_task,
                status="success",
                network=_distinct_network("accepted-second"),
                objective_value=200.0,
            ),
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=_task(
                    method="network_evolution_method",
                    task_id="esm-duplicate",
                    approach_temperature=16.0,
                ),
                status="success",
                network=_network("accepted-duplicate"),
                objective_value=110.0,
            ),
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=esm_best_task,
                status="success",
                network=_network("accepted-best"),
                objective_value=100.0,
            ),
            HeatExchangerNetworkSynthesisTaskOutcome(
                task=_task(
                    method="network_evolution_method",
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
        method="network_evolution_method",
        ranked_networks=(),
    )


def _feasible_balance_network(
    run_id: str,
    *,
    total_annual_cost: float = 150.0,
    cold_utility_duty: float = 500.0,
    recovery_area: float = 50.0,
    source_metadata: dict[str, object] | None = None,
) -> HeatExchangerNetwork:
    recovery_duty = 500.0
    return HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="R1",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                period_states=(
                    _period_state(
                        duty=recovery_duty,
                        source_inlet_temperature=500.0,
                        source_outlet_temperature=450.0,
                        sink_inlet_temperature=300.0,
                        sink_outlet_temperature=350.0,
                    ),
                ),
                area=recovery_area,
            ),
            HeatExchanger(
                exchanger_id="CU1",
                kind=HeatExchangerKind.COLD_UTILITY,
                source_stream="H1",
                sink_stream="CU",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.UTILITY,
                period_states=(
                    _period_state(
                        duty=cold_utility_duty,
                        source_inlet_temperature=450.0,
                        source_outlet_temperature=400.0,
                        sink_inlet_temperature=290.0,
                        sink_outlet_temperature=300.0,
                    ),
                ),
                area=25.0,
            ),
        ),
        run_id=run_id,
        method="network_evolution_method",
        stage_count=1,
        total_annual_cost=total_annual_cost,
        utility_cost=50.0,
        capital_cost=100.0,
        summary_metrics={
            "total_units": 2,
            "recovery_units": 1,
            "hot_utility_units": 0,
            "cold_utility_units": 1,
            "recovery_load": recovery_duty,
            "hot_utility_load": 0.0,
            "cold_utility_load": cold_utility_duty,
        },
        solver_axis_metadata={
            "axis_maps": {
                "hot_process_streams": {"H1": 0},
                "cold_process_streams": {"C1": 0},
            }
        },
        source_metadata=source_metadata
        or {
            "hot_stream_heat_capacity_flowrates": [10.0],
            "cold_stream_heat_capacity_flowrates": [10.0],
            "hot_stream_heat_transfer_coefficients": [1.0],
            "cold_stream_heat_transfer_coefficients": [1.0],
            "hot_stream_supply_temperatures": [500.0],
            "hot_stream_target_temperatures": [400.0],
            "cold_stream_supply_temperatures": [300.0],
            "cold_stream_target_temperatures": [350.0],
        },
    )


def _require_plotly() -> None:
    pytest.importorskip("plotly")


def _period_state(
    *,
    duty: float,
    active: bool = True,
    approach_temperatures: tuple[float, ...] = (),
    source_inlet_temperature: float | None = None,
    source_outlet_temperature: float | None = None,
    sink_inlet_temperature: float | None = None,
    sink_outlet_temperature: float | None = None,
) -> HeatExchangerPeriodState:
    return HeatExchangerPeriodState(
        period_id="0",
        period_idx=0,
        duty=duty,
        active=active,
        approach_temperatures=approach_temperatures,
        source_inlet_temperature=source_inlet_temperature,
        source_outlet_temperature=source_outlet_temperature,
        sink_inlet_temperature=sink_inlet_temperature,
        sink_outlet_temperature=sink_outlet_temperature,
    )


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
            _recovery("tiny", "H2", "C1", 2, 0.001),
        ),
        run_id=run_id,
        method="network_evolution_method",
        stage_count=3,
        solver_axis_metadata={
            "axis_maps": {
                "hot_process_streams": {"H1": 0, "H2": 1},
                "cold_process_streams": {"C1": 0, "C2": 1},
            },
        },
    )


def _distinct_network(run_id: str) -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(
            _recovery("E1", "H1", "C1", 1, 1200.0),
            _recovery("E2", "H2", "C2", 2, 900.0),
            _recovery("E3", "H1", "C1", 3, 700.0),
            _recovery("E4", "H2", "C2", 3, 300.0),
            _hot_utility("Utilities.HU1", "C2", 200.0),
            _cold_utility("H2", "Cold Utility/CU1", 250.0),
        ),
        run_id=run_id,
        method="network_evolution_method",
        stage_count=3,
    )


def _stage_gap_network(run_id: str) -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(
            _recovery("E1", "H1", "C1", 1, 1200.0),
            _recovery("E2", "H2", "C2", 3, 900.0),
            _hot_utility("Utilities.HU1", "C2", 200.0),
            _cold_utility("H2", "Cold Utility/CU1", 250.0),
        ),
        run_id=run_id,
        method="network_evolution_method",
        stage_count=3,
    )


def _stage_compressed_network(run_id: str) -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(
            _recovery("E1", "H1", "C1", 1, 1200.0),
            _recovery("E2", "H2", "C2", 2, 900.0),
            _hot_utility("Utilities.HU1", "C2", 200.0),
            _cold_utility("H2", "Cold Utility/CU1", 250.0),
        ),
        run_id=run_id,
        method="network_evolution_method",
        stage_count=3,
    )


def _plain_four_stream_network() -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(
            _recovery("E1", "H1", "C1", 1, 1200.0),
            _recovery("E2", "H2", "C2", 2, 900.0),
        ),
        run_id="plain-four-stream",
        method="network_evolution_method",
        stage_count=2,
    )


def _two_stream_network() -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(_recovery("E1", "H1", "C1", 1, 1200.0),),
        run_id="two-stream",
        method="network_evolution_method",
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
        method="network_evolution_method",
        stage_count=3,
    )


def _supply_side_hot_utility_network() -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="E1",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                period_states=(
                    _period_state(
                        duty=1200.0,
                        source_inlet_temperature=650.0,
                        source_outlet_temperature=570.0,
                        sink_inlet_temperature=300.0,
                        sink_outlet_temperature=390.0,
                    ),
                ),
            ),
            HeatExchanger(
                exchanger_id="HU1-C1",
                kind=HeatExchangerKind.HOT_UTILITY,
                source_stream="HU1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.UTILITY,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                period_states=(
                    _period_state(
                        duty=350.0,
                        sink_inlet_temperature=270.0,
                        sink_outlet_temperature=300.0,
                    ),
                ),
            ),
        ),
        run_id="supply-side-hot-utility",
        method="network_evolution_method",
        stage_count=1,
    )


def _out_of_order_midpoint_network() -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="low-temp",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                period_states=(
                    _period_state(
                        duty=800.0,
                        source_inlet_temperature=520.0,
                        source_outlet_temperature=500.0,
                        sink_inlet_temperature=290.0,
                        sink_outlet_temperature=330.0,
                    ),
                ),
            ),
            HeatExchanger(
                exchanger_id="high-source-low-sink",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H2",
                sink_stream="C2",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                period_states=(
                    _period_state(
                        duty=900.0,
                        source_inlet_temperature=720.0,
                        source_outlet_temperature=680.0,
                        sink_inlet_temperature=500.0,
                        sink_outlet_temperature=560.0,
                    ),
                ),
            ),
            HeatExchanger(
                exchanger_id="high-source-high-sink",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H3",
                sink_stream="C3",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                period_states=(
                    _period_state(
                        duty=850.0,
                        source_inlet_temperature=690.0,
                        source_outlet_temperature=650.0,
                        sink_inlet_temperature=430.0,
                        sink_outlet_temperature=470.0,
                    ),
                ),
            ),
        ),
        run_id="out-of-order-midpoints",
        method="network_evolution_method",
        stage_count=1,
    )


def _temperature_scaled_network() -> HeatExchangerNetwork:
    return HeatExchangerNetwork(
        exchangers=(
            HeatExchanger(
                exchanger_id="E1",
                kind=HeatExchangerKind.RECOVERY,
                source_stream="H1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                stage=1,
                period_states=(
                    _period_state(
                        duty=1200.0,
                        source_inlet_temperature=700.0,
                        source_outlet_temperature=600.0,
                        sink_inlet_temperature=300.0,
                        sink_outlet_temperature=400.0,
                    ),
                ),
            ),
            HeatExchanger(
                exchanger_id="HU1-C1",
                kind=HeatExchangerKind.HOT_UTILITY,
                source_stream="HU1",
                sink_stream="C1",
                source_stream_role=HeatExchangerStreamRole.UTILITY,
                sink_stream_role=HeatExchangerStreamRole.PROCESS,
                period_states=(
                    _period_state(
                        duty=200.0,
                        sink_inlet_temperature=200.0,
                        sink_outlet_temperature=300.0,
                    ),
                ),
            ),
            HeatExchanger(
                exchanger_id="H1-CU1",
                kind=HeatExchangerKind.COLD_UTILITY,
                source_stream="H1",
                sink_stream="CU1",
                source_stream_role=HeatExchangerStreamRole.PROCESS,
                sink_stream_role=HeatExchangerStreamRole.UTILITY,
                period_states=(
                    _period_state(
                        duty=250.0,
                        source_inlet_temperature=600.0,
                        source_outlet_temperature=500.0,
                    ),
                ),
            ),
        ),
        run_id="temperature-scaled",
        method="network_evolution_method",
        stage_count=1,
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
        period_states=(
            _period_state(
                duty=duty,
                source_inlet_temperature=650.0,
                source_outlet_temperature=source_outlet_temperature,
                sink_inlet_temperature=300.0,
                sink_outlet_temperature=400.0,
            ),
        ),
        area=123.4,
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
        period_states=(
            _period_state(
                duty=duty,
                sink_inlet_temperature=365.0,
                sink_outlet_temperature=400.0,
            ),
        ),
        area=45.6,
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
        period_states=(
            _period_state(
                duty=duty,
                source_inlet_temperature=560.0,
                source_outlet_temperature=550.0,
            ),
        ),
        area=67.8,
    )


def _arrow_x_positions(diagram) -> list[tuple[float, float]]:
    return sorted(diagram.ax.stream_bounds)


def _marker_sizes(diagram) -> list[float]:
    return [
        float(line.kwargs["markersize"])
        for line in diagram.ax.lines
        if line.kwargs.get("marker")
    ]


def _line_widths(diagram) -> list[float]:
    return [
        float(line.kwargs["lw"]) for line in diagram.ax.lines if "lw" in line.kwargs
    ]


def _recovery_line_widths(diagram) -> list[float]:
    return [
        float(line.kwargs["lw"])
        for line in diagram.ax.lines
        if line.kwargs.get("marker")
        and line.kwargs.get("color") == _RECOVERY_MATCH_COLOR
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
