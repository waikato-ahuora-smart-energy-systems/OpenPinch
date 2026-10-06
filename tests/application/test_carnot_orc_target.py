"""Public ORC targeting through ``problem.target.carnot_orc``."""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.orc.search import OrcTargetingError
from OpenPinch.domain.enums import TargetType
from OpenPinch.domain.targets import DirectOrcTarget


def _problem(*, hot_duty: float = 20000.0) -> PinchProblem:
    # A hot stream with far more heat than the cold stream needs, so most of
    # it is surplus below the pinch.
    return PinchProblem(
        {
            "streams": [
                {
                    "zone": "Site",
                    "name": "Hot",
                    "t_supply": 250.0,
                    "t_target": 40.0,
                    "heat_flow": hot_duty,
                },
                {
                    "zone": "Site",
                    "name": "Cold",
                    "t_supply": 20.0,
                    "t_target": 200.0,
                    "heat_flow": 5000.0,
                },
            ],
            "utilities": [
                {"name": "HU", "type": "Hot", "t_supply": 270.0, "t_target": 269.9},
                {"name": "CU", "type": "Cold", "t_supply": 15.0, "t_target": 15.1},
            ],
        },
        project_name="Site",
    )


def test_carnot_orc_turns_surplus_into_power_without_extra_hot_utility():
    problem = _problem()
    base = problem.target.direct_heat_integration()

    target = problem.target.carnot_orc(stages=2, maximum_iterations=50)

    assert isinstance(target, DirectOrcTarget)
    assert target.type == TargetType.DORC.value
    assert target.orc_n_stages == 2
    assert target.orc_net_power > 0.0
    assert target.orc_total_annualized_cost_change < 0.0
    assert target.orc_heat_in == pytest.approx(
        target.orc_net_power + target.orc_condenser_duty
    )
    assert target.hot_utility_target == pytest.approx(base.hot_utility_target)
    assert target.cold_utility_target == pytest.approx(
        base.cold_utility_target - target.orc_heat_in, rel=1e-8
    )
    assert all(t > 30.0 for t in target.orc_evaporating_temperatures)


def test_carnot_orc_settings_reach_the_design():
    problem = _problem()

    target = problem.target.carnot_orc(
        condensing_temperature=40.0,
        minimum_lift=20.0,
        load_fraction=0.5,
        maximum_iterations=50,
    )

    base = problem.target.direct_heat_integration()
    assert target.orc_condensing_temperature == 40.0
    assert min(target.orc_evaporating_temperatures) >= 60.0 - 1e-9
    assert target.orc_heat_in <= 0.5 * base.cold_utility_target + 1e-6


def test_no_surplus_below_the_pinch_gives_no_orc_target():
    problem = PinchProblem(
        {
            "streams": [
                {
                    "zone": "Site",
                    "name": "Cold",
                    "t_supply": 20.0,
                    "t_target": 200.0,
                    "heat_flow": 5000.0,
                },
            ],
            "utilities": [
                {"name": "HU", "type": "Hot", "t_supply": 270.0, "t_target": 269.9},
                {"name": "CU", "type": "Cold", "t_supply": 15.0, "t_target": 15.1},
            ],
        },
        project_name="Site",
    )

    assert problem.target.carnot_orc() is None


def test_an_orc_that_does_not_pay_is_reported():
    problem = _problem()

    with pytest.raises(OrcTargetingError, match="No beneficial ORC"):
        problem.target.carnot_orc(
            options={"COSTING_ORC_EQUIPMENT_COST": 1e12},
            maximum_iterations=20,
        )


def test_orc_results_are_reported_with_units():
    problem = _problem()
    target = problem.target.carnot_orc(maximum_iterations=50)

    rows = [
        row
        for row in problem.results.targets
        if row.target_method == "Organic Rankine Cycle"
    ]

    assert len(rows) == 1
    row = rows[0]
    assert row.orc_model == "carnot"
    assert row.orc_net_power.unit == "kW"
    assert row.orc_net_power.value == pytest.approx(target.orc_net_power)
    assert row.orc_thermal_efficiency.unit == "%"
    assert row.orc_thermal_efficiency.value == pytest.approx(
        100.0 * target.orc_thermal_efficiency
    )
    assert row.orc_total_annualized_cost_change.unit == "$/y"
    assert row.orc_evaporating_temperatures == target.orc_evaporating_temperatures
    assert row.Qc.value == pytest.approx(target.cold_utility_target)


def test_simulated_orc_reports_its_fluid_and_beats_no_orc():
    pytest.importorskip("CoolProp")
    problem = _problem()
    base = problem.target.direct_heat_integration()

    target = problem.target.organic_rankine_cycle(
        fluids=["Isopentane"],
        stages=1,
        maximum_restarts=2,
        maximum_iterations=40,
    )

    assert target.orc_model == "simulated"
    assert target.orc_fluid == "Isopentane"
    assert target.orc_net_power > 0.0
    assert target.orc_total_annualized_cost_change < 0.0
    assert target.hot_utility_target == pytest.approx(base.hot_utility_target)
    assert target.cold_utility_target == pytest.approx(
        base.cold_utility_target - target.orc_heat_in, rel=1e-8, abs=1e-6
    )
    row = next(
        row
        for row in problem.results.targets
        if row.target_method == "Organic Rankine Cycle"
    )
    assert row.orc_fluid == "Isopentane"


def test_orc_graphs_show_the_gcc_and_net_loads_with_the_orc():
    from OpenPinch.domain.enums import GraphType, ProblemTableLabel

    problem = _problem()
    target = problem.target.carnot_orc(stages=2, maximum_iterations=50)

    gcc = target.graphs[GraphType.GCC_ORC.value]
    nlp = target.graphs[GraphType.NLP_ORC.value]
    # The ORC lowers the cascade by exactly the heat it takes, all of it below
    # its hottest evaporator, and none of it above the pinch.
    drop = gcc[ProblemTableLabel.H_NET_A] - gcc[ProblemTableLabel.H_NET_ORC]
    assert drop.max() == pytest.approx(target.orc_heat_in)
    assert drop.min() == pytest.approx(0.0, abs=1e-9)
    assert gcc[ProblemTableLabel.H_NET_ORC].min() >= -1e-6
    evaporators = nlp[ProblemTableLabel.H_COLD_ORC]
    assert evaporators.max() == pytest.approx(0.0, abs=1e-9)
    assert evaporators.min() == pytest.approx(-target.orc_heat_in)

    for plot, kind in (
        (problem.plot.grand_composite_curve_with_orc, GraphType.GCC_ORC),
        (problem.plot.net_load_profiles_with_orc, GraphType.NLP_ORC),
    ):
        graph = plot(return_graph_data=True)
        assert graph["type"] == kind.value
        assert graph["segments"]
