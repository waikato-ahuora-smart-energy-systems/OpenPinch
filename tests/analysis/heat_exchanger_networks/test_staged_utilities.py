"""Heaters and coolers between the stages of a stage-wise network.

A utility exchanger at a stream end can only serve the stream's hottest (or
coldest) span. A level that targeting assigns to a span between two
recovery matches needs a heater (or cooler) between the stages. With
``staged_utilities`` the stage-wise model offers one at every interior stage
boundary, in series between the stage it leaves and the stage it feeds.
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.extraction.service import (
    extract_heat_exchanger_network,
)
from OpenPinch.analysis.heat_exchanger_networks.models._base.execution import _value
from OpenPinch.analysis.heat_exchanger_networks.models.stagewise import (
    StageWiseModel,
)
from OpenPinch.analysis.heat_exchanger_networks.reporting.verification import (
    verify_network_feasibility,
)
from OpenPinch.analysis.heat_exchanger_networks.solver import backend
from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
    problem_to_solver_arrays,
)
from OpenPinch.analysis.heat_exchanger_networks.solver.dependencies import (
    MissingSynthesisDependencyError,
    MissingSynthesisSolverError,
    require_solver_binary,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain.enums import HeatExchangerKind

D_TMIN = 10.0


def _highs_available() -> bool:
    try:
        backend.require_solver_backend("highs")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


pytestmark = pytest.mark.skipif(
    "not _highs_available()", reason="needs highspy and pyomo"
)


def _quantity(value, unit):
    return {"value": value, "unit": unit}


def _stream(name, supply, target, heat_flow):
    return {
        "zone": "Site/Plant",
        "name": name,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "heat_flow": (
            {"values": heat_flow, "unit": "kW"}
            if isinstance(heat_flow, list)
            else _quantity(heat_flow, "kW")
        ),
        "dt_cont": _quantity(D_TMIN / 2, "delta_degC"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
    }


def _utility(name, kind, supply, target, price):
    return {
        "name": name,
        "type": kind,
        "t_supply": _quantity(supply, "degC"),
        "t_target": _quantity(target, "degC"),
        "dt_cont": _quantity(2.5, "delta_degC"),
        "price": _quantity(price, "$/MWh"),
        "htc": _quantity(1.0, "kW/m^2/delta_degC"),
        "heat_flow": None,
    }


def _heating_problem(periods: int = 1) -> PinchProblem:
    """C1 (60 to 200 C) is heated by H2 low, H1 high, and 40 kW between.

    H1 (220 to 150 C) can heat C1 from 130 to 200 C and H2 (100 to 70 C)
    from 60 to 90 C. Between them C1 needs 40 kW from 90 to 130 C, which LPS
    (140 C) can give; at C1's hot end only HPS (250 C) can.
    """

    def flows(heat_flow):
        return heat_flow if periods == 1 else [heat_flow] * periods

    options = (
        {}
        if periods == 1
        else {"PROBLEM_PERIOD_IDS": [f"p{n}" for n in range(periods)]}
    )
    return PinchProblem(
        source={
            "options": options,
            "streams": [
                _stream("H1", 220.0, 150.0, flows(70.0)),
                _stream("H2", 100.0, 70.0, flows(30.0)),
                _stream("C1", 60.0, 200.0, flows(140.0)),
            ],
            "utilities": [
                _utility("LPS", "Hot", 140.0, 139.9, 20.0),
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                _utility("CW", "Cold", 20.0, 25.0, 5.0),
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "Plant", "type": "Process Zone"}],
            },
        }
    )


def _solved(problem: PinchProblem, *, staged_utilities: bool) -> StageWiseModel:
    model = StageWiseModel(
        name="staged-utilities",
        framework="ESM",
        solver="highs",
        solver_arrays=problem_to_solver_arrays(problem, D_TMIN),
        stages=2,
        dTmin=D_TMIN,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal="utility costs",
        non_isothermal_model=False,
        integers=True,
        tol=1e-6,
        **({"staged_utilities": True} if staged_utilities else {}),
    )
    model.optimise(False)
    assert model.mSuccess == 1
    return model


def _hot_utility_duties(model) -> dict[str, float]:
    """Each hot utility's duty (kW), stream-end and staged heaters together."""
    names = ("LPS", "HPS")
    duties = dict.fromkeys(names, 0.0)
    for row in model.Q_h_by_period[0]:
        for u, duty in enumerate(row):
            duties[names[u]] += _value(duty)
    for row in getattr(model, "Q_hs_by_period", [()])[0]:
        for u, by_stage in enumerate(row):
            duties[names[u]] += sum(_value(duty) for duty in by_stage)
    return duties


def test_without_staged_utilities_the_heat_comes_from_the_stream_end():
    duties = _hot_utility_duties(_solved(_heating_problem(), staged_utilities=False))

    assert duties["LPS"] == pytest.approx(0.0, abs=1e-6)
    assert duties["HPS"] == pytest.approx(40.0, rel=1e-6)


def test_a_staged_heater_serves_the_span_between_two_recovery_stages():
    model = _solved(_heating_problem(), staged_utilities=True)

    duties = _hot_utility_duties(model)
    assert duties["LPS"] == pytest.approx(40.0, rel=1e-6)
    assert duties["HPS"] == pytest.approx(0.0, abs=1e-6)
    (c1,) = range(model.J)
    assert _value(model.Q_hs_by_period[0][c1][0][1]) == pytest.approx(40.0, rel=1e-6)
    assert _value(model.T_c_mid_by_period[0][c1][1]) == pytest.approx(
        model.T_c_in_period[0][c1] + 30.0, abs=1e-6
    )
    assert _value(model.T_c_by_period[0][c1][1]) == pytest.approx(
        model.T_c_in_period[0][c1] + 70.0, abs=1e-6
    )


def _cooling_problem() -> PinchProblem:
    """H1 (200 to 60 C) heats C2 high, C1 low, and gives 40 kW between.

    H1 can heat C2 (160 to 190 C) from 200 to 170 C and C1 (50 to 120 C)
    from 130 to 60 C. Between them H1 gives 40 kW from 170 to 130 C, which
    raises LP steam (120 C); at H1's cold end only cooling water can take it.
    """
    return PinchProblem(
        source={
            "streams": [
                _stream("H1", 200.0, 60.0, 140.0),
                _stream("C1", 50.0, 120.0, 70.0),
                _stream("C2", 160.0, 190.0, 30.0),
            ],
            "utilities": [
                _utility("HPS", "Hot", 250.0, 249.9, 40.0),
                _utility("LPG", "Cold", 120.0, 120.1, 1.0),
                _utility("CW", "Cold", 20.0, 25.0, 5.0),
            ],
            "zone_tree": {
                "name": "Site",
                "type": "Site",
                "children": [{"name": "Plant", "type": "Process Zone"}],
            },
        }
    )


def _cold_utility_duties(model) -> dict[str, float]:
    """Each cold utility's duty (kW), stream-end and staged coolers together."""
    names = ("LPG", "CW")
    duties = dict.fromkeys(names, 0.0)
    for row in model.Q_c_by_period[0]:
        for u, duty in enumerate(row):
            duties[names[u]] += _value(duty)
    for row in getattr(model, "Q_cs_by_period", [()])[0]:
        for u, by_stage in enumerate(row):
            duties[names[u]] += sum(_value(duty) for duty in by_stage)
    return duties


def test_without_staged_utilities_the_cooling_is_at_the_stream_end():
    duties = _cold_utility_duties(_solved(_cooling_problem(), staged_utilities=False))

    assert duties["LPG"] == pytest.approx(0.0, abs=1e-6)
    assert duties["CW"] == pytest.approx(40.0, rel=1e-6)


def test_a_staged_cooler_serves_the_span_between_two_recovery_stages():
    model = _solved(_cooling_problem(), staged_utilities=True)

    duties = _cold_utility_duties(model)
    assert duties["LPG"] == pytest.approx(40.0, rel=1e-6)
    assert duties["CW"] == pytest.approx(0.0, abs=1e-6)
    (h1,) = range(model.I)
    assert _value(model.Q_cs_by_period[0][h1][0][1]) == pytest.approx(40.0, rel=1e-6)
    assert _value(model.T_h_mid_by_period[0][h1][1]) == pytest.approx(
        model.T_h_in_period[0][h1] - 30.0, abs=1e-6
    )
    assert _value(model.T_h_by_period[0][h1][1]) == pytest.approx(
        model.T_h_in_period[0][h1] - 70.0, abs=1e-6
    )


def _couenne_available() -> bool:
    try:
        require_solver_binary("couenne", purpose="staged utility regression")
    except (MissingSynthesisDependencyError, MissingSynthesisSolverError):
        return False
    return True


@pytest.mark.skipif("not _couenne_available()", reason="needs couenne")
@pytest.mark.parametrize("periods", [1, 2])
@pytest.mark.parametrize("goal", ["total cost", "variable total cost"])
def test_total_cost_prices_a_staged_heaters_area(goal, periods):
    model = StageWiseModel(
        name="staged-utilities-total-cost",
        framework="TDM",
        solver="couenne",
        solver_arrays=problem_to_solver_arrays(_heating_problem(periods), D_TMIN),
        stages=2,
        dTmin=D_TMIN,
        z_restriction=None,
        min_dqda=0.0,
        minimisation_goal=goal,
        non_isothermal_model=False,
        integers=True,
        tol=1e-6,
        staged_utilities=True,
    )
    model.optimise(False)
    assert model.mSuccess == 1

    duties = _hot_utility_duties(model)
    assert duties["LPS"] == pytest.approx(40.0, rel=1e-4)
    assert duties["HPS"] == pytest.approx(0.0, abs=1e-4)
    (c1,) = range(model.J)
    assert model.area_hs[c1][0][1] > 0.0
    ok, reasons = model.verify()
    assert ok, reasons


def _network(model, problem):
    return extract_heat_exchanger_network(
        model, problem_to_solver_arrays(problem, D_TMIN), run_id="staged"
    )


def test_a_staged_heater_is_extracted_between_its_stages():
    problem = _heating_problem()
    network = _network(_solved(problem, staged_utilities=True), problem)

    heaters = [
        exchanger
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.HOT_UTILITY
    ]
    (heater,) = heaters
    # C1 has just left (1-based) stage 2, its colder stage.
    assert heater.exchanger_id.endswith(":S2")
    assert heater.stage == 2
    assert heater.source_stream.endswith("LPS")
    state = heater.state()
    assert state.duty == pytest.approx(40.0, rel=1e-6)
    assert state.sink_inlet_temperature == pytest.approx(90.0 + 273.15, abs=1e-6)
    assert state.sink_outlet_temperature == pytest.approx(130.0 + 273.15, abs=1e-6)
    (low_match,) = [
        exchanger
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.RECOVERY
        and exchanger.source_stream.endswith("H2")
    ]
    assert low_match.state().sink_outlet_temperature == pytest.approx(
        90.0 + 273.15, abs=1e-6
    )
    assert verify_network_feasibility(network) == ()


def test_a_staged_cooler_is_extracted_between_its_stages():
    problem = _cooling_problem()
    network = _network(_solved(problem, staged_utilities=True), problem)

    (cooler,) = [
        exchanger
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.COLD_UTILITY
    ]
    # H1 has just left stage 1, its hotter stage.
    assert cooler.exchanger_id.endswith(":S1")
    assert cooler.stage == 1
    assert cooler.sink_stream.endswith("LPG")
    state = cooler.state()
    assert state.duty == pytest.approx(40.0, rel=1e-6)
    assert state.source_inlet_temperature == pytest.approx(170.0 + 273.15, abs=1e-6)
    assert state.source_outlet_temperature == pytest.approx(130.0 + 273.15, abs=1e-6)
    (high_match,) = [
        exchanger
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.RECOVERY
        and exchanger.sink_stream.endswith("C2")
    ]
    assert high_match.state().source_outlet_temperature == pytest.approx(
        170.0 + 273.15, abs=1e-6
    )
    assert verify_network_feasibility(network) == ()
