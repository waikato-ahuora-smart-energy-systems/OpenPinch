"""Heat pump moves for retrofit: candidates, screen and solver arrays."""

from __future__ import annotations

import pytest


def test_a_candidate_evaporates_below_the_hot_target_and_condenses_above_the_cold():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pumps import (
        carnot_rating,
        heat_pump_candidates,
    )

    # (supply, target, heat capacity flowrate), K and kW/K.
    ends = {
        "H1": (400.0, 330.0, 10.0),
        "C1": (300.0, 360.0, 8.0),
        "C2": (290.0, 310.0, 5.0),
    }

    (candidate,) = heat_pump_candidates(ends, approach=5.0, rating=carnot_rating(0.5))

    # H1 to C2 needs no lift (310 + 5 is below 330 - 5): a match does that.
    assert (candidate.hot, candidate.cold) == ("H1", "C1")
    assert candidate.evaporating == pytest.approx(325.0)
    assert candidate.condensing == pytest.approx(365.0)
    # The HPR service's Carnot heating COP: 1 + eta_II T_evap / lift.
    assert candidate.cop == pytest.approx(1.0 + 0.5 * 325.0 / 40.0, rel=1e-12)
    assert candidate.refrigerant is None


def test_a_real_fluid_rating_takes_the_best_fluid_within_the_cycle_limits():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pumps import (
        vapour_compression_rating,
    )
    from OpenPinch.analysis.heat_pumps.cycles.limits import CycleLimits
    from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_cycle import (
        VapourCompressionCycle,
    )

    limits = CycleLimits(max_pressure_ratio=15.0, min_suction_pressure=0.5e5)
    settings = {"dtcont": 1.0, "eta_comp": 0.7, "eta_motor": 0.95, "limits": limits}
    rate = vapour_compression_rating(["water", "ammonia", "isobutane"], **settings)

    cop, refrigerant = rate(340.0, 380.0)

    # Water evaporates at 0.27 bar here, below the suction pressure limit;
    # ammonia beats isobutane.
    assert refrigerant == "ammonia"
    cycle = VapourCompressionCycle()
    cycle.solve(
        340.0 - 273.15, 380.0 - 273.15, refrigerant="ammonia", Q_heat=1.0, **settings
    )
    assert cop == pytest.approx(cycle.COP_h, rel=1e-9)
    assert rate(340.0, 380.0)[0] < 1.0 + 340.0 / 40.0


def test_a_real_fluid_rating_serves_nothing_below_every_fluids_triple_point():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pumps import (
        vapour_compression_rating,
    )
    from OpenPinch.analysis.heat_pumps.cycles.limits import CycleLimits

    rate = vapour_compression_rating(
        ["ammonia"], dtcont=1.0, eta_comp=0.7, eta_motor=0.95, limits=CycleLimits()
    )

    # Ammonia freezes at 195 K.
    assert rate(150.0, 250.0) is None


def test_a_pair_no_cycle_can_serve_is_left_out():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pumps import (
        heat_pump_candidates,
    )

    ends = {"H1": (400.0, 330.0, 10.0), "C1": (300.0, 360.0, 8.0)}

    assert heat_pump_candidates(ends, approach=5.0, rating=lambda e, c: None) == []


def test_the_screen_ranks_heat_pumps_by_the_utility_they_could_relieve():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pumps import (
        HeatPumpCandidate,
        screen_heat_pumps,
    )
    from OpenPinch.domain._heat_exchanger.period_state import (
        HeatExchangerPeriodState,
    )
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork

    structure = HeatExchangerNetwork.from_structure(
        recovery=[("H1", "C1", 1)],
        heaters=["C1", "C2"],
        coolers=["H1"],
        hot_utility="HU",
        cold_utility="CU",
    )
    duties = {"H1->CU": 90.0, "HU->C1": 200.0, "HU->C2": 1.0, "H1->C1": 50.0}
    network = structure.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(
                    update={
                        "period_states": (
                            HeatExchangerPeriodState(
                                period_id="0",
                                period_idx=0,
                                duty=duties[f"{e.source_stream}->{e.sink_stream}"],
                            ),
                        )
                    }
                )
                for e in structure.exchangers
            )
        }
    )
    to_c1 = HeatPumpCandidate("H1", "C1", 325.0, 365.0, 4.0)
    to_c2 = HeatPumpCandidate("H1", "C2", 325.0, 365.0, 4.0)
    from_h2 = HeatPumpCandidate("H2", "C1", 325.0, 365.0, 4.0)

    # The condenser can deliver the cooler duty times COP / (COP - 1), up
    # to the cold stream's heater duty: 120 kW to C1, 1 kW to C2, none from
    # H2, which has no cooler.
    candidates = [to_c2, from_h2, to_c1]
    assert screen_heat_pumps(network, candidates, limit=5) == [to_c1, to_c2]
    assert screen_heat_pumps(network, candidates, limit=1) == [to_c1]


def test_heat_pump_utilities_join_the_solver_arrays_at_their_temperatures():
    import json

    import numpy as np

    from OpenPinch.analysis.heat_exchanger_networks.context import (
        ensure_target_results,
    )
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pumps import (
        HeatPumpCandidate,
        condenser_name,
        evaporator_name,
        with_heat_pump_utilities,
    )
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        problem_to_solver_arrays,
    )
    from OpenPinch.application.problem import PinchProblem
    from OpenPinch.resources import read_sample_case

    problem = PinchProblem(
        json.loads(read_sample_case("Five-stream-Ciric-and-Floudas-1989.json"))
    )
    ensure_target_results(problem, {}, None)
    arrays = problem_to_solver_arrays(problem, 10.0)
    candidate = HeatPumpCandidate("H1", "C1", 325.0, 365.0, 4.0)
    electricity = 830.0  # $/kW/y

    extended = with_heat_pump_utilities(arrays, [candidate], electricity)

    data = extended.arrays
    axes = extended.axis_maps
    condenser = condenser_name(candidate)
    evaporator = evaporator_name(candidate)
    hot = axes["hot_utilities"][condenser]
    cold = axes["cold_utilities"][evaporator]
    assert (hot, cold) == (1, 1)
    # Each condenses or evaporates over 1 K at its temperature.
    assert data["T_hu_in_period"][0][hot] == pytest.approx(365.0)
    assert data["T_hu_out_period"][0][hot] == pytest.approx(364.0)
    assert data["T_cu_in_period"][0][cold] == pytest.approx(324.0)
    assert data["T_cu_out_period"][0][cold] == pytest.approx(325.0)
    # The condenser's heat costs its compressor work; the evaporator's is free.
    assert data["hu_cost_period"][0][hot] == pytest.approx(electricity / 4.0)
    assert data["cu_cost_period"][0][cold] == pytest.approx(0.0)
    assert data["hot_utility_segment_price_period"][0][hot][0] == pytest.approx(
        electricity / 4.0
    )
    assert data["hot_utility_segment_t_in_period"][0][hot][0] == pytest.approx(365.0)
    assert not data["hot_utility_parent_segmented"][hot]
    assert not data["cold_utility_parent_segmented"][cold]
    assert axes["hot_utility_segments"][condenser + ".S1"] == 1
    assert extended.utility_identities["hot_utilities"][hot] == condenser
    assert extended.stream_identities["cold_utility_segments"][1] == evaporator + ".S1"
    # Every utility array grows by one along its utility axis; the rest is
    # untouched, and so are the original arrays.
    for name, values in arrays.arrays.items():
        grown = data[name]
        if grown.shape == values.shape:
            np.testing.assert_array_equal(grown, values, err_msg=name)
        else:
            assert sum(b - a for a, b in zip(values.shape, grown.shape)) == 1, name
            original = grown[tuple(slice(0, n) for n in values.shape)]
            np.testing.assert_array_equal(original, values, err_msg=name)
    assert len(arrays.axis_maps["hot_utilities"]) == 1


def test_a_heat_pump_condenser_delivers_its_evaporator_heat_and_its_work():
    from dataclasses import replace

    from OpenPinch.analysis.heat_exchanger_networks.context import (
        ensure_target_results,
    )
    from OpenPinch.analysis.heat_exchanger_networks.duty_optimisation import (
        build_duty_optimisation_request,
        fixed_network_structure,
    )
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pumps import (
        HeatPumpCandidate,
        condenser_name,
        evaporator_name,
        with_heat_pump_utilities,
    )
    from OpenPinch.analysis.heat_exchanger_networks.solver.arrays import (
        problem_to_solver_arrays,
    )
    from OpenPinch.application.problem import PinchProblem
    from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
    from tests.support.scipy_nlp import scipy_fixed_structure_model

    def stream(name, cp, supply, target):
        return {
            "name": name,
            "zone": "Site/Process A",
            "heat_capacity_flowrate": {"unit": "kW/delta_degC", "value": cp},
            "heat_flow": {"unit": "kW", "value": abs(cp * (target - supply))},
            "htc": {"unit": "kW/m^2/K", "value": 1.0},
            "t_supply": {"unit": "K", "value": supply},
            "t_target": {"unit": "K", "value": target},
        }

    def utility(name, kind, supply, target, price):
        return {
            "name": name,
            "type": kind,
            "heat_flow": None,
            "htc": {"unit": "kW/m^2/K", "value": 1.0},
            "price": {"unit": "$/MWh", "value": price},
            "t_supply": {"unit": "K", "value": supply},
            "t_target": {"unit": "K", "value": target},
        }

    # H1 can heat C1 to about 490 K; steam (HPS) does the rest unless the
    # heat pump, condensing at 660 K for a third of steam's price, does.
    problem = PinchProblem(
        {
            "streams": [
                stream("H1", 10.0, 500.0, 300.0),
                stream("C1", 5.0, 450.0, 650.0),
            ],
            "utilities": [
                utility("HPS", "Hot", 700.0, 700.0, 80.0),
                utility("CW", "Cold", 290.0, 310.0, 15.0),
            ],
            "options": {"HENS_DT_CONT_MULTIPLIERS": [1.0]},
        }
    )
    ensure_target_results(problem, {}, None)
    arrays = problem_to_solver_arrays(problem, 10.0)
    cop = 3.0
    candidate = HeatPumpCandidate("H1", "C1", 280.0, 660.0, cop)
    steam_price = float(arrays.arrays["hu_cost_period"][0][0])
    arrays = with_heat_pump_utilities(arrays, [candidate], steam_price)
    names = {
        name.rsplit(".", 1)[-1]: name
        for axis in ("hot_utilities", "cold_utilities")
        for name in arrays.axis_maps[axis]
    }
    network = HeatExchangerNetwork.from_structure(
        recovery=[("H1", "C1", 1)],
        heaters=[("C1", names["HPS"]), ("C1", condenser_name(candidate))],
        coolers=[("H1", names["CW"]), ("H1", evaporator_name(candidate))],
    )
    request = build_duty_optimisation_request(
        network, objective="cost", min_approach_temperature=10.0
    )
    structure = fixed_network_structure(request, arrays.axis_maps)
    utilities = structure.spec.utilities

    def slot(side, index):
        return next(
            e for e, m in enumerate(utilities) if (m.side, m.utility) == (side, index)
        )

    evaporator, condenser, steam = slot("cold", 1), slot("hot", 1), slot("hot", 0)
    spec = replace(structure.spec, heat_pumps=((evaporator, condenser, cop),))
    model = scipy_fixed_structure_model()(
        name="heat pump",
        solver="ipopt-pyomo",
        solver_arrays=arrays,
        spec=spec,
        minimisation_goal="utility cost",
        default_approach=10.0,
    )
    model.optimise()

    assert model.mSuccess == 1
    lifted = model.utility_results[evaporator].periods[0].duty
    delivered = model.utility_results[condenser].periods[0].duty
    # The condenser delivers the evaporator's heat plus the work, COP /
    # (COP - 1) times it, and at a third of steam's price it takes all the
    # heating the recovery match leaves.
    assert delivered > 100.0
    assert delivered == pytest.approx(lifted * cop / (cop - 1.0), rel=1e-4)
    assert model.utility_results[steam].periods[0].duty == pytest.approx(0.0, abs=1e-3)
