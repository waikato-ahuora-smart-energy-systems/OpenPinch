"""Heat pump targets for a retrofit: the ideal and the as-piped target."""

from __future__ import annotations

import json

import numpy as np
import pytest

from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.resources import read_sample_case

CASE = "Five-stream-Ciric-and-Floudas-1989.json"
KELVIN = 273.15
# The case's temperature contribution of every process stream.
DT_CONT = 5.0


def _solved_network() -> HeatExchangerNetwork:
    """Ciric and Floudas's existing network with the baseline's solved
    states (temperatures in K): one heater on C2, coolers on H1 and H2."""

    structure = HeatExchangerNetwork.from_structure(
        recovery=[("H3", "C2", 1), ("H1", "C1", 1), ("H2", "C2", 2), ("H3", "C1", 2)],
        heaters=["C2"],
        coolers=["H1", "H2"],
        hot_utility="HU",
        cold_utility="CU",
    )
    # (duty, source in, source out, sink in, sink out)
    states = {
        "cold-utility:H1->CU": (13603.6, 409.68, 350.15, 293.15, 313.15),
        "cold-utility:H2->CU": (1665.1, 442.77, 361.15, 293.15, 313.15),
        "hot-utility:HU->C2": (17518.7, 773.15, 772.15, 448.81, 538.15),
    }

    def solved(exchanger):
        if exchanger.exchanger_id not in states:
            return exchanger
        duty, source_in, source_out, sink_in, sink_out = states[exchanger.exchanger_id]
        state = HeatExchangerPeriodState(
            period_id="0",
            period_idx=0,
            duty=duty,
            source_inlet_temperature=source_in,
            source_outlet_temperature=source_out,
            sink_inlet_temperature=sink_in,
            sink_outlet_temperature=sink_out,
        )
        return exchanger.model_copy(update={"period_states": (state,)})

    return structure.model_copy(
        update={"exchangers": tuple(solved(e) for e in structure.exchangers)}
    )


def _problem() -> PinchProblem:
    from OpenPinch.analysis.heat_exchanger_networks.context import (
        ensure_target_results,
    )

    problem = PinchProblem(json.loads(read_sample_case(CASE)))
    ensure_target_results(problem, {}, None)
    return problem


def test_the_as_piped_profiles_are_the_existing_coolers_and_heaters():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
        as_piped_profiles,
    )

    T_vals, H_hot, H_cold = as_piped_profiles(_problem(), _solved_network())

    # Shifted temperatures (degC), hottest first: the heater's outlet raised
    # and the coldest cooler outlet lowered by the stream's contribution.
    assert T_vals[0] == pytest.approx(538.15 - KELVIN + DT_CONT)
    assert T_vals[-1] == pytest.approx(350.15 - KELVIN - DT_CONT)
    assert np.all(np.diff(T_vals) < 0.0)
    # The sink is the heater's duty, the source the coolers', unnetted.
    assert abs(H_cold).max() == pytest.approx(17518.7)
    assert abs(H_hot).max() == pytest.approx(13603.6 + 1665.1)
    # The sink lies wholly above the heater's inlet, the source below the
    # hottest cooler inlet.
    assert np.all(H_cold[T_vals <= 448.81 - KELVIN + DT_CONT] == 0.0)
    assert np.all(H_hot[T_vals >= 442.77 - KELVIN - DT_CONT] == 0.0)


def test_a_cooler_on_a_stream_the_problem_lacks_is_an_error():
    from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
        as_piped_profiles,
    )

    network = _solved_network()
    network = network.model_copy(
        update={
            "exchangers": tuple(
                e.model_copy(update={"source_stream": "H9"})
                if e.source_stream == "H1"
                else e
                for e in network.exchangers
            )
        }
    )

    with pytest.raises(ValueError, match="H9"):
        as_piped_profiles(_problem(), network)


def test_the_ideal_and_as_piped_targets_relieve_their_own_utilities(monkeypatch):
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.retrofit import (
        heat_pump_targets,
    )
    from OpenPinch.domain.enums import ProblemTableLabel, TargetType

    # The HPR service, faked: each target delivers 1000 kW at two levels
    # for 300 kW of work, 100 kW of its heat from ambient air.
    calls = []

    def fake_target(*, T_vals, H_hot, H_cold, config, args=None):
        calls.append((np.asarray(H_hot), np.asarray(H_cold), args))
        return SimpleNamespace(
            Q_cond=np.array([600.0, 400.0]),
            T_cond=np.array([270.0, 200.0]),
            Q_evap=np.array([700.0]),
            T_evap=np.array([90.0]),
            Q_amb_cold=100.0,
            w_net=300.0,
            cop_h=1000.0 / 300.0,
            hpr_total_annualized_cost=None,
        )

    monkeypatch.setattr(
        heat_pump_targets, "compute_profile_heat_pump_target", fake_target
    )
    problem = _problem()

    targets = heat_pump_targets.retrofit_heat_pump_targets(
        problem, _solved_network(), args={"maximum_iterations": 7}
    )

    direct = problem.master_zone.targets[TargetType.DI.value]
    ideal, piped = calls
    np.testing.assert_array_equal(ideal[1], direct.pt[ProblemTableLabel.H_NET_COLD])
    assert piped[1].max() == pytest.approx(17518.7)
    assert ideal[2] == piped[2] == {"maximum_iterations": 7}
    # The ideal starts from the process's utility targets, the as-piped
    # from the existing heaters and coolers.
    assert targets.ideal.hot_utility_before == pytest.approx(
        direct.hot_utility_target
    )
    assert targets.ideal.cold_utility_before == pytest.approx(
        direct.cold_utility_target
    )
    piped = targets.as_piped
    assert piped.hot_utility_before == pytest.approx(17518.7)
    assert piped.hot_utility_after == pytest.approx(17518.7 - 1000.0)
    assert piped.cold_utility_before == pytest.approx(13603.6 + 1665.1)
    # Only process heat relieves a cooler; the air's does not.
    assert piped.cold_utility_after == pytest.approx(13603.6 + 1665.1 - 600.0)
    assert (piped.work, piped.cop) == (300.0, pytest.approx(1000.0 / 300.0))
    assert piped.ambient_heat == 100.0
    assert piped.condensing_temperatures == (270.0, 200.0)
    assert piped.condenser_duties == (600.0, 400.0)
    assert (piped.evaporating_temperatures, piped.evaporator_duties) == (
        (90.0,),
        (700.0,),
    )
    # What keeping to the existing pipework costs in hot utility and work.
    assert targets.hot_utility_gap == pytest.approx(
        piped.hot_utility_after - targets.ideal.hot_utility_after
    )
