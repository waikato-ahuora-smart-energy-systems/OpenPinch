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
