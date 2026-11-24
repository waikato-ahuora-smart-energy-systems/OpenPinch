"""The as-piped profiles of a network over several periods.

A heat pump designed for an existing network of several periods serves its
coolers and heaters as they run in each period. Each period's profiles are
read from that period's solved states, all on one temperature grid, as a
shared heat pump design needs.
"""

from __future__ import annotations

import json

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_targets import (
    as_piped_period_profiles,
)
from OpenPinch.application.problem import PinchProblem
from OpenPinch.domain._heat_exchanger.period_state import HeatExchangerPeriodState
from OpenPinch.domain.heat_exchanger_network import HeatExchangerNetwork
from OpenPinch.resources import read_sample_case
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import CASE

PERIODS = ("base", "peak")
# (duty, source in, source out, sink in, sink out) per period: in the peak
# H1's cooler takes less and starts colder, and the heater takes more.
STATES = {
    "cold-utility:H1->CU": (
        (13603.6, 409.68, 350.15, 293.15, 313.15),
        (9000.0, 395.15, 350.15, 293.15, 313.15),
    ),
    "cold-utility:H2->CU": (
        (1665.1, 442.77, 361.15, 293.15, 313.15),
        (1665.1, 442.77, 361.15, 293.15, 313.15),
    ),
    "hot-utility:HU->C2": (
        (17518.7, 773.15, 772.15, 448.81, 538.15),
        (20000.0, 773.15, 772.15, 440.15, 538.15),
    ),
}


def two_period_network() -> HeatExchangerNetwork:
    """Ciric and Floudas's existing network solved in two periods."""

    structure = HeatExchangerNetwork.from_structure(
        recovery=[("H3", "C2", 1), ("H1", "C1", 1), ("H2", "C2", 2), ("H3", "C1", 2)],
        heaters=["C2"],
        coolers=["H1", "H2"],
        hot_utility="HU",
        cold_utility="CU",
    )

    def solved(exchanger):
        if exchanger.exchanger_id not in STATES:
            return exchanger
        states = tuple(
            HeatExchangerPeriodState(
                period_id=period,
                period_idx=index,
                duty=duty,
                source_inlet_temperature=source_in,
                source_outlet_temperature=source_out,
                sink_inlet_temperature=sink_in,
                sink_outlet_temperature=sink_out,
            )
            for index, (period, (duty, source_in, source_out, sink_in, sink_out)) in (
                enumerate(zip(PERIODS, STATES[exchanger.exchanger_id]))
            )
        )
        return exchanger.model_copy(update={"period_states": states})

    return structure.model_copy(
        update={"exchangers": tuple(solved(e) for e in structure.exchangers)}
    )


def two_period_problem() -> PinchProblem:
    source = json.loads(read_sample_case(CASE))
    source["options"] = (source.get("options") or {}) | {
        "PROBLEM_PERIOD_IDS": list(PERIODS),
        "PROBLEM_PERIOD_WEIGHTS": [0.5, 0.5],
    }
    return PinchProblem(source)


def test_there_are_profiles_for_every_period():
    profiles = as_piped_period_profiles(two_period_problem(), two_period_network())

    assert list(profiles) == list(PERIODS)
