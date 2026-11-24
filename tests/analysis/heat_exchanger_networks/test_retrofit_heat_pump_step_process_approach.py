"""With no approach given, a heat pump match stands off by its process
stream's own contribution.

A search without ``min_approach_temperature`` holds each exchanger to its
two streams' contributions. The matches that serve a heat pump were
placed at ``THERMAL_DT_CONT`` plus ``HPR_DT_CONT`` whatever the process
stream's own contribution. Here that is 2 K: an evaporator at 130 degC
passed it against H1's cooler, which starts at 136.5 degC, though H1
carries 10 K and the pair needs 11 K. Only H2's cooler, from 169.6 degC,
is far enough above it.
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_streams import (
    HeatPumpStep,
)
from OpenPinch.analysis.heat_exchanger_networks.retrofit.search import (
    RetrofitSettings,
    _heat_pump_root,
)
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.stream import Stream
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import (
    _solved_network,
)


def _stream(name: str, supply: float, target: float) -> Stream:
    return Stream(
        name=name, supply_temperature=supply, target_temperature=target, heat_flow=1.0
    )


def test_an_evaporator_stands_off_a_cooler_by_its_stream_contribution():
    config = Configuration()
    config.update_values(THERMAL_DT_CONT=1.0, HPR_DT_CONT=1.0)
    network = _solved_network()
    step = HeatPumpStep(
        record=SimpleNamespace(machine="HP"),
        condensers=(),
        evaporators=(_stream("HP_C1", 129.99, 130.0),),
        ambient_sources=(),
        base=SimpleNamespace(network=network),
    )
    evaluator = SimpleNamespace(
        config=config,
        settings=RetrofitSettings(approach=None),
        names={},
        contributions={"H1": 10.0, "H2": 1.0, "C2": 1.0},
    )

    state = _heat_pump_root(network, step, evaluator)

    served = sorted(m.match.hot for m in state.moves if m.match.cold == "HP_C1")
    assert served == ["H2"]
