"""With no approach given, air stands off by its own contribution.

A search without ``min_approach_temperature`` holds each exchanger to its
two streams' contributions. The air a heat pump draws on carries the
ambient contribution (``HPR_DT_ENV_CONT``, 10 K by default) and an
evaporator the heat pump's own (``HPR_DT_CONT``, 1 K), so an evaporator
takes air only 11 K below it, not at the process streams' approach
(``THERMAL_DT_CONT`` plus ``HPR_DT_CONT``).

Here the process approach is 2 K: an evaporator at 5 degC would pass it
against air at 15 degC, but stands 9.99 K off and gets no air; one at
3 degC, 11.99 K off, does.
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


def test_air_stands_off_an_evaporator_by_the_two_contributions():
    config = Configuration()
    config.update_values(THERMAL_DT_CONT=1.0, HPR_DT_CONT=1.0, HPR_DT_ENV_CONT=10.0)
    network = _solved_network()
    step = HeatPumpStep(
        record=SimpleNamespace(machine="HP"),
        condensers=(),
        evaporators=(
            _stream("HP_C1", 4.99, 5.0),
            _stream("HP_C2", 2.99, 3.0),
        ),
        ambient_sources=(_stream("AIR_H1", 15.0, 14.99),),
        base=SimpleNamespace(network=network),
    )
    evaluator = SimpleNamespace(
        config=config, settings=RetrofitSettings(approach=None), names={}
    )

    state = _heat_pump_root(network, step, evaluator)

    air = sorted(m.match.cold for m in state.moves if m.match.hot == "AIR_H1")
    assert air == ["HP_C2"]
