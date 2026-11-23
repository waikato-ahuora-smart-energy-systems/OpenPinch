"""A heat pump step carries the air its evaporators draw on.

The step lists the machine's ambient source streams beside its
condensers and evaporators. The search's first state gives each one its
matches with the evaporators, as part of the same step, and they belong
to the machine like its other matches.
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
from OpenPinch.domain.stream import Stream
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import (
    _solved_network,
)


def _stream(name: str, supply: float, target: float) -> Stream:
    return Stream(
        name=name, supply_temperature=supply, target_temperature=target, heat_flow=1.0
    )


def test_the_first_state_matches_the_air_to_the_evaporator():
    network = _solved_network()
    step = HeatPumpStep(
        record=SimpleNamespace(machine="HP"),
        condensers=(),
        # At -5 degC, well over the 10 K approach below the air at both
        # ends of the match.
        evaporators=(_stream("HP_C1", -5.01, -5.0),),
        ambient_sources=(_stream("AIR_H1", 15.0, 14.99),),
        base=SimpleNamespace(network=network),
    )
    evaluator = SimpleNamespace(
        config=None, settings=RetrofitSettings(approach=10.0), names={}
    )

    state = _heat_pump_root(network, step, evaluator)

    air = [m for m in state.moves if m.match.hot == "AIR_H1"]
    assert [(m.match.cold, m.machine, m.repiped) for m in air] == [("HP_C1", "HP", ())]
