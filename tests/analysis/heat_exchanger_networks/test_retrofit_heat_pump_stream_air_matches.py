"""The air a heat pump draws on is matched to its evaporators.

Where the coolers reject too little heat, the as-piped target lets the
evaporators take the rest from ambient air. As streams, the air is a hot
stream of fixed duty beside the machine's own, and something has to cool
it: each evaporator colder than the air gets a match with it, in a new
stage after the network's last, the evaporators sharing the air between
them.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_streams import (
    with_heat_pump_matches,
)
from OpenPinch.domain.stream import Stream
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import (
    _solved_network,
)


def _stream(name: str, supply: float, target: float) -> Stream:
    return Stream(
        name=name, supply_temperature=supply, target_temperature=target, heat_flow=1.0
    )


def test_ambient_air_is_matched_to_every_evaporator_colder_than_it():
    _network, added = with_heat_pump_matches(
        _solved_network(),
        condensers=[],
        evaporators=[
            # Hotter than the air: it draws on the coolers alone.
            _stream("HP_C1", 99.99, 100.0),
            _stream("HP_C2", 4.99, 5.0),
            _stream("HP_C3", -5.01, -5.0),
        ],
        ambient_sources=[_stream("AIR_H1", 15.0, 14.99)],
        approach=10.0,
    )

    air = {
        p.match.cold: (p.match.new_stage, p.splits)
        for p in added
        if p.match.hot == "AIR_H1"
    }
    assert air == {"HP_C2": (True, ()), "HP_C3": (False, ("AIR_H1",))}
    stages = {p.match.stage for p in added if p.match.hot == "AIR_H1"}
    assert stages == {max(p.match.stage for p in added)}
