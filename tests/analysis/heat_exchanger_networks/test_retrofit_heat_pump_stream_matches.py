"""A heat pump's streams join a network with the matches that serve them.

In a retrofit the heat pump takes heat a cooler rejects and delivers heat
a heater supplies. As streams, each evaporator is matched to every hot
stream whose cooler starts hot enough for it, after that stream's last
recovery match, and each condenser to every cold stream whose heater
starts cold enough, ahead of its other matches. Evaporators take new
stages at the cold end, hottest first; condensers new stages at the hot
end, hottest at stage 1.

Ciric and Floudas's network cools H1 from 136.5 degC and H2 from
169.6 degC and heats C2 from 175.7 degC. With a 10 K approach an
evaporator at 150 degC can serve H2 alone, one at 100 degC both, and a
condenser at 280 degC serves C2.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_streams import (
    with_heat_pump_matches,
)
from OpenPinch.domain.enums import HeatExchangerKind
from OpenPinch.domain.stream import Stream
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import (
    _solved_network,
)


def _stream(name: str, supply: float, target: float) -> Stream:
    return Stream(
        name=name, supply_temperature=supply, target_temperature=target, heat_flow=1.0
    )


def test_each_heat_pump_stream_is_matched_where_a_cooler_or_heater_can_serve_it():
    network, added = with_heat_pump_matches(
        _solved_network(),
        condensers=[_stream("HP_H1", 280.0, 279.99)],
        evaporators=[
            _stream("HP_C2", 99.99, 100.0),
            _stream("HP_C1", 149.99, 150.0),
        ],
        approach=10.0,
    )

    recovery = {
        (e.source_stream, e.sink_stream): e.stage
        for e in network.exchangers
        if e.kind is HeatExchangerKind.RECOVERY
    }
    assert recovery == {
        ("HP_H1", "C2"): 1,
        ("H3", "C2"): 2,
        ("H1", "C1"): 2,
        ("H2", "C2"): 3,
        ("H3", "C1"): 3,
        ("H2", "HP_C1"): 4,
        ("H1", "HP_C2"): 5,
        ("H2", "HP_C2"): 5,
    }
    assert network.stage_count == 5
    by_id = {e.exchanger_id: (e.source_stream, e.sink_stream) for e in network.exchangers}
    assert sorted(by_id[exchanger_id] for exchanger_id in added) == [
        ("H1", "HP_C2"),
        ("H2", "HP_C1"),
        ("H2", "HP_C2"),
        ("HP_H1", "C2"),
    ]
