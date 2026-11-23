"""A heat pump match never takes an id the network already uses.

The new exchangers are named ``new:<machine>:<hot>-><cold>``. An
existing network may hold an exchanger of exactly that id (from an
earlier retrofit, say). The search keys areas and modifications by id,
so a second exchanger of the same id would merge with the first; the new
one takes ``:2`` instead, as any new match does.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_streams import (
    with_heat_pump_matches,
)
from OpenPinch.domain.stream import Stream
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import (
    _solved_network,
)


def test_a_match_whose_id_is_taken_gets_a_numbered_one():
    network = _solved_network()
    first, *rest = network.exchangers
    network = network.model_copy(
        update={
            "exchangers": (
                first.model_copy(update={"exchanger_id": "new:HP:H1->HP_C1"}),
                *rest,
            )
        }
    )

    result, added = with_heat_pump_matches(
        network,
        condensers=[],
        evaporators=[
            Stream(
                name="HP_C1",
                supply_temperature=99.99,
                target_temperature=100.0,
                heat_flow=1.0,
            )
        ],
        approach=10.0,
    )

    assert sorted(placed.exchanger_id for placed in added) == [
        "new:HP:H1->HP_C1:2",
        "new:HP:H2->HP_C1",
    ]
    ids = [e.exchanger_id for e in result.exchangers]
    assert len(ids) == len(set(ids))
