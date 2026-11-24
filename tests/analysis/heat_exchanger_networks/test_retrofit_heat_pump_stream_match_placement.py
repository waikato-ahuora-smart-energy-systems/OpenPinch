"""Each heat pump match says how it was placed.

One level of a heat pump can serve several process streams. Its first
match takes a new stage; the others join that stage, which splits the
level's own stream between them. A search prices a split and reports
whether a match made a stage, so each added match comes with the move
that placed it and the streams it split.

On Ciric and Floudas's network an evaporator at 100 degC serves H1 and
H2: the match on H1 makes the stage, the match on H2 joins it and splits
the evaporator's stream.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.retrofit.heat_pump_streams import (
    with_heat_pump_matches,
)
from OpenPinch.domain.stream import Stream
from tests.analysis.heat_exchanger_networks.test_retrofit_heat_pump_targets import (
    _solved_network,
)


def test_a_match_that_joins_its_levels_stage_splits_the_levels_stream():
    _network, added = with_heat_pump_matches(
        _solved_network(),
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

    placed = {
        (p.match.hot, p.match.cold): (p.match.stage, p.match.new_stage, p.splits)
        for p in added
    }
    assert placed == {
        ("H1", "HP_C1"): (3, True, ()),
        ("H2", "HP_C1"): (3, False, ("HP_C1",)),
    }
