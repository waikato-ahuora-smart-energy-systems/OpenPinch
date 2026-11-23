"""A side is split wherever its heat cascade closes, levels held or not.

A second pinch inside a pinch side needs no utility level to make it: a
heat pump's streams balance a side to zero utility at the condenser or the
evaporator level, and a process can have two pinches of its own. No heat
may cross such a pinch, so the side is designed as one region on each side
of it.
"""

from __future__ import annotations

from dataclasses import replace

from OpenPinch.analysis.heat_exchanger_networks.heuristic.regions import (
    pinch_regions,
)
from tests.analysis.heat_exchanger_networks.heuristic_sides import side, stream


def _with_shifts(pinch_side):
    """The side with every stream's temperature contribution read (0 K)."""

    return replace(
        pinch_side,
        hot=tuple(replace(s, contributions=(0.0,)) for s in pinch_side.hot),
        cold=tuple(replace(s, contributions=(0.0,)) for s in pinch_side.cold),
    )


def test_a_balanced_side_of_process_streams_is_split_where_its_cascade_closes():
    # Above a pinch at 100: H1 and C1 balance from 100 to 150, H2 and C2
    # from 150 to 250. Nothing crosses 150.
    balanced = _with_shifts(
        side(
            "above",
            hot=[
                stream("hot", 0, 2.0, 100.0, 100.0),
                stream("hot", 1, 1.0, 100.0, 150.0, at_pinch=False),
            ],
            cold=[
                stream("cold", 0, 2.0, 100.0, 100.0),
                stream("cold", 1, 1.0, 100.0, 150.0, at_pinch=False),
            ],
        )
    )

    regions = pinch_regions(balanced)

    assert [
        ([s.index for s in region.hot], [s.index for s in region.cold])
        for region in regions
    ] == [([0], [0]), ([1], [1])]
