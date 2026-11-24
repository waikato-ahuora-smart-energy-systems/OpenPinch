"""A cascade that closes at one end of a near-isothermal stream is a pinch.

A condensing or evaporating stream is written with a small glide (0.01 K).
A target that sizes it to close the cascade does so at one end of the
glide; at the other end the cascade is off by the glide times the net CP
of the streams around it, a few parts in 100,000 of the side's load. That
is still the pinch.
"""

from __future__ import annotations

from dataclasses import replace

from OpenPinch.analysis.heat_exchanger_networks.heuristic.regions import (
    pinch_regions,
)
from tests.analysis.heat_exchanger_networks.heuristic_sides import side, stream


def _with_shifts(pinch_side):
    return replace(
        pinch_side,
        hot=tuple(replace(s, contributions=(0.0,)) for s in pinch_side.hot),
        cold=tuple(replace(s, contributions=(0.0,)) for s in pinch_side.cold),
    )


def test_an_evaporator_sized_to_the_pocket_splits_the_side_at_its_level():
    # Below a pinch at 100: H1 (CP 4) cools from 100 to 20 and C1 (CP 2)
    # heats from 20 to 100. Between 100 and 60.01 H1 gives 159.96 kW and C1
    # takes 79.98 kW; an evaporator from 60 to 60.01 takes the other
    # 79.98 kW. At 60 the cascade carries 0.02 kW of the side's 560 kW.
    below = _with_shifts(
        side(
            "below",
            hot=[stream("hot", 0, 4.0, 320.0, 100.0)],
            cold=[
                stream("cold", 0, 2.0, 160.0, 100.0),
                stream("cold", 1, 7998.0, 79.98, 60.01, at_pinch=False),
            ],
        )
    )

    regions = pinch_regions(below)

    assert [[s.index for s in region.cold] for region in regions] == [[0, 1], [0]]
