"""A side is cut where its cascade closes, not a step before.

On a large side a point just short of a pinch can carry little heat next
to the side's load: the pulp mill's above side carries 23 kW of 353 MW
0.1 K below its utility pinch. A cut there leaves that heat on the wrong
side, and every proposal of the side then fails. The cut belongs at the
pinch itself.
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


def test_a_point_carrying_a_little_heat_just_below_a_pinch_is_not_the_cut():
    # Above a pinch at 100, with a second pinch at 150. Between 149.9 and
    # 150 the hot stream H3 (CP 2.2) outruns the cold C3 (CP 2.0), so
    # 0.02 kW crosses 149.9: 5 parts in 100,000 of the side's 400 kW.
    balanced = _with_shifts(
        side(
            "above",
            hot=[
                stream("hot", 0, 2.0, 99.8, 100.0),
                stream("hot", 1, 1.0, 100.0, 150.0, at_pinch=False),
                stream("hot", 2, 2.2, 0.22, 149.9, at_pinch=False),
            ],
            cold=[
                stream("cold", 0, 99.82 / 49.9, 99.82, 100.0),
                stream("cold", 1, 1.0, 100.0, 150.0, at_pinch=False),
                stream("cold", 2, 2.0, 0.2, 149.9, at_pinch=False),
            ],
        )
    )

    regions = pinch_regions(balanced)

    assert [
        ([s.index for s in region.hot], [s.index for s in region.cold])
        for region in regions
    ] == [([0, 2], [0, 2]), ([1], [1])]
