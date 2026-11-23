"""A side's own utility counts in its cascade, entering at the far end.

A side whose levels are not held as streams is out of balance by its
utility duty. That heat enters beyond every stream (a heater at a cold
stream's hot end, a cooler at a hot stream's cold end), so it crosses every
temperature the side's streams reach, and the cascade closes only where it
and the streams' heat together leave nothing.
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


def test_a_side_is_not_split_where_its_utility_still_carries_heat_down():
    # Above a pinch at 100: H1 gives 100 kW from 100 to 150, H2 20 kW from
    # 150 to 170, and C1 takes 300 kW from 100 to 200, so the heater gives
    # 180 kW. At 150 the streams beyond are 130 kW short, and the heater
    # sends 50 kW down past it: not a pinch.
    heated = _with_shifts(
        side(
            "above",
            hot=[
                stream("hot", 0, 2.0, 100.0, 100.0),
                stream("hot", 1, 1.0, 20.0, 150.0, at_pinch=False),
            ],
            cold=[stream("cold", 0, 3.0, 300.0, 100.0)],
        )
    )

    regions = pinch_regions(heated)

    assert len(regions) == 1
