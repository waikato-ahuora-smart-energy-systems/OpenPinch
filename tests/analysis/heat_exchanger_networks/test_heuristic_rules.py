"""Linnhoff pinch rules and stream splitting on hand-built pinch sides."""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.heuristic.rules import (
    cp_feasible,
    cp_rule_met,
    number_rule_met,
    pinch_streams,
    split_counts,
)
from tests.analysis.heat_exchanger_networks.heuristic_sides import side, stream


def _above(hot_cps, cold_cps):
    return side(
        "above",
        [stream("hot", i, cp, 100.0, 100.0) for i, cp in enumerate(hot_cps)],
        [stream("cold", j, cp, 100.0, 90.0) for j, cp in enumerate(cold_cps)],
    )


def test_textbook_side_meets_both_rules_without_splits():
    above = _above([2.0, 3.0], [3.0, 4.0])
    must, partners = pinch_streams(above)

    assert [s.kind for s in must] == ["hot", "hot"]
    assert number_rule_met(above)
    assert cp_rule_met(must[0], partners[0])
    assert not cp_rule_met(partners[1], must[0])
    assert split_counts(above) == ({0: 1, 1: 1}, {0: 1, 1: 1})
    assert cp_feasible(above, [(0, 0), (1, 1)])
    assert cp_feasible(above, [(0, 1), (1, 0)])


def test_cp_rule_failure_splits_the_hot_stream_above_the_pinch():
    above = _above([5.0], [3.0, 4.0])

    assert number_rule_met(above)
    assert split_counts(above) == ({0: 2}, {0: 1, 1: 1})
    assert not cp_feasible(above, [(0, 1)])
    assert cp_feasible(above, [(0, 0), (0, 1)])


def test_number_rule_failure_splits_the_largest_cp_cold_stream():
    above = _above([2.0, 3.0, 1.0], [10.0, 4.0])

    assert not number_rule_met(above)
    assert split_counts(above) == ({0: 1, 1: 1, 2: 1}, {0: 2, 1: 1})
    assert cp_feasible(above, [(0, 0), (1, 1), (2, 0)])


def test_below_the_pinch_the_rules_mirror_with_cold_streams_matched_first():
    below = side(
        "below",
        [stream("hot", 0, 10.0, 100.0, 100.0)],
        [stream("cold", j, cp, 100.0, 90.0) for j, cp in enumerate([2.0, 3.0, 1.0])],
    )
    must, partners = pinch_streams(below)

    assert [s.kind for s in must] == ["cold"] * 3
    assert [s.kind for s in partners] == ["hot"]
    assert split_counts(below) == ({0: 1, 1: 1, 2: 1}, {0: 3})
    assert cp_feasible(below, [(0, 0), (1, 0), (2, 0)])


def test_streams_away_from_the_pinch_take_no_part_in_the_pinch_rules():
    above = side(
        "above",
        [
            stream("hot", 0, 2.0, 100.0, 100.0),
            stream("hot", 1, 9.0, 50.0, 140.0, at_pinch=False),
        ],
        [stream("cold", 0, 3.0, 100.0, 90.0)],
    )

    must, _partners = pinch_streams(above)
    assert [s.index for s in must] == [0]
    assert number_rule_met(above)
