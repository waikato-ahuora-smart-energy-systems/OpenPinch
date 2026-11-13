"""Enumerating pinch-side topologies with the Linnhoff rules."""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_exchanger_networks.heuristic.enumeration import (
    pinch_assignments,
    propose_side_topologies,
)
from tests.analysis.heat_exchanger_networks.heuristic_sides import side, stream


def _four_stream_below():
    # Yee and Grossmann's four streams below the pinch at 14 K (583 K shifted).
    return side(
        "below",
        [stream("hot", 0, 10.0, 2200.0, 590.0), stream("hot", 1, 20.0, 4400.0, 590.0)],
        [
            stream("cold", 0, 15.0, 2490.0, 576.0),
            stream("cold", 1, 13.0, 1950.0, 500.0, at_pinch=False),
        ],
        approach=14.0,
    )


def _duties(proposal):
    duties = {(m.hot, m.cold, m.stage): m.duty for m in proposal.matches}
    return {key: pytest.approx(value) for key, value in duties.items()}


def test_pinch_matches_follow_the_cp_rule():
    # Below the pinch the cold stream needs a hot partner with a CP at least
    # its own: 20 kW/K, not 10.
    assignments = pinch_assignments(_four_stream_below(), limit=10)

    assert assignments[0] == ((0, 1),)
    assert ((0, 0),) not in assignments


def test_pinch_matches_tick_off_then_away_matches_then_utilities():
    proposals = propose_side_topologies(_four_stream_below(), limit=10)

    first = proposals[0]
    assert first.complete and first.stages == 1
    assert _duties(first) == {(1, 0, 0): 2490.0, (0, 1, 0): 1950.0}
    assert set(first.coolers) == {(0, 0), (1, 0)}
    assert first.heaters == ()


def test_a_match_away_from_the_pinch_sits_one_stage_further_out():
    above = side(
        "above",
        [
            stream("hot", 0, 2.0, 100.0, 100.0),
            stream("hot", 1, 1.0, 40.0, 150.0, at_pinch=False),
        ],
        [stream("cold", 0, 4.0, 300.0, 90.0)],
    )

    first = propose_side_topologies(above, limit=10)[0]

    assert first.stages == 2
    # Above the pinch, stage S - 1 is next to it.
    assert _duties(first) == {(0, 0, 1): 100.0, (1, 0, 0): 40.0}
    assert first.heaters == ((0, 0),)
    z, z_hu, z_cu = first.z_restriction(
        hot_count=2, cold_count=1, hot_utilities=1, cold_utilities=1
    )
    assert z == [[[0, 1]], [[1, 0]]]
    assert z_hu == [[1]] and z_cu == [[0], [0]]


def test_alternative_pinch_assignments_give_distinct_proposals_up_to_the_cap():
    above = side(
        "above",
        [stream("hot", i, cp, 100.0, 100.0) for i, cp in enumerate([2.0, 3.0])],
        [stream("cold", j, cp, 100.0, 90.0) for j, cp in enumerate([3.0, 4.0])],
    )

    proposals = propose_side_topologies(above, limit=10)

    assert len(proposals) == 2
    assert len({proposal.signature for proposal in proposals}) == 2
    assert {frozenset((m.hot, m.cold) for m in p.matches) for p in proposals} == {
        frozenset({(0, 0), (1, 1)}),
        frozenset({(0, 1), (1, 0)}),
    }
    assert len(propose_side_topologies(above, limit=1)) == 1


def test_a_split_must_stream_matches_two_partners_at_the_pinch():
    above = side(
        "above",
        [stream("hot", 0, 5.0, 100.0, 100.0)],
        [stream("cold", j, cp, 100.0, 90.0) for j, cp in enumerate([3.0, 4.0])],
    )

    first = propose_side_topologies(above, limit=10)[0]

    assert {(m.hot, m.cold, m.stage) for m in first.matches} == {(0, 0, 0), (0, 1, 0)}
    assert sum(m.duty for m in first.matches) == pytest.approx(100.0)


def test_an_optional_split_at_the_pinch_is_a_later_alternative():
    proposals = propose_side_topologies(_four_stream_below(), limit=10)

    pinch_matches = [
        {(m.hot, m.cold) for m in p.matches if m.cold == 0} for p in proposals
    ]
    assert pinch_matches[0] == {(1, 0)}
    # The cold pinch stream split across both hot streams.
    assert {(0, 0), (1, 0)} in pinch_matches[1:]


def _two_period_above():
    from dataclasses import replace

    base = side(
        "above",
        [stream("hot", 0, 2.0, 100.0, 100.0)],
        [stream("cold", 0, 3.0, 100.0, 90.0), stream("cold", 1, 4.0, 100.0, 90.0)],
    )

    def periods(s, active):
        return replace(
            s,
            loads=tuple(s.loads[0] if on else 0.0 for on in active),
            pinch_cps=tuple(s.pinch_cps[0] if on else 0.0 for on in active),
            at_pinch=tuple(on for on in active),
            heats=(s.heats[0],) * 2,
            temperatures=(s.temperatures[0],) * 2,
            weights=(0.5, 0.5),
        )

    return replace(
        base,
        hot=(periods(base.hot[0], (True, True)),),
        cold=(
            periods(base.cold[0], (True, False)),
            periods(base.cold[1], (False, True)),
        ),
        approach=base.approach * 2,
        weights=(0.5, 0.5),
    )


def test_a_multiperiod_proposal_is_the_union_of_each_periods_design():
    first = propose_side_topologies(_two_period_above(), limit=10)[0]

    assert {(m.hot, m.cold, m.stage) for m in first.matches} == {(0, 0, 0), (0, 1, 0)}
    assert first.complete
