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


def test_with_a_cap_of_one_the_most_cp_similar_pinch_match_is_kept():
    # The 6 kW/K cold stream's load would tick the hot stream off exactly,
    # but the 2.2 kW/K one is closer in CP.
    above = side(
        "above",
        [stream("hot", 0, 2.0, 100.0, 100.0)],
        [stream("cold", 0, 2.2, 300.0, 90.0), stream("cold", 1, 6.0, 100.0, 90.0)],
    )

    (kept,) = propose_side_topologies(above, limit=1)

    pinch = {(m.hot, m.cold) for m in kept.matches if m.stage == kept.stages - 1}
    assert pinch == {(0, 0)}
    assert pinch_assignments(above, limit=2) == [((0, 0),), ((0, 1),)]


def test_away_from_the_pinch_the_most_cp_similar_partner_comes_first():
    # The hot stream away from the pinch ticks off against either cold
    # stream; the 1.1 kW/K one is closer in CP than the 6 kW/K one.
    above = side(
        "above",
        [
            stream("hot", 0, 2.0, 100.0, 100.0),
            stream("hot", 1, 1.0, 40.0, 200.0, at_pinch=False),
        ],
        [
            stream("cold", 0, 2.0, 100.0, 90.0),
            stream("cold", 1, 6.0, 300.0, 95.0, at_pinch=False),
            stream("cold", 2, 1.1, 200.0, 95.0, at_pinch=False),
        ],
    )

    (kept,) = propose_side_topologies(above, limit=1)

    away = {(m.hot, m.cold) for m in kept.matches if m.hot == 1}
    assert away == {(1, 2)}


def test_every_partner_stream_is_offered_its_utilities():
    # A partner whose load the matches seem to cover still gets its heaters
    # or coolers: when the side model cannot reach the tick-off duties, the
    # load left must have somewhere to go (Six-stream Yee and Grossmann below
    # the pinch put its cooler on the wrong stream).
    below = side(
        "below",
        [stream("hot", 0, 20.0, 1000.0, 590.0)],
        [stream("cold", 0, 15.0, 1000.0, 576.0)],
        approach=14.0,
    )

    (proposal,) = propose_side_topologies(below, limit=10)

    assert proposal.complete
    assert proposal.coolers == ((0, 0),)


def test_an_unsplit_pinch_match_gets_no_relief_stage():
    # The relief stage lets split branches that mix at one temperature share
    # their duty with the next stage out; a match with no split gains nothing.
    from OpenPinch.analysis.heat_exchanger_networks.heuristic.enumeration import (
        _completions,
    )

    above = side(
        "above",
        [stream("hot", 0, 10.0, 100.0, 380.0)],
        [stream("cold", 0, 5.0, 100.0, 370.0)],
    )

    proposal = next(_completions(above, ((0, 0),), nodes=[100]))

    assert proposal.stages == 1


def test_a_must_stream_splits_into_enough_branches_for_its_cp():
    # Thirteen-stream Kim below the pinch: one cold stream of CP 289.9 needs
    # four hot partners; two branches broke the CP rule on every proposal.
    from OpenPinch.analysis.heat_exchanger_networks.heuristic.rules import (
        split_counts,
    )

    below = side(
        "below",
        [
            stream("hot", i, cp, 1000.0, 400.0)
            for i, cp in enumerate([100.0, 90.0, 80.0, 70.0])
        ],
        [stream("cold", 0, 289.9, 1000.0, 390.0)],
    )

    must_counts, _partner_counts = split_counts(below)

    assert must_counts == {0: 4}
    assert all(
        proposal.complete for proposal in propose_side_topologies(below, limit=3)
    )


def test_a_proposal_that_breaks_the_cp_rule_is_not_complete():
    below = side(
        "below",
        [stream("hot", i, cp, 1000.0, 400.0) for i, cp in enumerate([100.0, 90.0])],
        [stream("cold", 0, 500.0, 1000.0, 390.0)],
    )

    proposals = propose_side_topologies(below, limit=3)

    assert proposals and not any(proposal.complete for proposal in proposals)


def test_a_side_search_stays_within_one_node_budget(monkeypatch):
    # Each pinch-match set had its own budget of 2000 nodes per proposal, so
    # a side's work grew with the square of the cap.
    from OpenPinch.analysis.heat_exchanger_networks.heuristic import enumeration

    calls = [0]
    real_extend = enumeration._extend

    def counting(*args, **kwargs):
        calls[0] += 1
        yield from real_extend(*args, **kwargs)

    monkeypatch.setattr(enumeration, "_extend", counting)
    monkeypatch.setattr(enumeration, "_NODES_PER_PROPOSAL", 20)
    above = side(
        "above",
        [stream("hot", i, 2.0 + i, 100.0, 100.0) for i in range(6)]
        # A hot stream no cold stream can take: every proposal dead-ends.
        + [stream("hot", 6, 1.0, 500.0, 200.0, at_pinch=False)],
        # Partners keep load after the pinch matches, so the search branches
        # widely before it runs out of room for the last hot stream.
        [stream("cold", j, 3.0 + j, 150.0, 90.0) for j in range(6)],
    )

    propose_side_topologies(above, limit=5)

    assert calls[0] <= 20 * 5


def test_a_relief_stage_proposal_is_not_complete():
    # One cold stream at the pinch splits three ways and its mixed branches
    # break the approach, so the relief stage is added. Most such proposals
    # cannot be realised: they rank after complete ones.
    from OpenPinch.analysis.heat_exchanger_networks.heuristic.enumeration import (
        _completions,
    )

    above = side(
        "above",
        [
            stream("hot", 0, 6.0, 600.0, 380.0),
            stream("hot", 1, 4.0, 200.0, 380.0),
            stream("hot", 2, 6.0, 600.0, 380.0),
        ],
        [stream("cold", 0, 18.0, 3000.0, 370.0)],
    )

    proposal = next(_completions(above, ((0, 0), (1, 0), (2, 0)), nodes=[100]))

    assert proposal.stages == 2
    assert not proposal.complete


def test_a_wide_split_among_many_partners_finishes_quickly():
    # One must stream needing eight branches among 40 partners listed every
    # combination at every search node and did not finish.
    import time

    below = side(
        "below",
        [stream("hot", i, 10.0, 100.0, 400.0) for i in range(40)],
        [stream("cold", 0, 80.0, 1000.0, 390.0)],
    )

    start = time.perf_counter()
    proposals = propose_side_topologies(below, limit=10)

    assert proposals
    assert time.perf_counter() - start < 10.0


def test_a_fruitless_search_stops_once_its_incomplete_proposals_are_kept():
    # Every match is temperature-limited, so no proposal completes. Once the
    # cap of incomplete proposals was kept, the search still spent its whole
    # node budget (47 s here) looking for a complete one.
    import time

    above = side(
        "above",
        [
            stream("hot", i, 10.0, 5000.0, 150.0 + 60.0 * i, at_pinch=False)
            for i in range(3)
        ],
        [stream("cold", j, 1.0, 5000.0, 90.0, at_pinch=False) for j in range(5)],
    )

    start = time.perf_counter()
    proposals = propose_side_topologies(above, limit=10)

    assert proposals and not any(proposal.complete for proposal in proposals)
    assert time.perf_counter() - start < 5.0


def test_dead_ends_met_first_do_not_hide_the_complete_proposals_behind_them():
    # The stream nearest the pinch prefers the only cold stream the farthest
    # hot stream may use, so the first choice leads to hundreds of dead ends
    # and the second to complete designs, a few hundred nodes on.
    from dataclasses import replace

    hot = (
        [stream("hot", 0, 1.0, 100.0, 300.0, at_pinch=False)]
        + [
            stream("hot", 1 + i, 1.0, 100.0, 311.0 + i, at_pinch=False)
            for i in range(4)
        ]
        + [stream("hot", 5, 1.0, 100.0, 400.0, at_pinch=False)]
    )
    cold = [stream("cold", j, 1.0, 100.0, 100.0, at_pinch=False) for j in range(8)]
    allowed = [[False] * 8 for _hot in range(6)]
    allowed[0][0] = allowed[0][1] = allowed[5][0] = True
    for i in range(1, 5):
        allowed[i][2:] = [True] * 6
    above = replace(
        side("above", hot, cold),
        match_allowed=tuple(tuple(row) for row in allowed),
    )

    proposals = propose_side_topologies(above, limit=5)

    assert [proposal.complete for proposal in proposals] == [True] * 5
