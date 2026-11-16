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

    # One budget per search: no proposal completes, so the base search and
    # each fallback rung search the side, each within its own budget.
    assert calls[0] <= 20 * 5 * (1 + len(enumeration._FALLBACK_RUNGS))


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


def test_a_fruitless_search_stops_once_its_incomplete_proposals_are_kept(
    monkeypatch,
):
    # Every match is temperature-limited, so no proposal completes. Once the
    # cap of incomplete proposals was kept, the search still spent its whole
    # node budget (47 s here) looking for a complete one. Counting nodes, not
    # seconds, keeps the check independent of the machine (Pyodide is slow).
    from OpenPinch.analysis.heat_exchanger_networks.heuristic import enumeration

    nodes = [0]
    real_extend = enumeration._extend

    def counting(*args, **kwargs):
        nodes[0] += 1
        yield from real_extend(*args, **kwargs)

    monkeypatch.setattr(enumeration, "_extend", counting)
    above = side(
        "above",
        [
            stream("hot", i, 10.0, 5000.0, 150.0 + 60.0 * i, at_pinch=False)
            for i in range(3)
        ],
        [stream("cold", j, 1.0, 5000.0, 90.0, at_pinch=False) for j in range(5)],
    )

    proposals = propose_side_topologies(above, limit=10)

    assert proposals and not any(proposal.complete for proposal in proposals)
    # Well under the side's budget of 2000 nodes per proposal requested, in
    # the base search and in each fallback rung, which also stall.
    assert nodes[0] <= 2 * enumeration._STALL_NODES * (
        1 + len(enumeration._FALLBACK_RUNGS)
    )


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


def _ten_stream_ahmad_above():
    # Ahmad's ten streams above the pinch at 10 K (K, kW, kW/K). Matching the
    # must stream nearest the pinch first, no order of partners completes it.
    return side(
        "above",
        [
            stream("hot", 0, 156.3, 4532.7, 329.15),
            stream("hot", 1, 50.0, 3200.0, 329.15),
            stream("hot", 4, 1200.0, 6000.0, 358.15, at_pinch=False),
            stream("hot", 5, 50.0, 7500.0, 348.15, at_pinch=False),
        ],
        [
            stream("cold", 0, 466.7, 4200.3, 319.15),
            stream("cold", 1, 600.0, 6000.0, 328.15, at_pinch=False),
            stream("cold", 2, 180.0, 18000.0, 338.15, at_pinch=False),
            stream("cold", 3, 81.3, 10081.2, 319.15),
        ],
    )


def _nine_stream_above():
    # Linnhoff and Ahmad's nine streams above the pinch at 7.5 K. A must
    # stream is left with load no partner can take in series.
    return side(
        "above",
        [
            stream("hot", 0, 100.0, 16700.0, 433.15),
            stream("hot", 1, 160.0, 9600.0, 433.15),
            stream("hot", 2, 60.0, 3600.0, 433.15),
        ],
        [
            stream("cold", 0, 100.0, 14750.0, 425.65),
            stream("cold", 1, 70.0, 805.0, 425.65),
            stream("cold", 3, 60.0, 1050.0, 425.65),
            stream("cold", 4, 200.0, 29500.0, 425.65),
        ],
        approach=7.5,
    )


def _stages_keep_their_approach(pinch_side, proposal) -> bool:
    """Every match keeps its approach at both ends of its stage.

    Split branches leave a stage mixed, so each stream spans the whole of
    its duty in the stage, from its heat before the stage to its heat
    through it, both counted from its pinch end.
    """

    streams = {(s.kind, s.index): s for s in pinch_side.hot + pinch_side.cold}

    def layer(match):
        above = pinch_side.location == "above"
        return proposal.stages - 1 - match.stage if above else match.stage

    for match in proposal.matches:
        ends = []
        for bound in (
            lambda other: layer(other) < layer(match),
            lambda other: layer(other) <= layer(match),
        ):
            heat = {("hot", match.hot): 0.0, ("cold", match.cold): 0.0}
            for other in proposal.matches:
                if bound(other):
                    for key in (("hot", other.hot), ("cold", other.cold)):
                        if key in heat:
                            heat[key] += other.duty
            ends.append(heat)
        hot, cold = streams[("hot", match.hot)], streams[("cold", match.cold)]
        approach = pinch_side.approach[0][match.hot][match.cold] - 1e-6
        for heat in ends:
            if (
                hot.temperature(0, heat[("hot", match.hot)])
                - cold.temperature(0, heat[("cold", match.cold)])
                < approach
            ):
                return False
    return True


def test_a_side_short_of_complete_proposals_backtracks_over_the_must_stream_order():
    proposals = propose_side_topologies(_ten_stream_ahmad_above(), limit=10)

    assert proposals[0].complete


def test_a_must_stream_no_partner_takes_in_series_splits_a_partner_in_parallel():
    nine = _nine_stream_above()

    proposals = propose_side_topologies(nine, limit=10)

    complete = [proposal for proposal in proposals if proposal.complete]
    assert complete
    assert all(_stages_keep_their_approach(nine, proposal) for proposal in complete)


def test_the_fallback_searches_add_nothing_to_a_side_already_at_the_cap(monkeypatch):
    from OpenPinch.analysis.heat_exchanger_networks.heuristic import enumeration

    searches = []
    real_completions = enumeration._completions

    def counting(*args, **kwargs):
        searches.append(kwargs.get("rung"))
        yield from real_completions(*args, **kwargs)

    monkeypatch.setattr(enumeration, "_completions", counting)

    proposals = propose_side_topologies(_four_stream_below(), limit=1)

    assert proposals[0].complete
    assert set(searches) == {enumeration._BASE_RUNG}


def test_the_largest_feasible_duty_is_found_past_an_infeasible_stretch():
    # A cold stream heating fast (CP 0.5), then slowly (CP 10), then fast again
    # (CP 0.1), against a hot stream of CP 1, 10 K approach. The end gap dips
    # below the approach from 2 kW to 18.9 kW, recovers, and closes again at
    # 25 + 5.5 / 9 kW. Bisection from the 30 kW limit stops at 2 kW.
    from dataclasses import replace

    from OpenPinch.analysis.heat_exchanger_networks.heuristic.enumeration import (
        _feasible_duty,
        _State,
    )
    from OpenPinch.analysis.heat_exchanger_networks.heuristic.side_streams import (
        SideStream,
    )

    above = side(
        "above",
        [stream("hot", 0, 1.0, 30.0, 100.0)],
        [stream("cold", 0, 1.0, 30.0, 88.0)],
    )
    segmented = SideStream(
        kind="cold",
        index=0,
        loads=(30.0,),
        pinch_cps=(0.5,),
        at_pinch=(True,),
        heats=((0.0, 10.0, 25.0, 30.0),),
        temperatures=((88.0, 108.0, 109.5, 159.5),),
        weights=(1.0,),
    )
    above = replace(above, cold=(segmented,))
    hot, cold = above.hot[0], above.cold[0]
    state = _State(
        remaining={("hot", 0): 30.0, ("cold", 0): 30.0},
        used={("hot", 0): 0.0, ("cold", 0): 0.0},
        layer={},
        pairs=frozenset(),
        matches=(),
    )

    assert _feasible_duty(above, state, hot, cold) == pytest.approx(
        25.0 + 5.5 / 9.0, rel=1e-6
    )
