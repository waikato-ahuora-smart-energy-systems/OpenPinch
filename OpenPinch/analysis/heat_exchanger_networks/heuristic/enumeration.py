"""Enumerate pinch-side topologies with the Linnhoff pinch design rules.

Each proposal starts with a set of pinch matches that meets the CP and
number rules (splitting streams where they fail), ticks each pinch match off
at the smaller of its two loads, then matches the remaining must-stream loads
away from the pinch, nearest the pinch first, branching over the partners whose
temperatures allow it. Partners left with load take utilities. Proposals are
topologies only: the pinch side model realises their duties.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace
from itertools import combinations, islice
from typing import Iterator

from .rules import cp_feasible, cp_rule_met, pinch_streams, split_counts
from .side_streams import PinchSide, SideStream

# A load within this fraction of the stream's load counts as met.
LOAD_FRACTION_TOLERANCE = 1e-3
# DFS nodes allowed per proposal requested, across a side's search.
_NODES_PER_PROPOSAL = 2000
_BISECTION_STEPS = 40
# Pinch-match sets collected per one kept, before ordering by CP similarity.
_COLLECTED_PER_PROPOSAL = 4
# Search nodes spent without a complete proposal, once the incomplete ones
# are kept, before a side's search stops.
_STALL_NODES = 1000
# Partner combinations tried per must stream at one search node.
_COMBINATIONS_PER_NODE = 64


def cp_distance(cp_a: float, cp_b: float) -> float:
    """CP dissimilarity of a match, ``|ln(CP_a / CP_b)|``; 0 for equal CPs."""

    if cp_a <= 0.0 or cp_b <= 0.0:
        return math.inf
    return abs(math.log(cp_a / cp_b))


@dataclass(frozen=True)
class ProposedMatch:
    """A recovery match in a side's stage grid, with its tick-off duty."""

    hot: int
    cold: int
    stage: int
    duty: float


@dataclass(frozen=True)
class SideProposal:
    """One topology for one pinch side.

    ``heaters`` and ``coolers`` are allowed ``(stream, utility)`` pairs; the
    side model's cost objective picks among them. ``complete`` is false when
    some must-stream load found no partner, so the side model will likely
    find the proposal infeasible.
    """

    location: str
    stages: int
    matches: tuple[ProposedMatch, ...]
    heaters: tuple[tuple[int, int], ...]
    coolers: tuple[tuple[int, int], ...]
    complete: bool

    @property
    def signature(self) -> tuple:
        return (
            self.stages,
            tuple(sorted((m.hot, m.cold, m.stage) for m in self.matches)),
            tuple(sorted(self.heaters)),
            tuple(sorted(self.coolers)),
        )

    def z_restriction(
        self,
        *,
        hot_count: int,
        cold_count: int,
        hot_utilities: int,
        cold_utilities: int,
    ) -> list:
        """``[z, z_hu, z_cu]`` match restrictions for the side model."""

        z = [
            [[0 for _k in range(self.stages)] for _j in range(cold_count)]
            for _i in range(hot_count)
        ]
        for match in self.matches:
            z[match.hot][match.cold][match.stage] = 1
        z_hu = [[0] * hot_utilities for _j in range(cold_count)]
        for j, u in self.heaters:
            z_hu[j][u] = 1
        z_cu = [[0] * cold_utilities for _i in range(hot_count)]
        for i, u in self.coolers:
            z_cu[i][u] = 1
        return [z, z_hu, z_cu]


def pinch_assignments(
    side: PinchSide, *, limit: int
) -> list[tuple[tuple[int, int], ...]]:
    """Up to ``limit`` sets of pinch matches meeting the pinch rules.

    Each set holds ``(must index, partner index)`` matches. A must stream
    takes as many partners as it has branches and a partner hosts at most
    its branch count; split ratios must exist that meet the CP rule. The
    textbook split counts come first, fewer splits first; then, as
    alternatives, one must stream at a time split once more, largest CP
    first. When no set meets the rules, each partner may host one more
    branch; failing that, the CP rule is dropped.
    """

    must, partners = pinch_streams(side)
    if not must or not partners:
        return [()]
    must_counts, partner_counts = split_counts(side)
    for extra, check_cp in ((0, True), (1, True), (1, False)):
        capacity = {index: count + extra for index, count in partner_counts.items()}
        found = _assignments(
            side, must, partners, must_counts, capacity, limit=limit, check_cp=check_cp
        )
        if found:
            break
    else:
        return [()]
    for stream in sorted(must, key=lambda s: (-s.pinch_cp, s.index)):
        if len(found) >= limit:
            break
        more = dict(must_counts) | {stream.index: must_counts[stream.index] + 1}
        found += [
            edges
            for edges in _assignments(
                side,
                must,
                partners,
                more,
                capacity,
                limit=limit - len(found),
                check_cp=check_cp,
                exact=True,
            )
            if edges not in found
        ]
    return found[:limit]


def _assignments(
    side, must, partners, must_counts, capacity, *, limit, check_cp, exact=False
):
    # Hardest streams first: largest CP, then most branches.
    order = sorted(must, key=lambda s: (-must_counts[s.index], -s.pinch_cp, s.index))
    found: list[tuple[tuple[int, int], ...]] = []
    budget = [_NODES_PER_PROPOSAL * max(1, limit)]

    def options(stream: SideStream, used: dict[int, int]):
        allowed = [
            p
            for p in partners
            if used.get(p.index, 0) < capacity[p.index] and _allowed(side, stream, p)
        ]
        count = min(must_counts[stream.index], len(allowed))
        if count == 0 or (exact and count < must_counts[stream.index]):
            return []
        # Partners in order of preference, then a bounded number of their
        # combinations, generated lazily: listing every combination of many
        # partners at every node does not finish.
        allowed.sort(
            key=lambda p: (
                not cp_rule_met(stream, p),
                cp_distance(stream.pinch_cp, p.pinch_cp),
                p.index,
            )
        )
        combos = list(islice(combinations(allowed, count), _COMBINATIONS_PER_NODE))
        # One-to-one CP rule first, then the most similar CPs, then partners
        # whose load ticks the stream off most closely.
        return sorted(
            combos,
            key=lambda combo: (
                not all(cp_rule_met(stream, p) for p in combo),
                sum(cp_distance(stream.pinch_cp, p.pinch_cp) for p in combo),
                abs(sum(p.load for p in combo) - stream.load),
                tuple(p.index for p in combo),
            ),
        )

    def search(position: int, edges: list, used: dict[int, int]) -> None:
        # Collect past the cap, within the node budget, so the CP ordering
        # below decides which sets the cap keeps.
        if len(found) >= _COLLECTED_PER_PROPOSAL * limit or budget[0] <= 0:
            return
        budget[0] -= 1
        if position == len(order):
            found.append(tuple(sorted(edges)))
            return
        stream = order[position]
        for combo in options(stream, used):
            new_edges = edges + [(stream.index, p.index) for p in combo]
            if check_cp and not cp_feasible(side, new_edges):
                continue
            new_used = dict(used)
            for p in combo:
                new_used[p.index] = new_used.get(p.index, 0) + 1
            search(position + 1, new_edges, new_used)

    search(0, [], {})
    # Fewer splits first, then the most similar CPs over the whole set.
    by_index = {s.index: s for s in must}
    partner_by_index = {s.index: s for s in partners}
    return sorted(
        found,
        key=lambda edges: (
            _split_count(edges),
            sum(
                cp_distance(by_index[m].pinch_cp, partner_by_index[p].pinch_cp)
                for m, p in edges
            ),
            edges,
        ),
    )[:limit]


def _split_count(edges) -> int:
    musts = [m for m, _p in edges]
    partners = [p for _m, p in edges]
    return (len(musts) - len(set(musts))) + (len(partners) - len(set(partners)))


def propose_side_topologies(side: PinchSide, *, limit: int) -> list[SideProposal]:
    """Up to ``limit`` distinct topologies for one pinch side.

    With several periods, each period is designed on its own and a proposal
    is the union of one design per period (one shared topology), stages
    aligned by their distance from the pinch. Combinations of lower-ranked
    designs come first.
    """

    if limit <= 0:
        return []
    if side.periods == 1:
        return _single_period_topologies(side, limit=limit)
    designs = [
        _single_period_topologies(side.period_view(n), limit=limit)
        for n in range(side.periods)
    ]
    designs = [period for period in designs if period]
    if not designs:
        return []
    proposals: list[SideProposal] = []
    seen: set[tuple] = set()
    start = (0,) * len(designs)
    heap = [(0, start)]
    queued = {start}
    pops = 0
    while heap and len(proposals) < limit and pops < _NODES_PER_PROPOSAL * limit:
        pops += 1
        _total, ranks = heapq.heappop(heap)
        union = _union(side, [designs[n][r] for n, r in enumerate(ranks)])
        if union.signature not in seen:
            seen.add(union.signature)
            proposals.append(union)
        for n in range(len(designs)):
            if ranks[n] + 1 < len(designs[n]):
                following = ranks[:n] + (ranks[n] + 1,) + ranks[n + 1 :]
                if following not in queued:
                    queued.add(following)
                    heapq.heappush(heap, (sum(following), following))
    return sorted(proposals, key=lambda proposal: not proposal.complete)


def _union(side: PinchSide, designs: list[SideProposal]) -> SideProposal:
    """One topology holding every period design, stages aligned at the pinch."""

    stages = max(design.stages for design in designs)
    above = side.location == "above"
    matches: dict[tuple[int, int, int], ProposedMatch] = {}
    for design in designs:
        for match in design.matches:
            layer = design.stages - 1 - match.stage if above else match.stage
            stage = stages - 1 - layer if above else layer
            key = (match.hot, match.cold, stage)
            if key not in matches or matches[key].duty < match.duty:
                matches[key] = ProposedMatch(match.hot, match.cold, stage, match.duty)
    return SideProposal(
        location=side.location,
        stages=stages,
        matches=tuple(matches.values()),
        heaters=tuple(sorted({h for design in designs for h in design.heaters})),
        coolers=tuple(sorted({c for design in designs for c in design.coolers})),
        complete=all(design.complete for design in designs),
    )


def _single_period_topologies(side: PinchSide, *, limit: int) -> list[SideProposal]:
    """Up to ``limit`` distinct topologies for one single-period side.

    Every set of pinch matches gets its own depth-first search over the
    matches away from the pinch; the searches take turns, so the proposals
    vary the pinch design as well as the later choices. Complete proposals
    come first, and incomplete ones fill the cap only when too few are
    complete.
    """

    if limit <= 0:
        return []
    # One node budget for the whole side, shared by every pinch-match search.
    nodes = [_NODES_PER_PROPOSAL * limit]
    searches = [
        _completions(side, assignment, nodes=nodes)
        if not assignment or cp_feasible(side, assignment)
        # Pinch matches that break the CP rule (its last fallback) cannot
        # make a complete design, whatever follows.
        else (
            replace(proposal, complete=False)
            for proposal in _completions(side, assignment, nodes=nodes)
        )
        for assignment in pinch_assignments(side, limit=limit)
    ]
    complete: list[SideProposal] = []
    incomplete: list[SideProposal] = []
    seen: set[tuple] = set()
    # Node budget left when a proposal was last kept: once the incomplete
    # list is full, a search that spends _STALL_NODES without a complete
    # proposal is unlikely to find one.
    kept_at = nodes[0]
    stalled = False
    while searches and len(complete) < limit and not stalled:
        for search in list(searches):
            proposal = next(search, None)
            if proposal is None:
                searches.remove(search)
                continue
            if not proposal.complete and len(incomplete) >= limit:
                stalled = kept_at - nodes[0] >= _STALL_NODES
                if stalled:
                    break
                continue
            kept_at = nodes[0]
            if proposal.signature in seen:
                continue
            seen.add(proposal.signature)
            if proposal.complete:
                complete.append(proposal)
            elif len(incomplete) < limit:
                incomplete.append(proposal)
            if len(complete) >= limit:
                break
    return (complete + incomplete)[:limit]


def _allowed(side: PinchSide, must: SideStream, partner: SideStream) -> bool:
    hot, cold = _hot_cold(must, partner)
    return side.match_allowed[hot.index][cold.index]


def _hot_cold(a: SideStream, b: SideStream) -> tuple[SideStream, SideStream]:
    return (a, b) if a.kind == "hot" else (b, a)


@dataclass
class _State:
    remaining: dict
    used: dict
    layer: dict
    pairs: frozenset
    matches: tuple
    # A relief stage was added: most such proposals cannot be realised, so
    # they are never marked complete.
    relief: bool = False


def _completions(
    side: PinchSide, assignment, *, nodes: list[int]
) -> Iterator[SideProposal]:
    """Depth-first completions of one set of pinch matches.

    ``nodes`` is the side's remaining node budget, shared between searches.
    """

    must_kind = "hot" if side.location == "above" else "cold"
    streams = {(s.kind, s.index): s for s in side.hot + side.cold}
    state = _State(
        remaining={key: s.load for key, s in streams.items()},
        used={key: 0.0 for key in streams},
        layer={},
        pairs=frozenset(),
        matches=(),
    )
    partner_kind = "cold" if must_kind == "hot" else "hot"
    state = _tick_off(side, state, streams, assignment, must_kind, partner_kind)
    yield from _extend(side, state, streams, must_kind, nodes)


def _tick_off(side, state, streams, assignment, must_kind, partner_kind) -> _State:
    """Place the pinch matches next to the pinch, each at its tick-off duty.

    A split stream shares its load over its pinch matches in proportion to
    the other streams' CPs. The side model mixes split branches at one
    temperature, so each match must keep its approach between the streams'
    temperatures at both ends of the stage, not just its own slice. When a
    split's match cannot, every pinch match is also allowed one stage
    further out, and the side model shares the duty over the two stages.
    """

    must_edges: dict[int, list[int]] = {}
    partner_edges: dict[int, list[int]] = {}
    for m, p in assignment:
        must_edges.setdefault(m, []).append(p)
        partner_edges.setdefault(p, []).append(m)
    remaining = dict(state.remaining)
    used = dict(state.used)
    layer = dict(state.layer)
    matches = []
    for m, p in assignment:
        must = streams[(must_kind, m)]
        partner = streams[(partner_kind, p)]
        must_share = partner.pinch_cp / sum(
            streams[(partner_kind, q)].pinch_cp for q in must_edges[m]
        )
        partner_share = must.pinch_cp / sum(
            streams[(must_kind, q)].pinch_cp for q in partner_edges[p]
        )
        duty = min(must.load * must_share, partner.load * partner_share)
        for key in ((must_kind, m), (partner_kind, p)):
            remaining[key] -= duty
            used[key] += duty
            layer[key] = 0
        hot, cold = _hot_cold(must, partner)
        matches.append((hot.index, cold.index, 0, duty))
    split = len(must_edges) < len(assignment) or len(partner_edges) < len(assignment)
    if split and not _mixed_stage_feasible(side, streams, matches, used):
        relief = [(hot, cold, 1, 0.0) for hot, cold, _layer, _duty in matches]
        matches += relief
        for hot, cold, _layer, _duty in relief:
            layer[("hot", hot)] = layer[("cold", cold)] = 1
    return _State(
        remaining=remaining,
        used=used,
        layer=layer,
        pairs=frozenset((m, p) for m, p in assignment),
        matches=tuple(matches),
        relief=split and len(matches) > len(assignment),
    )


def _mixed_stage_feasible(side, streams, matches, stage_heat) -> bool:
    """Whether every match in the pinch stage keeps its approach at both ends.

    Each stream runs from its pinch end to ``stage_heat`` (its whole duty in
    the stage, as mixed branches leave at one temperature), scaled by each
    period's load.
    """

    for hot_index, cold_index, _layer, _duty in matches:
        hot, cold = streams[("hot", hot_index)], streams[("cold", cold_index)]
        for n in range(side.periods):
            if not (hot.active(n) and cold.active(n)):
                continue
            approach = side.approach[n][hot_index][cold_index] - 1e-6
            hot_far = stage_heat[("hot", hot_index)] * hot.loads[n] / hot.load
            cold_far = stage_heat[("cold", cold_index)] * cold.loads[n] / cold.load
            for hot_heat, cold_heat in ((0.0, 0.0), (hot_far, cold_far)):
                if (
                    hot.temperature(n, hot_heat) - cold.temperature(n, cold_heat)
                    < approach
                ):
                    return False
    return True


def _extend(side, state, streams, must_kind, nodes) -> Iterator[SideProposal]:
    if nodes[0] <= 0:
        return
    nodes[0] -= 1
    open_must = [
        s
        for (kind, _index), s in streams.items()
        if kind == must_kind and not _met(s, state.remaining[(kind, s.index)])
    ]
    if not open_must:
        yield _proposal(side, state, streams, must_kind, complete=not state.relief)
        return
    # The stream nearest the pinch first, so it takes the partners' slices
    # nearest the pinch before a stream far from it uses them up; then the
    # largest remaining load.
    must = min(
        open_must,
        key=lambda s: (
            _distance_from_pinch(side, s, state.used[(s.kind, s.index)]),
            -state.remaining[(s.kind, s.index)],
            s.index,
        ),
    )
    candidates = _candidates(side, state, streams, must, repeat=False)
    if not candidates:
        # A pair matched already may match again further out, as a second
        # exchanger in another stage, when no new partner can take the load.
        candidates = _candidates(side, state, streams, must, repeat=True)
    if not candidates:
        yield _proposal(side, state, streams, must_kind, complete=False)
        return
    must_remaining = state.remaining[(must.kind, must.index)]
    must_cp = must.cp_at(state.used[(must.kind, must.index)])
    # Tick-off first, then the most similar CP at the matched ends, then duty.
    candidates.sort(
        key=lambda item: (
            item[1] < must_remaining - _tolerance(must),
            cp_distance(
                must_cp, item[0].cp_at(state.used[(item[0].kind, item[0].index)])
            ),
            -item[1],
            item[0].index,
        )
    )
    for partner, duty in candidates:
        if nodes[0] <= 0:
            return
        yield from _extend(
            side, _place(state, must, partner, duty), streams, must_kind, nodes
        )


def _candidates(side, state, streams, must, *, repeat: bool):
    """Partners that can take some of the must stream's load, with duties.

    Without ``repeat`` only partners not yet matched with it; with it only
    those already matched.
    """

    candidates = []
    for partner in streams.values():
        if partner.kind == must.kind:
            continue
        if ((must.index, partner.index) in state.pairs) != repeat:
            continue
        if not _allowed(side, must, partner):
            continue
        if _met(partner, state.remaining[(partner.kind, partner.index)]):
            continue
        duty = _feasible_duty(side, state, must, partner)
        if duty > _tolerance(must):
            candidates.append((partner, duty))
    return candidates


def _distance_from_pinch(side, stream: SideStream, heat: float) -> float:
    """How far (K) a stream's next unmatched heat sits from the pinch side.

    Hot streams above the pinch and cold streams below it run away from the
    pinch as their heat from the pinch end grows; the mean is over the
    stream's active periods.
    """

    sign = 1.0 if side.location == "above" else -1.0
    temperatures = [
        sign * stream.temperature(n, heat * stream.loads[n] / stream.load)
        for n in range(side.periods)
        if stream.active(n)
    ]
    return sum(temperatures) / len(temperatures)


def _place(state: _State, must: SideStream, partner: SideStream, duty: float):
    keys = ((must.kind, must.index), (partner.kind, partner.index))
    layer = 1 + max(state.layer.get(key, -1) for key in keys)
    remaining = dict(state.remaining)
    used = dict(state.used)
    layers = dict(state.layer)
    for key in keys:
        remaining[key] -= duty
        used[key] += duty
        layers[key] = layer
    hot, cold = _hot_cold(must, partner)
    return _State(
        remaining=remaining,
        used=used,
        layer=layers,
        pairs=state.pairs | {(must.index, partner.index)},
        matches=state.matches + ((hot.index, cold.index, layer, duty),),
        relief=state.relief,
    )


def _feasible_duty(side, state, must, partner) -> float:
    """The largest duty, up to both remaining loads, with both ends at approach.

    The match takes the next slice of each stream from its pinch end. In
    each period where both streams run, the slice is scaled by that period's
    load, and both ends must be at least the pair's minimum approach apart.
    """

    hot, cold = _hot_cold(must, partner)
    limit = min(
        state.remaining[(must.kind, must.index)],
        state.remaining[(partner.kind, partner.index)],
    )

    def feasible(duty: float) -> bool:
        for n in range(side.periods):
            if not (hot.active(n) and cold.active(n)):
                continue
            approach = side.approach[n][hot.index][cold.index] - 1e-6
            for q in (0.0, duty):
                hot_heat = (
                    (state.used[("hot", hot.index)] + q) * hot.loads[n] / hot.load
                )
                cold_heat = (
                    (state.used[("cold", cold.index)] + q) * cold.loads[n] / cold.load
                )
                if (
                    hot.temperature(n, hot_heat) - cold.temperature(n, cold_heat)
                    < approach
                ):
                    return False
        return True

    if limit <= 0.0 or not feasible(0.0):
        return 0.0
    if feasible(limit):
        return limit
    low, high = 0.0, limit
    for _step in range(_BISECTION_STEPS):
        middle = (low + high) / 2.0
        if feasible(middle):
            low = middle
        else:
            high = middle
    return low


def _proposal(side, state, streams, must_kind, *, complete: bool) -> SideProposal:
    stages = 1 + max((match[2] for match in state.matches), default=0)
    above = side.location == "above"
    matches = tuple(
        ProposedMatch(
            hot=hot,
            cold=cold,
            stage=(stages - 1 - layer) if above else layer,
            duty=duty,
        )
        for hot, cold, layer, duty in state.matches
    )
    # Every partner may use its utilities, not just those with load left: if
    # the side model cannot reach the proposed duties, the load it leaves
    # must still have somewhere to go. The side model picks the cheapest.
    left = [s for (kind, _index), s in streams.items() if kind != must_kind]
    heaters = tuple(
        (s.index, u)
        for s in left
        if s.kind == "cold"
        for u, flag in enumerate(side.heater_allowed[s.index])
        if flag
    )
    coolers = tuple(
        (s.index, u)
        for s in left
        if s.kind == "hot"
        for u, flag in enumerate(side.cooler_allowed[s.index])
        if flag
    )
    return SideProposal(
        location=side.location,
        stages=stages,
        matches=matches,
        heaters=heaters,
        coolers=coolers,
        complete=complete,
    )


def _tolerance(stream: SideStream) -> float:
    return max(1e-9, LOAD_FRACTION_TOLERANCE * stream.load)


def _met(stream: SideStream, remaining: float) -> bool:
    return remaining <= _tolerance(stream)


__all__ = [
    "LOAD_FRACTION_TOLERANCE",
    "cp_distance",
    "ProposedMatch",
    "SideProposal",
    "pinch_assignments",
    "propose_side_topologies",
]
