"""Linnhoff pinch design rules and stream splitting for one pinch side.

At the pinch every stream that cannot use a utility on this side (hot streams
above the pinch, cold streams below) must be matched, the "must" streams. Their
partners are the other kind. The number rule asks for no more must streams than
partners; the CP rule for each pinch match asks for the must stream's CP to be
at most its partner's. Splits add branches: a partner split when the number
rule fails, a must split when no partner can take its CP.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable

from .side_streams import PinchSide, SideStream

# Relative slack on CP comparisons, so equal CPs meet the rule.
CP_TOLERANCE = 1e-6
_MAX_SPLIT_ROUNDS = 32


def pinch_streams(
    side: PinchSide,
) -> tuple[tuple[SideStream, ...], tuple[SideStream, ...]]:
    """The must streams and partners that reach the pinch."""

    hot = tuple(stream for stream in side.hot if stream.touches_pinch)
    cold = tuple(stream for stream in side.cold if stream.touches_pinch)
    return (hot, cold) if side.location == "above" else (cold, hot)


def number_rule_met(side: PinchSide) -> bool:
    must, partners = pinch_streams(side)
    return len(must) <= len(partners)


def cp_rule_met(must: SideStream, partner: SideStream) -> bool:
    """CP rule for one pinch match, in every period both reach the pinch."""

    return all(
        must.pinch_cps[n] <= partner.pinch_cps[n] * (1.0 + CP_TOLERANCE)
        for n in range(len(must.loads))
        if must.at_pinch[n] and partner.at_pinch[n]
    )


def split_counts(side: PinchSide) -> tuple[dict[int, int], dict[int, int]]:
    """Branches per must stream and per partner after the splitting rules.

    A must stream whose CP no partner can take is split into the fewest
    branches whose largest partner CPs together cover it (at least two, at
    most one per partner); then, while must branches outnumber partner
    branches, the partner with the largest CP per branch is split once more.
    Keys are stream indices.
    """

    must, partners = pinch_streams(side)
    must_counts = {stream.index: 1 for stream in must}
    partner_counts = {stream.index: 1 for stream in partners}
    if not partners:
        return must_counts, partner_counts
    for stream in must:
        if not any(cp_rule_met(stream, partner) for partner in partners):
            must_counts[stream.index] = _branches_covering(stream, partners)
    for _round in range(_MAX_SPLIT_ROUNDS):
        if sum(must_counts.values()) <= sum(partner_counts.values()):
            break
        largest = max(
            partners,
            key=lambda stream: stream.pinch_cp / partner_counts[stream.index],
        )
        partner_counts[largest.index] += 1
    return must_counts, partner_counts


def _branches_covering(stream: SideStream, partners) -> int:
    """Fewest branches (two or more) whose largest partner CPs cover the CP."""

    covered = 0.0
    count = 0
    for cp in sorted((partner.pinch_cp for partner in partners), reverse=True):
        covered += cp
        count += 1
        if count >= 2 and stream.pinch_cp <= covered * (1.0 + CP_TOLERANCE):
            break
    return max(2, count)


def cp_feasible(side: PinchSide, edges: Iterable[tuple[int, int]]) -> bool:
    """Whether split ratios exist that meet the CP rule on every pinch match.

    ``edges`` are ``(must index, partner index)`` pinch matches. A stream with
    several matches is split across them; the rule holds when, in every period,
    each must stream's CP can be shared over its matches without any partner
    receiving more than its own CP (a transport problem solved by max flow).
    Must streams with no match at the pinch in a period are not checked there.
    """

    must, partners = pinch_streams(side)
    must_by_index = {stream.index: stream for stream in must}
    partner_by_index = {stream.index: stream for stream in partners}
    edges = [(m, p) for m, p in edges if m in must_by_index and p in partner_by_index]
    for n in range(side.periods):
        period_edges = [
            (m, p)
            for m, p in edges
            if must_by_index[m].at_pinch[n] and partner_by_index[p].at_pinch[n]
        ]
        supplies = {m: must_by_index[m].pinch_cps[n] for m, _p in period_edges}
        capacities = {
            p: partner_by_index[p].pinch_cps[n] * (1.0 + CP_TOLERANCE)
            for _m, p in period_edges
        }
        demand = sum(supplies.values())
        if demand <= 0.0:
            continue
        flow = _max_flow(supplies, capacities, period_edges)
        if flow < demand * (1.0 - CP_TOLERANCE):
            return False
    return True


def _max_flow(supplies, capacities, edges) -> float:
    """Max flow source -> must -> partner -> sink, edges uncapacitated."""

    source, sink = ("s",), ("t",)
    residual: dict = {}

    def add(u, v, capacity):
        residual.setdefault(u, {}).setdefault(v, 0.0)
        residual.setdefault(v, {}).setdefault(u, 0.0)
        residual[u][v] += capacity

    for m, supply in supplies.items():
        add(source, ("m", m), supply)
    for p, capacity in capacities.items():
        add(("p", p), sink, capacity)
    unbounded = sum(supplies.values()) + 1.0
    for m, p in edges:
        add(("m", m), ("p", p), unbounded)

    total = 0.0
    while True:
        parents = {source: None}
        queue = deque([source])
        while queue and sink not in parents:
            u = queue.popleft()
            for v, capacity in residual.get(u, {}).items():
                if capacity > 1e-12 and v not in parents:
                    parents[v] = u
                    queue.append(v)
        if sink not in parents:
            return total
        path, v = [], sink
        while parents[v] is not None:
            path.append((parents[v], v))
            v = parents[v]
        push = min(residual[u][v] for u, v in path)
        for u, v in path:
            residual[u][v] -= push
            residual[v][u] += push
        total += push


__all__ = [
    "CP_TOLERANCE",
    "cp_feasible",
    "cp_rule_met",
    "number_rule_met",
    "pinch_streams",
    "split_counts",
]
