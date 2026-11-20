"""Canonical topology helpers for HEN synthesis task generation."""

from __future__ import annotations

import heapq
from collections.abc import Hashable, Iterable, Sequence
from typing import NamedTuple

from ....contracts.synthesis.topology import HeatExchangerNetworkTopologyRestriction


class StageNode(NamedTuple):
    """One exchanger placed on the stage axis.

    ``streams`` are the process streams it sits on, ``position`` orders it
    along each of them (equal positions on a shared stream are parallel
    branches), and ``order`` breaks ties between independent exchangers.
    """

    streams: frozenset[Hashable]
    position: float
    order: tuple


def canonical_stage_indices(nodes: Sequence[StageNode]) -> tuple[int, ...]:
    """Return a one-based canonical stage for each node.

    Nodes at one position that share a stream form a group, and each group
    takes a stage of its own. Groups follow every stream's order, and
    groups free to go next are placed by ``order``. The stages therefore
    depend only on the order of groups along each stream, not on how
    independent groups were numbered.
    """

    groups = _position_groups(nodes)
    along_stream: dict[Hashable, list[tuple[float, int]]] = {}
    for group, members in enumerate(groups):
        position = nodes[members[0]].position
        for stream in {stream for index in members for stream in nodes[index].streams}:
            along_stream.setdefault(stream, []).append((position, group))
    successors: list[set[int]] = [set() for _ in groups]
    for entries in along_stream.values():
        entries.sort()
        for (_, earlier), (_, later) in zip(entries, entries[1:], strict=False):
            successors[earlier].add(later)
    predecessors = [0] * len(groups)
    for following in successors:
        for group in following:
            predecessors[group] += 1

    keys = [min(nodes[index].order for index in members) for members in groups]
    ready = [
        (keys[group], group) for group in range(len(groups)) if not predecessors[group]
    ]
    heapq.heapify(ready)
    stages = [0] * len(nodes)
    stage = 0
    while ready:
        _, group = heapq.heappop(ready)
        stage += 1
        for index in groups[group]:
            stages[index] = stage
        for following in successors[group]:
            predecessors[following] -= 1
            if not predecessors[following]:
                heapq.heappush(ready, (keys[following], following))
    return tuple(stages)


def canonical_topology_restrictions(
    restrictions: Iterable[HeatExchangerNetworkTopologyRestriction],
    *,
    hot_stream_order: Sequence[str] | None = None,
    cold_stream_order: Sequence[str] | None = None,
) -> tuple[HeatExchangerNetworkTopologyRestriction, ...]:
    """Return restrictions in canonical grid-stage form.

    Matches that share a process hot or cold stream in one original stage
    stay in one canonical stage; every other group of matches takes a stage
    of its own, numbered by ``canonical_stage_indices``. Two numberings that
    give every stream the same match order therefore give the same result.
    Streams missing from the given orders sort by name after them.
    """

    values = tuple(restrictions)
    hot_index = _stream_index(
        hot_stream_order,
        (item.source_stream for item in values),
    )
    cold_index = _stream_index(
        cold_stream_order,
        (item.sink_stream for item in values),
    )
    orders = [_match_order(item, hot_index, cold_index) for item in values]
    stages = canonical_stage_indices(
        [
            StageNode(
                frozenset({("hot", item.source_stream), ("cold", item.sink_stream)}),
                item.stage,
                order,
            )
            for item, order in zip(values, orders, strict=True)
        ]
    )
    placed = sorted(
        zip(stages, orders, range(len(values)), strict=True),
    )
    return tuple(
        values[index].model_copy(update={"stage": stage}) for stage, _, index in placed
    )


def canonical_stage_count(
    restrictions: Iterable[HeatExchangerNetworkTopologyRestriction],
) -> int:
    """Return the stage count implied by canonical restrictions."""

    values = tuple(restrictions)
    if not values:
        raise ValueError("canonical topology restrictions require at least one match")
    return max(item.stage for item in values)


def topology_restriction_signature(
    restrictions: Iterable[HeatExchangerNetworkTopologyRestriction],
) -> tuple[tuple[str, str, int], ...]:
    """Return a duty-independent topology signature."""

    return tuple(
        sorted(
            (
                item.source_stream,
                item.sink_stream,
                item.stage,
            )
            for item in restrictions
        )
    )


def _position_groups(nodes: Sequence[StageNode]) -> list[list[int]]:
    """Indices of nodes joined by a shared stream at one position."""

    parent = list(range(len(nodes)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    first_at: dict[tuple[float, Hashable], int] = {}
    for index, node in enumerate(nodes):
        for stream in node.streams:
            other = first_at.setdefault((node.position, stream), index)
            parent[root(index)] = root(other)
    groups: dict[int, list[int]] = {}
    for index in range(len(nodes)):
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())


def _stream_index(
    preferred_order: Sequence[str] | None,
    fallback_order: Iterable[str],
) -> dict[str, int]:
    ordered: list[str] = []
    if preferred_order is not None:
        ordered.extend(str(item) for item in preferred_order)
    ordered.extend(sorted(str(item) for item in fallback_order))
    return {stream: index for index, stream in enumerate(dict.fromkeys(ordered))}


def _match_order(
    match: HeatExchangerNetworkTopologyRestriction,
    hot_index: dict[str, int],
    cold_index: dict[str, int],
) -> tuple[int, int, str, str]:
    return (
        hot_index.get(match.source_stream, len(hot_index)),
        cold_index.get(match.sink_stream, len(cold_index)),
        match.source_stream,
        match.sink_stream,
    )


__all__ = [
    "StageNode",
    "canonical_stage_count",
    "canonical_stage_indices",
    "canonical_topology_restrictions",
    "topology_restriction_signature",
]
