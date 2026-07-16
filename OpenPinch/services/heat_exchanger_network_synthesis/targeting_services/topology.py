"""Canonical topology helpers for HEN synthesis task generation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ....lib.schemas.synthesis.topology import HeatExchangerNetworkTopologyRestriction


def canonical_topology_restrictions(
    restrictions: Iterable[HeatExchangerNetworkTopologyRestriction],
    *,
    hot_stream_order: Sequence[str] | None = None,
    cold_stream_order: Sequence[str] | None = None,
) -> tuple[HeatExchangerNetworkTopologyRestriction, ...]:
    """Return restrictions in canonical grid-stage form.

    Empty recovery-stage gaps are removed. Within each occupied original stage,
    independent match groups are split into consecutive stages, while matches
    that share a process hot or cold stream stay in the same canonical stage.
    """

    values = tuple(restrictions)
    if not values:
        return ()

    hot_index = _stream_index(
        hot_stream_order,
        (item.source_stream for item in values),
    )
    cold_index = _stream_index(
        cold_stream_order,
        (item.sink_stream for item in values),
    )

    canonical: list[HeatExchangerNetworkTopologyRestriction] = []
    next_stage = 1
    for stage in sorted({item.stage for item in values}):
        stage_items = [item for item in values if item.stage == stage]
        components = _connected_match_components(
            stage_items,
            hot_index=hot_index,
            cold_index=cold_index,
        )
        for component in components:
            for item in sorted(
                component,
                key=lambda match: _match_order(match, hot_index, cold_index),
            ):
                canonical.append(item.model_copy(update={"stage": next_stage}))
            next_stage += 1
    return tuple(canonical)


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


def _connected_match_components(
    matches: Sequence[HeatExchangerNetworkTopologyRestriction],
    *,
    hot_index: dict[str, int],
    cold_index: dict[str, int],
) -> tuple[tuple[HeatExchangerNetworkTopologyRestriction, ...], ...]:
    unvisited = set(range(len(matches)))
    components: list[tuple[HeatExchangerNetworkTopologyRestriction, ...]] = []
    while unvisited:
        root = min(unvisited)
        stack = [root]
        unvisited.remove(root)
        component_indices = []
        while stack:
            index = stack.pop()
            component_indices.append(index)
            match = matches[index]
            neighbours = [
                other_index
                for other_index in tuple(unvisited)
                if _matches_conflict(match, matches[other_index])
            ]
            for other_index in neighbours:
                unvisited.remove(other_index)
                stack.append(other_index)

        component = tuple(matches[index] for index in component_indices)
        components.append(component)

    return tuple(
        sorted(
            components,
            key=lambda component: min(
                _match_order(match, hot_index, cold_index) for match in component
            ),
        )
    )


def _matches_conflict(
    left: HeatExchangerNetworkTopologyRestriction,
    right: HeatExchangerNetworkTopologyRestriction,
) -> bool:
    return (
        left.source_stream == right.source_stream
        or left.sink_stream == right.sink_stream
    )


def _stream_index(
    preferred_order: Sequence[str] | None,
    fallback_order: Iterable[str],
) -> dict[str, int]:
    ordered: list[str] = []
    if preferred_order is not None:
        ordered.extend(str(item) for item in preferred_order)
    ordered.extend(str(item) for item in fallback_order)
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
    "canonical_stage_count",
    "canonical_topology_restrictions",
    "topology_restriction_signature",
]
