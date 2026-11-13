"""Seed-network resolution for standalone HEN synthesis methods."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ..prepared_problem import PreparedProblem
from ..reporting.ranking import network_structure_signature
from ..solver.arrays import SEGMENT_PROFILE_VERSION
from .selection import ranked_networks


def resolve_seed_networks(
    problem: PreparedProblem,
    initial_networks: HeatExchangerNetwork | Sequence[HeatExchangerNetwork] | None,
    *,
    method_name: str,
    limit: int,
) -> tuple[HeatExchangerNetwork, ...]:
    """Return explicit seed networks or the cached design's networks.

    The cached design may come from any HEN method. Its selected network
    leads, followed by its other ranked topologies, up to ``limit`` in all.
    """

    if initial_networks is not None:
        return _normalise_seed_networks(initial_networks)

    cached = problem.results
    cached_design = None if cached is None else cached.design
    if cached_design is None:
        raise ValueError(
            f"{method_name} requires initial_networks or an existing cached "
            "design network."
        )
    networks = [cached_design.network]
    seen = {network_structure_signature(cached_design.network)}
    for outcome in ranked_networks(cached_design):
        if len(networks) >= limit:
            break
        signature = network_structure_signature(outcome.network)
        if signature not in seen:
            seen.add(signature)
            networks.append(outcome.network)
    if _problem_has_segments(problem) and any(
        network.source_metadata.get("segment_profile_version")
        != SEGMENT_PROFILE_VERSION
        for network in networks
    ):
        raise ValueError(
            f"{method_name} cannot reuse a cached network that lacks current "
            "segment-profile tensors; rerun the upstream HEN method."
        )
    return tuple(networks)


def _problem_has_segments(problem: PreparedProblem) -> bool:
    zone = problem.master_zone
    return zone is not None and any(stream.has_segments for stream in zone.all_streams)


def _normalise_seed_networks(
    initial_networks: HeatExchangerNetwork | Sequence[HeatExchangerNetwork],
) -> tuple[HeatExchangerNetwork, ...]:
    if isinstance(initial_networks, HeatExchangerNetwork | Mapping):
        # One network, or the JSON of one, rather than a sequence of them.
        networks = (initial_networks,)
    else:
        networks = tuple(initial_networks)
    if not networks:
        raise ValueError("initial_networks must contain at least one network.")
    networks = tuple(
        HeatExchangerNetwork.model_validate(network)
        if isinstance(network, Mapping)
        else network
        for network in networks
    )
    for network in networks:
        if not isinstance(network, HeatExchangerNetwork):
            raise TypeError("initial_networks must be HeatExchangerNetwork instances.")
    return networks


__all__ = ["resolve_seed_networks"]
