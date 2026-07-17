"""Seed-network resolution for standalone HEN synthesis methods."""

from __future__ import annotations

from collections.abc import Sequence

from ....application.problem import PinchProblem
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ..solver.arrays import SEGMENT_PROFILE_VERSION


def resolve_seed_networks(
    problem: PinchProblem,
    initial_networks: HeatExchangerNetwork | Sequence[HeatExchangerNetwork] | None,
    *,
    method_name: str,
    cached_source_method: str,
) -> tuple[HeatExchangerNetwork, ...]:
    """Return explicit seed networks or the expected cached upstream design."""

    if initial_networks is not None:
        return _normalise_seed_networks(initial_networks)

    cached = problem.results
    cached_design = None if cached is None else cached.design
    if cached_design is None:
        raise ValueError(
            f"{method_name} requires initial_networks or an existing cached "
            f"{cached_source_method} design network."
        )
    if cached_design.method != cached_source_method:
        raise ValueError(
            f"{method_name} requires a cached {cached_source_method} design when "
            f"initial_networks are omitted; cached design method is "
            f"{cached_design.method!r}."
        )
    if (
        _problem_has_segments(problem)
        and cached_design.network.source_metadata.get("segment_profile_version")
        != SEGMENT_PROFILE_VERSION
    ):
        raise ValueError(
            f"{method_name} cannot reuse a cached network that lacks current "
            "segment-profile tensors; rerun the upstream HEN method."
        )
    return (cached_design.network,)


def _problem_has_segments(problem: PinchProblem) -> bool:
    zone = problem.master_zone
    return zone is not None and any(stream.has_segments for stream in zone.all_streams)


def _normalise_seed_networks(
    initial_networks: HeatExchangerNetwork | Sequence[HeatExchangerNetwork],
) -> tuple[HeatExchangerNetwork, ...]:
    if isinstance(initial_networks, HeatExchangerNetwork):
        networks = (initial_networks,)
    else:
        networks = tuple(initial_networks)
    if not networks:
        raise ValueError("initial_networks must contain at least one network.")
    for network in networks:
        if not isinstance(network, HeatExchangerNetwork):
            raise TypeError("initial_networks must be HeatExchangerNetwork instances.")
    return networks


__all__ = ["resolve_seed_networks"]
