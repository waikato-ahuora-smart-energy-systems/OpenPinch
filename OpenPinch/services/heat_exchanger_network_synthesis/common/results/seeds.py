"""Seed-network resolution for standalone HEN synthesis methods."""

from __future__ import annotations

from collections.abc import Sequence

from .....classes.heat_exchanger_network import HeatExchangerNetwork
from .....classes.pinch_problem import PinchProblem


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
    return (cached_design.network,)


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
