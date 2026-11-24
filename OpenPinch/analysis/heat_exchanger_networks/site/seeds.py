"""Seeds for a site network from each process's pinch design networks.

A site network serves several processes. Each process is designed alone,
holding its utility levels at their targets; a seed then takes one network
per process at a dt_cont multiplier, joins their recovery matches and staged
heaters and coolers (which hold a stream's levels between its matches), and
records each process's level duties (``source_metadata["site_processes"]``)
so the evolution that refines it holds them and matches streams only within
a process (``models/_stagewise/level_targets.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ....domain.enums import HeatExchangerKind
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ..reporting.ranking import holds_utility_levels


def process_level_duties(network: HeatExchangerNetwork) -> dict[str, list[float]]:
    """The duty per period of each utility a process's network holds.

    Its level targets when it holds them; otherwise (a network without
    targets, or one that fell back to levels chosen by price) its own
    heater and cooler duties, by utility name.
    """

    targets = (network.source_metadata or {}).get("utility_level_targets")
    if targets and holds_utility_levels(network):
        return {
            name: [float(duty) for duty in duties] for name, duties in targets.items()
        }
    periods = len(network.exchangers[0].period_states) if network.exchangers else 1
    duties: dict[str, list[float]] = {}
    for exchanger in network.exchangers:
        if exchanger.kind is HeatExchangerKind.HOT_UTILITY:
            name = exchanger.source_stream
        elif exchanger.kind is HeatExchangerKind.COLD_UTILITY:
            name = exchanger.sink_stream
        else:
            continue
        by_period = duties.setdefault(name, [0.0] * periods)
        for state in exchanger.period_states:
            if state.active:
                by_period[state.period_idx] += float(state.duty)
    return duties


def site_seeds(
    pools: Mapping[str, Sequence[HeatExchangerNetwork]],
    *,
    level_held: Mapping[str, Sequence[HeatExchangerNetwork]] | None = None,
) -> tuple[HeatExchangerNetwork, ...]:
    """Seeds per dt_cont multiplier, in ascending order of multiplier.

    ``pools`` maps each process (its stream-key prefix) to the networks of
    its pinch design with levels chosen by price; ``level_held`` to those of
    its pinch design holding its utility level targets. A process with no
    network raises ``ValueError``. Seeds are made at each multiplier every
    process has a network at or above (all of them when there is none).
    At each multiplier (or, for a process without one, its nearest above)
    a ``"lowest_cost"`` seed takes each
    process's network with the fewest units, then the lowest total annual
    cost, from ``pools``; a ``"level_held"`` seed takes the same from the
    networks of ``level_held`` that hold their targets, and a process
    without one there at or above the multiplier takes its lowest-cost
    network. ``summary_metrics``
    record the kind as ``"site_seed"``; a seed equal to another is dropped.
    """

    for process, pool in pools.items():
        if not pool:
            raise ValueError(
                f"process {process!r} has no network to seed the site network; "
                "its pinch design found none."
            )
    level_held = level_held or {}
    held = {
        process: tuple(network for network in pool if _holds_targets(network))
        for process, pool in level_held.items()
    }
    multipliers = sorted(
        {
            _multiplier(network)
            for family in (pools, held)
            for pool in family.values()
            for network in pool
        }
    )
    # A network designed at a smaller multiplier can need more recovery
    # than a larger one's approach allows: no seed where a process has
    # nothing at or above its multiplier (unless that leaves none).
    covered = [m for m in multipliers if all(_covers(p, m) for p in pools.values())]
    seeds: list[HeatExchangerNetwork] = []
    seen: set[tuple] = set()
    for m in covered or multipliers:
        cheapest = {process: _fewest_units(pool, m) for process, pool in pools.items()}
        kinds = [("lowest_cost", cheapest)]
        if any(held.values()):
            kinds.insert(
                0,
                (
                    "level_held",
                    {
                        process: _fewest_units(held[process], m)
                        if _covers(held.get(process, ()), m)
                        else network
                        for process, network in cheapest.items()
                    },
                ),
            )
        for kind, networks in kinds:
            seed = _seed(networks, m, kind)
            key = (
                tuple(sorted(e.exchanger_id for e in seed.exchangers)),
                repr(seed.source_metadata["site_processes"]),
            )
            if key not in seen:
                seen.add(key)
                seeds.append(seed)
    if not any(_recovery(seed) for seed in seeds):
        raise ValueError(
            "no process network recovers heat, so there is no site network to "
            "evolve; design each process with heat_exchanger_network(zone=...)."
        )
    return tuple(seed for seed in seeds if _recovery(seed))


def _holds_targets(network: HeatExchangerNetwork) -> bool:
    targets = (network.source_metadata or {}).get("utility_level_targets")
    return bool(targets) and holds_utility_levels(network)


def _covers(pool: Sequence[HeatExchangerNetwork], m: float) -> bool:
    """Whether ``pool`` has a network at the multiplier ``m`` or above."""
    return any(_multiplier(network) >= m for network in pool)


def _fewest_units(
    pool: Sequence[HeatExchangerNetwork], m: float
) -> HeatExchangerNetwork:
    """At the multiplier nearest ``m`` at or above it (or nearest at all when
    there is none above): the network with the fewest units, then the
    lowest total annual cost (the first on a tie)."""
    candidates = [network for network in pool if _multiplier(network) >= m] or list(
        pool
    )
    distance = min(abs(_multiplier(network) - m) for network in candidates)
    nearest = [
        network for network in candidates if abs(_multiplier(network) - m) == distance
    ]
    return min(nearest, key=lambda network: (_units(network), _cost(network)))


def _units(network: HeatExchangerNetwork) -> int:
    units = (network.summary_metrics or {}).get("total_units")
    if isinstance(units, int | float) and not isinstance(units, bool):
        return int(units)
    return sum(
        1
        for exchanger in network.exchangers
        if any(state.active for state in exchanger.period_states)
    )


def _cost(network: HeatExchangerNetwork) -> float:
    tac = network.total_annual_cost
    return float("inf") if tac is None else float(tac)


def _seed(
    networks: Mapping[str, HeatExchangerNetwork], m: float, kind: str
) -> HeatExchangerNetwork:
    recovery = [
        exchanger for network in networks.values() for exchanger in _recovery(network)
    ]
    staged = [
        exchanger for network in networks.values() for exchanger in _staged(network)
    ]
    stages = [
        network.stage_count
        if network.stage_count is not None
        else max((exchanger.stage for exchanger in _recovery(network)), default=0)
        for network in networks.values()
    ]
    return HeatExchangerNetwork(
        exchangers=tuple(recovery + staged),
        run_id="site-seed",
        method="site_pinch_design",
        stage_count=max(stages, default=None) or None,
        summary_metrics={"dt_cont_multiplier": m, "site_seed": kind},
        source_metadata={
            "site_processes": {
                process: process_level_duties(network)
                for process, network in networks.items()
            }
        },
    )


def _recovery(network: HeatExchangerNetwork) -> list:
    return [
        exchanger
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.RECOVERY
        and exchanger.stage is not None
        and any(state.active for state in exchanger.period_states)
    ]


def _staged(network: HeatExchangerNetwork) -> list:
    """The network's heaters and coolers between stages."""
    return [
        exchanger
        for exchanger in network.exchangers
        if exchanger.kind is not HeatExchangerKind.RECOVERY
        and exchanger.stage is not None
        and any(state.active for state in exchanger.period_states)
    ]


def _multiplier(network: HeatExchangerNetwork) -> float:
    for metadata in (network.summary_metrics, network.source_metadata):
        value = (metadata or {}).get("dt_cont_multiplier")
        if isinstance(value, int | float):
            return float(value)
    return 1.0


__all__ = ["process_level_duties", "site_seeds"]
