"""Seeds for a site network from each process's pinch design networks.

A site network serves several processes. Each process is designed alone,
holding its utility levels at their targets; a seed then takes one network
per process at a dt_cont multiplier, joins their recovery matches, and
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
) -> tuple[HeatExchangerNetwork, ...]:
    """One seed per dt_cont multiplier, in ascending order.

    ``pools`` maps each process (its stream-key prefix) to its networks,
    best first. A seed takes each process's best network at the multiplier,
    or at the nearest one the process has.
    """

    multipliers = sorted(
        {_multiplier(network) for pool in pools.values() for network in pool}
    )
    seeds = tuple(
        _seed({process: _nearest(pool, m) for process, pool in pools.items()}, m)
        for m in multipliers
    )
    if not any(_recovery(seed) for seed in seeds):
        raise ValueError(
            "no process network recovers heat, so there is no site network to "
            "evolve; design each process with heat_exchanger_network(zone=...)."
        )
    return tuple(seed for seed in seeds if _recovery(seed))


def _seed(
    networks: Mapping[str, HeatExchangerNetwork], m: float
) -> HeatExchangerNetwork:
    recovery = [
        exchanger for network in networks.values() for exchanger in _recovery(network)
    ]
    stages = [
        network.stage_count
        if network.stage_count is not None
        else max((exchanger.stage for exchanger in _recovery(network)), default=0)
        for network in networks.values()
    ]
    return HeatExchangerNetwork(
        exchangers=tuple(recovery),
        run_id="site-seed",
        method="site_pinch_design",
        stage_count=max(stages, default=None) or None,
        summary_metrics={"dt_cont_multiplier": m},
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


def _multiplier(network: HeatExchangerNetwork) -> float:
    for metadata in (network.summary_metrics, network.source_metadata):
        value = (metadata or {}).get("dt_cont_multiplier")
        if isinstance(value, int | float):
            return float(value)
    return 1.0


def _nearest(pool: Sequence[HeatExchangerNetwork], m: float) -> HeatExchangerNetwork:
    """The first (best) network at the multiplier nearest ``m``."""
    distance = min(abs(_multiplier(network) - m) for network in pool)
    return next(
        network for network in pool if abs(_multiplier(network) - m) == distance
    )


__all__ = ["process_level_duties", "site_seeds"]
