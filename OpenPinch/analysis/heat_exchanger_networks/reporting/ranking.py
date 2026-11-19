"""Ranking helpers for heat exchanger network synthesis outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ....contracts.synthesis.topology import HeatExchangerNetworkTopologyRestriction
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ..targeting.topology import (
    canonical_topology_restrictions,
    topology_restriction_signature,
)
from .verification import is_network_feasible

if TYPE_CHECKING:
    from ....contracts.synthesis.result import HeatExchangerNetworkSynthesisResult
from ....contracts.synthesis.task import HeatExchangerNetworkSynthesisTaskOutcome


def rank_unique_network_outcomes(
    result: "HeatExchangerNetworkSynthesisResult",
    *,
    limit: int | None = None,
) -> tuple["HeatExchangerNetworkSynthesisTaskOutcome", ...]:
    """Return ranked successful outcomes with duplicate structures removed."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when supplied")

    unique_outcomes: list[HeatExchangerNetworkSynthesisTaskOutcome] = []
    seen_structures: set[tuple[tuple[Any, ...], ...]] = set()
    for outcome in _ranked_candidate_outcomes(result):
        structure = network_structure_signature(outcome.network)
        if structure in seen_structures:
            continue
        seen_structures.add(structure)
        unique_outcomes.append(outcome)
        if limit is not None and len(unique_outcomes) >= limit:
            break

    return tuple(unique_outcomes)


def network_structure_signature(
    network: HeatExchangerNetwork,
) -> tuple[tuple[Any, ...], ...]:
    """Return the stable topology signature used by grid-diagram plotting."""
    recovery_restrictions = canonical_topology_restrictions(
        (
            HeatExchangerNetworkTopologyRestriction(
                source_stream=exchanger.source_stream,
                sink_stream=exchanger.sink_stream,
                stage=int(exchanger.stage),
                duty=max(state.duty for state in exchanger.period_states),
            )
            for exchanger in network.exchangers
            if exchanger.kind.value == "recovery"
            and exchanger.stage is not None
            and exchanger.match_allowed
            and any(state.active for state in exchanger.period_states)
        ),
        hot_stream_order=tuple(
            dict.fromkeys(
                exchanger.source_stream
                for exchanger in network.exchangers
                if exchanger.kind.value == "recovery"
            )
        ),
        cold_stream_order=tuple(
            dict.fromkeys(
                exchanger.sink_stream
                for exchanger in network.exchangers
                if exchanger.kind.value == "recovery"
            )
        ),
    )
    links = [
        (
            "recovery",
            source_stream,
            sink_stream,
            stage,
        )
        for source_stream, sink_stream, stage in topology_restriction_signature(
            recovery_restrictions
        )
    ]
    links.extend(
        ("hot_utility", exchanger.source_stream, exchanger.sink_stream, None)
        for exchanger in network.exchangers
        if exchanger.kind.value == "hot_utility"
        and any(state.active for state in exchanger.period_states)
    )
    links.extend(
        ("cold_utility", exchanger.source_stream, exchanger.sink_stream, None)
        for exchanger in network.exchangers
        if exchanger.kind.value == "cold_utility"
        and any(state.active for state in exchanger.period_states)
    )
    return tuple(sorted(links))


# A utility's duty within this fraction of its level target (or this many
# kW of a small one) holds the level: the heuristic counts a load within
# 0.1% as met.
LEVEL_TOLERANCE = 1e-3


def holds_utility_levels(network: HeatExchangerNetwork) -> bool:
    """Whether each utility's duty meets its level target in every period.

    The targets are ``source_metadata["utility_level_targets"]`` (utility
    name to per-period kW), recorded by a design that holds the levels and
    by the tasks built on it. A network without them holds no level to miss.
    """
    targets = (network.source_metadata or {}).get("utility_level_targets")
    if not targets:
        return True
    duties: dict[str, dict[int, float]] = {}
    for exchanger in network.exchangers:
        if exchanger.kind.value == "hot_utility":
            name = exchanger.source_stream
        elif exchanger.kind.value == "cold_utility":
            name = exchanger.sink_stream
        else:
            continue
        for state in exchanger.period_states:
            by_period = duties.setdefault(name, {})
            by_period[state.period_idx] = by_period.get(state.period_idx, 0.0) + (
                float(state.duty) if state.active else 0.0
            )
    for name, by_period in targets.items():
        for period, target in enumerate(by_period):
            duty = duties.get(name, {}).get(period, 0.0)
            if abs(duty - float(target)) > LEVEL_TOLERANCE * max(1.0, abs(target)):
                return False
    return True


# A site network's total annual cost with site utility accounting
# (``../site/costing.py``): steam raised on a main is credited, not charged.
SITE_TOTAL_ANNUAL_COST = "site_total_annual_cost"


def ranking_cost(network: HeatExchangerNetwork | None, fallback):
    """The cost a network ranks by: its site total annual cost when it has
    one, else ``fallback`` (its total annual cost or objective)."""
    value = (network.summary_metrics if network is not None else {}).get(
        SITE_TOTAL_ANNUAL_COST
    )
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return fallback


def outcome_ranking_key(
    outcome: "HeatExchangerNetworkSynthesisTaskOutcome",
) -> tuple[bool, float, str]:
    """Return the one key used both to accept and to rank outcomes.

    Networks that hold their utility levels first (``holds_utility_levels``),
    then lowest objective (a site network's site total annual cost,
    ``ranking_cost``); ties go to the task id, so the choice never depends
    on the order in which tasks finished.
    """
    network = getattr(outcome, "network", None)
    objective = ranking_cost(network, outcome.objective_value)
    held = network is None or holds_utility_levels(network)
    return (
        not held,
        float("inf") if objective is None else float(objective),
        outcome.task.task_id or "",
    )


def _ranked_candidate_outcomes(
    result: "HeatExchangerNetworkSynthesisResult",
) -> tuple["HeatExchangerNetworkSynthesisTaskOutcome", ...]:
    candidates = [
        outcome
        for outcome in result.ranked_networks
        if outcome.status == "success"
        and outcome.network is not None
        and outcome.objective_value is not None
        and is_network_feasible(outcome.network)
    ]
    return tuple(sorted(candidates, key=outcome_ranking_key))


__all__ = [
    "LEVEL_TOLERANCE",
    "holds_utility_levels",
    "network_structure_signature",
    "SITE_TOTAL_ANNUAL_COST",
    "outcome_ranking_key",
    "ranking_cost",
    "rank_unique_network_outcomes",
]
