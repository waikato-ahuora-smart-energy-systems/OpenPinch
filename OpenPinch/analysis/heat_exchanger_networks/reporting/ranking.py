"""Ranking helpers for heat exchanger network synthesis outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ..targeting.topology import StageNode, canonical_stage_indices
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
    """Return the stable topology signature used by grid-diagram plotting.

    Recovery matches and staged heaters and coolers are numbered by
    ``canonical_stage_indices``: a staged heater at stage ``s`` sits on its
    cold stream at the boundary before stage ``s``, a staged cooler at
    stage ``s`` on its hot stream at the boundary after it. The signature
    therefore follows each stream's order of exchangers, however stages
    were numbered. A stream-end heater or cooler keeps stage ``None``.
    """
    placed: list[tuple[tuple[str, str, str], StageNode]] = []
    for exchanger in network.exchangers:
        if exchanger.stage is None or not any(
            state.active for state in exchanger.period_states
        ):
            continue
        kind = exchanger.kind.value
        link = (kind, exchanger.source_stream, exchanger.sink_stream)
        stage = int(exchanger.stage)
        if kind == "recovery" and exchanger.match_allowed:
            streams = {("hot", link[1]), ("cold", link[2])}
            node = StageNode(frozenset(streams), stage, (0, *link[1:]))
        elif kind == "hot_utility":
            node = StageNode(
                frozenset({("cold", link[2])}), stage - 0.5, (1, *link[1:])
            )
        elif kind == "cold_utility":
            node = StageNode(frozenset({("hot", link[1])}), stage + 0.5, (2, *link[1:]))
        else:
            continue
        placed.append((link, node))

    stages = canonical_stage_indices([node for _, node in placed])
    links = [(*link, stage) for (link, _), stage in zip(placed, stages, strict=True)]
    links.extend(
        (exchanger.kind.value, exchanger.source_stream, exchanger.sink_stream, None)
        for exchanger in network.exchangers
        if exchanger.stage is None
        and exchanger.kind.value in ("hot_utility", "cold_utility")
        and any(state.active for state in exchanger.period_states)
    )
    # An end heater (stage None) sorts before a staged one on the same pair.
    return tuple(sorted(links, key=lambda link: (*link[:3], _stage_order(link[3]))))


def _stage_order(stage) -> int:
    return -1 if stage is None else int(stage)


# A utility's duty within this fraction of its level target (or this many
# kW of a small one) holds the level: the heuristic counts a load within
# 0.1% as met.
LEVEL_TOLERANCE = 1e-3


def holds_utility_levels(network: HeatExchangerNetwork) -> bool:
    """Whether each utility's duty meets its level target in every period.

    The targets are ``source_metadata["utility_level_targets"]`` (utility
    name to per-period kW), recorded by a design that holds the levels and
    by the tasks built on it. A network without them holds no level to miss.
    A network held at least at its levels (``source_metadata
    ["utility_level_holds"] == "at_least"``) meets a target at or above it.
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
    at_least = network.source_metadata.get("utility_level_holds") == "at_least"
    for name, by_period in targets.items():
        for period, target in enumerate(by_period):
            excess = duties.get(name, {}).get(period, 0.0) - float(target)
            if at_least:
                excess = min(excess, 0.0)
            if abs(excess) > LEVEL_TOLERANCE * max(1.0, abs(target)):
                return False
    return True


# A site network's total annual cost with site utility accounting
# (``../models/_stagewise/site_utilities.py``): the site pays only for its
# net utility use.
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
