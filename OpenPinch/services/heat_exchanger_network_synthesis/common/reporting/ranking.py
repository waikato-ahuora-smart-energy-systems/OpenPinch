"""Ranking helpers for heat exchanger network synthesis outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .....classes.heat_exchanger_network import HeatExchangerNetwork
from .....lib.schemas.synthesis import HeatExchangerNetworkTopologyRestriction
from ...targeting_services.topology import (
    canonical_topology_restrictions,
    topology_restriction_signature,
)
from .verification import is_network_feasible

if TYPE_CHECKING:
    from .....lib.schemas.synthesis import (
        HeatExchangerNetworkSynthesisResult,
        HeatExchangerNetworkSynthesisTaskOutcome,
    )


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
    accepted_method = [
        outcome for outcome in candidates if outcome.task.method == result.method
    ]
    return tuple(
        sorted(
            accepted_method or candidates,
            key=lambda outcome: (
                outcome.objective_value,
                outcome.task.task_id or "",
            ),
        )
    )


__all__ = [
    "network_structure_signature",
    "rank_unique_network_outcomes",
]
