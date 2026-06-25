"""Ranking helpers for heat exchanger network synthesis outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .....classes.heat_exchanger_network import HeatExchangerNetwork
from .....lib.schemas.synthesis import HeatExchangerNetworkTopologyRestriction
from ....network_grid_diagram.builder import build_grid_model
from ...targeting_services.topology import (
    canonical_topology_restrictions,
    topology_restriction_signature,
)

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
    grid_model = build_grid_model(network)
    recovery_restrictions = canonical_topology_restrictions(
        (
            HeatExchangerNetworkTopologyRestriction(
                source_stream=match.source_stream,
                sink_stream=match.sink_stream,
                stage=int(match.stage),
                duty=float(match.duty),
            )
            for match in grid_model.recovery_matches
            if match.stage is not None
        ),
        hot_stream_order=grid_model.hot_streams,
        cold_stream_order=grid_model.cold_streams,
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
        ("hot_utility", match.source_stream, match.sink_stream, None)
        for match in grid_model.hot_utility_matches
    )
    links.extend(
        ("cold_utility", match.source_stream, match.sink_stream, None)
        for match in grid_model.cold_utility_matches
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
