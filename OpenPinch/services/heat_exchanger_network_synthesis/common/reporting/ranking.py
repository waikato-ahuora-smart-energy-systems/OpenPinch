"""Ranking helpers for heat exchanger network synthesis outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .....classes.heat_exchanger_network import HeatExchangerNetwork
from ....network_grid_diagram.builder import build_grid_model

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
    recovery_stage_index = {
        stage: index
        for index, stage in enumerate(
            dict.fromkeys(
                match.stage
                for match in grid_model.recovery_matches
                if match.stage is not None
            )
        )
    }
    links = [
        (
            "recovery",
            match.source_stream,
            match.sink_stream,
            recovery_stage_index[match.stage],
        )
        for match in grid_model.recovery_matches
        if match.stage is not None
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
