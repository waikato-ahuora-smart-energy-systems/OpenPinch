"""Ranking helpers for heat exchanger network synthesis outcomes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ....classes.heat_exchanger_network import HeatExchangerNetwork

if TYPE_CHECKING:
    from ....lib.schemas.synthesis import (
        HeatExchangerNetworkSynthesisResult,
        HeatExchangerNetworkSynthesisTaskOutcome,
    )

_STRUCTURE_DUTY_TOLERANCE_KW = 1.0


def rank_unique_network_outcomes(
    result: "HeatExchangerNetworkSynthesisResult",
    *,
    limit: int | None = None,
    duty_tolerance: float = _STRUCTURE_DUTY_TOLERANCE_KW,
) -> tuple["HeatExchangerNetworkSynthesisTaskOutcome", ...]:
    """Return ranked successful outcomes with duplicate structures removed."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1 when supplied")

    unique_outcomes: list[HeatExchangerNetworkSynthesisTaskOutcome] = []
    seen_structures: set[tuple[tuple[Any, ...], ...]] = set()
    for outcome in _ranked_candidate_outcomes(result):
        structure = network_structure_signature(
            outcome.network,
            duty_tolerance=duty_tolerance,
        )
        if structure in seen_structures:
            continue
        seen_structures.add(structure)
        unique_outcomes.append(outcome)
        if limit is not None and len(unique_outcomes) >= limit:
            break

    return tuple(unique_outcomes)


def network_structure_signature(
    network: HeatExchangerNetwork,
    *,
    duty_tolerance: float = _STRUCTURE_DUTY_TOLERANCE_KW,
) -> tuple[tuple[Any, ...], ...]:
    """Return a stable topology signature for active exchanger connections."""
    links = {
        (
            exchanger.kind.value,
            exchanger.source_stream,
            exchanger.sink_stream,
            exchanger.stage,
        )
        for exchanger in network.exchangers
        if exchanger.active
        and exchanger.match_allowed
        and exchanger.duty > duty_tolerance
    }
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
