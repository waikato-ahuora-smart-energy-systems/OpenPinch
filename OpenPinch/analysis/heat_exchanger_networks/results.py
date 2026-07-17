"""Ranking and selection operations for HEN synthesis results."""

from __future__ import annotations

from ...contracts.synthesis.result import HeatExchangerNetworkSynthesisResult


def ranked_networks(
    result: HeatExchangerNetworkSynthesisResult,
    n: int | None = None,
):
    """Return unique ranked network outcomes from one synthesis result."""
    from ...services.heat_exchanger_network_synthesis.common.reporting.ranking import (
        rank_unique_network_outcomes,
    )

    return rank_unique_network_outcomes(result, limit=n)


def network_for_rank(
    result: HeatExchangerNetworkSynthesisResult,
    solution_rank: int = 1,
):
    """Return a network by one-based rank without mutating the result."""
    if solution_rank < 1:
        raise IndexError("solution_rank is 1-based and must be at least 1")
    ranked = ranked_networks(result)
    if not ranked:
        if solution_rank == 1:
            return result.network
        raise IndexError("solution_rank 2 is unavailable; only 1 network is available")
    if solution_rank > len(ranked):
        raise IndexError(
            f"solution_rank {solution_rank} is unavailable; only "
            f"{len(ranked)} network(s) are available"
        )
    selected = ranked[solution_rank - 1]
    if selected.network is None:
        raise ValueError("selected ranked network outcome is missing network output")
    return selected.network


def select_network(
    result: HeatExchangerNetworkSynthesisResult,
    solution_rank: int = 1,
) -> HeatExchangerNetworkSynthesisResult:
    """Select one ranked outcome into the mutable synthesis result."""
    ranked = ranked_networks(result)
    network = network_for_rank(result, solution_rank)
    if not ranked:
        return result

    selected = ranked[solution_rank - 1]
    result.ranked_networks = ranked
    result.network = network
    result.task_id = selected.task.task_id
    result.solver_status = selected.solver_status
    result.method = selected.task.method
    result.stage_count = network.stage_count or selected.task.stage_count
    result.objective_values = {
        key: value
        for key, value in {
            "total_annual_cost": network.total_annual_cost,
            "utility_cost": network.utility_cost,
            "capital_cost": network.capital_cost,
        }.items()
        if value is not None
    }
    return result


__all__ = ["network_for_rank", "ranked_networks", "select_network"]
