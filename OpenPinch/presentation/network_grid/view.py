"""The design view of a HEN synthesis result as plain data."""

from __future__ import annotations

from ...analysis.heat_exchanger_networks.results.selection import (
    network_for_rank,
    ranked_networks,
)
from ...contracts.synthesis.result import HeatExchangerNetworkSynthesisResult
from ...contracts.synthesis.view import (
    HeatExchangerNetworkRankingRow,
    HeatExchangerNetworkUtilityDuty,
    HeatExchangerNetworkView,
)
from ...domain.enums import HeatExchangerKind
from ...domain.heat_exchanger_network import HeatExchangerNetwork
from .layout import build_grid_layout


def build_network_view(
    result: HeatExchangerNetworkSynthesisResult,
    *,
    rank: int = 1,
    period_id: str | None = None,
) -> HeatExchangerNetworkView:
    """Return the ranking, one ranked network's totals, duties and grid."""
    network = network_for_rank(result, rank)
    resolved_period_id = network.resolve_period_id(period_id)
    recovery, hot, cold = _totals(network, resolved_period_id)
    return HeatExchangerNetworkView(
        rank=rank,
        period_id=resolved_period_id,
        ranking=tuple(
            _ranking_row(index, outcome, resolved_period_id)
            for index, outcome in enumerate(ranked_networks(result), start=1)
        ),
        total_heat_recovery=recovery,
        total_hot_utility=hot,
        total_cold_utility=cold,
        unit_count=len(network.exchangers),
        utility_duties=_utility_duties(network, resolved_period_id),
        grid=build_grid_layout(network, period_id=resolved_period_id),
    )


def _ranking_row(rank, outcome, period_id) -> HeatExchangerNetworkRankingRow:
    network = outcome.network
    recovery, hot, cold = _totals(network, network.resolve_period_id(period_id))
    return HeatExchangerNetworkRankingRow(
        rank=rank,
        method=str(getattr(outcome.method, "value", outcome.method)),
        status=str(outcome.status),
        objective_value=outcome.objective_value,
        total_annual_cost=network.total_annual_cost,
        unit_count=len(network.exchangers),
        total_heat_recovery=recovery,
        total_hot_utility=hot,
        total_cold_utility=cold,
    )


def _totals(
    network: HeatExchangerNetwork, period_id: str | None
) -> tuple[float, float, float]:
    return tuple(
        float(network.total_duty(kind=kind, period_id=period_id))
        for kind in (
            HeatExchangerKind.RECOVERY,
            HeatExchangerKind.HOT_UTILITY,
            HeatExchangerKind.COLD_UTILITY,
        )
    )


def _utility_duties(
    network: HeatExchangerNetwork, period_id: str | None
) -> tuple[HeatExchangerNetworkUtilityDuty, ...]:
    duties: dict[tuple[str, str], float] = {}
    for kind, side in (
        (HeatExchangerKind.HOT_UTILITY, "hot"),
        (HeatExchangerKind.COLD_UTILITY, "cold"),
    ):
        for exchanger in network.exchangers:
            if exchanger.kind is not kind:
                continue
            name = exchanger.source_stream if side == "hot" else exchanger.sink_stream
            state = exchanger.state(period_id)
            duties.setdefault((name, side), 0.0)
            if state.active:
                duties[(name, side)] += float(state.duty)
    return tuple(
        HeatExchangerNetworkUtilityDuty(name=name, side=side, duty=duty)
        for (name, side), duty in duties.items()
    )


__all__ = ["build_network_view"]
