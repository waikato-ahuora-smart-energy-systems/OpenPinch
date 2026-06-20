"""Topology builder for heat exchanger network grid diagrams."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from ...classes.heat_exchanger import HeatExchanger
from ...classes.heat_exchanger_network import HeatExchangerNetwork
from ...lib.enums import HeatExchangerKind
from .constants import _DUTY_TOLERANCE_KW
from .models import GridDiagramMatch, HeatExchangerNetworkGridModel


def build_grid_model(
    network: HeatExchangerNetwork,
) -> HeatExchangerNetworkGridModel:
    """Normalize an OpenPinch network into the OpenHENS grid topology."""
    hot_streams: list[str] = []
    cold_streams: list[str] = []
    recovery_by_key: OrderedDict[tuple[str, str, int], GridDiagramMatch] = OrderedDict()
    hot_utility_by_cold_stream: OrderedDict[str, GridDiagramMatch] = OrderedDict()
    cold_utility_by_hot_stream: OrderedDict[str, GridDiagramMatch] = OrderedDict()
    seen_recovery_stages: set[int] = set()

    for exchanger in network.exchangers:
        if not _is_significant_match(exchanger):
            continue
        if exchanger.kind is HeatExchangerKind.RECOVERY:
            _append_unique(hot_streams, exchanger.source_stream)
            _append_unique(cold_streams, exchanger.sink_stream)
            if exchanger.stage is None:
                continue
            seen_recovery_stages.add(exchanger.stage)
            _add_or_accumulate(
                recovery_by_key,
                key=(exchanger.source_stream, exchanger.sink_stream, exchanger.stage),
                exchanger=exchanger,
            )
        elif exchanger.kind is HeatExchangerKind.HOT_UTILITY:
            _append_unique(cold_streams, exchanger.sink_stream)
            _add_or_accumulate(
                hot_utility_by_cold_stream,
                key=exchanger.sink_stream,
                exchanger=exchanger,
            )
        elif exchanger.kind is HeatExchangerKind.COLD_UTILITY:
            _append_unique(hot_streams, exchanger.source_stream)
            _add_or_accumulate(
                cold_utility_by_hot_stream,
                key=exchanger.source_stream,
                exchanger=exchanger,
            )

    if network.stage_count is not None:
        stages = tuple(range(1, network.stage_count + 1))
    else:
        stages = tuple(sorted(seen_recovery_stages)) or (1,)

    hot_index = {stream: index for index, stream in enumerate(hot_streams)}
    cold_index = {stream: index for index, stream in enumerate(cold_streams)}
    stage_index = {stage: index for index, stage in enumerate(stages)}
    recovery_matches = tuple(
        sorted(
            recovery_by_key.values(),
            key=lambda match: (
                stage_index.get(match.stage or 0, len(stages)),
                hot_index[match.source_stream],
                cold_index[match.sink_stream],
            ),
        )
    )
    hot_utility_matches = tuple(
        sorted(
            hot_utility_by_cold_stream.values(),
            key=lambda match: cold_index[match.sink_stream],
        )
    )
    cold_utility_matches = tuple(
        sorted(
            cold_utility_by_hot_stream.values(),
            key=lambda match: hot_index[match.source_stream],
        )
    )

    branch_counts: dict[tuple[str, int], int] = {}
    for stream in hot_streams:
        for stage in stages:
            count = sum(
                1
                for match in recovery_matches
                if match.source_stream == stream and match.stage == stage
            )
            if count > 1:
                branch_counts[(stream, stage)] = count
    for stream in cold_streams:
        for stage in stages:
            count = sum(
                1
                for match in recovery_matches
                if match.sink_stream == stream and match.stage == stage
            )
            if count > 1:
                branch_counts[(stream, stage)] = count

    return HeatExchangerNetworkGridModel(
        network=network,
        hot_streams=tuple(hot_streams),
        cold_streams=tuple(cold_streams),
        stages=stages,
        recovery_matches=recovery_matches,
        hot_utility_matches=hot_utility_matches,
        cold_utility_matches=cold_utility_matches,
        branch_counts=branch_counts,
    )


def _add_or_accumulate(
    matches: OrderedDict[Any, GridDiagramMatch],
    *,
    key: Any,
    exchanger: HeatExchanger,
) -> None:
    if key not in matches:
        matches[key] = _match(exchanger)
        return
    current = matches[key]
    matches[key] = GridDiagramMatch(
        exchanger=current.exchanger,
        source_stream=current.source_stream,
        sink_stream=current.sink_stream,
        stage=current.stage,
        duty=current.duty + exchanger.duty,
    )



def _is_significant_match(exchanger: HeatExchanger) -> bool:
    return (
        exchanger.active
        and exchanger.match_allowed
        and exchanger.duty > _DUTY_TOLERANCE_KW
    )


def _match(exchanger: HeatExchanger) -> GridDiagramMatch:
    return GridDiagramMatch(
        exchanger=exchanger,
        source_stream=exchanger.source_stream,
        sink_stream=exchanger.sink_stream,
        stage=exchanger.stage,
        duty=exchanger.duty,
    )


def _append_unique(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


__all__ = ["build_grid_model"]
