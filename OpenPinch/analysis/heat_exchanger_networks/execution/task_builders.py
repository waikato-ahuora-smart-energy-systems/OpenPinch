"""Shared task-builder utilities for HEN synthesis methods."""

from __future__ import annotations

from typing import Sequence

from ....contracts.synthesis.common import SynthesisMethod
from ....contracts.synthesis.task import HeatExchangerNetworkSynthesisTaskOutcome
from ....contracts.synthesis.topology import HeatExchangerNetworkTopologyRestriction
from ....domain.enums import HeatExchangerKind
from ....domain.heat_exchanger_network import HeatExchangerNetwork
from ..errors import WorkflowContractError
from .settings import SynthesisWorkflowSettings


def required_topology_restrictions_from_outcome(
    outcome: HeatExchangerNetworkSynthesisTaskOutcome,
    downstream_method: SynthesisMethod,
) -> tuple[HeatExchangerNetworkTopologyRestriction, ...]:
    """Return downstream topology restrictions or fail the workflow contract."""
    if outcome.network is None:
        raise WorkflowContractError(
            f"Successful {outcome.task.method} task {outcome.task.task_id} cannot "
            f"spawn {downstream_method} tasks without a HeatExchangerNetwork."
        )

    return topology_restrictions_from_network(
        outcome.network,
        downstream_method=downstream_method,
        source_method=outcome.task.method,
        source_task_id=outcome.task.task_id,
    )


def topology_restrictions_from_network(
    network: HeatExchangerNetwork,
    *,
    downstream_method: SynthesisMethod,
    source_method: SynthesisMethod | str = "seed_network",
    source_task_id: str | None = None,
) -> tuple[HeatExchangerNetworkTopologyRestriction, ...]:
    """Return topology restrictions from an existing network seed."""
    restrictions = tuple(
        HeatExchangerNetworkTopologyRestriction(
            source_stream=exchanger.source_stream,
            sink_stream=exchanger.sink_stream,
            stage=exchanger.stage,
            duty=max(state.duty for state in exchanger.period_states),
        )
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.RECOVERY
        and any(state.active for state in exchanger.period_states)
        and exchanger.match_allowed
        and exchanger.stage is not None
    )
    if not restrictions:
        source = (
            f"Successful {source_method} task {source_task_id}"
            if source_task_id is not None
            else str(source_method)
        )
        raise WorkflowContractError(
            f"{source} cannot spawn {downstream_method} tasks without recovery "
            "topology restrictions."
        )
    return restrictions


def stage_count_from_network(
    network: HeatExchangerNetwork,
    *,
    downstream_method: SynthesisMethod,
) -> int:
    """Return a stage count from network metadata or active recovery stages."""
    if network.stage_count is not None:
        return int(network.stage_count)
    stages = [
        int(exchanger.stage)
        for exchanger in network.exchangers
        if exchanger.kind is HeatExchangerKind.RECOVERY
        and any(state.active for state in exchanger.period_states)
        and exchanger.match_allowed
        and exchanger.stage is not None
    ]
    if not stages:
        raise WorkflowContractError(
            f"Seed network cannot spawn {downstream_method} tasks without a "
            "stage_count or staged active recovery exchangers."
        )
    return max(stages)


def approach_temperature_from_network(
    network: HeatExchangerNetwork,
    settings: SynthesisWorkflowSettings,
) -> float:
    """Return stable approach-temperature metadata for a seeded method task."""
    value = network.summary_metrics.get("approach_temperature")
    if isinstance(value, int | float) and value > 0.0:
        return float(value)
    value = network.source_metadata.get("solver_dTmin")
    if isinstance(value, int | float) and value > 0.0:
        return float(value)
    for exchanger in network.exchangers:
        for state in exchanger.period_states:
            for approach_temperature in state.approach_temperatures:
                if approach_temperature > 0.0:
                    return float(approach_temperature)
    return float(settings.approach_temperatures[0])


def derivative_threshold_from_network(network: HeatExchangerNetwork) -> float | None:
    """Return positive derivative-threshold metadata from a seed network."""
    value = network.summary_metrics.get("derivative_threshold")
    if isinstance(value, int | float) and value > 0.0:
        return float(value)
    return None


def _successful_method(
    outcome: HeatExchangerNetworkSynthesisTaskOutcome,
    method: SynthesisMethod,
) -> bool:
    return outcome.status == "success" and outcome.task.method == method


def _required_stage_count(
    outcome: HeatExchangerNetworkSynthesisTaskOutcome,
    downstream_method: SynthesisMethod,
) -> int:
    stage_count = (
        outcome.network.stage_count
        if outcome.network is not None
        else outcome.task.stage_count
    )
    if stage_count is None:
        raise WorkflowContractError(
            f"Successful {outcome.task.method} task {outcome.task.task_id} cannot "
            f"spawn {downstream_method} tasks without a stage count."
        )
    return int(stage_count)


def _outcome_map(
    outcomes: Sequence[HeatExchangerNetworkSynthesisTaskOutcome],
) -> dict[str, HeatExchangerNetworkSynthesisTaskOutcome]:
    return {
        outcome.task.task_id: outcome
        for outcome in outcomes
        if outcome.task.task_id is not None
    }


__all__ = [
    "approach_temperature_from_network",
    "derivative_threshold_from_network",
    "required_topology_restrictions_from_outcome",
    "stage_count_from_network",
    "topology_restrictions_from_network",
]
