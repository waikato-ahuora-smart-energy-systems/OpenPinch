"""Steady-state controllability analysis for heat exchanger networks."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from ...classes.heat_exchanger import HeatExchanger
from ...lib.enums import HeatExchangerKind, HeatExchangerStreamRole
from .models import (
    HeatExchangerNetworkControllabilityActuator,
    HeatExchangerNetworkControllabilityComponents,
    HeatExchangerNetworkControllabilityEndpoint,
    HeatExchangerNetworkControllabilityPairing,
    HeatExchangerNetworkControllabilityResult,
)

if TYPE_CHECKING:
    from ...classes.heat_exchanger_network import HeatExchangerNetwork

_COMPOSITE_WEIGHTS = {
    "rank": 0.25,
    "pairing": 0.20,
    "authority": 0.15,
    "conditioning": 0.15,
    "redundancy": 0.15,
    "thermal_margin": 0.10,
}


def quantify_heat_exchanger_network_controllability(
    network: "HeatExchangerNetwork",
    *,
    period_id: str | None = None,
    active_only: bool = True,
    include_utility_actuators: bool = True,
    minimum_interaction: float = 1e-9,
    minimum_approach_temperature: float = 5.0,
    desired_redundancy: int = 2,
    rank_tolerance: float | None = None,
    condition_warning_threshold: float = 25.0,
) -> HeatExchangerNetworkControllabilityResult:
    """Return a 0-1 controllability assessment for a solved HEN.

    The service builds a steady-state interaction matrix from available network
    data. Rows are process-stream outlet temperatures, columns are practical
    manipulated variables: recovery bypass fractions and utility flow rates.
    Entries are duty-normalised thermal authority values, which provide a
    deterministic controllability proxy when no dynamic HEN model is available.
    """

    _validate_options(
        minimum_interaction=minimum_interaction,
        minimum_approach_temperature=minimum_approach_temperature,
        desired_redundancy=desired_redundancy,
        rank_tolerance=rank_tolerance,
        condition_warning_threshold=condition_warning_threshold,
    )

    resolved_period_id = network.resolve_period_id(period_id)
    if resolved_period_id is None:
        raise ValueError("period_id cannot be resolved for an empty-period network")
    exchangers = tuple(
        exchanger
        for exchanger in network.exchangers
        if exchanger.state(resolved_period_id).active or not active_only
    )
    outputs = _build_outputs(exchangers, period_id=resolved_period_id)
    actuators = _build_actuators(
        exchangers,
        period_id=resolved_period_id,
        include_utility_actuators=include_utility_actuators,
    )
    matrix = _build_interaction_matrix(outputs, actuators)
    matrix_rank, singular_values, condition_number, conditioning_score = (
        _matrix_diagnostics(matrix, rank_tolerance)
    )
    pairings, pairing_score = _best_pairings(
        outputs,
        actuators,
        matrix,
        minimum_interaction=minimum_interaction,
    )
    rank_score = _rank_score(matrix_rank, len(outputs))
    authority_score = _authority_score(matrix)
    redundancy_score = _redundancy_score(
        matrix,
        desired_redundancy=desired_redundancy,
        minimum_interaction=minimum_interaction,
    )
    thermal_margin_score, minimum_observed_approach = _thermal_margin_score(
        exchangers,
        period_id=resolved_period_id,
        minimum_approach_temperature=minimum_approach_temperature,
    )

    components = HeatExchangerNetworkControllabilityComponents(
        rank=rank_score,
        pairing=pairing_score,
        authority=authority_score,
        conditioning=conditioning_score,
        redundancy=redundancy_score,
        thermal_margin=thermal_margin_score,
    )
    score = _composite_score(components)
    diagnostics = _diagnostics(
        outputs=outputs,
        actuators=actuators,
        matrix=matrix,
        matrix_rank=matrix_rank,
        condition_number=condition_number,
        condition_warning_threshold=condition_warning_threshold,
        conditioning_score=conditioning_score,
        include_utility_actuators=include_utility_actuators,
        minimum_approach_temperature=minimum_approach_temperature,
        minimum_observed_approach=minimum_observed_approach,
        thermal_margin_score=thermal_margin_score,
        minimum_interaction=minimum_interaction,
    )

    return HeatExchangerNetworkControllabilityResult(
        score=score,
        rating=_rating(score),
        components=components,
        outputs=outputs,
        actuators=actuators,
        interaction_matrix=_matrix_to_tuple(matrix),
        pairings=pairings,
        matrix_rank=matrix_rank,
        condition_number=condition_number,
        singular_values=tuple(float(value) for value in singular_values),
        minimum_approach_temperature=minimum_observed_approach,
        diagnostics=diagnostics,
    )


def _validate_options(
    *,
    minimum_interaction: float,
    minimum_approach_temperature: float,
    desired_redundancy: int,
    rank_tolerance: float | None,
    condition_warning_threshold: float,
) -> None:
    if not math.isfinite(minimum_interaction) or minimum_interaction < 0.0:
        raise ValueError("minimum_interaction must be finite and non-negative")
    if (
        not math.isfinite(minimum_approach_temperature)
        or minimum_approach_temperature < 0.0
    ):
        raise ValueError("minimum_approach_temperature must be finite and non-negative")
    if desired_redundancy < 1:
        raise ValueError("desired_redundancy must be at least 1")
    if rank_tolerance is not None and (
        not math.isfinite(rank_tolerance) or rank_tolerance < 0.0
    ):
        raise ValueError("rank_tolerance must be finite and non-negative")
    if (
        not math.isfinite(condition_warning_threshold)
        or condition_warning_threshold <= 0.0
    ):
        raise ValueError("condition_warning_threshold must be positive and finite")


def _build_outputs(
    exchangers: tuple[HeatExchanger, ...],
    *,
    period_id: str,
) -> tuple[HeatExchangerNetworkControllabilityEndpoint, ...]:
    totals: OrderedDict[str, dict[str, float | int | str]] = OrderedDict()

    for exchanger in exchangers:
        state = exchanger.state(period_id)
        for side, stream_id in _process_endpoints(exchanger):
            output_id = f"{side}:{stream_id}"
            if output_id not in totals:
                totals[output_id] = {
                    "stream_id": stream_id,
                    "side": side,
                    "total_duty": 0.0,
                    "exchanger_count": 0,
                }
            totals[output_id]["total_duty"] = float(
                totals[output_id]["total_duty"]
            ) + float(state.duty)
            totals[output_id]["exchanger_count"] = (
                int(totals[output_id]["exchanger_count"]) + 1
            )

    return tuple(
        HeatExchangerNetworkControllabilityEndpoint(
            output_id=output_id,
            stream_id=str(values["stream_id"]),
            side=values["side"],  # type: ignore[arg-type]
            exchanger_count=int(values["exchanger_count"]),
            total_duty=float(values["total_duty"]),
        )
        for output_id, values in totals.items()
        if float(values["total_duty"]) > 0.0
    )


def _build_actuators(
    exchangers: tuple[HeatExchanger, ...],
    *,
    period_id: str,
    include_utility_actuators: bool,
) -> tuple[HeatExchangerNetworkControllabilityActuator, ...]:
    actuators = []
    seen_ids: dict[str, int] = {}

    for index, exchanger in enumerate(exchangers, start=1):
        state = exchanger.state(period_id)
        if state.duty <= 0.0:
            continue
        if (
            not include_utility_actuators
            and exchanger.kind is not HeatExchangerKind.RECOVERY
        ):
            continue

        actuator_id = _actuator_id(exchanger, index, seen_ids)
        actuators.append(
            HeatExchangerNetworkControllabilityActuator(
                actuator_id=actuator_id,
                exchanger_id=exchanger.exchanger_id,
                kind=exchanger.kind,
                source_stream=exchanger.source_stream,
                sink_stream=exchanger.sink_stream,
                stage=exchanger.stage,
                manipulated_variable=_manipulated_variable(exchanger.kind),
                duty=state.duty,
            )
        )

    return tuple(actuators)


def _actuator_id(
    exchanger: HeatExchanger,
    index: int,
    seen_ids: dict[str, int],
) -> str:
    base_id = exchanger.exchanger_id or (
        f"{exchanger.kind.value}:{exchanger.source_stream}"
        f"->{exchanger.sink_stream}:{exchanger.stage or index}"
    )
    count = seen_ids.get(base_id, 0)
    seen_ids[base_id] = count + 1
    if count == 0:
        return base_id
    return f"{base_id}#{count + 1}"


def _manipulated_variable(
    kind: HeatExchangerKind,
) -> str:
    if kind is HeatExchangerKind.RECOVERY:
        return "recovery_bypass_fraction"
    if kind is HeatExchangerKind.HOT_UTILITY:
        return "hot_utility_flow"
    return "cold_utility_flow"


def _build_interaction_matrix(
    outputs: tuple[HeatExchangerNetworkControllabilityEndpoint, ...],
    actuators: tuple[HeatExchangerNetworkControllabilityActuator, ...],
) -> np.ndarray:
    matrix = np.zeros((len(outputs), len(actuators)), dtype=float)
    if not outputs or not actuators:
        return matrix

    for output_index, output in enumerate(outputs):
        if output.total_duty <= 0.0:
            continue
        for actuator_index, actuator in enumerate(actuators):
            if not _actuator_affects_output(actuator, output):
                continue
            matrix[output_index, actuator_index] = actuator.duty / output.total_duty
    return np.clip(matrix, 0.0, 1.0)


def _actuator_affects_output(
    actuator: HeatExchangerNetworkControllabilityActuator,
    output: HeatExchangerNetworkControllabilityEndpoint,
) -> bool:
    if output.side == "source":
        return actuator.source_stream == output.stream_id
    return actuator.sink_stream == output.stream_id


def _matrix_diagnostics(
    matrix: np.ndarray,
    rank_tolerance: float | None,
) -> tuple[int, tuple[float, ...], float | None, float]:
    if matrix.size == 0:
        return 0, (), None, 0.0

    singular_values_array = np.linalg.svd(matrix, compute_uv=False)
    singular_values = tuple(float(value) for value in singular_values_array)
    rank = int(np.linalg.matrix_rank(matrix, tol=rank_tolerance))
    if singular_values_array.size == 0 or singular_values_array[0] == 0.0:
        return rank, singular_values, None, 0.0

    tolerance = rank_tolerance
    if tolerance is None:
        tolerance = (
            max(matrix.shape) * np.finfo(float).eps * float(singular_values_array[0])
        )
    positive = singular_values_array[singular_values_array > tolerance]
    if positive.size == 0:
        return rank, singular_values, None, 0.0

    condition_number = float(singular_values_array[0] / positive[-1])
    conditioning_score = float(positive[-1] / singular_values_array[0])
    return rank, singular_values, condition_number, conditioning_score


def _best_pairings(
    outputs: tuple[HeatExchangerNetworkControllabilityEndpoint, ...],
    actuators: tuple[HeatExchangerNetworkControllabilityActuator, ...],
    matrix: np.ndarray,
    *,
    minimum_interaction: float,
) -> tuple[tuple[HeatExchangerNetworkControllabilityPairing, ...], float]:
    if not outputs or not actuators:
        return (), 0.0

    row_indices, column_indices = linear_sum_assignment(-matrix)
    pairings = []
    selected_strength = 0.0
    for row_index, column_index in zip(row_indices, column_indices):
        interaction = float(matrix[row_index, column_index])
        selected_strength += interaction
        if interaction > minimum_interaction:
            pairings.append(
                HeatExchangerNetworkControllabilityPairing(
                    output_id=outputs[row_index].output_id,
                    actuator_id=actuators[column_index].actuator_id,
                    interaction=interaction,
                )
            )

    pairing_score = selected_strength / len(outputs)
    return tuple(pairings), float(np.clip(pairing_score, 0.0, 1.0))


def _rank_score(matrix_rank: int, output_count: int) -> float:
    if output_count == 0:
        return 0.0
    return float(np.clip(matrix_rank / output_count, 0.0, 1.0))


def _authority_score(matrix: np.ndarray) -> float:
    if matrix.size == 0 or matrix.shape[0] == 0:
        return 0.0
    return float(np.clip(np.mean(np.max(matrix, axis=1)), 0.0, 1.0))


def _redundancy_score(
    matrix: np.ndarray,
    *,
    desired_redundancy: int,
    minimum_interaction: float,
) -> float:
    if matrix.size == 0 or matrix.shape[0] == 0:
        return 0.0
    if desired_redundancy == 1:
        return 1.0

    non_zero_counts = np.sum(matrix > minimum_interaction, axis=1)
    row_scores = np.clip(
        (non_zero_counts - 1) / (desired_redundancy - 1),
        0.0,
        1.0,
    )
    return float(np.mean(row_scores))


def _thermal_margin_score(
    exchangers: tuple[HeatExchanger, ...],
    *,
    period_id: str,
    minimum_approach_temperature: float,
) -> tuple[float | None, float | None]:
    margins = [
        min(exchanger.state(period_id).approach_temperatures)
        for exchanger in exchangers
        if exchanger.state(period_id).approach_temperatures
    ]
    if not margins:
        return None, None

    minimum_observed = float(min(margins))
    if minimum_approach_temperature == 0.0:
        return 1.0, minimum_observed

    margin_scores = [
        min(max(float(margin) / minimum_approach_temperature, 0.0), 1.0)
        for margin in margins
    ]
    return float(np.mean(margin_scores)), minimum_observed


def _composite_score(
    components: HeatExchangerNetworkControllabilityComponents,
) -> float:
    values = components.model_dump()
    weighted_sum = 0.0
    total_weight = 0.0
    for name, weight in _COMPOSITE_WEIGHTS.items():
        value = values[name]
        if value is None:
            continue
        weighted_sum += float(value) * weight
        total_weight += weight
    return float(np.clip(weighted_sum / total_weight, 0.0, 1.0))


def _diagnostics(
    *,
    outputs: tuple[HeatExchangerNetworkControllabilityEndpoint, ...],
    actuators: tuple[HeatExchangerNetworkControllabilityActuator, ...],
    matrix: np.ndarray,
    matrix_rank: int,
    condition_number: float | None,
    condition_warning_threshold: float,
    conditioning_score: float,
    include_utility_actuators: bool,
    minimum_approach_temperature: float,
    minimum_observed_approach: float | None,
    thermal_margin_score: float | None,
    minimum_interaction: float,
) -> tuple[str, ...]:
    diagnostics = []
    if not outputs:
        diagnostics.append("no process-stream outlet temperatures were found")
    if not actuators:
        diagnostics.append("no manipulated variables were found")
    if outputs and len(actuators) < len(outputs):
        diagnostics.append(
            "fewer manipulated variables than process-stream outlet temperatures"
        )
    if outputs and matrix_rank < len(outputs):
        diagnostics.append(
            "interaction matrix is rank deficient for full outlet-temperature control"
        )
    if condition_number is not None and condition_number > condition_warning_threshold:
        diagnostics.append(
            "interaction matrix is poorly conditioned "
            f"(condition number {condition_number:g})"
        )
    elif outputs and actuators and conditioning_score == 0.0:
        diagnostics.append("interaction matrix has no usable singular direction")
    if not include_utility_actuators:
        diagnostics.append("utility flow actuators were excluded from the analysis")
    if thermal_margin_score is None:
        diagnostics.append("approach-temperature margins were unavailable")
    elif (
        minimum_observed_approach is not None
        and minimum_observed_approach < minimum_approach_temperature
    ):
        diagnostics.append(
            "minimum approach temperature is below the requested margin "
            f"({minimum_observed_approach:g} < {minimum_approach_temperature:g})"
        )
    if matrix.size:
        row_actuator_counts = np.sum(matrix > minimum_interaction, axis=1)
        single_actuator_rows = int(np.sum(row_actuator_counts == 1))
        if single_actuator_rows:
            diagnostics.append(
                f"{single_actuator_rows} outlet temperature(s) have no spare actuator"
            )
    return tuple(diagnostics)


def _rating(score: float) -> str:
    if score >= 0.75:
        return "strong"
    if score >= 0.50:
        return "moderate"
    if score >= 0.25:
        return "weak"
    return "poor"


def _matrix_to_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


def _process_endpoints(exchanger: HeatExchanger) -> tuple[tuple[str, str], ...]:
    endpoints = []
    if exchanger.source_stream_role is HeatExchangerStreamRole.PROCESS:
        endpoints.append(("source", exchanger.source_stream))
    if exchanger.sink_stream_role is HeatExchangerStreamRole.PROCESS:
        endpoints.append(("sink", exchanger.sink_stream))
    return tuple(endpoints)


__all__ = ["quantify_heat_exchanger_network_controllability"]
