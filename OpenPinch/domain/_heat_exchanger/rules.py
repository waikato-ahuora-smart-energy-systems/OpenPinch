"""Validation rules shared by heat exchanger domain records and their schemas.

The runtime models in :mod:`OpenPinch.domain.heat_exchanger` /
:mod:`OpenPinch.domain.heat_exchanger_network` and the transport schemas in
:mod:`OpenPinch.contracts.input` enforce the same invariants; the rule logic
lives here once so both layers stay in step.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from ..enums import HeatExchangerKind, StreamID

EXCHANGER_IDENTITY_MESSAGE = "stream and exchanger identities must be non-empty strings"
EXCHANGER_NUMERIC_MESSAGE = "numeric exchanger values must be finite and non-negative"
NETWORK_IDENTITY_MESSAGE = "network metadata identities must be non-empty strings"
NETWORK_NUMERIC_MESSAGE = "network numeric values must be finite and non-negative"

EXPECTED_STREAM_ROLES: dict[HeatExchangerKind, tuple[StreamID, StreamID]] = {
    HeatExchangerKind.RECOVERY: (StreamID.Process, StreamID.Process),
    HeatExchangerKind.HOT_UTILITY: (StreamID.Utility, StreamID.Process),
    HeatExchangerKind.COLD_UTILITY: (StreamID.Process, StreamID.Utility),
}


def optional_identity(value: str | None, message: str) -> str | None:
    """Return a stripped identity, ``None`` unchanged, or raise ``message``."""

    if value is None:
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError(message)
    return value.strip()


def optional_non_negative_finite(value: float | None, message: str) -> float | None:
    """Return a finite non-negative float, ``None`` unchanged, or raise ``message``."""

    if value is None:
        return value
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(message)
    return float(value)


def check_period_states(period_states: Sequence[Any]) -> None:
    """Require contiguous ``period_idx`` ordering and unique ``period_id`` values."""

    expected_indices = tuple(range(len(period_states)))
    actual_indices = tuple(state.period_idx for state in period_states)
    if actual_indices != expected_indices:
        raise ValueError("period_states must be ordered by contiguous period_idx")
    period_ids = tuple(state.period_id for state in period_states)
    if len(set(period_ids)) != len(period_ids):
        raise ValueError("period_states must use unique period_id values")


def check_direction_semantics(
    *,
    kind: HeatExchangerKind,
    source_stream: str,
    sink_stream: str,
    source_stream_role: StreamID,
    sink_stream_role: StreamID,
    stage: int | None,
) -> None:
    """Require distinct endpoints, kind-consistent roles and recovery stages."""

    if source_stream == sink_stream:
        raise ValueError("source_stream and sink_stream must be distinct")

    expected_source_role, expected_sink_role = EXPECTED_STREAM_ROLES[kind]
    if (
        source_stream_role != expected_source_role
        or sink_stream_role != expected_sink_role
    ):
        raise ValueError(
            f"{kind.value} exchangers must link "
            f"{expected_source_role.value} -> {expected_sink_role.value}"
        )

    if kind is HeatExchangerKind.RECOVERY and stage is None:
        raise ValueError("recovery exchangers must include a synthesis stage")


def check_summary_metrics(value: Mapping[Any, Any]) -> None:
    """Require non-empty string metric names and finite float metric values."""

    for metric_name, metric_value in value.items():
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("summary metric names must be non-empty strings")
        if isinstance(metric_value, float) and not math.isfinite(metric_value):
            raise ValueError("summary metric values must be finite")


def check_period_alignment(period_id_sequences: Iterable[tuple[str, ...]]) -> None:
    """Require every exchanger to share the first exchanger's ordered period ids."""

    sequences = iter(period_id_sequences)
    ordered = next(sequences, None)
    if ordered is None:
        return
    for current in sequences:
        if current != ordered:
            raise ValueError(
                "all exchangers in a network must use the same ordered period_ids"
            )
