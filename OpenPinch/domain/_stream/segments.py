"""Parent-owned segment normalization and transaction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_WIRE_TO_RUNTIME = {
    "t_supply": "supply_temperature",
    "t_target": "target_temperature",
    "p_supply": "supply_pressure",
    "p_target": "target_pressure",
    "h_supply": "supply_enthalpy",
    "h_target": "target_enthalpy",
    "dt_cont": "delta_t_contribution",
    "dt_cont_multiplier": "delta_t_contribution_multiplier",
    "htc": "heat_transfer_coefficient",
}


def normalise_segment_input(segment: Any, *, segment_type: type):
    """Return one detached internal segment from a supported parent input."""
    from .profile import detached_segment

    if isinstance(segment, segment_type):
        return detached_segment(segment, segment_type=segment_type)

    raw_segment = segment
    if hasattr(raw_segment, "model_dump") and not isinstance(raw_segment, Mapping):
        raw_segment = raw_segment.model_dump(mode="python")
    if not isinstance(raw_segment, Mapping):
        raise TypeError(
            "Stream segments must be internal records, mappings, or objects "
            "providing model_dump()."
        )

    values = {
        _WIRE_TO_RUNTIME.get(key, key): value for key, value in raw_segment.items()
    }
    values.pop("segment_index", None)
    if values.get("name") is None:
        values.pop("name", None)
    return segment_type(**values)


def normalise_segment_inputs(segments, *, segment_type: type) -> tuple:
    """Normalize an ordered iterable before parent validation and attachment."""
    if isinstance(segments, (str, bytes, Mapping)):
        raise TypeError("segments must be an ordered iterable of segment inputs.")
    try:
        return tuple(
            normalise_segment_input(segment, segment_type=segment_type)
            for segment in segments
        )
    except TypeError:
        raise
    except Exception:
        raise


def update_all_value_attributes(parent, attr_name: str, value, *, idx=None) -> None:
    """Apply one value mutation to every child in a single transaction."""
    update_transaction(
        parent,
        {
            segment_index: {attr_name: value}
            for segment_index in range(parent.segment_count)
        },
        idx=idx,
    )


def update_transaction(
    parent,
    updates: Mapping[int, Mapping[str, object]],
    *,
    idx=None,
):
    """Clone, mutate, validate, and commit a sparse set of child updates."""
    if not parent.has_segments:
        raise ValueError(f"Stream {parent.name!r} has no explicit segments.")
    normalised: dict[int, dict[str, object]] = {}
    for segment_index, changes in updates.items():
        if isinstance(segment_index, bool) or not isinstance(segment_index, int):
            raise TypeError("Segment update indexes must be integers.")
        if segment_index < 0 or segment_index >= parent.segment_count:
            raise IndexError(f"Segment index {segment_index} is out of range.")
        if not isinstance(changes, Mapping):
            raise TypeError("Each segment update must be an attribute mapping.")
        validated_changes: dict[str, object] = {}
        for attr_name, value in changes.items():
            if not isinstance(attr_name, str):
                raise TypeError("Segment update attribute names must be strings.")
            if attr_name in {
                "is_active",
                "is_process_stream",
                "fluid_name",
                "fluid_phase",
            }:
                raise ValueError(
                    f"{attr_name!r} is controlled by parent stream {parent.name!r}."
                )
            internal_name = parent._resolve_attr_name(attr_name)
            if internal_name not in parent._CORE_VALUE_ATTRS:
                raise ValueError(
                    f"Attribute {attr_name!r} is not mutable segment state."
                )
            validated_changes[attr_name] = value
        normalised[segment_index] = validated_changes

    if not normalised:
        return

    candidates = [
        detached(segment, segment_type=type(segment)) for segment in parent._segments
    ]
    for segment_index, changes in normalised.items():
        candidate = candidates[segment_index]
        for attr_name, value in changes.items():
            if idx is None:
                candidate.set_value_attr(attr_name, value)
            else:
                candidate.set_value_attr_at_idx(attr_name, value, idx=idx)
    parent.replace_segments(candidates)


def replace(parent, segments, *, segment_type: type) -> None:
    """Normalize, validate, attach, and atomically replace child records."""
    candidates = normalise_segment_inputs(segments, segment_type=segment_type)
    parent._validate_segments(candidates)
    previous = parent._segments
    try:
        for segment in previous:
            segment._owner = None
        parent._segments = candidates
        for index, segment in enumerate(parent._segments):
            segment._owner = parent
            segment._segment_index = index
            segment._is_process_stream = parent._is_process_stream
            segment._active = parent._active
            segment._fluid_name = parent._fluid_name
            segment._fluid_phase = parent._fluid_phase
            segment._dt_cont_multiplier = parent._dt_cont_multiplier
            if parent._period_ids is not None and parent._weights is not None:
                segment.set_period_context(
                    period_ids=parent._period_ids,
                    weights=parent._weights,
                    num_periods=parent._num_periods,
                )
        sync_aggregate(parent)
    except Exception:
        for segment in candidates:
            segment._owner = None
        parent._segments = previous
        for index, segment in enumerate(previous):
            segment._owner = parent
            segment._segment_index = index
        raise


def detached(segment, *, segment_type: type):
    """Return a detached copy of one internal segment record."""
    from .profile import detached_segment

    return detached_segment(segment, segment_type=segment_type)


def validate(parent, segments: tuple) -> None:
    """Validate child ordering, continuity, direction, and period shape."""
    from ...domain.configuration import tol
    from .profile import validate_segments

    validate_segments(
        segments,
        parent_num_periods=parent._num_periods,
        tolerance=tol,
    )


def sync_aggregate(parent) -> None:
    """Synchronize parent aggregate values from authoritative children."""
    if not parent._segments:
        return
    from .profile import aggregate_segments

    aggregate = aggregate_segments(
        parent._segments,
        parent_num_periods=parent._num_periods,
    )
    parent._syncing_segments = True
    try:
        assignments = {
            "supply_temperature": aggregate.t_supply,
            "target_temperature": aggregate.t_target,
            "supply_pressure": aggregate.p_supply,
            "target_pressure": aggregate.p_target,
            "supply_enthalpy": aggregate.h_supply,
            "target_enthalpy": aggregate.h_target,
            "delta_t_contribution": aggregate.dt_cont,
            "heat_flow": parent._build_value(aggregate.heat_flow, unit="kW"),
            "heat_transfer_coefficient": parent._build_value(
                aggregate.effective_htc,
                unit="kW/m^2/delta_degC",
            ),
            "price": aggregate.price,
        }
        for attr_name, value in assignments.items():
            parent.set_value_attr(attr_name, value, update_derived=False)
        parent._validate_num_periods()
        parent.update_derived_properties()
    finally:
        parent._syncing_segments = False
    parent._bump_numeric_revision()
