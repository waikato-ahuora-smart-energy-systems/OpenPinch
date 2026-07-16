"""Parent-owned segment normalization and transaction helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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

    values = dict(raw_segment)
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
