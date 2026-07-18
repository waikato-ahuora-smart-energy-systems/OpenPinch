"""Filtered subset construction for parent stream collections."""

from __future__ import annotations

from copy import copy

from ..enums import StreamType


def build_stream_subset(
    collection,
    *,
    target_type: str | None,
    include_process_streams: bool,
    include_utility_streams: bool,
    invert_utility: bool,
    sort_attr: str | None,
):
    """Return a context-preserving filtered collection of existing streams."""
    if invert_utility:
        include_process_streams = False
        include_utility_streams = True

    subset = type(collection)()
    subset._period_ids = collection._period_ids
    subset._weights = collection._weights
    subset._num_periods = collection._num_periods
    subset._sort_spec = collection._sort_spec
    subset._rebuild_sort_key()
    subset._sort_reverse = collection._sort_reverse

    for key, stream in collection._streams.items():
        if stream.is_process_stream:
            if include_process_streams and (
                target_type is None or stream.stream_type == target_type
            ):
                subset._streams[key] = stream
            continue
        if not include_utility_streams:
            continue
        if invert_utility:
            opposite_type = (
                StreamType.Cold.value
                if target_type == StreamType.Hot.value
                else StreamType.Hot.value
            )
            if stream.stream_type != opposite_type:
                continue
            inverted_stream = copy(stream)
            inverted_stream.invert()
            subset._streams[key] = inverted_stream
        elif target_type is None or stream.stream_type == target_type:
            subset._streams[key] = stream

    if sort_attr is None:
        subset._sort_spec = collection._sort_spec
        subset._rebuild_sort_key()
        subset._sort_reverse = collection._sort_reverse
    else:
        subset.set_sort_key(sort_attr, reverse=collection._sort_reverse)
    subset._needs_sort = True
    return subset
