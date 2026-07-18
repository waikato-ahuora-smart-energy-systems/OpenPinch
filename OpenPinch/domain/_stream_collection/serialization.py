"""Stable reporting serialization for parent stream collections."""

from __future__ import annotations

from typing import Any


def collection_to_dict(
    collection,
    *,
    idx: int | None,
    expand_segments: bool,
) -> dict[str, list[Any]]:
    """Serialize streams or expanded segments in standard reporting order."""
    ordered_items = sorted(
        collection._streams.items(),
        key=lambda item: collection._dict_sort_key(item[1], idx),
    )
    columns = [
        "name",
        "category",
        "type",
        "is_process_stream",
        "t_supply",
        "t_target",
        "heat_flow",
        "dt_cont",
        "dt_cont_multiplier",
        "htc",
        "active",
    ]
    if expand_segments:
        columns = [
            "parent_key",
            "parent_name",
            "segment_index",
            "segment_identity",
            *columns,
        ]
    report_streams = []
    for parent_key, stream in ordered_items:
        if expand_segments and stream.has_segments:
            report_streams.extend(
                (
                    parent_key,
                    stream.name,
                    index,
                    f"{parent_key}.S{index + 1}",
                    segment,
                )
                for index, segment in enumerate(stream.segments)
            )
        else:
            report_streams.append((parent_key, stream.name, None, None, stream))
    rows = [
        {
            "parent_key": parent_key,
            "parent_name": parent_name,
            "segment_index": segment_index,
            "segment_identity": segment_identity,
            "name": stream.name,
            "category": collection._dict_category(stream),
            "type": stream.stream_type,
            "is_process_stream": stream.is_process_stream,
            "t_supply": collection._value_at_idx(stream._t_supply, idx),
            "t_target": collection._value_at_idx(stream._t_target, idx),
            "heat_flow": collection._value_at_idx(stream._heat_flow, idx),
            "dt_cont": collection._value_at_idx(stream._dt_cont, idx),
            "dt_cont_multiplier": stream.delta_t_contribution_multiplier,
            "htc": collection._value_at_idx(stream._htc, idx),
            "active": stream.is_active,
        }
        for parent_key, parent_name, segment_index, segment_identity, stream in (
            report_streams
        )
    ]
    return {column: [row[column] for row in rows] for column in columns}
