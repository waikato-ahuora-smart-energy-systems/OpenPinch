"""Grand-composite and utility-profile graph construction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from ...domain.configuration import tol
from ...domain.enums import ArrowHead, LineColour, ProblemTableLabel, StreamLoc
from .composite import clean_composite_curve
from .metadata import _GraphSeriesMeta
from .primitives import (
    _column_key,
    _column_to_list,
    _create_curve,
    _normalise_graph_fields,
    _normalise_graph_values,
    _series_meta_from_key,
    _should_plot_series,
    _streamloc_colour,
)

GCC_VERTICAL_TOL = 1e-3


def _normalise_gcc_flags(flag, count: int) -> list[bool]:
    flags = _normalise_graph_values(
        flag if flag is not None else False,
        count,
        "`value_field` and `is_utility_profile` must have the same length.",
    )
    try:
        return [bool(item) for item in flags]
    except TypeError as exc:
        raise ValueError(
            "`is_utility_profile` values must be coercible to bool."
        ) from exc


def _build_gcc_segments(
    y_vals: Iterable[float],
    x_vals: Iterable[float],
    *,
    series_id: str,
    meta: _GraphSeriesMeta,
    is_utility_profile: bool,
    decolour: bool,
) -> list[dict]:
    y_vals, x_vals = clean_composite_curve(y_vals, x_vals)
    counts: dict[StreamLoc, int] = defaultdict(int)
    segments: list[dict] = []
    for stream_loc, is_vertical, x_seg, y_seg in _iter_gcc_segment_slices(
        x_vals,
        y_vals,
        is_utility_profile,
        meta.preferred_stream_loc,
    ):
        counts[stream_loc] += 1
        colour = (
            LineColour.Black.value
            if is_vertical or decolour
            else _streamloc_colour(stream_loc)
        )
        segments.append(
            _create_curve(
                title=_format_segment_title(meta, counts[stream_loc]),
                colour=colour,
                x_vals=x_seg,
                y_vals=y_seg,
                arrow=ArrowHead.NO_ARROW.value,
                series_label=meta.label,
                series_id=series_id,
                series_description=meta.description,
                is_vertical=is_vertical,
                is_utility_stream=stream_loc in {StreamLoc.HotU, StreamLoc.ColdU},
            )
        )
    return segments


def _iter_gcc_segment_slices(
    x_vals: list[float],
    y_vals: list[float],
    is_utility_profile: bool,
    preferred_stream_loc: StreamLoc | None,
):
    start, end = _segment_bounds(x_vals)
    index = start
    while index < end:
        classified = _classify_segment(
            x_vals[index] - x_vals[index + 1],
            is_utility_profile,
        )
        next_index = index + 1
        while next_index < end:
            next_class = _classify_segment(
                x_vals[next_index] - x_vals[next_index + 1],
                is_utility_profile,
            )
            if next_class != classified:
                break
            next_index += 1
        raw_location = _segment_streamloc(classified)
        is_vertical = raw_location == StreamLoc.Unassigned
        stream_location = (
            preferred_stream_loc
            if is_vertical and preferred_stream_loc
            else raw_location
        )
        yield (
            stream_location,
            is_vertical,
            x_vals[index : next_index + 1],
            y_vals[index : next_index + 1],
        )
        index = next_index


def _segment_bounds(x_vals: list[float]) -> tuple[int, int]:
    start = next(
        (
            index
            for index in range(len(x_vals) - 1)
            if abs(x_vals[index] - x_vals[index + 1]) > tol
        ),
        0,
    )
    end = next(
        (
            index
            for index in range(len(x_vals) - 1, 0, -1)
            if abs(x_vals[index] - x_vals[index - 1]) > tol
        ),
        len(x_vals) - 1,
    )
    return start, end


def _format_segment_title(meta: _GraphSeriesMeta, index: int) -> str:
    base = meta.description or meta.label or "Segment"
    return f"{base} {index}"


def _make_gcc_graph(
    graph_title: str,
    key: str,
    data,
    label: str,
    *,
    value_field,
    name: str | None = None,
    is_utility_profile: bool = False,
    decolour: bool = False,
) -> dict:
    temperatures = _column_to_list(data, ProblemTableLabel.T)
    fields = _normalise_graph_fields(value_field)
    flags = _normalise_gcc_flags(is_utility_profile, len(fields))
    segments: list[dict] = []
    for field, utility_flag in zip(fields, flags):
        column_key = _column_key(field)
        x_vals = _column_to_list(data, column_key)
        if _should_plot_series(x_vals):
            segments.extend(
                _build_gcc_segments(
                    temperatures,
                    x_vals,
                    series_id=f"{key}:{column_key}",
                    meta=_series_meta_from_key(column_key),
                    is_utility_profile=utility_flag,
                    decolour=decolour,
                )
            )
    return {
        "type": key,
        "name": name or f"{label}: {graph_title}",
        "segments": segments,
    }


def _classify_segment(enthalpy_diff: float, is_utility_profile: bool) -> StreamLoc:
    if abs(enthalpy_diff) <= GCC_VERTICAL_TOL:
        return StreamLoc.Unassigned
    if enthalpy_diff > 0:
        return StreamLoc.ColdS if not is_utility_profile else StreamLoc.HotU
    if enthalpy_diff < 0:
        return StreamLoc.HotS if not is_utility_profile else StreamLoc.ColdU
    return StreamLoc.Unassigned


def _segment_streamloc(segment_type: StreamLoc) -> StreamLoc:
    """Return one canonical segment stream location."""
    if not isinstance(segment_type, StreamLoc):
        raise TypeError("segment_type must be a StreamLoc value")
    return segment_type
