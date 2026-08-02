"""Composite-curve cleanup and serialized graph construction."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...domain.configuration import tol
from ...domain.enums import (
    ArrowHead,
    GraphType,
    LineColour,
    ProblemTableLabel,
    StreamLoc,
)
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


def clean_composite_curve_ends(
    y_vals: np.ndarray | list,
    x_vals: np.ndarray | list,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove redundant constant-enthalpy points from curve ends."""
    y_vals = np.asarray(y_vals)
    x_vals = np.asarray(x_vals)
    if np.all(np.isclose(x_vals, 0.0, atol=tol)) or np.abs(x_vals.var()) < tol:
        return np.array([]), np.array([])
    start_mask = ~np.isclose(x_vals, x_vals[0] * np.ones(len(x_vals)), atol=tol)
    end_mask = ~np.isclose(x_vals, x_vals[-1] * np.ones(len(x_vals)), atol=tol)
    start = np.flatnonzero(start_mask)[0] - 1
    end = np.flatnonzero(end_mask)[-1] + 1
    return y_vals[start : end + 1], x_vals[start : end + 1]


def clean_composite_curve(
    y_array: np.ndarray | list,
    x_array: np.ndarray | list,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove collinear and repeated points from one composite curve."""
    y_vals, x_vals = clean_composite_curve_ends(y_array, x_array)
    if len(x_vals) > 1:
        duplicate = (x_vals[1:] == x_vals[:-1]) & (y_vals[1:] == y_vals[:-1])
        keep = np.concatenate(([True], ~duplicate))
        x_vals = x_vals[keep]
        y_vals = y_vals[keep]
    if len(x_vals) <= 2:
        return y_vals, x_vals

    x_clean, y_clean = [x_vals[0]], [y_vals[0]]
    for index in range(1, len(x_vals) - 1):
        x1, x2, x3 = x_vals[index - 1 : index + 2]
        y1, y2, y3 = y_vals[index - 1 : index + 2]
        if x1 == x3:
            if x1 != x2:
                x_clean.append(x2)
                y_clean.append(y2)
        else:
            interpolated = y1 + (y3 - y1) * (x2 - x1) / (x3 - x1)
            if abs(y2 - interpolated) > tol:
                x_clean.append(x2)
                y_clean.append(y2)
    x_clean.append(x_vals[-1])
    y_clean.append(y_vals[-1])
    if abs(x_clean[0] - x_clean[1]) < tol:
        x_clean.pop(0)
        y_clean.pop(0)
    final = len(x_clean) - 1
    if abs(x_clean[final] - x_clean[final - 1]) < tol:
        x_clean.pop(final)
        y_clean.pop(final)
    return np.asarray(y_clean), np.asarray(x_clean)


def _make_composite_graph(
    graph_title: str,
    key: str,
    data,
    label: str,
    *,
    value_field,
    stream_type,
    name: str | None = None,
    include_arrows: bool = True,
    decolour: bool = False,
) -> dict[str, Any]:
    temperatures = _column_to_list(data, ProblemTableLabel.T)
    fields = _normalise_graph_fields(value_field)
    stream_types = _normalise_graph_values(
        stream_type,
        len(fields),
        "`value_field` and `stream_type` must have the same length.",
    )
    segments: list[dict] = []
    for field, stream_loc in zip(fields, stream_types):
        column_key = _column_key(field)
        x_vals = _column_to_list(data, column_key)
        if _should_plot_series(x_vals):
            segments.extend(
                _graph_cc(
                    key,
                    stream_loc,
                    temperatures,
                    x_vals,
                    column_key=column_key,
                    include_arrows=include_arrows,
                    decolour=decolour,
                )
            )
    return {
        "type": key,
        "name": name or f"{label}: {graph_title}",
        "segments": segments,
    }


def _graph_cc(
    key: str,
    stream_loc: StreamLoc,
    y_vals: list[float],
    x_vals: list[float],
    *,
    column_key: str | None = None,
    include_arrows: bool = True,
    decolour: bool = False,
) -> list[dict]:
    """Build one shifted or unshifted composite-curve segment."""
    y_vals, x_vals = clean_composite_curve(y_vals, x_vals)
    if not isinstance(stream_loc, StreamLoc):
        raise TypeError("stream_loc must be a StreamLoc member.")

    title_map = {
        StreamLoc.HotS: "Hot CC",
        StreamLoc.ColdS: "Cold CC",
        StreamLoc.HotU: "Hot Utility",
        StreamLoc.ColdU: "Cold Utility",
    }
    if stream_loc not in title_map:
        raise ValueError("Unrecognised composite curve stream location.")
    arrow_map = (
        {
            StreamLoc.HotS: ArrowHead.START.value,
            StreamLoc.HotU: ArrowHead.START.value,
            StreamLoc.ColdS: ArrowHead.END.value,
            StreamLoc.ColdU: ArrowHead.END.value,
        }
        if key == GraphType.TSP.value
        else {
            StreamLoc.HotS: ArrowHead.END.value,
            StreamLoc.HotU: ArrowHead.END.value,
            StreamLoc.ColdS: ArrowHead.START.value,
            StreamLoc.ColdU: ArrowHead.START.value,
        }
    )
    if stream_loc not in title_map:
        raise ValueError("Unrecognised composite curve stream location.")
    colour = LineColour.Black.value if decolour else _streamloc_colour(stream_loc)
    arrow = arrow_map[stream_loc] if include_arrows else ArrowHead.NO_ARROW.value
    meta = _composite_series_meta(stream_loc, column_key)
    return [
        _create_curve(
            title=meta.composite_title,
            colour=colour,
            arrow=arrow,
            x_vals=x_vals,
            y_vals=y_vals,
            series_label=meta.label if column_key is not None else None,
            series_id=f"{key}:{column_key}" if column_key is not None else None,
            series_description=meta.description if column_key is not None else None,
        )
    ]


def _composite_series_meta(
    stream_loc: StreamLoc,
    column_key: str | None,
) -> _GraphSeriesMeta:
    if column_key is not None:
        meta = _series_meta_from_key(column_key)
        if meta.composite_title is not None:
            return meta
    default_title = {
        StreamLoc.HotS: "Hot CC",
        StreamLoc.ColdS: "Cold CC",
        StreamLoc.HotU: "Hot Utility",
        StreamLoc.ColdU: "Cold Utility",
    }[stream_loc]
    return _GraphSeriesMeta(
        label=default_title,
        description=default_title,
        composite_title=default_title,
    )


__all__ = ["clean_composite_curve", "clean_composite_curve_ends"]
