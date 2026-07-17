"""Shared graph-data normalization and curve primitives."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...domain.configuration import tol
from ...domain.enums import ArrowHead, LineColour, StreamLoc
from ...domain.problem_table import ProblemTable
from .metadata import GRAPH_SERIES_META
from .metadata import GraphSeriesMeta as _GraphSeriesMeta

DECIMAL_PLACES = 2


def _normalise_graph_fields(value_field) -> list:
    if isinstance(value_field, str) or not hasattr(value_field, "__iter__"):
        return [value_field]
    return list(value_field)


def _normalise_graph_values(value, count: int, mismatch_message: str) -> list:
    if isinstance(value, str) or not hasattr(value, "__iter__"):
        values = [value] * count
    else:
        values = list(value)
    if len(values) != count:
        raise ValueError(mismatch_message)
    return values


def _column_key(field) -> str:
    return getattr(field, "value", field)


def _series_meta_from_key(column_key: str) -> _GraphSeriesMeta:
    meta = GRAPH_SERIES_META.get(column_key)
    if meta is not None:
        return meta
    label = str(column_key)
    return _GraphSeriesMeta(label, label)


def _should_plot_series(values: list[float]) -> bool:
    """Return whether a graph series contains meaningful finite values."""
    try:
        numeric = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        numeric = np.array(
            [float(value) for value in values if value is not None],
            dtype=float,
        )
    finite = numeric[np.isfinite(numeric)]
    return bool(finite.size and np.any(np.abs(finite) > tol))


def _column_to_list(data, column_key: str) -> list[float]:
    """Return one graph-data column as a detached Python list."""
    if not isinstance(column_key, str):
        column_key = getattr(column_key, "value", column_key)
    try:
        if isinstance(data, ProblemTable):
            column = data[column_key]
        elif (
            hasattr(data, "col")
            and hasattr(data, "columns")
            and column_key in getattr(data, "columns", [])
        ):
            column = data.col[column_key]
        else:
            column = data[column_key]
    except (KeyError, AttributeError, TypeError) as exc:
        raise KeyError(f"Column '{column_key}' not found in graph data.") from exc
    if hasattr(column, "to_list"):
        return column.to_list()
    if hasattr(column, "tolist"):
        return column.tolist()
    return list(column)


def _streamloc_colour(stream_loc: StreamLoc) -> int:
    """Return the default colour for one stream location."""
    colours = {
        StreamLoc.HotS: LineColour.HotS.value,
        StreamLoc.ColdS: LineColour.ColdS.value,
        StreamLoc.HotU: LineColour.HotU.value,
        StreamLoc.ColdU: LineColour.ColdU.value,
        StreamLoc.Unassigned: LineColour.Black.value,
    }
    return colours.get(stream_loc, LineColour.Other.value)


def _create_curve(
    title: str,
    colour: int,
    x_vals,
    y_vals,
    arrow=ArrowHead.NO_ARROW.value,
    series_label: str | None = None,
    series_id: str | None = None,
    series_description: str | None = None,
    is_vertical: bool | None = None,
    is_utility_stream: bool | None = None,
) -> dict[str, Any]:
    """Create one serialized graph curve from coordinate values."""
    curve: dict[str, Any] = {"title": title, "colour": colour, "arrow": arrow}
    optional_fields = {
        "series": series_label,
        "series_id": series_id,
        "series_description": series_description,
        "is_vertical": None if is_vertical is None else bool(is_vertical),
        "is_utility_stream": (
            None if is_utility_stream is None else bool(is_utility_stream)
        ),
    }
    curve.update({key: value for key, value in optional_fields.items() if value is not None})
    curve["data_points"] = [
        {"x": round(x, DECIMAL_PLACES), "y": round(y, DECIMAL_PLACES)}
        for x, y in zip(x_vals, y_vals)
        if x is not None and y is not None
    ]
    return curve
