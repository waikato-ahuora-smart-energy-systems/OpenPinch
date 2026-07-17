"""Numeric snapshot helpers for :class:`StreamCollection` internals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..enums import ST
from ..value import Value


@dataclass(frozen=True)
class StreamCollectionNumericView:
    """Dense numeric snapshot of stream properties for internal kernels."""

    t_min: np.ndarray
    t_max: np.ndarray
    t_min_star: np.ndarray
    t_max_star: np.ndarray
    cp: np.ndarray
    rcp: np.ndarray
    heat_flow: np.ndarray
    dt_cont: np.ndarray
    is_hot: np.ndarray
    is_cold: np.ndarray
    active: np.ndarray
    parent_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=int))
    segment_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=int))
    parent_name: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=object)
    )
    parent_key: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=object))
    period_index: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=int))


def build_numeric_view(
    streams: list[object],
    idx: int | None = None,
    *,
    keys: list[str] | None = None,
):
    """Build a dense numeric view from stream objects."""
    n_streams = len(streams)
    t_min = np.empty(n_streams, dtype=float)
    t_max = np.empty(n_streams, dtype=float)
    t_min_star = np.empty(n_streams, dtype=float)
    t_max_star = np.empty(n_streams, dtype=float)
    cp = np.empty(n_streams, dtype=float)
    rcp = np.empty(n_streams, dtype=float)
    heat_flow = np.empty(n_streams, dtype=float)
    dt_cont = np.empty(n_streams, dtype=float)
    is_hot = np.empty(n_streams, dtype=bool)
    is_cold = np.empty(n_streams, dtype=bool)
    active = np.empty(n_streams, dtype=bool)

    for stream_idx, stream in enumerate(streams):
        t_min[stream_idx] = value_at_idx(stream._t_min, idx)
        t_max[stream_idx] = value_at_idx(stream._t_max, idx)
        t_min_star[stream_idx] = value_at_idx(stream._t_min_star, idx)
        t_max_star[stream_idx] = value_at_idx(stream._t_max_star, idx)
        cp[stream_idx] = value_at_idx(stream._cp, idx)
        rcp[stream_idx] = value_at_idx(stream._rcp_prod, idx)
        heat_flow[stream_idx] = value_at_idx(stream._heat_flow, idx)
        dt_cont[stream_idx] = value_at_idx(stream._dt_cont, idx)
        is_hot[stream_idx] = stream.type == ST.Hot.value
        is_cold[stream_idx] = stream.type == ST.Cold.value
        active[stream_idx] = bool(stream.active)

    return StreamCollectionNumericView(
        t_min=t_min,
        t_max=t_max,
        t_min_star=t_min_star,
        t_max_star=t_max_star,
        cp=cp,
        rcp=rcp,
        heat_flow=heat_flow,
        dt_cont=dt_cont,
        is_hot=is_hot,
        is_cold=is_cold,
        active=active,
        parent_index=np.arange(n_streams, dtype=int),
        segment_index=np.full(n_streams, -1, dtype=int),
        parent_name=np.asarray(
            [
                getattr(stream, "name", str(index))
                for index, stream in enumerate(streams)
            ],
            dtype=object,
        ),
        parent_key=np.asarray(
            keys
            if keys is not None
            else [
                getattr(stream, "name", str(index))
                for index, stream in enumerate(streams)
            ],
            dtype=object,
        ),
        period_index=np.full(n_streams, 0 if idx is None else int(idx), dtype=int),
    )


def build_segment_numeric_view(
    streams: list[object],
    idx: int | None = None,
    *,
    keys: list[str] | None = None,
):
    """Build a thermal view expanded to explicit segments with parent metadata."""
    expanded = []
    parent_index = []
    segment_index = []
    parent_name = []
    parent_key = []
    for stream_index, stream in enumerate(streams):
        children = getattr(stream, "segments", ())
        rows = children if children else (stream,)
        for child_index, row in enumerate(rows):
            expanded.append(row)
            parent_index.append(stream_index)
            segment_index.append(child_index if children else -1)
            parent_name.append(getattr(stream, "name", str(stream_index)))
            parent_key.append(
                keys[stream_index]
                if keys is not None
                else getattr(stream, "name", str(stream_index))
            )
    view = build_numeric_view(expanded, idx)
    return StreamCollectionNumericView(
        t_min=view.t_min,
        t_max=view.t_max,
        t_min_star=view.t_min_star,
        t_max_star=view.t_max_star,
        cp=view.cp,
        rcp=view.rcp,
        heat_flow=view.heat_flow,
        dt_cont=view.dt_cont,
        is_hot=view.is_hot,
        is_cold=view.is_cold,
        active=view.active,
        parent_index=np.asarray(parent_index, dtype=int),
        segment_index=np.asarray(segment_index, dtype=int),
        parent_name=np.asarray(parent_name, dtype=object),
        parent_key=np.asarray(parent_key, dtype=object),
        period_index=np.full(
            len(expanded),
            0 if idx is None else int(idx),
            dtype=int,
        ),
    )


def value_at_idx(value: Value | None, idx: int | None = None) -> float:
    """Return one canonical numeric magnitude from a scalar or multiperiod value."""
    if value is None:
        return np.nan
    magnitudes = value._quantity.magnitude
    if magnitudes.size == 1:
        return float(magnitudes[0])
    period_idx = 0 if idx is None else int(idx)
    return float(magnitudes[period_idx])
