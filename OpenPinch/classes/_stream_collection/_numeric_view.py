"""Numeric snapshot helpers for :class:`StreamCollection` internals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...lib.enums import ST
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


def build_numeric_view(streams: list[object], idx: int | None = None):
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
    )


def value_at_idx(value: Value | None, idx: int | None = None) -> float:
    """Return one canonical numeric magnitude from a scalar or stateful value."""
    if value is None:
        return np.nan
    magnitudes = value._quantity.magnitude
    if magnitudes.size == 1:
        return float(magnitudes[0])
    state_idx = 0 if idx is None else int(idx)
    return float(magnitudes[state_idx])
