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
    return StreamCollectionNumericView(
        t_min=np.asarray(
            [value_at_idx(stream._t_min, idx) for stream in streams],
            dtype=float,
        ),
        t_max=np.asarray(
            [value_at_idx(stream._t_max, idx) for stream in streams],
            dtype=float,
        ),
        t_min_star=np.asarray(
            [value_at_idx(stream._t_min_star, idx) for stream in streams],
            dtype=float,
        ),
        t_max_star=np.asarray(
            [value_at_idx(stream._t_max_star, idx) for stream in streams],
            dtype=float,
        ),
        cp=np.asarray(
            [value_at_idx(stream._cp, idx) for stream in streams],
            dtype=float,
        ),
        rcp=np.asarray(
            [value_at_idx(stream._rcp_prod, idx) for stream in streams],
            dtype=float,
        ),
        heat_flow=np.asarray(
            [value_at_idx(stream._heat_flow, idx) for stream in streams],
            dtype=float,
        ),
        dt_cont=np.asarray(
            [value_at_idx(stream._dt_cont, idx) for stream in streams],
            dtype=float,
        ),
        is_hot=np.asarray(
            [stream.type == ST.Hot.value for stream in streams],
            dtype=bool,
        ),
        is_cold=np.asarray(
            [stream.type == ST.Cold.value for stream in streams],
            dtype=bool,
        ),
        active=np.asarray([stream.active for stream in streams], dtype=bool),
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
