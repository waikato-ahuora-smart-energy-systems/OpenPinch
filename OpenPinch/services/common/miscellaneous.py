"""Shared numerical helpers."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ...lib.config import tol

__all__ = [
    "delta_vals",
    "delta_with_zero_at_start",
    "g_ineq_penalty",
    "get_state_index",
    "interp_with_plateaus",
    "linear_interpolation",
    "make_monotonic",
]


def get_state_index(
    state_ids: dict[str, int] | None,
    args: dict | None,
) -> Tuple[int, str | None]:
    sid = None if not isinstance(args, dict) else args.get("state_id")
    sid = None if sid is None else str(sid)
    raw_idx = None if not isinstance(args, dict) else args.get("idx")
    explicit_idx = None if raw_idx is None else int(raw_idx)

    lookup = {} if state_ids is None else state_ids

    if sid is not None:
        if lookup and sid not in lookup:
            raise ValueError(
                f"state_id {sid!r} was not found on this collection. "
                f"Available states: {', '.join(lookup)}."
            )
        resolved_idx = lookup.get(sid, 0)
        if explicit_idx is not None and explicit_idx != resolved_idx:
            raise ValueError(
                f"state_id {sid!r} resolves to idx {resolved_idx}, "
                f"but idx {explicit_idx} was also provided."
            )
        return resolved_idx, sid

    if explicit_idx is not None:
        if explicit_idx < 0:
            raise ValueError("idx must be a non-negative integer.")
        if lookup and explicit_idx not in set(lookup.values()):
            raise ValueError(
                f"idx {explicit_idx} was not found on this collection. "
                f"Available indices: {', '.join(str(idx) for idx in lookup.values())}."
            )
        return explicit_idx, None

    return 0, None


def linear_interpolation(
    xi: float, x1: float, x2: float, y1: float, y2: float
) -> float:
    """Estimate ``y`` at ``xi`` using two known points and linear interpolation."""
    if x1 == x2:
        raise ValueError(
            "Cannot perform interpolation when x1 == x2 (undefined slope)."
        )
    m = (y1 - y2) / (x1 - x2)
    c = y1 - m * x1
    yi = m * xi + c
    return yi


def delta_with_zero_at_start(x: np.ndarray) -> np.ndarray:
    """Compute successive differences and prepend a zero entry."""
    return np.insert(delta_vals(x), 0, 0.0)


def delta_vals(x: np.ndarray, descending_vals: bool = True) -> np.ndarray:
    """Compute difference between successive entries in a column."""
    deltas = x[:-1] - x[1:] if descending_vals else x[1:] - x[:-1]
    deltas[np.abs(deltas) <= tol] = 0.0
    return deltas


def interp_with_plateaus(
    h_vals: np.ndarray,
    t_vals: np.ndarray,
    targets: np.ndarray,
    side: str,
    tol: float = 1e-6,
) -> np.ndarray:
    """Interpolate temperatures while respecting vertical curve segments."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    h_vals = np.asarray(h_vals, dtype=float)
    t_vals = np.asarray(t_vals, dtype=float)
    targets = np.asarray(targets, dtype=float)

    if h_vals.size == 1:
        return np.full_like(targets, t_vals[0], dtype=float)

    h_monotonic = make_monotonic(h_vals, side, tol)
    return np.interp(targets, h_monotonic, t_vals)


def make_monotonic(h_vals: np.ndarray, side: str, tol: float = 1e-6) -> np.ndarray:
    """Adjust repeated values to become strictly increasing for interpolation."""
    adjusted = np.asarray(h_vals, dtype=float).copy()
    if adjusted.size <= 1:
        return adjusted

    eps = tol * 0.5
    # Identify the start of each strictly increasing block
    diff = np.abs(np.diff(adjusted)) > tol
    starts = np.flatnonzero(np.concatenate(([True], diff)))
    n = adjusted.size
    lengths = np.diff(np.append(starts, n))

    if np.all(lengths == 1):
        return adjusted

    # Compute position within each block using vectorised repetition
    within_block = np.arange(n) - np.repeat(starts, lengths)
    block_lengths = np.repeat(lengths, lengths)
    mask = block_lengths > 1

    offsets = np.zeros_like(adjusted)
    if side == "right":
        offsets[mask] = (block_lengths[mask] - 1 - within_block[mask]) * eps
        adjusted[mask] -= offsets[mask]
    else:  # side == "left"
        offsets[mask] = within_block[mask] * eps
        adjusted[mask] += offsets[mask]

    return adjusted


def g_ineq_penalty(
    g: float | list | np.ndarray,
    *,
    eta: float = 0.01,
    rho: float = 10,
    form: str = "square",
) -> np.float64:
    """Return a penalty value for an inequality-constraint residual."""
    g = np.asarray(g, dtype=float)
    if (
        form.lower() == "square_root_smoothing"
        or form.lower() == "square root smoothing"
    ):
        p = 0.5 * rho * (g + ((g) ** 2 + (eta) ** 2) ** 0.5)
    elif form.lower() == "square":
        p = rho * (g**2)
    else:
        raise ValueError("Unrecognised penalty function form selection.")

    if isinstance(p, float):
        return np.float64(p)
    elif isinstance(p, np.ndarray):
        return p.sum()
    else:
        raise ValueError(
            "Return of the penalty function failed due to unrecognised type."
        )
