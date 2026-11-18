"""Split a level-holding pinch side at its utility pinches.

With each utility level held at its target, the side's streams and levels
balance, and the heat cascade can close at temperatures between the process
pinch and the side's far end: utility pinches. No heat may cross one, so
each region between them is designed on its own, from its end nearer the
process pinch, and the regions' designs are stacked, nearest the pinch
first.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .side_streams import PINCH_TOLERANCE, PinchSide, SideStream

# The cascade closes where the heat crossing is within this fraction of the
# side's load: targeting rounds each level to about this.
CASCADE_TOLERANCE = 1e-6


def utility_pinch_regions(side: PinchSide) -> list[PinchSide]:
    """A single-period side split at its utility pinches, nearest first.

    A side without utility levels among its streams, or whose cascade does
    not close between its ends, is one region: itself.
    """

    if side.periods != 1 or not _has_levels(side):
        return [side]
    streams = [s for s in side.hot + side.cold if s.active(0)]
    if not streams or any(not s.contributions for s in streams):
        return [side]
    cuts = _utility_pinches(side, streams)
    if not cuts:
        return [side]
    bounds = [None, *cuts, None]
    regions = []
    for near, far in zip(bounds[:-1], bounds[1:]):
        region = _region(side, streams, near, far)
        if region.hot or region.cold:
            regions.append(region)
    return regions


def _has_levels(side: PinchSide) -> bool:
    return any(s.index >= side.hot_count for s in side.hot) or any(
        s.index >= side.cold_count for s in side.cold
    )


def _away(side: PinchSide, stream: SideStream, temperature: float) -> float:
    """A stream's heat beyond ``temperature`` (shifted), away from the pinch."""

    shifted = np.asarray(stream.shifted(0), dtype=float)
    heats = np.asarray(stream.heats[0], dtype=float)
    # Shifted temperatures rise away from the pinch above it, fall below it.
    if side.location == "above":
        from_pinch = float(np.interp(temperature, shifted, heats))
    else:
        from_pinch = float(np.interp(-temperature, -shifted, heats))
    return stream.loads[0] - from_pinch


def _crossing(side: PinchSide, streams, temperature: float) -> float:
    """The heat the cascade carries across ``temperature`` toward the pinch.

    Above the pinch, hot heat beyond it flows down; below the pinch, the
    cold beyond it draws heat down across it.
    """

    hot = sum(_away(side, s, temperature) for s in streams if s.kind == "hot")
    cold = sum(_away(side, s, temperature) for s in streams if s.kind == "cold")
    return hot - cold if side.location == "above" else cold - hot


def _utility_pinches(side: PinchSide, streams) -> list[float]:
    """Interior shifted temperatures where the cascade closes, nearest first."""

    above = side.location == "above"
    points = sorted(
        {t for s in streams for t in s.shifted(0)}, key=lambda t: t if above else -t
    )
    total = sum(s.loads[0] for s in streams)
    tolerance = CASCADE_TOLERANCE * max(total, 1.0)
    cuts: list[float] = []
    closed = False
    for temperature in points[1:-1]:
        near = sum(s.loads[0] - _away(side, s, temperature) for s in streams)
        away = sum(_away(side, s, temperature) for s in streams)
        if near <= tolerance or away <= tolerance:
            closed = False
            continue
        if _crossing(side, streams, temperature) <= tolerance:
            # One cut per closed stretch: no stream runs within it.
            if not closed:
                cuts.append(temperature)
            closed = True
        else:
            closed = False
    return cuts


def _region(side: PinchSide, streams, near, far) -> PinchSide:
    """The side between two cuts (``None``: the pinch or the far end)."""

    hot, cold = [], []
    for stream in streams:
        piece = _slice(side, stream, near, far)
        if piece is not None:
            (hot if piece.kind == "hot" else cold).append(piece)
    return replace(side, hot=tuple(hot), cold=tuple(cold))


def _slice(side: PinchSide, stream: SideStream, near, far) -> SideStream | None:
    """The part of a stream between two cuts, its heat counted from ``near``."""

    load = stream.loads[0]
    start = 0.0 if near is None else load - _away(side, stream, near)
    end = load if far is None else load - _away(side, stream, far)
    if end - start <= max(1e-9, 1e-9 * load):
        return None
    heats = np.asarray(stream.heats[0], dtype=float)
    temperatures = np.asarray(stream.temperatures[0], dtype=float)
    inner = [q for q in heats if start < q < end]
    points = [start, *inner, end]
    sliced = [float(np.interp(q, heats, temperatures)) for q in points]
    span = abs(sliced[1] - sliced[0])
    cp = (points[1] - points[0]) / span if span > 0.0 else 0.0
    sign = -1.0 if stream.kind == "hot" else 1.0
    pinch = None if near is None else near - sign * stream.contributions[0]
    at_pinch = (
        stream.at_pinch[0]
        if near is None
        else abs(sliced[0] - pinch) <= PINCH_TOLERANCE
    )
    return replace(
        stream,
        loads=(end - start,),
        pinch_cps=(cp,),
        at_pinch=(at_pinch,),
        heats=(tuple(q - start for q in points),),
        temperatures=(tuple(sliced),),
    )


__all__ = ["CASCADE_TOLERANCE", "utility_pinch_regions"]
