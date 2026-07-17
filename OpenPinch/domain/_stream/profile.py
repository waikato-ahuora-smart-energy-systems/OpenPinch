"""Stateless ordered-profile helpers for segmented thermal streams."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..value import Value


@dataclass(frozen=True)
class SegmentSpec:
    """Construction data for one segment produced from a heat profile."""

    name: str
    t_supply: float
    t_target: float
    heat_flow: Value
    segment_index: int


@dataclass(frozen=True)
class SegmentAggregateState:
    """Aggregate core values calculated from an ordered segment sequence."""

    t_supply: Value | None
    t_target: Value | None
    p_supply: Value | None
    p_target: Value | None
    h_supply: Value | None
    h_target: Value | None
    dt_cont: Value | None
    heat_flow: np.ndarray
    effective_htc: np.ndarray
    price: Value | None


def temperature_heat_segment_specs(
    *,
    name: str,
    points,
    heat_scale: float,
    heat_unit: str,
    dt_diff_max: float | None,
    tolerance: float,
) -> tuple[SegmentSpec, ...]:
    """Validate ordered heat-profile points and return segment construction data."""
    profile = np.asarray(points, dtype=float)
    if profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 2:
        raise ValueError(
            "A temperature-heat profile requires at least two [H, T] points."
        )
    if not np.isfinite(profile).all():
        raise ValueError("Temperature-heat profile points must be finite.")
    if not np.isfinite(heat_scale) or heat_scale <= 0.0:
        raise ValueError("heat_scale must be finite and positive.")
    heat_steps = np.diff(profile[:, 0])
    if np.any(np.abs(heat_steps) <= tolerance) or not np.all(
        np.sign(heat_steps) == np.sign(heat_steps[0])
    ):
        raise ValueError(
            "Temperature-heat profile coordinates must be strictly monotonic."
        )
    is_hot = bool(profile[0, 1] > profile[-1, 1])
    if dt_diff_max is not None:
        from .linearisation import get_piecewise_data_points

        profile = get_piecewise_data_points(
            curve=profile,
            is_hot_stream=is_hot,
            dt_diff_max=float(dt_diff_max),
        )

    return tuple(
        SegmentSpec(
            name=f"{name}.S{index + 1}",
            t_supply=float(start[1]),
            t_target=float(end[1]),
            heat_flow=Value(
                abs(float(end[0] - start[0])) * float(heat_scale),
                heat_unit,
            ),
            segment_index=index,
        )
        for index, (start, end) in enumerate(zip(profile[:-1], profile[1:]))
    )


def detached_segment(segment, *, segment_type: type):
    """Clone a segment without carrying its parent ownership reference."""
    if not isinstance(segment, segment_type):
        raise TypeError("Segmented streams require StreamSegment children.")
    clone = object.__new__(segment_type)
    clone.__dict__ = {
        key: deepcopy(value)
        for key, value in segment.__dict__.items()
        if key != "_owner"
    }
    clone._owner = None
    return clone


def validate_segments(
    segments: Sequence[Any],
    *,
    parent_num_periods: int | None,
    tolerance: float,
) -> None:
    """Validate an ordered segment profile across every operating period."""
    if not segments:
        raise ValueError("A segmented stream requires at least one segment.")
    period_sizes = [segment._period_vector_size() for segment in segments]
    state_size = max([parent_num_periods or 1, *period_sizes])
    if any(size not in {1, state_size} for size in period_sizes):
        raise ValueError(
            "All segment values must use one shared operating-period count."
        )
    direction: int | None = None
    for index, segment in enumerate(segments):
        supply = segment._value_array(segment._t_supply, size=state_size)
        target = segment._value_array(segment._t_target, size=state_size)
        duty = segment._value_array(segment._heat_flow, size=state_size)
        htc = segment._value_array(segment._htc, size=state_size)
        if not np.isfinite(supply).all() or not np.isfinite(target).all():
            raise ValueError(f"Segment {index + 1} temperatures must be finite.")
        if not np.isfinite(duty).all() or np.any(duty <= tolerance):
            raise ValueError(
                f"Segment {index + 1} heat flow must be positive and finite."
            )
        if not np.isfinite(htc).all() or np.any(htc <= tolerance):
            raise ValueError(f"Segment {index + 1} HTC must be positive and finite.")
        delta = target - supply
        if np.any(np.abs(delta) <= tolerance):
            raise ValueError(
                f"Segment {index + 1} must span a non-zero temperature range."
            )
        signs = np.sign(delta)
        if not np.all(signs == signs[0]):
            raise ValueError(
                f"Segment {index + 1} changes thermal direction across periods."
            )
        segment_direction = int(signs[0])
        if direction is None:
            direction = segment_direction
        elif segment_direction != direction:
            raise ValueError("All segments must have the same hot or cold direction.")
        if index:
            previous_target = segments[index - 1]._value_array(
                segments[index - 1]._t_target,
                size=state_size,
            )
            if not np.allclose(
                previous_target,
                supply,
                atol=tolerance,
                rtol=0.0,
            ):
                raise ValueError(
                    f"Segment {index} target temperature must match segment "
                    f"{index + 1} supply temperature in every period."
                )


def aggregate_segments(
    segments: Sequence[Any],
    *,
    parent_num_periods: int | None,
) -> SegmentAggregateState:
    """Calculate parent core values from an already validated segment profile."""
    state_size = max(
        parent_num_periods or 1,
        *(segment._period_vector_size() for segment in segments),
    )
    first = segments[0]
    last = segments[-1]
    duties = np.asarray(
        [
            segment._value_array(segment._heat_flow, size=state_size)
            for segment in segments
        ]
    )
    htcs = np.asarray(
        [segment._value_array(segment._htc, size=state_size) for segment in segments]
    )
    duty = np.sum(duties, axis=0)
    resistance = np.sum(
        np.divide(duties, htcs, out=np.zeros_like(duties), where=htcs > 0.0),
        axis=0,
    )
    effective_htc = np.divide(
        duty,
        resistance,
        out=np.zeros_like(duty),
        where=resistance > 0.0,
    )
    prices = np.asarray(
        [segment._value_array(segment._price, size=state_size) for segment in segments]
    )
    effective_price = np.divide(
        np.sum(prices * duties, axis=0),
        duty,
        out=np.zeros_like(duty),
        where=duty > 0.0,
    )
    return SegmentAggregateState(
        t_supply=first._t_supply,
        t_target=last._t_target,
        p_supply=first._p_supply,
        p_target=last._p_target,
        h_supply=first._h_supply,
        h_target=last._h_target,
        dt_cont=first._dt_cont,
        heat_flow=duty,
        effective_htc=effective_htc,
        price=Value(effective_price, unit="$/MW/h"),
    )


__all__: list[str] = []
