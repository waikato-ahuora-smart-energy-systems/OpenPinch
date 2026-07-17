from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hypothesis import strategies as st

from OpenPinch.domain.configuration import tol


@dataclass(frozen=True)
class ZeroDutyStreamSideCase:
    """Generated zero-duty process side with a near-degenerate property profile."""

    is_condenser: bool
    duty: float
    mass_flow: float
    profile: tuple[tuple[float, float], tuple[float, float]]


@st.composite
def zero_duty_stream_side_cases(draw) -> ZeroDutyStreamSideCase:
    """Generate finite zero-duty condenser and evaporator profile cases."""
    is_condenser = draw(st.booleans())
    duty = draw(
        st.floats(
            min_value=-tol,
            max_value=tol,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    mass_flow = draw(
        st.floats(
            min_value=1e-6,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    enthalpy = draw(
        st.floats(
            min_value=1e4,
            max_value=1e6,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    temperature = draw(
        st.floats(
            min_value=-100.0,
            max_value=350.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    temperature_span = draw(
        st.floats(
            min_value=0.01,
            max_value=50.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    target_temperature = (
        temperature - temperature_span
        if is_condenser
        else temperature + temperature_span
    )
    profile = (
        (enthalpy, temperature),
        (float(np.nextafter(enthalpy, np.inf)), target_temperature),
    )
    return ZeroDutyStreamSideCase(
        is_condenser=is_condenser,
        duty=duty,
        mass_flow=mass_flow,
        profile=profile,
    )
