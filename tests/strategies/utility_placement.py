"""Reusable Hypothesis strategies for utility-placement contracts."""

from __future__ import annotations

from hypothesis import strategies as st

from OpenPinch.contracts.utility_placement import (
    QuantityInterval,
    QuantityValue,
    UtilityLevelKind,
    UtilityLevelTemplate,
    UtilityPlacementRequest,
    UtilitySide,
)

finite_magnitudes = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
)
valid_isothermal_counts = st.integers(min_value=2, max_value=8)
valid_sensible_counts = st.integers(min_value=0, max_value=6)
invalid_counts = st.one_of(
    st.booleans(),
    st.integers(max_value=-1),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
)


@st.composite
def quantity_intervals(draw, *, unit: str = "degC"):
    """Generate finite ordered intervals, including fixed coordinates."""
    lower = draw(finite_magnitudes)
    width = draw(
        st.floats(
            min_value=0.0,
            max_value=500.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return QuantityInterval(lower=lower, upper=lower + width, unit=unit)


@st.composite
def isothermal_templates(draw, *, side: UtilitySide = UtilitySide.HOT):
    """Generate one valid isothermal template."""
    name = draw(st.from_regex(r"[a-z][a-z0-9_]{0,12}", fullmatch=True))
    span = draw(
        st.floats(
            min_value=0.001,
            max_value=50.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return UtilityLevelTemplate(
        name=name,
        side=side,
        kind=UtilityLevelKind.ISOTHERMAL,
        fixed_span=QuantityValue(value=span, unit="delta_degC"),
    )


@st.composite
def count_only_requests(draw):
    """Generate valid thermodynamic requests that use template generation."""
    return UtilityPlacementRequest(
        isothermal_level_count=draw(valid_isothermal_counts),
        sensible_level_count=draw(valid_sensible_counts),
    )


@st.composite
def residual_profile_envelopes(draw):
    """Generate aligned descending temperatures and finite residual profiles."""
    temperatures = tuple(
        float(value)
        for value in sorted(
            draw(
                st.lists(
                    st.integers(min_value=-100, max_value=400),
                    min_size=3,
                    max_size=10,
                    unique=True,
                )
            ),
            reverse=True,
        )
    )
    profile_strategy = st.lists(
        st.integers(min_value=0, max_value=10_000),
        min_size=len(temperatures),
        max_size=len(temperatures),
    ).map(lambda values: tuple(float(value) for value in values))
    return temperatures, draw(profile_strategy), draw(profile_strategy)


@st.composite
def duty_split_cases(draw):
    """Generate one bounded utility-family dispatch case."""
    level_count = draw(st.integers(min_value=2, max_value=8))
    fractions = draw(
        st.lists(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=level_count - 1,
            max_size=level_count - 1,
        )
    )
    total_duty = draw(
        st.floats(
            min_value=0.001,
            max_value=100_000.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return tuple(fractions), total_duty


__all__ = [
    "count_only_requests",
    "duty_split_cases",
    "finite_magnitudes",
    "invalid_counts",
    "isothermal_templates",
    "quantity_intervals",
    "residual_profile_envelopes",
    "valid_isothermal_counts",
    "valid_sensible_counts",
]
