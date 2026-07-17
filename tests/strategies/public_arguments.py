"""Generated precedence cases for public workflow arguments."""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import strategies as st

from OpenPinch.application._problem.arguments import MISSING


@dataclass(frozen=True)
class ArgumentPrecedenceCase:
    named: object
    options: object
    config: object
    default: object


PUBLIC_ARGUMENT_VALUES = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=0, max_value=10_000),
    st.text(min_size=0, max_size=12),
)


@st.composite
def argument_precedence_cases(draw) -> ArgumentPrecedenceCase:
    """Generate values at every precedence layer, including explicit falsey values."""
    optional_value = st.one_of(st.just(MISSING), PUBLIC_ARGUMENT_VALUES)
    return ArgumentPrecedenceCase(
        named=draw(optional_value),
        options=draw(optional_value),
        config=draw(optional_value),
        default=draw(PUBLIC_ARGUMENT_VALUES),
    )


__all__ = ["ArgumentPrecedenceCase", "argument_precedence_cases"]
