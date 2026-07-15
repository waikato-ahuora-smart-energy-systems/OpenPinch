from __future__ import annotations

from hypothesis import strategies as st

from OpenPinch import StreamSegment
from OpenPinch.classes import Stream


@st.composite
def segmented_streams(draw) -> Stream:
    """Generate valid ordered scalar hot or cold segmented streams."""
    segment_count = draw(st.integers(min_value=1, max_value=5))
    is_hot = draw(st.booleans())
    spans = draw(
        st.lists(
            st.floats(
                min_value=1.0,
                max_value=80.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=segment_count,
            max_size=segment_count,
        )
    )
    duties = draw(
        st.lists(
            st.floats(
                min_value=0.1,
                max_value=1000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=segment_count,
            max_size=segment_count,
        )
    )
    start = draw(
        st.floats(
            min_value=100.0 if is_hot else -50.0,
            max_value=500.0 if is_hot else 200.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    segments = []
    current = start
    for index, (span, duty) in enumerate(zip(spans, duties)):
        target = current - span if is_hot else current + span
        segments.append(
            StreamSegment(
                name=f"S{index + 1}",
                t_supply=current,
                t_target=target,
                heat_flow=duty,
            )
        )
        current = target
    return Stream(name="Generated", segments=segments)
