"""Domain-specific strategies for external input contracts."""

from __future__ import annotations

from hypothesis import strategies as st


@st.composite
def target_input_payloads(draw):
    """Generate one thermally valid process-stream input."""
    is_hot = draw(st.booleans())
    low = draw(st.floats(min_value=-50.0, max_value=250.0, allow_nan=False))
    span = draw(st.floats(min_value=0.1, max_value=300.0, allow_nan=False))
    supply, target = (low + span, low) if is_hot else (low, low + span)
    heat_flow = draw(
        st.floats(
            min_value=0.1,
            max_value=1_000_000.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return {
        "streams": [
            {
                "zone": "Process",
                "name": "Generated stream",
                "t_supply": supply,
                "t_target": target,
                "heat_flow": heat_flow,
            }
        ],
        "utilities": [],
    }


__all__ = ["target_input_payloads"]
