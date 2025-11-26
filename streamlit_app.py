"""Streamlit entry point for exploring OpenPinch analysis results.

Run with ``streamlit run streamlit_app.py`` to load the bundled demonstration
case and launch the interactive dashboard defined in
``OpenPinch.analysis.graphing``.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from OpenPinch import PinchProblem

# Default case. Update this path if you want to switch datasets.
DEFAULT_PROBLEM_FILE = Path("examples/OpenPinchWkbs/pulp_mill.xlsb")


@st.cache_resource
def _load_problem(problem_path: str) -> PinchProblem:
    """Load and solve the pinch problem once per Streamlit session."""
    return PinchProblem(problem_path, run=True)


def main() -> None:
    problem_path = DEFAULT_PROBLEM_FILE
    if not problem_path.exists():
        st.error(
            f"Problem file not found at {problem_path}. "
            "Update DEFAULT_PROBLEM_FILE to point to a valid case."
        )
        st.stop()

    problem = _load_problem(str(problem_path))
    problem.render_streamlit_dashboard(
        page_title="OpenPinch Dashboard",
        value_rounding=2,
    )


if __name__ == "__main__":
    main()
