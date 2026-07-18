"""Generated ordering properties for the public workspace case batch."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from OpenPinch import PinchProblem, PinchWorkspace


@settings(max_examples=12, deadline=None)
@given(
    st.lists(
        st.integers(min_value=0, max_value=100).map(lambda value: f"case_{value}"),
        min_size=1,
        max_size=4,
        unique=True,
    )
)
def test_case_batch_preserves_requested_order_without_mutating_case_order(
    case_names,
):
    workspace = PinchWorkspace("basic_pinch.json", project_name="Site")
    for name in case_names:
        workspace.scenario(name)
    stored_order = workspace.list_cases()

    requested = list(reversed(case_names))
    with patch.object(
        PinchProblem,
        "_execute_targeting",
        lambda self, **kwargs: SimpleNamespace(name=self.project_name),
    ):
        outcome = workspace.cases(requested).target.direct_heat_integration()

    assert list(outcome.results) == requested
    assert not outcome.errors
    assert workspace.list_cases() == stored_order
