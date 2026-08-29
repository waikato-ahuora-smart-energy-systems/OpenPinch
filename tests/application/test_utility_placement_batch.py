"""Ordered workspace utility-placement batch tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from OpenPinch import PinchWorkspace
from OpenPinch.application._problem.accessors.target import _TargetAccessor


def test_workspace_batch_mirrors_placement_and_preserves_active_case(
    monkeypatch,
) -> None:
    workspace = PinchWorkspace(source="chocolate_factory.json")
    workspace.scenario("scenario", activate=False)
    active = workspace.active_case_name

    calls = 0

    def fake(self, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("case failure")
        return self._problem.project_name

    monkeypatch.setattr(_TargetAccessor, "utility_placement", fake)
    batch = workspace.cases(("baseline", "scenario")).target.utility_placement(
        isothermal=2
    )

    assert tuple(batch.results) == ("baseline",)
    assert tuple(batch.errors) == ("scenario",)
    assert workspace.active_case_name == active


def test_workspace_batch_all_periods_uses_shared_placement_surface(monkeypatch) -> None:
    workspace = PinchWorkspace(source="chocolate_factory.json")
    seen = []

    def fake(self, **kwargs):
        seen.append(kwargs)
        return tuple(self._problem.period_ids)

    monkeypatch.setattr(_TargetAccessor, "utility_placement", fake)
    batch = workspace.cases(("baseline",)).target.all_periods.utility_placement(
        isothermal=2
    )
    assert batch.results["baseline"] == ("0",)
    assert seen[0]["period_ids"] == ("0",)


def test_workspace_batch_forwards_maximum_duties_unchanged(monkeypatch) -> None:
    workspace = PinchWorkspace(source="chocolate_factory.json")
    maximum_duties = {
        "hot_iso_1": {
            "values": [10.0],
            "period_ids": ["0"],
            "unit": "kW",
        }
    }
    seen = []

    def fake(self, **kwargs):
        seen.append(kwargs)
        return self._problem.project_name

    monkeypatch.setattr(_TargetAccessor, "utility_placement", fake)
    batch = workspace.cases(("baseline",)).target.all_periods.utility_placement(
        isothermal=2,
        maximum_duties=maximum_duties,
    )

    assert tuple(batch.results) == ("baseline",)
    assert seen == [
        {
            "isothermal": 2,
            "maximum_duties": maximum_duties,
            "period_ids": ("0",),
        }
    ]


@given(order=st.permutations(("baseline", "scenario", "third")))
@settings(deadline=None, max_examples=6)
def test_workspace_batch_preserves_generated_case_order(order) -> None:
    workspace = PinchWorkspace(source="chocolate_factory.json")
    workspace.scenario("scenario", activate=False)
    workspace.scenario("third", activate=False)
    active = workspace.active_case_name
    sources = {
        name: workspace.case(name).to_problem_json() for name in workspace.list_cases()
    }

    with patch.object(
        _TargetAccessor,
        "utility_placement",
        lambda self, **_kwargs: self._problem.project_name,
    ):
        outcome = workspace.cases(order).target.utility_placement(
            isothermal=2
        )

    assert tuple(outcome.results) == tuple(order)
    assert workspace.active_case_name == active
    assert {
        name: workspace.case(name).to_problem_json() for name in workspace.list_cases()
    } == sources


@given(period_ids=st.permutations(("winter", "summer", "shoulder")))
def test_all_period_placement_preserves_canonical_period_order(period_ids) -> None:
    problem = SimpleNamespace(
        period_ids={period_id: index for index, period_id in enumerate(period_ids)}
    )
    target = _TargetAccessor(problem)
    seen = {}

    def fake(self, **kwargs):
        seen.update(kwargs)
        return kwargs["period_ids"]

    with patch.object(_TargetAccessor, "utility_placement", fake):
        result = target.all_periods.utility_placement(isothermal=2)

    assert result == tuple(period_ids)
    assert seen["period_ids"] == tuple(period_ids)
