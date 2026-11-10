"""Closed public signatures for HPR load kinks."""

from __future__ import annotations

import inspect

import pytest

from OpenPinch import PinchProblem, PinchWorkspace
from OpenPinch.application.workspace import (
    _CaseBatchAllPeriodsTargetAccessor,
    _CaseBatchTargetAccessor,
)

_SELECTED = ("zone", "period_id", "is_refrigeration", "steepening")
_ALL_PERIODS = ("zone", "is_refrigeration", "steepening", "workers")
_DEFAULTS = {
    "zone": None,
    "period_id": None,
    "is_refrigeration": False,
    "steepening": 2.0,
    "workers": 1,
}


@pytest.mark.parametrize(
    "method,names",
    [
        (PinchProblem().target.hpr_load_kinks, _SELECTED),
        (PinchProblem().target.all_periods.hpr_load_kinks, _ALL_PERIODS),
        (_CaseBatchTargetAccessor.hpr_load_kinks, ("self",) + _SELECTED),
        (_CaseBatchAllPeriodsTargetAccessor.hpr_load_kinks, ("self",) + _ALL_PERIODS),
    ],
)
def test_signatures_are_keyword_only_with_documented_defaults(method, names) -> None:
    parameters = inspect.signature(method).parameters

    assert tuple(parameters) == names
    for name, parameter in parameters.items():
        if name == "self":
            continue
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == _DEFAULTS[name]


def test_workspace_delegates_the_exact_active_case_signatures() -> None:
    workspace = PinchWorkspace("basic_pinch.json", project_name="Site")

    assert inspect.signature(workspace.target.hpr_load_kinks) == inspect.signature(
        workspace.case().target.hpr_load_kinks
    )
    assert inspect.signature(
        workspace.target.all_periods.hpr_load_kinks
    ) == inspect.signature(workspace.case().target.all_periods.hpr_load_kinks)
