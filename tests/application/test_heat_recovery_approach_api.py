"""Closed public signatures for inverse heat-recovery targeting."""

from __future__ import annotations

import inspect

from OpenPinch import PinchProblem, PinchWorkspace
from OpenPinch.application.workspace import (
    _CaseBatchAllPeriodsTargetAccessor,
    _CaseBatchTargetAccessor,
)


def _signature_names(callable_) -> tuple[str, ...]:
    return tuple(inspect.signature(callable_).parameters)


def test_selected_period_signature_is_exact() -> None:
    method = PinchProblem().target.heat_recovery_approach_temperature
    parameters = inspect.signature(method).parameters

    assert _signature_names(method) == ("heat_recovery", "zone", "period_id")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )
    assert parameters["zone"].default is None
    assert parameters["period_id"].default is None


def test_all_period_signature_is_exact() -> None:
    method = PinchProblem().target.all_periods.heat_recovery_approach_temperature
    parameters = inspect.signature(method).parameters

    assert _signature_names(method) == ("heat_recovery", "zone", "workers")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )
    assert parameters["zone"].default is None
    assert parameters["workers"].default == 1


def test_workspace_delegates_the_exact_active_case_signatures() -> None:
    workspace = PinchWorkspace("basic_pinch.json", project_name="Site")

    assert inspect.signature(
        workspace.target.heat_recovery_approach_temperature
    ) == inspect.signature(workspace.case().target.heat_recovery_approach_temperature)
    assert inspect.signature(
        workspace.target.all_periods.heat_recovery_approach_temperature
    ) == inspect.signature(
        workspace.case().target.all_periods.heat_recovery_approach_temperature
    )


def test_batch_signatures_are_explicit() -> None:
    selected = inspect.signature(
        _CaseBatchTargetAccessor.heat_recovery_approach_temperature
    ).parameters
    all_periods = inspect.signature(
        _CaseBatchAllPeriodsTargetAccessor.heat_recovery_approach_temperature
    ).parameters

    assert tuple(selected) == ("self", "heat_recovery", "zone", "period_id")
    assert tuple(all_periods) == ("self", "heat_recovery", "zone", "workers")
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in selected.items()
        if name != "self"
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in all_periods.items()
        if name != "self"
    )
