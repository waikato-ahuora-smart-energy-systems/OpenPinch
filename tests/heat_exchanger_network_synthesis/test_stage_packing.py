"""Stage-packing constraint tests using static topology fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from OpenPinch.services.heat_exchanger_network_synthesis.unit_models import (
    packed_pinch_design,
    packed_stagewise,
)
from OpenPinch.services.heat_exchanger_network_synthesis.unit_models.stage_packing import (
    add_recovery_stage_packing_constraints,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "synthesis_stage_packing_cases.json"
)


class _FakeExpression:
    def __init__(self, name: str, payload=None):
        self.name = name
        self.payload = payload

    def __mul__(self, other):
        return ("mul", self, other)

    def __rmul__(self, other):
        return ("mul", other, self)

    def __le__(self, other):
        return ("le", self, other)

    def __ge__(self, other):
        return ("ge", self, other)


class _FakeGekkoLikeModel:
    def __init__(self):
        self.params = []
        self.variables = []
        self.intermediates = []
        self.equations = []

    def Param(self, *, value, name: str):
        param = _FakeExpression(name, value)
        self.params.append(param)
        return param

    def Var(self, *, value, ub, lb, integer, name: str):
        variable = _FakeExpression(
            name,
            {"value": value, "ub": ub, "lb": lb, "integer": integer},
        )
        self.variables.append(variable)
        return variable

    def Intermediate(self, expression, *, name: str):
        intermediate = _FakeExpression(name, expression)
        self.intermediates.append(intermediate)
        return intermediate

    def Equation(self, expression):
        self.equations.append(expression)
        return expression

    def sum(self, expressions):
        return ("sum", tuple(expressions))


def _load_stage_packing_case(name: str) -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))[name]


def _fake_model_from_case(case: dict):
    q_max = case["q_max"]
    z_allowed = case["z_allowed"]
    stage_count = len(z_allowed[0][0])
    stream_rows = range(len(q_max))
    stream_cols = range(len(q_max[0]))
    return SimpleNamespace(
        integers=case["integers"],
        tol=case["tol"],
        I=len(q_max),
        J=len(q_max[0]),
        S=stage_count,
        Q_max=q_max,
        z_allowed=z_allowed,
        Q_r=[
            [
                [_FakeExpression(f"Q_r_{i}_{j}_{k}") for k in range(stage_count)]
                for j in stream_cols
            ]
            for i in stream_rows
        ],
        z=[
            [
                [_FakeExpression(f"z_{i}_{j}_{k}") for k in range(stage_count)]
                for j in stream_cols
            ]
            for i in stream_rows
        ],
        m=_FakeGekkoLikeModel(),
    )


@pytest.mark.parametrize(
    "model",
    [
        SimpleNamespace(integers=False, S=3),
        SimpleNamespace(integers=True, S=1),
    ],
)
def test_stage_packing_skips_non_integer_or_single_stage_models(model):
    add_recovery_stage_packing_constraints(model)

    assert not hasattr(model, "recovery_stage_active")
    assert not hasattr(model, "recovery_stage_duty")


def test_stage_packing_skips_models_without_positive_stage_capacity():
    model = _fake_model_from_case(_load_stage_packing_case("zero_capacity_case"))

    add_recovery_stage_packing_constraints(model)

    assert model.m.equations == []
    assert not hasattr(model, "recovery_stage_active")
    assert not hasattr(model, "recovery_stage_duty")


def test_stage_packing_adds_contiguous_stage_constraints_from_static_fixture():
    model = _fake_model_from_case(_load_stage_packing_case("positive_capacity_case"))

    add_recovery_stage_packing_constraints(model)

    assert [expr.name for expr in model.recovery_stage_active] == [
        "recovery_stage_active_0",
        "recovery_stage_active_1",
        "recovery_stage_active_2",
    ]
    assert [expr.name for expr in model.recovery_stage_duty] == [
        "recovery_stage_duty_0",
        "recovery_stage_duty_1",
        "recovery_stage_duty_2",
    ]
    assert [expr.name for expr in model.m.variables] == ["recovery_stage_active_0"]
    assert [expr.name for expr in model.m.params] == [
        "recovery_stage_active_1",
        "recovery_stage_duty_1",
        "recovery_stage_active_2",
        "recovery_stage_duty_2",
    ]
    assert [expr.name for expr in model.m.intermediates] == ["recovery_stage_duty_0"]
    assert len(model.m.equations) == 5
    assert model.m.equations[0][0] == "ge"
    assert model.m.equations[-1][0] == "ge"


def test_stage_packed_model_wrappers_call_parent_setup_before_packing(monkeypatch):
    events = []

    monkeypatch.setattr(
        packed_pinch_design.PinchDecompModel,
        "set_stage_wise_superstructure",
        lambda self: events.append(("pinch_parent", type(self).__name__)),
    )
    monkeypatch.setattr(
        packed_stagewise.StageWiseModel,
        "set_stage_wise_superstructure",
        lambda self: events.append(("stagewise_parent", type(self).__name__)),
    )
    monkeypatch.setattr(
        packed_pinch_design,
        "add_recovery_stage_packing_constraints",
        lambda self: events.append(("pinch_packing", type(self).__name__)),
    )
    monkeypatch.setattr(
        packed_stagewise,
        "add_recovery_stage_packing_constraints",
        lambda self: events.append(("stagewise_packing", type(self).__name__)),
    )

    pinch_model = packed_pinch_design.StagePackedPinchDecompModel.__new__(
        packed_pinch_design.StagePackedPinchDecompModel
    )
    stagewise_model = packed_stagewise.StagePackedStageWiseModel.__new__(
        packed_stagewise.StagePackedStageWiseModel
    )

    pinch_model.set_stage_wise_superstructure()
    stagewise_model.set_stage_wise_superstructure()

    assert events == [
        ("pinch_parent", "StagePackedPinchDecompModel"),
        ("pinch_packing", "StagePackedPinchDecompModel"),
        ("stagewise_parent", "StagePackedStageWiseModel"),
        ("stagewise_packing", "StagePackedStageWiseModel"),
    ]
