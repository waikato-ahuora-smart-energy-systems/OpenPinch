"""Contracts for heat pump and refrigeration load kinks."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from OpenPinch.contracts.hpr_load_kinks import (
    HprLoadKink,
    HprLoadKinkKind,
    HprLoadKinksResult,
)
from OpenPinch.contracts.schema import contract_json_schemas


def _vertex(load_fraction, kind, slope_after=10.0) -> dict:
    return {
        "load_fraction": load_fraction,
        "duty": {"value": 1_000.0 * load_fraction, "unit": "kW"},
        "temperature": {"value": 60.0, "unit": "degC"},
        "slope_before": {"value": 20.0, "unit": "delta_degC"},
        "slope_after": None
        if slope_after is None
        else {"value": slope_after, "unit": "delta_degC"},
        "kind": kind,
    }


def _result(**overrides) -> HprLoadKinksResult:
    payload = {
        "scope": "Site",
        "period_id": "0",
        "is_refrigeration": False,
        "steepening": 2.0,
        "maximum_duty": {"value": 1_000.0, "unit": "kW"},
        "vertices": [
            _vertex(0.25, "minor", 25.0),
            _vertex(0.5, "step", None),
            _vertex(0.75, "steepens", 80.0),
            _vertex(0.9, "flattens", 5.0),
        ],
        "logical_load_fractions": [0.5, 0.75, 1.0],
    }
    payload.update(overrides)
    return HprLoadKinksResult.model_validate(payload)


def test_result_is_frozen_strict_finite_and_json_round_trippable() -> None:
    result = _result()

    assert result.vertices[1].kind is HprLoadKinkKind.STEP
    assert result.vertices[1].slope_after is None
    assert json.loads(result.model_dump_json())["logical_load_fractions"] == [
        0.5,
        0.75,
        1.0,
    ]
    assert HprLoadKinksResult.model_validate_json(result.model_dump_json()) == result

    with pytest.raises(ValidationError, match="frozen"):
        result.steepening = 3.0
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _result(unknown=True)
    with pytest.raises(ValidationError, match="finite"):
        _result(maximum_duty={"value": float("inf"), "unit": "kW"})


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"scope": " "}, "must not be empty"),
        ({"steepening": 1.0}, "greater than 1"),
        ({"maximum_duty": {"value": 0.0, "unit": "kW"}}, "positive"),
        ({"maximum_duty": {"value": 1.0, "unit": "degC"}}, "compatible with kW"),
        ({"logical_load_fractions": [0.5, 1.0]}, "step and steepens"),
        (
            {"vertices": [_vertex(0.5, "step", None), _vertex(0.4, "minor")]},
            "increasing",
        ),
    ],
)
def test_result_rejects_inconsistent_payloads(overrides, match) -> None:
    with pytest.raises(ValidationError, match=match):
        _result(**overrides)


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"load_fraction": 1.0}, "between 0 and 1"),
        ({"load_fraction": 0.0}, "between 0 and 1"),
        ({"kind": "step"}, "step has no slope after"),
        ({"slope_after": None}, "only a step"),
        ({"temperature": {"value": 1.0, "unit": "kW"}}, "compatible with degC"),
    ],
)
def test_vertex_rejects_inconsistent_payloads(overrides, match) -> None:
    payload = _vertex(0.5, "minor") | overrides
    with pytest.raises(ValidationError, match=match):
        HprLoadKink.model_validate(payload)


def test_result_schema_lists_vertices_and_logical_loads() -> None:
    schema = contract_json_schemas()["HprLoadKinksResult"]

    assert {"vertices", "logical_load_fractions", "maximum_duty"} <= set(
        schema["properties"]
    )
