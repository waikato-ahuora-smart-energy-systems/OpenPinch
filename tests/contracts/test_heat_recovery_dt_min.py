"""Contracts for inverse heat-recovery dt_min targeting."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from OpenPinch.contracts.heat_recovery_dt_min import (
    HeatRecoveryDtMinResult,
    HeatRecoveryDtMinStatus,
    HeatRecoveryQuantity,
)


def _result(**overrides) -> HeatRecoveryDtMinResult:
    payload = {
        "scope": "Site/Process",
        "period_id": "0",
        "dt_min": {"value": 10.0, "unit": "delta_degC"},
        "requested_heat_recovery": {"value": 4_000.0, "unit": "kW"},
        "achieved_heat_recovery": {"value": 4_000.0000001, "unit": "kW"},
        "thermodynamic_limit": {"value": 5_000.0, "unit": "kW"},
        "heat_recovery_residual": {"value": 1e-7, "unit": "kW"},
        "status": "solved",
        "iterations": 27,
    }
    payload.update(overrides)
    return HeatRecoveryDtMinResult.model_validate(payload)


def test_result_is_frozen_strict_finite_and_json_round_trippable() -> None:
    result = _result()

    assert result.status is HeatRecoveryDtMinStatus.SOLVED
    assert result.dt_min == HeatRecoveryQuantity(
        value=10.0,
        unit="delta_degC",
    )
    assert json.loads(result.model_dump_json())["period_id"] == "0"
    assert (
        HeatRecoveryDtMinResult.model_validate_json(result.model_dump_json()) == result
    )

    with pytest.raises(ValidationError, match="frozen"):
        result.iterations = 28
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _result(unknown=True)
    with pytest.raises(ValidationError, match="finite"):
        _result(dt_min={"value": float("inf"), "unit": "delta_degC"})


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("scope", " ", "must not be empty"),
        ("period_id", "", "must not be empty"),
        ("iterations", -1, "non-negative"),
        ("iterations", True, "integer"),
        ("status", "unknown", "Input should be"),
    ],
)
def test_result_rejects_invalid_metadata(field, value, match) -> None:
    with pytest.raises(ValidationError, match=match):
        _result(**{field: value})


def test_quantity_requires_explicit_nonempty_unit() -> None:
    with pytest.raises(ValidationError, match="unit"):
        HeatRecoveryQuantity(value=1.0, unit=" ")
    with pytest.raises(ValidationError, match="finite number"):
        HeatRecoveryQuantity(value=True, unit="kW")
