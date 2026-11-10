"""Contracts for single-stage refrigerant screening."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from OpenPinch.contracts.hpr_refrigerant_screen import (
    HprRefrigerantScreenResult,
    RefrigerantScreenVerdict,
)


def _quantity(value, unit):
    return {"value": value, "unit": unit}


def _row(fluid, verdict, rank=None, *, saturated=True, vhc=4000.0):
    row = {
        "fluid": fluid,
        "T_evap": _quantity(0.0, "degC"),
        "T_cond": _quantity(85.0, "degC"),
        "T_crit": _quantity(132.4, "degC"),
        "verdict": verdict,
        "rank": rank,
    }
    if saturated:
        row |= {
            "p_evap": _quantity(4.3, "bar"),
            "p_cond": _quantity(46.0, "bar"),
            "pressure_ratio": 10.7,
            "suction_vhc": _quantity(vhc, "kJ/m^3"),
        }
    return row


def _result(**overrides) -> HprRefrigerantScreenResult:
    payload = {
        "source": "explicit_levels",
        "T_evap_levels": [_quantity(0.0, "degC")],
        "T_cond_levels": [_quantity(85.0, "degC")],
        "dt_cont": _quantity(5.0, "delta_degC"),
        "max_pressure_ratio": 15.0,
        "min_suction_pressure": _quantity(0.5, "bar"),
        "critical_margin": _quantity(15.0, "delta_degC"),
        "rows": [
            _row("Ammonia", "practical", 1),
            _row("n-Butane", "practical", 2, vhc=1060.0),
            _row("n-Propane", "near critical"),
            _row("CarbonDioxide", "above critical", saturated=False),
        ],
    }
    payload.update(overrides)
    return HprRefrigerantScreenResult.model_validate(payload)


def test_result_is_frozen_strict_finite_and_json_round_trippable() -> None:
    result = _result()

    assert result.rows[2].verdict is RefrigerantScreenVerdict.NEAR_CRITICAL
    assert [row.fluid for row in result.practical] == ["Ammonia", "n-Butane"]
    assert json.loads(result.model_dump_json())["source"] == "explicit_levels"
    assert (
        HprRefrigerantScreenResult.model_validate_json(result.model_dump_json())
        == result
    )
    with pytest.raises(ValidationError, match="frozen"):
        result.source = "carnot_target"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _result(unknown=True)
    with pytest.raises(ValidationError, match="finite"):
        _result(max_pressure_ratio=float("inf"))


@pytest.mark.parametrize(
    "row,match",
    [
        (_row("Ammonia", "practical"), "practical rows need a rank"),
        (_row("n-Propane", "near critical", 1), "only practical rows"),
        (_row("Ammonia", "practical", 0), "rank must be positive"),
        (_row("Ammonia", "above critical", saturated=False) | {"rank": 1}, "only"),
        (_row("CarbonDioxide", "above critical"), "no saturation properties"),
        (_row("n-Propane", "over ratio limit", saturated=False), "needs saturation"),
    ],
)
def test_rows_keep_rank_and_properties_consistent_with_the_verdict(row, match):
    with pytest.raises(ValidationError, match=match):
        _result(rows=[row])
