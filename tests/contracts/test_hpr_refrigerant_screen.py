"""Contracts for refrigerant screening of HPR layouts."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from OpenPinch.contracts.hpr import HPRTopologyIdentifier
from OpenPinch.contracts.hpr_refrigerant_screen import (
    HprRefrigerantScreenResult,
    RefrigerantScreenVerdict,
)


def _quantity(value, unit):
    return {"value": value, "unit": unit}


def _row(fluid, verdict, rank=None, *, saturated=True, vhc=4000.0):
    row = {
        "fluid": fluid,
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


def _stage(T_evap, T_cond, rows, kind="vapour_compression"):
    return {
        "kind": kind,
        "T_evap": _quantity(T_evap, "degC"),
        "T_cond": _quantity(T_cond, "degC"),
        "rows": rows,
    }


TOP = _stage(
    80.0,
    120.0,
    [
        _row("Ammonia", "practical", 1),
        _row("n-Butane", "near critical", 2, vhc=1060.0),
        _row("CarbonDioxide", "above critical", saturated=False),
    ],
)
BOTTOM = _stage(
    0.0,
    85.0,
    [
        _row("Ammonia", "practical", 1),
        _row("n-Butane", "practical", 2, vhc=1060.0),
        _row("Isobutane", "practical", 3, vhc=900.0),
    ],
)
MVR = _stage(115.0, 135.0, [_row("Water", "practical", 1)], kind="mvr")


def _layout(stages, method="cascade_vapour_compression", **overrides):
    return {
        "method": method,
        "T_cond_levels": [_quantity(120.0, "degC"), _quantity(85.0, "degC")],
        "T_evap_levels": [_quantity(0.0, "degC")],
        "stages": stages,
        "n_fluid_sets": 6,
    } | overrides


def _result(**overrides) -> HprRefrigerantScreenResult:
    payload = {
        "source": "explicit_levels",
        "topology": "cascade",
        "is_heat_pumping": True,
        "T_evap_levels": [_quantity(0.0, "degC")],
        "T_cond_levels": [_quantity(120.0, "degC"), _quantity(85.0, "degC")],
        "dt_cont": _quantity(5.0, "delta_degC"),
        "dt_cascade_hx": _quantity(5.0, "delta_degC"),
        "max_pressure_ratio": 15.0,
        "min_suction_pressure": _quantity(0.5, "bar"),
        "critical_margin": _quantity(5.0, "delta_degC"),
        "max_stage_lift": _quantity(70.0, "delta_degC"),
        "layouts": [_layout([TOP, BOTTOM])],
    }
    payload.update(overrides)
    return HprRefrigerantScreenResult.model_validate(payload)


def test_result_is_frozen_strict_finite_and_json_round_trippable() -> None:
    result = _result()

    layout = result.layouts[0]
    assert layout.method is HPRTopologyIdentifier.CASCADE_VAPOUR_COMPRESSION
    assert layout.stages[0].rows[1].verdict is RefrigerantScreenVerdict.NEAR_CRITICAL
    assert layout.stages[0].feasible == ("Ammonia", "n-Butane")
    assert json.loads(result.model_dump_json())["layouts"][0]["n_fluid_sets"] == 6
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


def test_fluid_sets_expand_the_vapour_compression_stages_hottest_first():
    result = _result()

    assert result.layouts[0].fluid_sets()[:2] == (
        ("Ammonia", "Ammonia"),
        ("Ammonia", "n-Butane"),
    )
    assert len(result.layouts[0].fluid_sets()) == 6

    with_mvr = _result(
        layouts=[
            _layout([MVR, BOTTOM], method="vapour_compression_mvr", n_fluid_sets=3)
        ]
    )
    assert with_mvr.layouts[0].fluid_sets() == (
        ("Ammonia",),
        ("n-Butane",),
        ("Isobutane",),
    )


def test_fluid_sets_keep_the_critical_temperature_from_rising_down_the_stages():
    hot = _stage(
        80.0,
        120.0,
        [
            _row("Water", "practical", 1) | {"T_crit": _quantity(373.9, "degC")},
            _row("Ammonia", "practical", 2, vhc=900.0),
        ],
    )
    cold = _stage(
        0.0,
        85.0,
        [
            _row("Ammonia", "practical", 1),
            _row("Water", "practical", 2, vhc=10.0)
            | {"T_crit": _quantity(373.9, "degC")},
        ],
    )
    result = _result(layouts=[_layout([hot, cold], n_fluid_sets=3)])

    assert result.layouts[0].fluid_sets() == (
        ("Water", "Ammonia"),
        ("Water", "Water"),
        ("Ammonia", "Ammonia"),
    )
    with pytest.raises(ValidationError, match="n_fluid_sets must be 3"):
        _result(layouts=[_layout([hot, cold], n_fluid_sets=4)])


def test_feasible_layouts_keep_those_with_a_fluid_set():
    blocked = _stage(80.0, 120.0, [_row("Water", "over ratio limit")])
    result = _result(
        layouts=[_layout([blocked, BOTTOM], n_fluid_sets=0), _layout([TOP, BOTTOM])]
    )

    assert result.layouts[0].fluid_sets() == ()
    assert result.feasible_layouts == (result.layouts[1],)


@pytest.mark.parametrize(
    "row,match",
    [
        (_row("Ammonia", "practical"), "feasible rows need a rank"),
        (_row("n-Propane", "over ratio limit", 1), "only feasible rows"),
        (_row("Ammonia", "practical", 0), "rank must be positive"),
        (_row("Ammonia", "above critical", saturated=False) | {"rank": 1}, "only"),
        (_row("CarbonDioxide", "above critical"), "no saturation properties"),
        (_row("n-Propane", "over ratio limit", saturated=False), "needs saturation"),
    ],
)
def test_rows_keep_rank_and_properties_consistent_with_the_verdict(row, match):
    with pytest.raises(ValidationError, match=match):
        _result(layouts=[_layout([_stage(0.0, 85.0, [row])], n_fluid_sets=1)])


def test_layouts_check_their_stages_and_fluid_set_count():
    with pytest.raises(ValidationError, match="n_fluid_sets"):
        _result(layouts=[_layout([TOP, BOTTOM], n_fluid_sets=5)])
    with pytest.raises(ValidationError, match="ranked"):
        unordered = _stage(
            0.0, 85.0, [_row("Ammonia", "practical", 2), _row("Butane", "practical", 1)]
        )
        _result(layouts=[_layout([unordered], n_fluid_sets=2)])
    with pytest.raises(ValidationError, match="water"):
        wet = _stage(115.0, 135.0, [_row("Ammonia", "practical", 1)], kind="mvr")
        _result(layouts=[_layout([wet, BOTTOM], n_fluid_sets=3)])
    with pytest.raises(ValidationError, match="added levels"):
        _result(
            layouts=[_layout([TOP, BOTTOM], added_levels=[_quantity(100.0, "degC")])]
        )
    assert (
        _result(
            layouts=[_layout([TOP, BOTTOM], added_levels=[_quantity(85.0, "degC")])]
        )
        .layouts[0]
        .added_levels[0]
        .value
        == 85.0
    )
    with pytest.raises(ValidationError, match="at least 1"):
        _result(layouts=[_layout([], n_fluid_sets=1)])
