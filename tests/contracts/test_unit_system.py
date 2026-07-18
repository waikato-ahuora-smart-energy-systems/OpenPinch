"""Unit-system coercion tests backed by static fixture payloads."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import OpenPinch.contracts.units as unit_system
from OpenPinch.domain.configuration import Configuration
from OpenPinch.domain.value import Value
from tests.support.paths import FIXTURES_ROOT

FIXTURE_PATH = FIXTURES_ROOT / "unit_system_cases.json"


def _unit_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_unit_system_config_override_resolution_paths():
    fixture = _unit_fixture()
    cfg = Configuration(
        options={
            "INPUT_UNIT_TEMPERATURE": "K",
            "OUTPUT_UNIT_HEAT_FLOW": "MW",
        }
    )

    assert unit_system._config_units(None, attr_name="input_unit_overrides") == {}
    assert (
        unit_system._config_units(
            {"input_unit_overrides": fixture["input_overrides"]},
            attr_name="input_unit_overrides",
        )
        == fixture["input_overrides"]
    )
    assert (
        unit_system._config_units(
            fixture["input_overrides"],
            attr_name="input_unit_overrides",
        )
        == fixture["input_overrides"]
    )
    config_input_units = unit_system._config_units(
        cfg,
        attr_name="input_unit_overrides",
    )
    assert config_input_units["temperature"] == "K"
    assert config_input_units["heat_flow"] == "kW"

    assert (
        unit_system._resolve_override(
            key="t_supply",
            unit_groups=("temperature",),
            overrides={"t_supply": None, "temperature": " K "},
        )
        == "K"
    )
    assert (
        unit_system._resolve_override(
            key="t_supply",
            unit_groups=("temperature",),
            overrides={"temperature": " "},
        )
        is None
    )


def test_unit_system_unit_detection_and_value_magnitudes_from_static_fixture():
    fixture = _unit_fixture()

    assert unit_system._normalise_unit_text(None) is None
    assert unit_system._normalise_unit_text(" ") is None
    assert unit_system._normalise_unit_text("-") == "dimensionless"
    assert unit_system._unit_from_mapping({"unit": " kW "}) == "kW"
    assert unit_system._unit_from_value_like(Value(1.0)) == "dimensionless"
    assert unit_system._unit_from_value_like({"value": 1.0, "unit": ""}) is None
    assert (
        unit_system._unit_from_value_like(SimpleNamespace(**fixture["scalar_object"]))
        == "kW"
    )
    assert unit_system._has_explicit_unit(Value(1.0)) is False
    assert unit_system._has_explicit_unit({"value": 1.0, "unit": "kW"}) is True

    class DumpableValue:
        def model_dump(self, *, mode: str):
            assert mode == "python"
            return fixture["period_value"]

    assert unit_system._value_magnitudes(Value([1.0, 2.0], unit="kW")).tolist() == [
        1.0,
        2.0,
    ]
    assert unit_system._value_magnitudes(fixture["period_value"]) == [1.0, 2.0]
    assert unit_system._value_magnitudes({"value": 7.0, "unit": "kW"}) == 7.0
    assert unit_system._value_magnitudes(
        SimpleNamespace(**fixture["period_object"])
    ) == [3.0, 6.0]
    assert (
        unit_system._value_magnitudes(SimpleNamespace(**fixture["scalar_object"]))
        == 12.0
    )
    assert unit_system._value_magnitudes(DumpableValue()) == [1.0, 2.0]
    assert unit_system._value_magnitudes(9.0) == 9.0

    inferred_value = unit_system._ensure_value(Value(2.0), user_unit="kW")
    assert inferred_value.value == pytest.approx(2.0)
    assert inferred_value.unit == "kW"


def test_standardise_input_value_uses_canonical_and_override_units():
    fixture = _unit_fixture()

    assert unit_system.standardise_input_value(None, field_name="t_supply") is None
    assert unit_system.standardise_input_value(5.0, field_name="unknown").value == (
        pytest.approx(5.0)
    )

    temperature = unit_system.standardise_input_value(
        fixture["temperature_value"],
        field_name="t_supply",
    )
    assert temperature.value == pytest.approx(25.0)
    assert temperature.unit == "degC"

    temperature_from_override = unit_system.standardise_input_value(
        298.15,
        field_name="t_supply",
        config={"input_unit_overrides": fixture["input_overrides"]},
    )
    assert temperature_from_override.value == pytest.approx(25.0)
    assert temperature_from_override.unit == "degC"

    delta_temperature = unit_system.standardise_input_value(
        fixture["delta_temperature_value"],
        field_name="dt_cont",
    )
    assert delta_temperature.value == pytest.approx(5.0)
    assert delta_temperature.unit == "delta_degC"


@pytest.mark.parametrize("field_name", ["t_supply", "t_target"])
def test_temperature_unit_group_override_applies_to_each_input_field(field_name):
    value = unit_system.standardise_input_value(
        298.15,
        field_name=field_name,
        config={"input_unit_overrides": {"temperature": "K"}},
    )

    assert value.value == pytest.approx(25.0)
    assert value.unit == "degC"


@pytest.mark.parametrize("metric_name", ["Qh", "Qc", "Qr", "utility_heat_flow"])
def test_heat_flow_unit_group_override_applies_to_each_output_metric(metric_name):
    value = unit_system.coerce_output_value(
        Value(1000.0, "kW"),
        metric_name=metric_name,
        config={"output_unit_overrides": {"heat_flow": "MW"}},
    )

    assert value.value == pytest.approx(1.0)
    assert value.unit == "MW"


def test_coerce_output_value_uses_default_and_configured_display_units():
    fixture = _unit_fixture()

    assert unit_system.coerce_output_value(None, metric_name="Qh") is None

    unknown_metric = unit_system.coerce_output_value(0.5, metric_name="custom_metric")
    assert unknown_metric.value == pytest.approx(0.5)
    assert unknown_metric.unit == "-"

    heat_flow = unit_system.coerce_output_value(
        Value(1000.0, "kW"),
        metric_name="Qh",
        config={"output_unit_overrides": fixture["output_overrides"]},
    )
    assert heat_flow.value == pytest.approx(1.0)
    assert heat_flow.unit == "MW"

    percent = unit_system.coerce_output_value(0.5, metric_name="degree_of_integration")
    assert percent.value == pytest.approx(50.0)
    assert percent.unit == "%"

    temporary_rule = unit_system.OutputUnitRule(
        "dimensionless",
        None,
        unit_groups=("raw_fraction",),
    )
    unit_system.OUTPUT_UNIT_RULES["raw_fraction"] = temporary_rule
    try:
        raw_fraction = unit_system.coerce_output_value(0.5, metric_name="raw_fraction")
    finally:
        del unit_system.OUTPUT_UNIT_RULES["raw_fraction"]

    assert raw_fraction.value == pytest.approx(0.5)
    assert raw_fraction.unit == "-"
