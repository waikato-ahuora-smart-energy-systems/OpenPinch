"""Editor metadata for the editable configuration fields."""

from __future__ import annotations

from OpenPinch.presentation.configuration import configuration_field_metadata


def _field(name: str):
    return next(field for field in configuration_field_metadata() if field.name == name)


def test_field_metadata_gives_label_unit_description_and_default():
    field = _field("THERMAL_DT_CONT")

    assert field.label == "Minimum contribution temperature difference"
    assert field.unit == "delta_degC"
    assert field.default == 5.0
    assert "dt_cont" in field.description


def test_every_editable_field_has_its_own_label_and_description():
    for field in configuration_field_metadata():
        assert field.label != field.name.replace("_", " ").title(), field.name
        assert field.description.endswith("."), field.name


def test_numeric_list_fields_are_reported_as_number_lists():
    for name in (
        "PROBLEM_PERIOD_WEIGHTS",
        "HENS_APPROACH_TEMPERATURES",
        "HENS_DT_CONT_MULTIPLIERS",
        "HENS_DERIVATIVE_THRESHOLDS",
        "HENS_STAGE_SELECTION",
    ):
        field = _field(name)
        assert (field.field_type, field.multiple) == ("number_list", True), name

    assert _field("PROBLEM_PERIOD_IDS").field_type == "string_list"
    assert _field("HPR_LOAD_PERIOD_VALUES").field_type == "object"


def test_positive_fields_report_an_exclusive_minimum():
    from OpenPinch.domain.configuration_fields import USER_CONFIG_FIELD_SPECS

    positive = [name for name, spec in USER_CONFIG_FIELD_SPECS.items() if spec.positive]
    assert positive
    for name in positive:
        field = _field(name)
        assert field.numeric_min == 0.0, name
        assert field.numeric_min_exclusive is True, name

    assert _field("THERMAL_DT_CONT").numeric_min_exclusive is False
