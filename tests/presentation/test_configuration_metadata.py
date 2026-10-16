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
