"""ZoneType keeps its published member names."""

from OpenPinch.domain.enums import ZoneType


def test_unit_operation_is_reachable_by_both_names():
    assert ZoneType.UO.value == "Unit Operation"
    assert ZoneType["O"] is ZoneType.UO
    assert getattr(ZoneType, "O") is ZoneType.UO
    assert ZoneType("Unit Operation") is ZoneType.UO
    assert ZoneType.UO.name == "UO"
