"""Additional branch coverage tests for Zone."""

from OpenPinch.classes.zone import Zone


def test_zone_active_property_and_duplicate_suffix_increment():
    root = Zone("Root")
    assert root.active is True

    first = Zone("Child")
    first.hot_streams = [1]
    root.add_zone(first, sub=True)

    already_taken = Zone("Child_1")
    already_taken.hot_streams = [9]
    root.add_zone(already_taken, sub=True)

    duplicate = Zone("Child")
    duplicate.hot_streams = [2]
    root.add_zone(duplicate, sub=True)

    assert "Child_2" in root.subzones
