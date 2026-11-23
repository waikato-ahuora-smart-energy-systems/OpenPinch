"""What ``heat_pump=`` takes: a cycle name, or a mapping of the target's
arguments. Anything else is refused by name, before any design runs."""

from __future__ import annotations

import pytest

from OpenPinch.application.heat_pump_design import heat_pump_request


def test_a_cycle_name_and_a_mapping_both_name_the_cycle():
    assert heat_pump_request("vapour_compression") == {"cycle": "vapour_compression"}
    assert heat_pump_request({"levels": {"condensers": 2}}) == {
        "cycle": "carnot",
        "levels": {"condensers": 2},
    }


def test_an_argument_the_target_does_not_take_is_refused_by_name():
    with pytest.raises(ValueError, match="'condensers'.*cycle, load, levels"):
        heat_pump_request({"cycle": "carnot", "condensers": 2})


def test_anything_but_a_name_or_a_mapping_is_refused():
    with pytest.raises(TypeError, match="cycle name or a mapping"):
        heat_pump_request(True)
