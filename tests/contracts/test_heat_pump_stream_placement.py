"""A heat pump stream says where its machine is placed.

A heat pump on the process exchanges heat with process streams; one on
the utility system exchanges heat with utility levels. The two need
opposite match rules, so the mark carries the placement, as the targeting
call does (``placement="process"`` or ``"utility"``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from OpenPinch.contracts.input import HeatPumpStreamSchema


def test_a_heat_pump_stream_is_placed_on_the_process_unless_it_says_utility():
    assert HeatPumpStreamSchema(machine="HP", role="condenser").placement == "process"
    assert (
        HeatPumpStreamSchema(
            machine="HP", role="condenser", placement="utility"
        ).placement
        == "utility"
    )
    with pytest.raises(ValidationError, match="placement"):
        HeatPumpStreamSchema(machine="HP", role="condenser", placement="site")
