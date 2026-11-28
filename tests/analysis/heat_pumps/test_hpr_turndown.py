"""A shared heat pump turns down on duty and pressure ratio, not lift.

A shared vapour compression design is one installed machine, designed once
and operated in each period (review finding 19, decided 2026-11-28). Its
compressors turn down on their installed volume flow (HPR_TURNDOWN_DUTY)
and design pressure ratio (HPR_TURNDOWN_PRESSURE_RATIO); the lift floor of
the shared design vector, HPR_TURNDOWN, is retired.
"""

from __future__ import annotations

from OpenPinch.domain.configuration_fields import CONFIG_FIELD_SPECS


def test_the_lift_turndown_is_retired():
    assert "HPR_TURNDOWN" not in CONFIG_FIELD_SPECS
    assert CONFIG_FIELD_SPECS["HPR_TURNDOWN_DUTY"].default == 0.25
    assert CONFIG_FIELD_SPECS["HPR_TURNDOWN_PRESSURE_RATIO"].default == 0.25
