"""A Carnot target that serves nothing is an error, not a success.

A threshold problem (one cold stream, no heat source) with ambient air off
leaves a heat pump nothing to draw on. The Carnot target came back as a
success of 0 kW of work; the vapour compression target of the same case
raises that no heat pump improves on the baseline.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem
from OpenPinch.contracts.hpr import HPRTargetingError


def _quantity(value, unit="degC"):
    return {"value": value, "unit": unit}


def _threshold_problem() -> PinchProblem:
    return PinchProblem(
        source={
            "streams": [
                {
                    "zone": "Plant",
                    "name": "Water heating",
                    "t_supply": _quantity(60.0),
                    "t_target": _quantity(90.0),
                    "heat_flow": _quantity(500.0, "kW"),
                    "dt_cont": _quantity(5.0),
                    "htc": _quantity(1.0, "kW/m^2/degC"),
                }
            ],
            "utilities": [
                {
                    "name": name,
                    "type": kind,
                    "t_supply": _quantity(supply),
                    "t_target": _quantity(target),
                    "dt_cont": _quantity(1.0),
                    "price": _quantity(price, "$/MWh"),
                    "htc": _quantity(1.0, "kW/m^2/degC"),
                    "heat_flow": None,
                }
                for name, kind, supply, target, price in (
                    ("Steam", "Hot", 150.0, 149.9, 40.0),
                    ("CW", "Cold", 20.0, 25.0, 5.0),
                )
            ],
            "options": {"HPR_AMBIENT_AIR_ENABLED": False},
        },
        project_name="threshold",
    )


def test_a_carnot_heat_pump_with_nothing_to_draw_on_is_not_a_target():
    with pytest.raises(HPRTargetingError, match="improves on the baseline"):
        _threshold_problem().target.carnot_heat_pump(
            load_fraction=1.0, condensers=1, evaporators=1, maximum_restarts=1
        )
