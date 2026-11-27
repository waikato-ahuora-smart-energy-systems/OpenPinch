"""Heat a heat pump target rejects to ambient air is charged.

Every rejected heat ends at ambient and is charged at the ambient price
(decided 2026-11-27, review finding 20). The heat pump objectives charged
cooling water at its price ratio but left heat rejected to free ambient air
free, so a target could reject any surplus to air at no cost. Ambient air
as a source of heat stays free.
"""

from __future__ import annotations

import pytest

from OpenPinch import PinchProblem

_SHORT = {"maximum_restarts": 0, "maximum_evaluations": 1}


def test_the_objective_charges_heat_rejected_to_air_as_cooling_water():
    """The objective (power-equivalent per kW targeted) priced the work and
    left the heat this design rejects to air free; it is charged at the
    cooling water's price ratio, as rejection to ambient is."""
    target = PinchProblem("basic_pinch.json").target.carnot_heat_pump(
        load_fraction=1.0,
        condensers=1,
        evaporators=1,
        options={
            "HPR_AMBIENT_AIR_ENABLED": True,
            "COSTING_HPR_PRICE_RATIO_COOLING_WATER_TO_ELE": 0.025,
        },
        **_SHORT,
    )

    details = target.hpr_details
    assert details.Q_amb_cold > 1.0
    expected = (details.w_net + 0.025 * details.Q_amb_cold) / target.hpr_load.selected
    assert details.obj == pytest.approx(expected, rel=1e-6)


def test_a_vapour_compression_target_pays_to_reject_heat_to_air():
    """A vapour compression target's operating cost was its electricity
    alone with heat rejected to air; that heat is charged at the cooling
    water's price, the electricity price times its ratio."""
    target = PinchProblem("basic_pinch.json").target.vapour_compression_heat_pump(
        refrigerants=["Water"],
        load_fraction=0.25,
        condensers=1,
        evaporators=1,
        maximum_iterations=1,
        options={
            "HPR_AMBIENT_AIR_ENABLED": True,
            "COSTING_HPR_CAPITAL_RECOVERY_ENABLED": False,
            "HPR_MAX_PRESSURE_RATIO": 1e6,
            "HPR_SCREEN_MIN_SUCTION_PRESSURE": 0.0,
            "COSTING_HPR_ELE_PRICE": 100.0,
            "COSTING_ANNUAL_OP_TIME": 8000.0,
            "COSTING_HPR_PRICE_RATIO_COOLING_WATER_TO_ELE": 0.025,
        },
        **_SHORT,
    )

    details = target.hpr_details
    assert details.Q_amb_cold > 1.0
    electricity_and_air = details.w_net + 0.025 * details.Q_amb_cold
    assert float(target.hpr_operating_cost.to("$/y").value) == pytest.approx(
        electricity_and_air * 100.0 * 8000.0 / 1000.0, rel=1e-6
    )
