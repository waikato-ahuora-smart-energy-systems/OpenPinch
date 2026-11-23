"""The as-piped heat pump target can be a vapour compression machine.

The retrofit's stream step solves the as-piped target with the cycle it is
asked for: ``cycle="vapour_compression"`` is a cascade of vapour
compression cycles on ``HPR_REFRIGERANTS``, whatever the zone's
``HPR_TYPE``. The service is stubbed: a vapour compression search needs
hundreds of evaluations on this case, minutes in Pyodide.
"""

from __future__ import annotations

from OpenPinch.analysis.heat_exchanger_networks.retrofit import heat_pump_targets
from OpenPinch.domain.enums import HeatPumpAndRefrigerationCycle
from tests.analysis.heat_exchanger_networks.low_temperature_retrofit_case import (
    low_temperature_problem,
    solved_network,
)


def test_the_as_piped_target_solves_the_cycle_it_is_asked_for(monkeypatch):
    asked = []
    monkeypatch.setattr(
        heat_pump_targets,
        "compute_profile_heat_pump_target",
        lambda **arguments: asked.append(arguments["config"].hpr.type),
    )
    problem = low_temperature_problem()

    heat_pump_targets.as_piped_heat_pump(
        problem, solved_network(), cycle="vapour_compression"
    )
    heat_pump_targets.as_piped_heat_pump(problem, solved_network())

    assert asked == [
        HeatPumpAndRefrigerationCycle.CascadeVapourComp.value,
        problem.master_zone.config.hpr.type,
    ]
