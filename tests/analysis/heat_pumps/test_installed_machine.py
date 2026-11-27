"""A machine installed for several periods is each component's peak.

A shared multiperiod design is one machine: its compressor moves the most
any period asks of it and each exchanger has the most area any period
needs, which can come from different periods (review finding 19).
"""

from __future__ import annotations

from types import SimpleNamespace

from OpenPinch.analysis.heat_pumps.common.capital import (
    CompressorDuty,
    MachineComponents,
    installed_machine,
)
from OpenPinch.analysis.heat_pumps.common.machine_capital import machine_components

from .helpers import _base_args


def _stage(*, ihx_duty: float):
    return SimpleNamespace(
        refrigerant="Ammonia",
        suction_volume_flow=100.0,
        shaft_power=40.0,
        ihx_duty=ihx_duty,
        ihx_lmtd=10.0,
        Q_cas_heat=0.0,
        Q_heat=100.0,
        Q_cool=80.0,
    )


def test_a_component_idle_in_a_period_keeps_its_place():
    # process_areas: condenser, evaporator, air cooler, air heater.
    args = _base_args(n_cond=1, n_evap=1)
    running = machine_components(
        [[_stage(ihx_duty=20.0)]], process_areas=(30.0, 40.0, 5.0, 0.0), args=args
    )
    idle = machine_components(
        [[_stage(ihx_duty=0.0)]], process_areas=(30.0, 40.0, 0.0, 0.0), args=args
    )

    assert len(idle[0].exchanger_areas) == len(running[0].exchanger_areas)
    assert idle[0].exchanger_areas[-2] == running[0].exchanger_areas[-2] == 40.0


def test_each_component_takes_its_own_peak():
    base = MachineComponents(
        compressors=(CompressorDuty("Ammonia", 100.0, 40.0),),
        exchanger_areas=(50.0, 10.0),
    )
    peak = MachineComponents(
        compressors=(CompressorDuty("Ammonia", 60.0, 55.0),),
        exchanger_areas=(30.0, 40.0),
    )

    assert installed_machine([base, peak]) == MachineComponents(
        compressors=(CompressorDuty("Ammonia", 100.0, 55.0),),
        exchanger_areas=(50.0, 40.0),
    )
