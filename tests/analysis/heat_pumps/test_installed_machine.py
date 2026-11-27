"""A machine installed for several periods is each component's peak.

A shared multiperiod design is one machine: its compressor moves the most
any period asks of it and each exchanger has the most area any period
needs, which can come from different periods (review finding 19).
"""

from __future__ import annotations

from OpenPinch.analysis.heat_pumps.common.capital import (
    CompressorDuty,
    MachineComponents,
    installed_machine,
)


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
