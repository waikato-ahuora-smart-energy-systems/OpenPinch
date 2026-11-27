"""A shared design's capital is its machine sized component by component.

The capital took each machine's most expensive period. A compressor that
peaks in one period and an exchanger that peaks in another were then
priced at the smaller of the two machines (review finding 19).
"""

from __future__ import annotations

import pytest

from OpenPinch.analysis.heat_pumps.common.capital import (
    CompressorDuty,
    MachineComponents,
    installed_capital,
    installed_machine,
)
from OpenPinch.analysis.heat_pumps.optimisation_adapter import (
    aggregate_hpr_period_results,
)
from OpenPinch.contracts.hpr import HPRBackendResult, HPRThermoArtifacts
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.value import Value


def _period(machine: MachineComponents) -> HPRBackendResult:
    capital = installed_capital(machine)
    return HPRBackendResult(
        obj=1.0,
        utility_tot=10.0,
        w_net=1.0,
        Q_ext_heat=0.0,
        Q_ext_cold=0.0,
        hpr_operating_cost=Value(100.0, "$/y"),
        hpr_capital_cost=Value(capital, "$"),
        hpr_annualized_capital_cost=Value(0.1 * capital, "$/y"),
        hpr_machine_capital_costs=(capital,),
        hpr_machine_annualized_capital_costs=(0.1 * capital,),
        hpr_machine_components=(machine,),
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
        cop_h=3.0,
        amb_streams=StreamCollection(),
        artifacts=HPRThermoArtifacts(hpr_streams=StreamCollection()),
    )


def test_the_shared_capital_is_each_components_peak():
    base = MachineComponents(
        compressors=(CompressorDuty("Ammonia", 400.0, 140.0),),
        exchanger_areas=(0.0, 0.0, 80.0, 0.0, 20.0, 0.0),
    )
    peak = MachineComponents(
        compressors=(CompressorDuty("Ammonia", 250.0, 160.0),),
        exchanger_areas=(0.0, 0.0, 50.0, 0.0, 90.0, 0.0),
    )

    weighted, _ = aggregate_hpr_period_results(
        {"base": _period(base), "peak": _period(peak)}, [0.5, 0.5]
    )

    installed = installed_capital(installed_machine([base, peak]))
    assert installed > max(installed_capital(base), installed_capital(peak))
    assert weighted.hpr_capital_cost.value == pytest.approx(installed)
    assert weighted.hpr_annualized_capital_cost.value == pytest.approx(0.1 * installed)
