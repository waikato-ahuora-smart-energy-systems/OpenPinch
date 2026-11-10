"""Compressor efficiency models inside the vapour compression cycle models."""

import pytest

from OpenPinch.analysis.heat_pumps.common.compressor_efficiency import (
    CompressorModelError,
)
from OpenPinch.analysis.heat_pumps.cycles.mechanical_vapour_recompression_cycle import (
    MechanicalVapourRecompressionCycle,
)
from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_cycle import (
    VapourCompressionCycle,
)

pytest.importorskip("CoolProp")


def _solve(eta_comp):
    cycle = VapourCompressionCycle()
    work = cycle.solve(
        T_evap=10.0,
        T_cond=60.0,
        dtcont=0.0,
        dT_superheat=5.0,
        dT_ihx_gas_side=0.0,
        eta_comp=eta_comp,
        refrigerant="AMMONIA",
        Q_heat=1000.0,
    )
    return cycle, work


def test_a_compressor_function_sets_the_efficiency_the_cycle_applies():
    cycle, work = _solve(lambda conditions: 0.6)
    _, constant_work = _solve(0.6)

    assert work == constant_work
    assert cycle.eta_comp == 0.6


def test_a_compressor_function_sees_the_suction_and_discharge_conditions():
    seen = []

    def recording(conditions):
        seen.append(conditions)
        return 0.7

    cycle, _ = _solve(recording)

    (conditions,) = seen
    suction, discharge = cycle.cycle_states[0], cycle.cycle_states[1]
    assert conditions.refrigerant == "AMMONIA"
    assert conditions.P_suction == pytest.approx(suction["P"] / 1000.0, rel=1e-12)
    assert conditions.P_discharge == pytest.approx(discharge["P"] / 1000.0, rel=1e-12)
    assert conditions.T_suction == pytest.approx(15.0, abs=1e-6)
    assert conditions.T_evap_dew == pytest.approx(10.0, abs=1e-6)
    assert conditions.T_cond_dew == pytest.approx(60.0, abs=1e-6)
    assert conditions.rho_suction < conditions.rho_evap_dew < conditions.rho_cond_dew


@pytest.mark.parametrize("eta", [0.0, 1.2, float("nan")])
def test_a_compressor_function_must_return_an_efficiency_in_range(eta):
    # Not a ValueError, which the targeting search reads as an infeasible point.
    def out_of_range(conditions):
        return eta

    with pytest.raises(CompressorModelError, match="out_of_range returned"):
        _solve(out_of_range)


def test_an_mvr_stage_takes_a_compressor_function():
    def solve(eta_mvr_comp):
        stage = MechanicalVapourRecompressionCycle()
        work = stage.solve_from_source_heat(
            T_evap=100.0,
            T_cond=130.0,
            Q_source=1000.0,
            eta_mvr_comp=eta_mvr_comp,
            fluid="Water",
        )
        return stage, work

    stage, work = solve(lambda conditions: 0.6)

    assert work == solve(0.6)[1]
    assert stage.eta_mvr_comp == 0.6
