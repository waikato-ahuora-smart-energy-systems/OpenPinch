"""Practical limits on a vapour-compression stage.

Liquid injection holds the gas entering the condenser to a maximum superheat,
and the stage is penalised for a wet discharge or for running outside its
pressure ratio, pressure and lift limits.
"""

from __future__ import annotations

import CoolProp.CoolProp as CP
import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.cycles.cascade_vapour_compression_cycle as cascade_cycle_module
import OpenPinch.analysis.heat_pumps.cycles.parallel_vapour_compression_cycles as parallel_cycle_module
import OpenPinch.analysis.heat_pumps.cycles.vapour_compression_mvr_cascade as vc_mvr_cycle_module
from OpenPinch.analysis.heat_pumps.common import preprocessing as hp_pre
from OpenPinch.analysis.heat_pumps.common.shared import cycle_limits
from OpenPinch.analysis.heat_pumps.cycles.cascade_vapour_compression_cycle import (
    CascadeVapourCompressionCycle,
)
from OpenPinch.analysis.heat_pumps.cycles.limits import CycleLimits
from OpenPinch.analysis.heat_pumps.cycles.parallel_vapour_compression_cycles import (
    ParallelVapourCompressionCycles,
)
from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_cycle import (
    VapourCompressionCycle,
)
from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_mvr_cascade import (
    VapourCompressionMvrCascade,
)
from OpenPinch.analysis.heat_pumps.targeting import (
    cascade_vapour_compression as cascade_targeting,
)
from OpenPinch.analysis.heat_pumps.targeting import (
    parallel_vapour_compression as parallel_targeting,
)
from OpenPinch.analysis.heat_pumps.targeting import (
    vapour_compression_mvr as vc_mvr_targeting,
)
from OpenPinch.contracts.hpr import HPRParsedState
from OpenPinch.domain.configuration import Configuration


def _water_stage(limits=None):
    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=115.8,
        T_cond=183.6,
        dtcont=0.0,
        refrigerant="Water",
        dT_ihx_gas_side=0.0,
        Q_heat=1000.0,
        limits=limits,
    )
    assert cycle.solved
    return cycle


def _condenser_profile_C(cycle):
    streams = cycle.build_stream_collection(include_cond=True)
    (stream,) = list(streams)
    segments = stream.segments if stream.has_segments else (stream,)
    temperatures = [
        float(np.ravel(getattr(s, attr).value)[0])
        for s in segments
        for attr in ("supply_temperature", "target_temperature")
    ]
    duty = sum(float(np.ravel(s.heat_flow.value)[0]) for s in segments)
    return max(temperatures), duty


def test_liquid_injection_caps_the_gas_entering_the_condenser():
    plain = _water_stage()
    injected = _water_stage(CycleLimits(max_discharge_superheat=50.0))

    T_dew = CP.PropsSI("T", "P", injected.Ps[1], "Q", 1, "Water") - 273.15
    T_top, duty = _condenser_profile_C(injected)
    T_top_plain, duty_plain = _condenser_profile_C(plain)

    assert T_top_plain > T_dew + 200.0
    assert T_top == pytest.approx(T_dew + 50.0, abs=0.5)
    # Injected condensate is evaporated by the hot gas, so the condenser
    # rejects the same heat and the compressor does the same work.
    assert duty == pytest.approx(duty_plain, rel=1e-6)
    assert injected.work == pytest.approx(plain.work, rel=1e-9)


def test_liquid_injection_survives_a_cascade_stage_settling_its_subcooling():
    # A small process duty beside cascade heat cannot take the requested
    # subcooling, so the stage re-solves; the injection must still apply.
    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=0.0,
        T_cond=80.0,
        dtcont=5.0,
        dT_subcool=78.0,
        refrigerant="Ammonia",
        dT_ihx_gas_side=200.0,
        Q_heat=5.0,
        Q_cas_heat=100.0,
        limits=CycleLimits(max_discharge_superheat=50.0),
    )

    assert cycle.dT_subcool < 78.0
    assert cycle.liquid_injection_ratio > 0.0


def _stage(refrigerant, T_evap, T_cond, *, dT_ihx=0.0, limits=None, Q_heat=1000.0):
    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        refrigerant=refrigerant,
        dT_ihx_gas_side=dT_ihx,
        Q_heat=Q_heat,
        limits=limits,
    )
    assert cycle.solved
    return cycle


def test_a_wet_discharge_is_penalised():
    # Isopentane compressed from saturated vapour ends two-phase (quality
    # 0.85 at 19.6 -> 122.2 degC); the gas short of its dew point is the penalty.
    wet = _stage("Isopentane", 19.6, 122.2)

    h_dew = CP.PropsSI("H", "P", wet.Ps[1], "Q", 1, "Isopentane")
    assert float(np.sum(wet.penalty)) == pytest.approx(
        (h_dew - wet.Hs[1]) * wet.m_dot, rel=1e-6
    )
    assert float(np.sum(wet.penalty)) > 0.0


def test_a_dry_discharge_is_not_penalised():
    assert float(np.sum(_stage("Ammonia", 9.4, 35.9).penalty)) == 0.0


def _pressures(refrigerant, T_evap, T_cond):
    P_lo = CP.PropsSI("P", "T", T_evap + 273.15, "Q", 1, refrigerant)
    P_hi = CP.PropsSI("P", "T", T_cond + 273.15, "Q", 0, refrigerant)
    return P_lo, P_hi


@pytest.mark.parametrize(
    ("refrigerant", "T_evap", "T_cond", "limits", "relative_violation"),
    [
        (
            "Water",
            115.8,
            183.6,
            CycleLimits(max_pressure_ratio=5.0),
            lambda P_lo, P_hi: P_hi / P_lo / 5.0 - 1.0,
        ),
        (
            "Ammonia",
            19.8,
            105.9,
            CycleLimits(max_discharge_pressure=60e5),
            lambda P_lo, P_hi: P_hi / 60e5 - 1.0,
        ),
        (
            "Water",
            9.7,
            35.9,
            CycleLimits(min_suction_pressure=0.5e5),
            lambda P_lo, P_hi: 0.5e5 / P_lo - 1.0,
        ),
        (
            "Ammonia",
            30.0,
            32.0,
            CycleLimits(min_lift=5.0),
            lambda P_lo, P_hi: 1.0 - 2.0 / 5.0,
        ),
    ],
)
def test_a_stage_outside_a_limit_is_penalised_by_its_relative_violation(
    refrigerant, T_evap, T_cond, limits, relative_violation
):
    stage = _stage(refrigerant, T_evap, T_cond, limits=limits)
    P_lo, P_hi = _pressures(refrigerant, T_evap, T_cond)

    assert float(np.sum(stage.penalty)) == pytest.approx(
        stage.Q_cond * relative_violation(P_lo, P_hi), rel=1e-6
    )


@pytest.mark.parametrize(
    "limits",
    [
        CycleLimits(max_pressure_ratio=15.0),
        CycleLimits(max_discharge_pressure=60e5),
        CycleLimits(min_suction_pressure=0.5e5),
        CycleLimits(min_lift=5.0),
    ],
)
def test_a_stage_within_its_limits_is_not_penalised(limits):
    assert float(np.sum(_stage("Ammonia", 9.4, 35.9, limits=limits).penalty)) == 0.0


def test_carbon_dioxide_has_its_own_discharge_pressure_limit():
    # CO2 condensing at 20 degC runs at 57 bar.
    limits = CycleLimits(max_discharge_pressure=50e5, max_discharge_pressure_co2=140e5)

    assert float(np.sum(_stage("CO2", -10.0, 20.0, limits=limits).penalty)) == 0.0
    assert float(np.sum(_stage("Ammonia", -10.0, 90.0, limits=limits).penalty)) > 0.0


_LIMITS = CycleLimits(max_discharge_superheat=50.0, min_lift=5.0)


def _recording_stage_class(seen):
    class RecordingStage(VapourCompressionCycle):
        def solve(self, *args, **kwargs):
            seen.append(kwargs.get("limits"))
            return super().solve(*args, **kwargs)

    return RecordingStage


def _solve_cascade():
    CascadeVapourCompressionCycle().solve(
        T_evap=np.array([5.0]),
        T_cond=np.array([80.0, 45.0]),
        dtcont=0.0,
        refrigerant="R1234ze(E)",
        dT_ihx_gas_side=0.0,
        dt_cascade_hx=5.0,
        Q_heat=np.array([100.0, 50.0]),
        limits=_LIMITS,
    )


def _solve_parallel():
    ParallelVapourCompressionCycles().solve(
        T_evap=np.array([5.0, 10.0]),
        T_cond=np.array([80.0, 45.0]),
        dtcont=0.0,
        refrigerant="R1234ze(E)",
        dT_ihx_gas_side=0.0,
        Q_heat=np.array([100.0, 50.0]),
        limits=_LIMITS,
    )


def _solve_vc_mvr():
    VapourCompressionMvrCascade().solve(
        T_evap_vc=np.array([20.0, 30.0]),
        T_cond_vc=np.array([90.0, 80.0]),
        dT_lift_mvr=np.array([15.0, 12.0]),
        Q_heat_vc=np.array([1000.0, 500.0]),
        mvr_source_split=0.25,
        mvr_process_split=np.array([0.4]),
        refrigerant=["R134A", "R134A"],
        dt_cascade_hx=5.0,
        limits=_LIMITS,
    )


@pytest.mark.parametrize(
    ("module", "solve"),
    [
        (cascade_cycle_module, _solve_cascade),
        (parallel_cycle_module, _solve_parallel),
        (vc_mvr_cycle_module, _solve_vc_mvr),
    ],
)
def test_multi_stage_cycles_hold_every_stage_to_the_limits(monkeypatch, module, solve):
    seen = []
    monkeypatch.setattr(module, "VapourCompressionCycle", _recording_stage_class(seen))

    solve()

    assert seen
    assert seen == [_LIMITS] * len(seen)


def _target_inputs(options):
    return hp_pre.construct_HPRTargetInputs(
        Q_hpr_target=50.0,
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_hot=np.array([0.0, -50.0, -100.0]),
        H_cold=np.array([100.0, 50.0, 0.0]),
        config=Configuration(options=options),
    )


def test_the_default_configuration_sets_the_practical_limits():
    assert cycle_limits(_target_inputs({})) == CycleLimits(
        max_discharge_superheat=50.0,
        max_pressure_ratio=15.0,
        max_discharge_pressure=60e5,
        max_discharge_pressure_co2=140e5,
        min_suction_pressure=0.5e5,
        min_lift=5.0,
    )


def test_each_limit_is_configurable():
    limits = cycle_limits(
        _target_inputs(
            {
                "HPR_MAX_DISCHARGE_SUPERHEAT": 40.0,
                "HPR_MAX_PRESSURE_RATIO": 10.0,
                "HPR_MAX_DISCHARGE_PRESSURE": 52.0,
                "HPR_MAX_DISCHARGE_PRESSURE_CO2": 130.0,
                "HPR_SCREEN_MIN_SUCTION_PRESSURE": 0.3,
                "HPR_MIN_STAGE_LIFT": 8.0,
            }
        )
    )

    assert limits == CycleLimits(
        max_discharge_superheat=40.0,
        max_pressure_ratio=10.0,
        max_discharge_pressure=52e5,
        max_discharge_pressure_co2=130e5,
        min_suction_pressure=0.3e5,
        min_lift=8.0,
    )


class _RecordingSolve(Exception):
    pass


def _recording_cycle_class(seen):
    class RecordingCycle:
        def solve(self, **kwargs):
            seen.update(kwargs)
            raise _RecordingSolve

    return RecordingCycle


_STATE = HPRParsedState(
    T_evap=np.array([10.0]),
    T_cond=np.array([60.0]),
    dT_subcool=np.zeros(1),
    dT_ihx_gas_side=np.zeros(1),
)


@pytest.mark.parametrize(
    ("module", "builder", "cycle_class"),
    [
        (
            parallel_targeting,
            "_build_and_solve_parallel_cycles",
            "ParallelVapourCompressionCycles",
        ),
        (
            cascade_targeting,
            "_build_and_solve_cascade_cycle",
            "CascadeVapourCompressionCycle",
        ),
    ],
)
def test_vapour_compression_targeting_passes_the_limits_to_the_cycles(
    monkeypatch, module, builder, cycle_class
):
    seen = {}
    monkeypatch.setattr(module, cycle_class, _recording_cycle_class(seen))
    args = _target_inputs({"HPR_MIN_STAGE_LIFT": 8.0})

    with pytest.raises(_RecordingSolve):
        getattr(module, builder)(_STATE.model_copy(), args, True)

    assert seen.get("limits") == cycle_limits(args)


def test_vc_mvr_targeting_passes_the_limits_to_the_cascade(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        vc_mvr_targeting, "VapourCompressionMvrCascade", _recording_cycle_class(seen)
    )
    monkeypatch.setattr(
        vc_mvr_targeting,
        "_parse_vc_mvr_state_variables",
        lambda x, args: _STATE.model_copy(),
    )
    monkeypatch.setattr(
        vc_mvr_targeting,
        "_unpack_vc_mvr_state",
        lambda state_vars, args: dict.fromkeys(
            (
                "T_evap_vc",
                "T_cond_vc",
                "dT_lift_mvr",
                "mvr_source_split",
                "mvr_process_split",
                "dT_subcool_vc",
                "dT_subcool_mvr",
            )
        ),
    )
    args = _target_inputs({"HPR_MIN_STAGE_LIFT": 8.0})

    with pytest.raises(_RecordingSolve):
        vc_mvr_targeting._compute_vc_mvr_system_obj(np.zeros(1), args)

    assert seen.get("limits") == cycle_limits(args)
