"""Second-law property tests for vapour-compression cycles with an IHX.

The internal heat exchanger (IHX) moves heat inside the cycle, so it can
never lift a heating COP above the Carnot limit. These tests guard against
compressing from an enthalpy/entropy pair that belongs to different states.

With no superheat and no subcooling, every external heat output leaves at or
above the lowest condensing temperature and every external heat input enters
at or below the highest evaporating temperature. An entropy balance then gives
``COP_h <= T_h / (T_h - T_c)`` with ``T_h = min(T_cond)`` and
``T_c = max(T_evap)`` in kelvin.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from OpenPinch.analysis.heat_pumps.cycles.cascade_vapour_compression_cycle import (
    CascadeVapourCompressionCycle,
)
from OpenPinch.analysis.heat_pumps.cycles.parallel_vapour_compression_cycles import (
    ParallelVapourCompressionCycles,
)
from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_cycle import (
    VapourCompressionCycle,
)
from OpenPinch.analysis.heat_pumps.cycles.vapour_compression_mvr_cascade import (
    VapourCompressionMvrCascade,
)

pytest.importorskip("CoolProp")

_SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
_REFRIGERANTS = st.sampled_from(["R134a", "R1234ze(E)", "Ammonia", "Propane"])
_T_EVAP = st.floats(min_value=-10.0, max_value=30.0)
_LIFT = st.floats(min_value=20.0, max_value=55.0)
_ETA = st.floats(min_value=0.5, max_value=1.0)
_DT_IHX = st.floats(min_value=0.0, max_value=25.0)
# Small tolerance for CoolProp round-off at eta_comp == 1.
_RTOL = 1e-6


def _carnot_cop_h(T_cond_C, T_evap_C) -> float:
    T_h = float(np.min(T_cond_C)) + 273.15
    T_c = float(np.max(T_evap_C)) + 273.15
    return T_h / (T_h - T_c)


def _solve_vc(T_evap, T_cond, eta, refrigerant, dT_ihx) -> VapourCompressionCycle:
    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        eta_comp=eta,
        refrigerant=refrigerant,
        dT_ihx_gas_side=dT_ihx,
        Q_heat=1000.0,
    )
    assert cycle.solved
    return cycle


@_SETTINGS
@given(T_evap=_T_EVAP, lift=_LIFT, eta=_ETA, refrigerant=_REFRIGERANTS, dT_ihx=_DT_IHX)
def test_vc_heating_cop_never_exceeds_carnot(T_evap, lift, eta, refrigerant, dT_ihx):
    T_cond = T_evap + lift
    cycle = _solve_vc(T_evap, T_cond, eta, refrigerant, dT_ihx)

    assert cycle.COP_h <= _carnot_cop_h(T_cond, T_evap) * (1.0 + _RTOL)


@_SETTINGS
@given(
    T_evap=_T_EVAP,
    lift=_LIFT,
    eta=_ETA,
    refrigerant=_REFRIGERANTS,
    dT_ihx=st.lists(_DT_IHX, min_size=2, max_size=2).map(sorted),
)
def test_vc_discharge_temperature_does_not_decrease_with_ihx_duty(
    T_evap, lift, eta, refrigerant, dT_ihx
):
    T_cond = T_evap + lift
    small = _solve_vc(T_evap, T_cond, eta, refrigerant, dT_ihx[0])
    large = _solve_vc(T_evap, T_cond, eta, refrigerant, dT_ihx[1])

    T_discharge_small = small._cycle_states[1]["T"]
    T_discharge_large = large._cycle_states[1]["T"]
    assert T_discharge_large >= T_discharge_small - 1e-6


@_SETTINGS
@given(
    T_evap=st.lists(_T_EVAP, min_size=2, max_size=2),
    lift=st.lists(_LIFT, min_size=2, max_size=2),
    eta=_ETA,
    refrigerant=_REFRIGERANTS,
    dT_ihx=st.lists(_DT_IHX, min_size=2, max_size=2),
)
def test_parallel_vc_heating_cop_never_exceeds_carnot(
    T_evap, lift, eta, refrigerant, dT_ihx
):
    T_evap = np.array(T_evap)
    T_cond = T_evap + np.array(lift)
    cycle = ParallelVapourCompressionCycles()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        eta_comp=eta,
        refrigerant=refrigerant,
        dT_ihx_gas_side=np.array(dT_ihx),
        Q_heat=np.array([1000.0, 500.0]),
    )
    assert cycle.solved

    # Each subcycle obeys its own Carnot limit, so the aggregate cannot
    # beat the best of them.
    carnot = max(_carnot_cop_h(Tc, Te) for Tc, Te in zip(T_cond, T_evap))
    assert cycle.COP_h <= carnot * (1.0 + _RTOL)


@_SETTINGS
@given(
    T_evap=_T_EVAP,
    lifts=st.lists(st.floats(min_value=15.0, max_value=30.0), min_size=2, max_size=2),
    eta=_ETA,
    refrigerant=_REFRIGERANTS,
    dT_ihx=st.lists(_DT_IHX, min_size=2, max_size=2),
)
def test_cascade_vc_heating_cop_never_exceeds_carnot(
    T_evap, lifts, eta, refrigerant, dT_ihx
):
    T_cond_low = T_evap + lifts[0]
    T_cond = np.array([T_cond_low + lifts[1], T_cond_low])
    cycle = CascadeVapourCompressionCycle()
    cycle.solve(
        T_evap=np.array([T_evap]),
        T_cond=T_cond,
        dtcont=0.0,
        dt_cascade_hx=2.0,
        eta_comp=eta,
        refrigerant=refrigerant,
        dT_ihx_gas_side=np.array(dT_ihx),
        Q_heat=np.array([1000.0, 500.0]),
    )
    assert cycle.solved

    assert cycle.COP_h <= _carnot_cop_h(T_cond, T_evap) * (1.0 + _RTOL)


@_SETTINGS
@given(
    T_evap=st.floats(min_value=0.0, max_value=30.0),
    lift=st.floats(min_value=25.0, max_value=50.0),
    dT_lift_mvr=st.floats(min_value=5.0, max_value=20.0),
    eta=_ETA,
    refrigerant=st.sampled_from(["R134a", "R1234ze(E)"]),
    dT_ihx=_DT_IHX,
)
def test_vc_mvr_heating_cop_never_exceeds_carnot(
    T_evap, lift, dT_lift_mvr, eta, refrigerant, dT_ihx
):
    T_cond = T_evap + lift
    cycle = VapourCompressionMvrCascade()
    cycle.solve(
        T_evap_vc=np.array([T_evap]),
        T_cond_vc=np.array([T_cond]),
        dT_lift_mvr=np.array([dT_lift_mvr]),
        Q_heat_vc=np.array([1000.0]),
        mvr_source_split=0.25,
        mvr_process_split=np.array([]),
        dT_subcool_vc=np.array([0.0]),
        dT_subcool_mvr=np.array([0.0]),
        dT_ihx_gas_side_vc=np.array([dT_ihx]),
        eta_comp=eta,
        eta_mvr_comp=eta,
        refrigerant=[refrigerant],
        mvr_fluid=["Water"],
        dt_cascade_hx=2.0,
    )
    assert cycle.solved

    assert cycle.COP_h <= _carnot_cop_h(T_cond, T_evap) * (1.0 + _RTOL)
