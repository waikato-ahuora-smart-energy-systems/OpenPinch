"""Regression tests for the simple heat pump cycle classes."""

from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given

import OpenPinch.services.heat_pump_integration.unit_models.vapour_compression_cycle as shp_mod
from OpenPinch.lib.enums import *
from OpenPinch.services.heat_pump_integration.unit_models.vapour_compression_cycle import (
    VapourCompressionCycle,
)
from tests.strategies.heat_pump_cycles import zero_duty_stream_side_cases

pytest.importorskip("CoolProp")


class _FakeFluid:
    def keyed_output(self, _key):
        return 500.0


class _FakeState:
    def __init__(self, *, h=0.0, s=1.0, p=1.0, T=300.0):
        self._h = h
        self._s = s
        self._p = p
        self._T = T

    def hmass(self):
        return self._h

    def smass(self):
        return self._s

    def p(self):
        return self._p

    def T(self):
        return self._T


@pytest.mark.parametrize(
    "is_condenser",
    [True, False],
    ids=["condenser", "evaporator"],
)
def test_zero_process_duty_skips_roundoff_profile(is_condenser):
    """A one-ULP property residue must not create a zero-duty process stream."""
    cycle = VapourCompressionCycle()
    cycle._solved = True
    cycle._m_dot = 0.005
    cycle._Q_heat = 0.0
    cycle._Q_cool = 0.0
    enthalpy = 230_289.987
    temperature = 22.0
    target_temperature = temperature - 0.01 if is_condenser else temperature + 0.01
    profile = np.array(
        [
            [enthalpy, temperature],
            [np.nextafter(enthalpy, np.inf), target_temperature],
        ]
    )
    profile_calls = 0

    def _profile_with_roundoff():
        nonlocal profile_calls
        profile_calls += 1
        return profile

    if is_condenser:
        cycle._build_condenser_profile = _profile_with_roundoff
    else:
        cycle._build_evaporator_profile = _profile_with_roundoff

    streams = cycle.build_stream_collection(
        include_cond=is_condenser,
        include_evap=not is_condenser,
    )

    assert len(streams) == 0
    assert profile_calls == 0


@given(zero_duty_stream_side_cases())
def test_zero_process_duty_omission_invariant(case):
    """Every process duty within project tolerance omits its derived stream."""
    cycle = VapourCompressionCycle()
    cycle._solved = True
    cycle._m_dot = case.mass_flow
    cycle._Q_heat = case.duty if case.is_condenser else None
    cycle._Q_cool = None if case.is_condenser else case.duty
    profile_calls = 0

    def _generated_profile():
        nonlocal profile_calls
        profile_calls += 1
        return np.asarray(case.profile)

    if case.is_condenser:
        cycle._build_condenser_profile = _generated_profile
    else:
        cycle._build_evaporator_profile = _generated_profile

    streams = cycle.build_stream_collection(
        include_cond=case.is_condenser,
        include_evap=not case.is_condenser,
    )

    assert len(streams) == 0
    assert profile_calls == 0


def _validate_results(
    cycle: VapourCompressionCycle,
    T_evap,
    T_cond,
    dT_superheat,
    dT_subcool,
):
    """Validate results for this test module."""
    cond_streams = cycle.build_stream_collection(include_cond=True)
    evap_streams = cycle.build_stream_collection(include_evap=True)

    assert len(cycle.Hs) == VapourCompressionCycle.STATECOUNT
    assert len(cycle.Ts) == VapourCompressionCycle.STATECOUNT
    assert len(cycle.Ps) == VapourCompressionCycle.STATECOUNT
    assert np.isclose(cycle.q_cond - cycle.q_evap - cycle.w_net, 0)
    assert np.isclose(cycle.Q_cond - cycle.Q_evap - cycle.work, 0)
    assert np.isclose(cycle.Q_cond - cycle.Q_heat - cycle.Q_cas_heat, 0)
    assert len(cond_streams) <= 1
    assert len(evap_streams) <= 1
    assert all(stream.has_segments for stream in [*cond_streams, *evap_streams])
    assert np.isclose(sum([s.heat_flow for s in cond_streams]) - cycle.Q_heat, 0)
    if cond_streams:
        assert np.isclose(
            min(
                segment.t_target
                for stream in cond_streams
                for segment in stream.segments
            )
            - (T_cond - dT_subcool),
            0,
            atol=0.02,
        )
    if evap_streams:
        assert np.isclose(
            evap_streams[-1].t_target - (T_evap + dT_superheat), 0, atol=0.02
        )

    if evap_streams and evap_streams[0].type == ST.Cold.value:
        assert np.isclose(sum([s.heat_flow for s in evap_streams]) - cycle.Q_cool, 0)
    elif evap_streams:
        assert np.isclose(
            sum([s.heat_flow for s in evap_streams]) - cycle.Q_cool * -1, 0
        )
    else:
        assert np.isclose(cycle.Q_cool, 0)

    assert cycle.COP_h >= 1 if cycle.q_evap >= 0 else cycle.COP_h < 1
    assert cycle.COP_r >= 0 if cycle.q_evap >= 0 else cycle.COP_r < 0


def test_heat_pump_cycle_case_1():
    T_evap = -15  # degC, evaporator saturation temperature
    T_cond = 35  # degC, condenser saturation temperature
    dT_superheat = 5.0  # K superheat
    dT_subcool = 5.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dT_ihx_gas_side=5.0,
        eta_comp=0.75,
        refrigerant="R134a",
        Q_heat=1000,
        is_heat_pump=True,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_requires_dtcont():
    cycle = VapourCompressionCycle()

    with pytest.raises(TypeError, match="dtcont"):
        cycle.solve(T_evap=-15.0, T_cond=35.0, refrigerant="R134a")


def test_heat_pump_cycle_exposes_kelvin_saturation_temperature_properties():
    cycle = VapourCompressionCycle()
    cycle.temperature_unit = "K"
    cycle._T_evap_sat_vap = 280.0
    cycle._T_cond_sat_liq = 330.0

    assert cycle.T_evap_sat_vap == pytest.approx(280.0)
    assert cycle.T_cond_sat_liq == pytest.approx(330.0)


def test_heat_pump_cycle_rejects_evaporator_pressure_above_condenser(monkeypatch):
    monkeypatch.setattr(
        VapourCompressionCycle,
        "_get_fluid_state",
        staticmethod(lambda _value: _FakeFluid()),
    )
    cycle = VapourCompressionCycle()
    pressures = iter([2.0, 1.0])
    monkeypatch.setattr(
        cycle, "_get_P_sat_from_T", lambda *_args, **_kwargs: next(pressures)
    )

    with pytest.raises(ValueError, match="Evaporator pressure"):
        cycle.solve(
            T_evap=20.0,
            T_cond=60.0,
            dtcont=0.0,
            refrigerant="R134a",
        )


def test_heat_pump_cycle_penalties_precede_condenser_enthalpy_guard(monkeypatch):
    monkeypatch.setattr(
        VapourCompressionCycle,
        "_get_fluid_state",
        staticmethod(lambda _value: _FakeFluid()),
    )
    cycle = VapourCompressionCycle()
    monkeypatch.setattr(cycle, "_get_P_sat_from_T", lambda *_args, **_kwargs: 1.0)

    quality_states = iter(
        [
            _FakeState(T=280.0),
            _FakeState(h=110.0),
            _FakeState(T=330.0),
        ]
    )
    temperature_states = iter(
        [
            _FakeState(h=100.0, T=285.0),
            _FakeState(h=150.0, T=380.0),
            _FakeState(h=120.0, T=340.0),
            _FakeState(h=90.0, T=320.0),
        ]
    )

    monkeypatch.setattr(
        cycle,
        "_compute_state_from_pressure_quality",
        lambda *_args, **_kwargs: next(quality_states),
    )
    monkeypatch.setattr(
        cycle,
        "_compute_state_from_pressure_temperature",
        lambda *_args, **_kwargs: next(temperature_states),
    )
    monkeypatch.setattr(
        cycle,
        "_compute_compressor_outlet_state",
        lambda **_kwargs: _FakeState(h=100.0, T=360.0),
    )
    monkeypatch.setattr(
        cycle,
        "_compute_state_from_pressure_enthalpy",
        lambda **kwargs: _FakeState(h=kwargs["h"], T=310.0),
    )

    with pytest.raises(ValueError, match="Condenser cannot"):
        cycle.solve(
            T_evap=20.0,
            T_cond=80.0,
            dtcont=0.0,
            refrigerant="R134a",
            dT_ihx_gas_side=100.0,
        )

    assert len(cycle._penalty) == 2


def test_heat_pump_cycle_case_2():
    T_evap = -15  # degC, evaporator saturation temperature
    T_cond = 35  # degC, condenser saturation temperature
    dT_superheat = 0.0  # K superheat
    dT_subcool = 0.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dT_ihx_gas_side=5.0,
        eta_comp=0.75,
        refrigerant="R134a",
        Q_heat=1000,
        is_heat_pump=True,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_3():
    T_evap = -15  # degC, evaporator saturation temperature
    T_cond = 35  # degC, condenser saturation temperature
    dT_superheat = 0.0  # K superheat
    dT_subcool = 0.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dT_ihx_gas_side=5.0,
        eta_comp=0.75,
        refrigerant="R134a",
        Q_heat=0,
        is_heat_pump=True,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_4():
    T_evap = -15  # degC, evaporator saturation temperature
    T_cond = 35  # degC, condenser saturation temperature
    dT_superheat = 5.0  # K superheat
    dT_subcool = 0.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dT_ihx_gas_side=0,
        eta_comp=0.75,
        refrigerant="R134a",
        Q_heat=1000,
        is_heat_pump=True,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_5():
    T_evap = 5  # degC, evaporator saturation temperature
    T_cond = 170  # degC, condenser saturation temperature
    dT_superheat = 10.0  # K superheat
    dT_subcool = 10.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dT_ihx_gas_side=0,
        eta_comp=0.75,
        refrigerant="R601",
        Q_heat=1000,
        is_heat_pump=True,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_6():
    T_evap = 5  # degC, evaporator saturation temperature
    T_cond = 170  # degC, condenser saturation temperature
    dT_superheat = 0.0  # K superheat
    dT_subcool = 0.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dT_ihx_gas_side=0,
        eta_comp=0.75,
        refrigerant="R601",
        Q_heat=1000,
        is_heat_pump=True,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_7():
    T_evap = 5  # degC, evaporator saturation temperature
    T_cond = 170  # degC, condenser saturation temperature
    dT_superheat = 10.0  # K superheat
    dT_subcool = 10.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dT_ihx_gas_side=10,
        eta_comp=0.75,
        refrigerant="R601",
        Q_heat=1000,
        is_heat_pump=True,
    )

    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)


def test_heat_pump_cycle_case_8():
    T_evap = 5  # degC, evaporator saturation temperature
    T_cond = 170  # degC, condenser saturation temperature
    dT_superheat = 10.0  # K superheat
    dT_subcool = 10.0  # K subcooling

    cycle1 = VapourCompressionCycle()
    cycle1.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dT_ihx_gas_side=10,
        eta_comp=0.75,
        refrigerant="R601",
        Q_heat=500,
        Q_cas_heat=500,
        is_heat_pump=True,
    )
    _validate_results(cycle1, T_evap, T_cond, dT_superheat, dT_subcool)

    cycle2 = VapourCompressionCycle()
    cycle2.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dT_ihx_gas_side=10,
        eta_comp=0.75,
        refrigerant="R601",
        Q_heat=500,
        Q_cas_heat=500,
        is_heat_pump=True,
    )
    _validate_results(cycle2, T_evap, T_cond, dT_superheat, dT_subcool)

    assert np.isclose(cycle2.m_dot, cycle1.m_dot, 0.0)
    assert np.isclose(cycle2.Q_cond, cycle1.Q_cond, 0.0)
    assert np.isclose(cycle2.work, cycle1.work, 0.0)


def test_heat_pump_cycle_case_9():
    T_evap = 5  # degC, evaporator saturation temperature
    T_cond = 170  # degC, condenser saturation temperature
    dT_superheat = 10.0  # K superheat
    dT_subcool = 10.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dT_ihx_gas_side=10,
        eta_comp=0.75,
        refrigerant="R601",
        Q_heat=500,
        Q_cas_heat=500,
        Q_cool=300,
        is_heat_pump=True,
    )
    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)
    assert np.isclose(cycle.Q_evap, cycle.Q_cool + cycle.Q_cas_cool, 0.0)
    assert np.isclose(cycle.Q_cond, cycle.Q_heat + cycle.Q_cas_heat, 0.0)

    streams = cycle.build_stream_collection(include_cond=True, include_evap=True)
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_cold_streams()]), cycle.Q_cool, 0
    )
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_hot_streams()]), cycle.Q_heat, 0
    )


def test_heat_pump_cycle_case_10():
    T_evap = 0.0  # degC, evaporator saturation temperature
    T_cond = 24  # degC, condenser saturation temperature
    dT_superheat = 5.0  # K superheat
    dT_subcool = 2.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dT_ihx_gas_side=10,
        eta_comp=0.7,
        refrigerant="R134A",
        Q_heat=0.0,
        Q_cas_heat=900,
        Q_cool=np.nan,
        is_heat_pump=True,
    )
    _validate_results(cycle, T_evap, T_cond, dT_superheat, dT_subcool)
    assert np.isclose(cycle.Q_evap, cycle.Q_cool + cycle.Q_cas_cool, 0.0)
    assert np.isclose(cycle.Q_cond, cycle.Q_heat + cycle.Q_cas_heat, 0.0)

    streams = cycle.build_stream_collection(include_cond=True, include_evap=True)
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_cold_streams()]), cycle.Q_cool, 0
    )
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_hot_streams()]), cycle.Q_heat, 0
    )


def test_heat_pump_cycle_with_zeotropic_mixture_generates_gliding_profiles():
    T_evap = 0.0  # degC, evaporator dew temperature
    T_cond = 35.0  # degC, condenser dew temperature
    dT_superheat = 5.0  # K superheat
    dT_subcool = 3.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dT_ihx_gas_side=5.0,
        eta_comp=0.75,
        refrigerant="R407C",
        Q_heat=1000.0,
        is_heat_pump=True,
    )

    cond_streams = cycle.build_stream_collection(include_cond=True)
    evap_streams = cycle.build_stream_collection(include_evap=True)

    assert cycle.solved is True
    assert cycle.refrigerant == "R407C"
    assert np.isclose(cycle.Q_cond - cycle.Q_evap - cycle.work, 0.0)
    assert np.isclose(cycle.Q_cond - cycle.Q_heat - cycle.Q_cas_heat, 0.0)
    assert np.isclose(sum(s.heat_flow for s in cond_streams), cycle.Q_heat, 0.0)
    assert np.isclose(sum(s.heat_flow for s in evap_streams), cycle.Q_cool, 0.0)

    # Zeotropic blends should preserve glide across the phase-change profiles.
    cond_segments = [segment for stream in cond_streams for segment in stream.segments]
    evap_segments = [segment for stream in evap_streams for segment in stream.segments]
    assert min(s.t_target for s in cond_segments) < T_cond
    assert max(s.t_target for s in cond_segments) > T_cond
    assert min(s.t_supply for s in evap_segments) < T_evap
    assert max(s.t_target for s in evap_segments) > T_evap
    assert all(abs(s.t_supply - s.t_target) > 0.05 for s in cond_segments)
    assert all(abs(s.t_supply - s.t_target) > 0.05 for s in evap_segments)


def test_heat_pump_cycle_accepts_binary_mole_fraction_refrigerant_string():
    refrigerant = "HEOS::R32[0.5]&R125[0.5]"

    cycle = VapourCompressionCycle()
    cycle.state = refrigerant

    assert cycle.state.fluid_names() == ["R32", "R125"]
    assert np.allclose(cycle.state.get_mole_fractions(), [0.5, 0.5])


def test_heat_pump_cycle_with_binary_mole_fraction_refrigerant_solves():
    refrigerant = "HEOS::R32[0.5]&R125[0.5]"
    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=0.0,
        T_cond=35.0,
        dtcont=0.0,
        dT_superheat=5.0,
        dT_subcool=3.0,
        dT_ihx_gas_side=5.0,
        eta_comp=0.75,
        refrigerant=refrigerant,
        Q_heat=1000.0,
        is_heat_pump=True,
    )

    cond_streams = cycle.build_stream_collection(include_cond=True)
    evap_streams = cycle.build_stream_collection(include_evap=True)

    assert cycle.solved is True
    assert cycle.refrigerant == refrigerant
    assert cycle.state.fluid_names() == ["R32", "R125"]
    assert np.allclose(cycle.state.get_mole_fractions(), [0.5, 0.5])
    assert len(cycle.Hs) == VapourCompressionCycle.STATECOUNT
    assert np.isclose(cycle.Q_cond - cycle.Q_evap - cycle.work, 0.0)
    assert np.isclose(cycle.Q_cond - cycle.Q_heat - cycle.Q_cas_heat, 0.0)
    assert np.isclose(sum(s.heat_flow for s in cond_streams), cycle.Q_heat, 0.0)
    assert np.isclose(sum(s.heat_flow for s in evap_streams), cycle.Q_cool, 0.0)
    assert cycle.COP_h > 1.0


def test_refrigeration_cycle_uses_q_cool_as_primary_duty():
    T_evap = -15  # degC, evaporator saturation temperature
    T_cond = 35  # degC, condenser saturation temperature
    dT_superheat = 5.0  # K superheat
    dT_subcool = 5.0  # K subcooling

    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=T_evap,
        T_cond=T_cond,
        dtcont=0.0,
        dT_superheat=dT_superheat,
        dT_subcool=dT_subcool,
        dT_ihx_gas_side=5.0,
        eta_comp=0.75,
        refrigerant="R134a",
        Q_cool=1000.0,
        Q_cas_cool=150.0,
        Q_heat=np.nan,
        is_heat_pump=False,
    )

    assert len(cycle.Hs) == VapourCompressionCycle.STATECOUNT
    assert np.isclose(cycle.Q_cond - cycle.Q_evap - cycle.work, 0.0)
    assert np.isclose(cycle.Q_evap, cycle.Q_cool + cycle.Q_cas_cool, 0.0)
    assert np.isclose(cycle.Q_heat, cycle.Q_cond, 0.0)
    assert np.isclose(cycle.Q_cas_heat, 0.0, 0.0)
    assert cycle.COP_r > 0.0
    assert cycle.COP_h > 1.0

    streams = cycle.build_stream_collection(include_cond=True, include_evap=True)
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_cold_streams()]), cycle.Q_cool, 0
    )
    assert np.isclose(
        sum([s.heat_flow for s in streams.get_hot_streams()]), cycle.Q_heat, 0
    )


def test_refrigeration_cycle_caps_q_heat_to_available_q_cond():
    cycle = VapourCompressionCycle()
    cycle.solve(
        T_evap=-10.0,
        T_cond=30.0,
        dtcont=0.0,
        dT_superheat=3.0,
        dT_subcool=2.0,
        dT_ihx_gas_side=5.0,
        eta_comp=0.75,
        refrigerant="R134a",
        Q_cool=600.0,
        Q_cas_cool=0.0,
        Q_heat=1e9,
        is_heat_pump=False,
    )

    assert cycle.Q_heat <= cycle.Q_cond + 1e-6
    assert np.isclose(cycle.Q_cas_heat, 0.0, 0.0)
    assert np.isclose(cycle.Q_cond, cycle.Q_heat + cycle.Q_cas_heat, 0.0)


# ===== Merged from test_simple_heat_pump_cycle_extra.py =====
"""Additional branch coverage tests for VapourCompressionCycle."""


class _CycleStore(dict):
    def __getitem__(self, key):
        return super().get(key, 0.0)


def test_simple_cycle_property_getters_and_require_solution():
    hp = VapourCompressionCycle()
    hp._cycle_states = "cycle-states"
    hp._q_cas_cool = 1.0
    hp._q_cool = 2.0
    hp._q_cas_heat = 3.0
    hp._q_heat = 4.0
    hp._penalty = 0.5
    hp._dt_diff_max = 0.25
    hp._refrigerant = "water"
    hp._T_evap = 25.0
    hp._T_cond = 70.0
    hp._dT_superheat = 5.0
    hp._dT_subcool = 2.0
    hp._eta_comp = 0.8
    hp._ihx_gas_dt = 7.0

    assert hp.cycle_states == "cycle-states"
    assert hp.q_cas_cool == 1.0
    assert hp.q_cool == 2.0
    assert hp.q_cas_heat == 3.0
    assert hp.q_heat == 4.0
    assert hp.penalty == 0.5
    assert hp.dt_diff_max == 0.25
    assert hp.refrigerant == "water"
    assert hp.T_evap == 25.0
    assert hp.T_cond == 70.0
    assert hp.dT_superheat == 5.0
    assert hp.dT_subcool == 2.0
    assert hp.eta_comp == 0.8
    assert hp.dT_ihx_gas_side == 7.0

    hp._solved = False
    with pytest.raises(RuntimeError, match="Solve the cycle"):
        hp._require_solution()


def test_validate_solve_inputs_and_get_psat_high_temperature_branch():
    hp = VapourCompressionCycle()
    with pytest.raises(ValueError, match="A fluid must be specified"):
        hp._validate_solve_inputs(refrigerant=None)

    hp._refrigerant = "water"
    hp._t_crit = 350.0
    hp._d_crit = 1.0
    assert np.isclose(hp._get_P_sat_from_T(T=360.0), 62193.56549054144)


def test_build_evaporator_profile_subcooled_segment_branch(monkeypatch):
    class _ProfileState:
        def __init__(self, h):
            self._h = float(h)

        def hmass(self):
            return self._h

        def T(self):
            return 273.15 + self._h

    hp = VapourCompressionCycle()
    hp._solved = True
    hp._p_crit = 10.0
    hp._dt_diff_max = 0.5

    monkeypatch.setattr(
        VapourCompressionCycle,
        "Hs",
        property(lambda self: [0.0, 0.0, 0.0, 1.0, 0.0, 5.0]),
    )
    monkeypatch.setattr(VapourCompressionCycle, "Ps", property(lambda self: [1.0]))
    monkeypatch.setattr(
        shp_mod,
        "get_piecewise_data_points",
        lambda curve, is_hot_stream, dt_diff_max: curve,
    )
    monkeypatch.setattr(
        VapourCompressionCycle,
        "_compute_state_from_pressure_quality",
        lambda self, P, Q: _ProfileState(4.0 if Q == 1.0 else 2.0),
    )
    monkeypatch.setattr(
        VapourCompressionCycle,
        "_compute_state_from_pressure_enthalpy",
        lambda self, P, h: _ProfileState(h),
    )

    out = hp._build_evaporator_profile()
    assert isinstance(out, np.ndarray)
    assert out.shape[0] > 0


def test_simple_heat_pump_state_and_cycle_state_setters():
    hp = VapourCompressionCycle()

    assert hp.system is not None
    hp.state = "HEOS::Water"
    assert hp.state is not None
    assert hp.solved is False

    bad = [{}]
    bad[0]["H"] = 1.0
    with pytest.raises(ValueError, match="Expected exactly"):
        hp.cycle_states = bad

    good = [{} for _ in range(hp.STATECOUNT)]
    for i in range(hp.STATECOUNT):
        good[i]["H"] = 1_000.0 + i
        good[i]["S"] = 10.0 + i
        good[i]["P"] = 100_000.0 + i
        good[i]["T"] = 300.0 + i
    hp.cycle_states = good
    hp._solved = True

    assert hp.solved is True
    assert len(hp.state_points) == hp.STATECOUNT
    assert len(hp.Hs) == hp.STATECOUNT
    assert len(hp.Ss) == hp.STATECOUNT
    assert len(hp.Ts) == hp.STATECOUNT
    assert len(hp.Ps) == hp.STATECOUNT


def test_simple_heat_pump_cop_zero_division_and_misc_helpers(monkeypatch):
    hp = VapourCompressionCycle()
    hp._solved = True
    hp._q_cond = 10.0
    hp._q_evap = 5.0
    hp._w_net = 0.0
    with pytest.raises(ZeroDivisionError):
        _ = hp.COP_h
    with pytest.raises(ZeroDivisionError):
        _ = hp.COP_r

    hp.dtcont = 3.0
    assert hp.dtcont == 3.0
    assert hp._convert_C_to_K(10.0) == pytest.approx(283.15)
    assert hp._convert_K_to_C(283.15) == pytest.approx(10.0)

    hp._cycle_states = hp._new_cycle_states()
    state = hp._compute_state_from_pressure_temperature(100_000.0, 300.0, phase=1)
    hp._save_cycle_state(state, 0)

    assert hp._cycle_states[0]["H"] == pytest.approx(112653.67968857559)


def test_simple_heat_pump_celsius_temperature_properties_handle_values_and_none():
    hp = VapourCompressionCycle()
    hp.temperature_unit = "C"
    hp._T_evap = None
    hp._T_evap_sat_vap = None
    hp._T_cond = 363.15
    hp._T_cond_sat_liq = None

    assert hp.T_evap is None
    assert hp.T_evap_sat_vap is None
    assert hp.T_cond == pytest.approx(90.0)
    assert hp.T_cond_sat_liq is None

    hp._T_evap = 283.15
    hp._T_evap_sat_vap = 282.15
    hp._T_cond_sat_liq = 360.15

    assert hp.T_evap == pytest.approx(10.0)
    assert hp.T_evap_sat_vap == pytest.approx(9.0)
    assert hp.T_cond_sat_liq == pytest.approx(87.0)


def test_simple_heat_pump_single_phase_solver_brackets_and_reports_failure():
    low_hit = _LinearSinglePhaseCycle()._solve_single_phase_state_from_pressure_target(
        P=1.0,
        target=10.0,
        property_name="hmass",
        T_low=10.0,
        T_high=20.0,
    )
    high_hit = _LinearSinglePhaseCycle()._solve_single_phase_state_from_pressure_target(
        P=1.0,
        target=20.0,
        property_name="hmass",
        T_low=10.0,
        T_high=20.0,
    )
    expanded_low = (
        _LinearSinglePhaseCycle()._solve_single_phase_state_from_pressure_target(
            P=1.0,
            target=5.0,
            property_name="hmass",
            T_low=20.0,
            T_high=30.0,
            expand="low",
        )
    )

    assert low_hit.T() == pytest.approx(10.0)
    assert high_hit.T() == pytest.approx(20.0)
    assert expanded_low.T() == pytest.approx(5.0)

    with pytest.raises(ValueError, match="Could not bracket"):
        _LinearSinglePhaseCycle(
            constant_property=1.0
        )._solve_single_phase_state_from_pressure_target(
            P=1.0,
            target=0.0,
            property_name="hmass",
            T_low=10.0,
            T_high=20.0,
        )


class _LinearSinglePhaseState:
    def __init__(self, constant_property: float | None = None) -> None:
        self._temperature = 0.0
        self._constant_property = constant_property

    def update(self, _input_pair, _pressure, temperature):
        self._temperature = float(temperature)

    def hmass(self):
        if self._constant_property is not None:
            return self._constant_property
        return self._temperature


class _LinearSinglePhaseCycle(VapourCompressionCycle):
    def __init__(self, constant_property: float | None = None) -> None:
        super().__init__()
        self._refrigerant = "linear"
        self._constant_property = constant_property

    def _get_fluid_state(self, value):
        del value
        return _LinearSinglePhaseState(self._constant_property)

    def _compute_state_from_pressure_temperature(self, P, T, *, phase=1.0):
        del P, phase
        return SimpleNamespace(T=lambda: float(T))
