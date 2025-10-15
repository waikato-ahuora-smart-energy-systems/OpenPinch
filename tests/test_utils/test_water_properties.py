from OpenPinch.utils.water_properties import *


def test_unit_conversions():
    assert toSIunit_p(1.0) == 100000
    assert fromSIunit_p(100000) == 1.0
    assert toSIunit_T(0.0) == 273.15
    assert fromSIunit_T(373.15) == 100.0
    assert toSIunit_h(1.0) == 1000
    assert fromSIunit_h(1000) == 1.0
    assert toSIunit_s(2.5) == 2.5
    assert fromSIunit_s(2.5) == 2.5


def test_thermo_functions():
    P = 1.0  # bar
    T = 100.0  # degC
    h = h_pT(P, T)
    s = s_ph(P, h)

    assert round(Tsat_p(P), 1) == 99.6
    assert round(psat_T(T), 2) == 1.01  # approx saturation pressure of water at 100°C
    assert round(hV_p(P), -2) in range(2600, 2800)
    assert round(hL_p(P), -2) in range(400, 500)
    assert round(h_pT(P, T), 0) == round(h, 0)
    assert round(h_ps(P, s), 0) == round(h, 0)
    assert round(s_ph(P, h), 2) == round(s, 2)
