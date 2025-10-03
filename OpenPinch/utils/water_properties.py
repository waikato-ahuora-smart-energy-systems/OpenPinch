"""Convenience wrappers around CoolProp for common water property queries."""

from CoolProp.CoolProp import PropsSI

FLUID = "Water"


def Tsat_p(P):
    """Saturation temperature (°C) at pressure ``P`` (bar)."""
    P = toSIunit_p(P)
    T = PropsSI("T", "P", P, "Q", 1, FLUID)
    return fromSIunit_T(T)


def psat_T(T):
    """Saturation pressure (bar) at temperature ``T`` (°C)."""
    T = toSIunit_T(T)
    p = PropsSI("P", "T", T, "Q", 1, FLUID)
    return fromSIunit_p(p)


def hV_p(P):
    """Vapour enthalpy (kJ/kg) at pressure ``P`` (bar)."""
    P = toSIunit_p(P)
    h = PropsSI("H", "P", P, "Q", 1, FLUID)
    return fromSIunit_h(h)


def hL_p(P):
    """Liquid enthalpy (kJ/kg) at pressure ``P`` (bar)."""
    P = toSIunit_p(P)
    h = PropsSI("H", "P", P, "Q", 0, FLUID)
    return fromSIunit_h(h)


def h_pT(P, T):
    """Specific enthalpy (kJ/kg) at ``(P, T)`` where ``P`` is bar and ``T`` is °C."""
    P = toSIunit_p(P)
    T = toSIunit_T(T)
    h = PropsSI("H", "P", P, "T", T, FLUID)
    return fromSIunit_h(h)


def h_ps(P, s):
    """Specific enthalpy (kJ/kg) at pressure ``P`` (bar) and entropy ``s`` (kJ/kg/K)."""
    P = toSIunit_p(P)
    s = toSIunit_s(s)
    h = PropsSI("H", "P", P, "S", s, FLUID)
    return fromSIunit_h(h)


def s_ph(P, H):
    """Specific entropy (kJ/kg/K) at pressure ``P`` (bar) and enthalpy ``H`` (kJ/kg)."""
    P = toSIunit_p(P)
    H = toSIunit_h(H)
    s = PropsSI("S", "H", H, "P", P, FLUID)
    return fromSIunit_s(s)


"""'
***********************************************************************************************************
*2 Units                                                                                                  *
***********************************************************************************************************
"""


def toSIunit_p(Ins):
    """Convert bar to Pa."""
    # Translate bar to Pa
    if Ins == None:
        Ins = 0
    return Ins * 100000


def fromSIunit_p(Ins):
    """Convert Pa to bar."""
    # Translate MPa to bar
    if Ins == None:
        Ins = 0
    return Ins / 100000


def toSIunit_T(Ins):
    """Convert °C to Kelvin."""
    # Translate degC to Kelvin
    if Ins == None:
        Ins = 0
    return Ins + 273.15


def fromSIunit_T(Ins):
    """Convert Kelvin to °C."""
    # Translate Kelvin to degC
    if Ins == None:
        Ins = 0
    return Ins - 273.15


def toSIunit_h(Ins):
    """Convert kJ/kg to J/kg."""
    if Ins == None:
        Ins = 0
    return Ins * 1000


def fromSIunit_h(Ins):
    """Convert J/kg to kJ/kg."""
    if Ins == None:
        Ins = 0
    return Ins / 1000


def toSIunit_s(Ins):
    """Identity conversion for entropy; maintained for API symmetry."""
    return Ins


def fromSIunit_s(Ins):
    """Identity conversion for entropy; maintained for API symmetry."""
    return Ins
