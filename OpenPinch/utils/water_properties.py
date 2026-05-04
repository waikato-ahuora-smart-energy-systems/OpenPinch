"""Convenience wrappers around CoolProp for common water property queries."""

from CoolProp.CoolProp import PropsSI

FLUID = "Water"

__all__ = [
    "Tsat_p",
    "fromSIunit_T",
    "fromSIunit_h",
    "fromSIunit_p",
    "fromSIunit_s",
    "hL_p",
    "hV_p",
    "h_pT",
    "h_ps",
    "psat_T",
    "s_ph",
    "toSIunit_T",
    "toSIunit_h",
    "toSIunit_p",
    "toSIunit_s",
]


def Tsat_p(P):
    """Saturation temperature (degC) at pressure ``P`` (bar)."""
    P = toSIunit_p(P)
    T = PropsSI("T", "P", P, "Q", 1, FLUID)
    return fromSIunit_T(T)


def psat_T(T):
    """Saturation pressure (bar) at temperature ``T`` (degC)."""
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
    """Specific enthalpy (kJ/kg) at ``(P, T)`` where ``P`` is bar and ``T`` is degC."""
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


def toSIunit_p(Ins):
    """Convert bar to Pa."""
    if Ins is None:
        Ins = 0
    return Ins * 100000


def fromSIunit_p(Ins):
    """Convert Pa to bar."""
    if Ins is None:
        Ins = 0
    return Ins / 100000


def toSIunit_T(Ins):
    """Convert degC to Kelvin."""
    if Ins is None:
        Ins = 0
    return Ins + 273.15


def fromSIunit_T(Ins):
    """Convert Kelvin to degC."""
    if Ins is None:
        Ins = 0
    return Ins - 273.15


def toSIunit_h(Ins):
    """Convert kJ/kg to J/kg."""
    if Ins is None:
        Ins = 0
    return Ins * 1000


def fromSIunit_h(Ins):
    """Convert J/kg to kJ/kg."""
    if Ins is None:
        Ins = 0
    return Ins / 1000


def toSIunit_s(Ins):
    """Identity conversion for entropy; maintained for API symmetry."""
    return Ins


def fromSIunit_s(Ins):
    """Identity conversion for entropy; maintained for API symmetry."""
    return Ins
