from CoolProp.CoolProp import PropsSI

FLUID = 'Water'

def Tsat_p(P):
    P = toSIunit_p(P)
    T = PropsSI('T', 'P', P, 'Q', 1, FLUID)
    return fromSIunit_T(T)

def psat_T(T):
    T = toSIunit_T(T)
    p = PropsSI('P', 'T', T, 'Q', 1, FLUID)
    return fromSIunit_p(p)

def hV_p(P):
    P = toSIunit_p(P)
    h = PropsSI('H', 'P', P, 'Q', 1, FLUID)
    return fromSIunit_h(h)

def hL_p(P):
    P = toSIunit_p(P)
    h = PropsSI('H', 'P', P, 'Q', 0, FLUID)
    return fromSIunit_h(h)

def h_pT(P, T):
    P = toSIunit_p(P)
    T = toSIunit_T(T)
    h = PropsSI('H', 'P', P, 'T', T, FLUID)
    return fromSIunit_h(h)

def h_ps(P, s):
    P = toSIunit_p(P)
    s = toSIunit_s(s)
    h = PropsSI('H', 'P', P, "S", s, FLUID)
    return fromSIunit_h(h)

def s_ph(P, H):
    P = toSIunit_p(P)
    H = toSIunit_h(H)
    s = PropsSI("S", "H", H, "P", P, FLUID)
    return fromSIunit_s(s)


"""'***********************************************************************************************************
'*6 Units                                                                                      *
'***********************************************************************************************************
"""


def toSIunit_p(Ins):
    # Translate bar to Pa
    if Ins == None:
        Ins = 0
    return Ins * 100000

def fromSIunit_p(Ins):
    # Translate MPa to bar
    if Ins == None:
        Ins = 0
    return Ins / 100000

def toSIunit_T(Ins):
    # Translate degC to Kelvin
    if Ins == None:
        Ins = 0
    return Ins + 273.15

def fromSIunit_T(Ins):
    # Translate Kelvin to degC
    if Ins == None:
        Ins = 0
    return Ins - 273.15

def toSIunit_h(Ins):
    if Ins == None:
        Ins = 0
    return Ins * 1000 

def fromSIunit_h(Ins):
    if Ins == None:
        Ins = 0
    return Ins / 1000 

def toSIunit_s(Ins):
    return Ins

def fromSIunit_s(Ins):
    return Ins