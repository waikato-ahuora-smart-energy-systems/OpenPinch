import math
from ..lib.enums import HeatExchangerTypes as HX


def HX_Eff(Arrangement, Ntu, c, Passes=None, Rows=None, Cmin_Phase=None):
    if Passes == None:
        Passes = 1

    Ntu = Ntu / Passes
    if Ntu > 0 and c >= 0:
        # Counter Flow - Single Pass Effectiveness
        if Arrangement == HX.CF.value:
            # test = c * math.exp(-Ntu * (1 - c))
            if c != 1 and c * math.exp(-Ntu * (1 - c)) != 1:
                eff = (1 - math.exp(-Ntu * (1 - c))) / (1 - c * math.exp(-Ntu * (1 - c)))
            else:
                eff = Ntu / (1 + Ntu)
        # Parallel Flow - Single Pass Effectiveness
        elif Arrangement == HX.PF.value:
            eff = (1 - math.exp(-Ntu * (1 + c))) / (1 + c)
        # Cross Flow - Both Streams Unmixed Effectiveness
        elif Arrangement == HX.CrFUU:
            if Rows == None or Cmin_Phase == None:
                eff = CrossflowUnmixedEff1(Ntu, c)
            else:
                eff = CrossflowUnmixedEff2(Ntu, c, Rows, Cmin_Phase)
        # Cross Flow - Both Streams Mixed Effectiveness
        elif Arrangement == HX.CrFMM:
            eff = (1 / (1 - math.exp(-Ntu)) + c / (1 - math.exp(-Ntu * c)) - 1 / Ntu) ** -1
        # Cross Flow - Stream Cmax Unmixed Effectiveness
        elif Arrangement == HX.CrFMUmax:
            eff = 1 - math.exp(-1 / c * (1 - math.exp(-Ntu * c)))
        # Cross Flow - Stream Cmin Unmixed Effectiveness
        elif Arrangement == HX.CrFMUmin:
            eff = 1 / c * (1 - math.exp(-c * (1 - math.exp(-Ntu))))
        # Shell and Tube - One Shell Pass; 2,4,6, etc., Tube Passes Effectiveness
        elif Arrangement == HX.ShellTube.value:
            d = (1 + c ** 2) ** 0.5
            eff = 2 / ((1 + c) + d ** 0.5 * Coth(Ntu * d / 2))
        # Condensing or Evaporating of One Fluid
        elif Arrangement == HX.CondEvap:
            eff = 1 - math.exp(-Ntu)
        else:
            eff = HX_Eff(HX.CF.value, Ntu, c, 1)
    else:
        eff = 0

    if Passes > 1:
        Eff_p = eff
        return MultiPassEff(Eff_p, c, Passes)
    else:
        return eff


def HX_NTU(Arrangement, eff, c, Passes=None):
    if Passes == None:
        Passes = 1

    if Passes > 1:
        Eff_p = MultiPassNTU(eff, c, Passes)
        eff = Eff_p

    if eff > 0 and eff < 1:
        # Counter Flow - Single Pass Effectiveness
        if Arrangement == HX.CF.value:
            if c != 1:
                Ntu = 1 / (1 - c) * math.log((1 - eff * c) / (1 - eff))
            else:
                Ntu = eff / (1 - eff)
        # Parallel Flow - Single Pass Effectiveness
        elif Arrangement == HX.PF.value:
            Ntu = -math.log(1 - eff * (1 + c)) / (1 + c)
        # Cross Flow - Both Streams Unmixed NTU
        elif Arrangement == HX.CrFUU:
            Ntu = HX_NTU_Numerical(Arrangement, eff, c)
        # Cross Flow - Both Streams Mixed NTU
        elif Arrangement == HX.CrFMM:
            Ntu = HX_NTU_Numerical(Arrangement, eff, c)
        # Cross Flow - Stream Cmax Unmixed NTU
        elif Arrangement == HX.CrFMUmax:
            Ntu = -1 / c * math.log(1 + c * math.log(1 - eff))
        # Cross Flow - Stream Cmin Unmixed NTU
        elif Arrangement == HX.CrFMUmin:
            Ntu = -math.log(1 + 1 / c * math.log(1 - eff * c))
        # Shell and Tube - One Shell Pass; 2,4,6, etc., Tube Passes NTU
        elif Arrangement == HX.ShellTube.value:
            D1 = 1 + c - (1 + c ** 2) ** (1 / 4)
            D2 = 1 + c + (1 + c ** 2) ** (1 / 4)
            Ntu = (1 + c ** 2) ** -0.5 * math.log((2 - eff * D1) / (2 - eff * D2))
        # Condensing or Evaporating of One Fluid
        elif Arrangement == HX.CondEvap:
            Ntu = -math.log(1 - eff)
        else:
            Ntu = -1
    else:
        Ntu = 0

    return Ntu * Passes


def CalcAreaUE(Arrangement, U, C_p, T_p1, T_p2, T_u1, T_u2, Passes):
    Q = C_p * abs(T_p1 - T_p2)
    C_u = Q / abs(T_u1 - T_u2)
    if C_p < C_u:
        eff = Q / C_p / abs(T_p1 - T_u1)
        c = C_p / C_u
        Ntu = HX_NTU(Arrangement, eff, c, Passes)
        return Ntu * C_p / U
    else:
        eff = Q / C_u / abs(T_p1 - T_u1)
        c = C_u / C_p
        Ntu = HX_NTU(Arrangement, eff, c, Passes)
        return Ntu * C_u / U


def eNTU_slope_Numerical(Arrangement, Ntu, c, Passes):
    dx = 1e-6
    if Ntu > 0:
        return (HX_Eff(Arrangement, Ntu + dx, c, Passes) - HX_Eff(Arrangement, Ntu, c, Passes)) / dx


def Coth(R):
    return (math.exp(2 * R) + 1) / (math.exp(2 * R) - 1)


def MultiPassEff(eff, c, Passes):
    if c != 1:
        return (((1 - eff * c) / (1 - eff)) ** Passes - 1) / (((1 - eff * c) / (1 - eff)) ** Passes - c)
    else:
        return Passes * eff / (1 + eff * (Passes - 1))


def MultiPassNTU(Eff_p, c, Passes):
    if c != 1:
        return (((1 - Eff_p * c) / (1 - Eff_p)) ** (1 / Passes) - 1) / (((1 - Eff_p * c) / (1 - Eff_p)) ** (1 / Passes) - c)
    else:
        return Eff_p / (Passes - Eff_p * (Passes - 1))


def CrossflowUnmixedEff1(Ntu, c):
    Sum_Pn = 0
    for i in range(1, 21):
        Pn = 0
        for j in range(1, i):
            Pn = Pn + c ** i / math.factorial(i + 1) * (i - j + 1) / math.factorial(j) * Ntu ** (i + j)
        Sum_Pn = Sum_Pn + Pn
    return 1 - math.exp(-Ntu) - math.exp(-(1 + c) * Ntu) * Sum_Pn


def CrossflowUnmixedEff2(Ntu, c, Rows, Cmin_fluid):
    # ESDU 86018
    if Cmin_fluid == 'Air':
        if Rows == 1:
            eff = 1 / c * (1 - math.exp(-c * (1 - math.exp(-Ntu))))
        elif Rows == 2:
            k = 1 - math.exp(-Ntu / 2)
            eff = 1 / c * (1 - math.exp(-2 * k * c) * (1 + c * k ** 2))
        elif Rows == 3:
            k = 1 - math.exp(-Ntu / 3)
            eff = 1 / c * (1 - math.exp(-3 * k * c) * (1 + c * k ** 2 * (3 - k) + (3 * c ** 2 * k ** 4) / 2))
        elif Rows == 4:
            k = 1 - math.exp(-Ntu / 4)
            eff = 1 / c * (1 - math.exp(-4 * k * c) * (1 + c * k ** 2 * (6 - 4 * k + k ** 2) + 4 * c ** 2 * k ** 4 * (2 - k) + (8 * c ** 3 * k ** 6) / 3))
        elif Rows > 4:
            eff = CrossflowUnmixedEff1(Ntu, c)
    else:
        if Rows == 1:
            eff = 1 - math.exp(-1 / c * (1 - math.exp(-Ntu * c)))
        elif Rows == 2:
            k = 1 - math.exp(-Ntu * c / 2)
            eff = 1 - math.exp(-2 * k / c) * (1 + (k ** 2) / c)
        elif Rows == 3:
            k = 1 - math.exp(-Ntu * c / 3)
            eff = 1 - math.exp(-3 * k / c) * (1 + k ** 2 * (3 - k) / c + (3 * k ** 4) / (2 * c ** 2))
        elif Rows == 4:
            k = 1 - math.exp(-Ntu * c / 4)
            eff = 1 - math.exp(-4 * k / c) * (1 + k ** 2 * (6 - 4 * k + k ** 2) / c + 4 * k ** 4 * (2 - k) / c ** 2 + (8 * k ** 6) / (3 * c ** 3))
        elif Rows > 4:
            eff = CrossflowUnmixedEff1(Ntu, c)

    return eff


def HX_NTU_Numerical(Arrangement, eff, c):
    NTU1 = 0.001
    NTU2 = 0.1
    eps = 1e-5
    f = 100.0
    count = 1
    F1 = eff - HX_Eff(Arrangement, NTU1, c)
    F2 = eff - HX_Eff(Arrangement, NTU2, c)
    while f > eps:
        a = (F1 - F2) / (NTU1 - NTU2)
        b = F1 - a * NTU1
        NTU3 = -b / a
        F3 = eff - HX_Eff(Arrangement, NTU3, c)
        f = abs(F3)
        NTU1 = NTU2
        F1 = F2
        NTU2 = NTU3
        F2 = F3
        count += 1
        if count > 50:
            raise ValueError('Solution does not converge')

    return NTU3
