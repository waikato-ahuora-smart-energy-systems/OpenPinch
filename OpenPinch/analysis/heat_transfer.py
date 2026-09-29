"""Heat-exchanger performance correlations used by area targeting routines."""

from __future__ import annotations

import math
from numbers import Integral

import numpy as np

from ..domain.enums import HeatExchangerTypes as HX

__all__ = [
    "CalcAreaUE",
    "Coth",
    "CrossflowUnmixedEff1",
    "CrossflowUnmixedEff2",
    "HX_Eff",
    "HX_NTU",
    "HX_NTU_Numerical",
    "MultiPassEff",
    "MultiPassNTU",
    "compute_LMTD_from_dts",
    "compute_LMTD_from_ts",
    "eNTU_slope_Numerical",
]


def compute_LMTD_from_dts(
    delta_T1: float | list | np.ndarray,
    delta_T2: float | list | np.ndarray,
) -> np.ndarray:
    """Return the LMTD for a counterflow heat exchanger from end-point deltas."""
    # Check temperature directions for counter-current assumption
    delta_T1 = np.array(delta_T1)
    delta_T2 = np.array(delta_T2)

    if delta_T1.round(6).min() <= 0 or delta_T2.round(6).min() <= 0:
        raise ValueError(
            f"Invalid temperature differences: ΔT1={delta_T1}, ΔT2={delta_T2}"
        )
    mask_equal = np.isclose(delta_T1, delta_T2, atol=1e-6)
    lmtd = np.empty_like(delta_T1, dtype=float)
    arithmetic = (delta_T1 + delta_T2) / 2
    np.copyto(lmtd, arithmetic, where=mask_equal)
    np.divide(
        delta_T1 - delta_T2,
        np.log(delta_T1 / delta_T2),
        out=lmtd,
        where=~mask_equal,
    )
    return lmtd


def compute_LMTD_from_ts(
    T_hot_in: float | list | np.ndarray,
    T_hot_out: float | list | np.ndarray,
    T_cold_in: float | list | np.ndarray,
    T_cold_out: float | list | np.ndarray,
) -> float:
    """Return the LMTD for a counterflow heat exchanger from temperatures."""
    T_hot_in = np.array(T_hot_in)
    T_hot_out = np.array(T_hot_out)
    T_cold_in = np.array(T_cold_in)
    T_cold_out = np.array(T_cold_out)

    # Check temperature directions for counter-current assumption
    if T_hot_in < T_hot_out:
        raise ValueError("Hot fluid must cool down (T_hot_in > T_hot_out)")
    if T_cold_out < T_cold_in:
        raise ValueError("Cold fluid must heat up (T_cold_out > T_cold_in)")

    return compute_LMTD_from_dts(
        T_hot_in - T_cold_out,  # Inlet diff (hottest hot - hottest cold)
        T_hot_out - T_cold_in,  # Outlet diff (coldest hot - coldest cold)
    )


def _arrangement(value) -> HX | None:
    """Return the arrangement as a ``HeatExchangerTypes`` member (enum or its value)."""
    if isinstance(value, HX):
        return value
    try:
        return HX(value)
    except ValueError:
        return None


def HX_Eff(Arrangement, Ntu, c, Passes=None, Rows=None, Cmin_Phase=None):
    """Return heat exchanger effectiveness for the specified arrangement/NTU/c ratio.

    ``Arrangement`` is a ``HeatExchangerTypes`` member or its string value; an
    unknown arrangement is treated as counter flow.
    """
    if Passes is None:
        Passes = 1
    Arrangement = _arrangement(Arrangement)

    Ntu = Ntu / Passes
    if Ntu > 0 and c >= 0:
        # c = 0: one stream has infinite capacity (phase change), and every
        # arrangement reduces to the same limit.
        if c == 0:
            eff = 1 - math.exp(-Ntu)
        # Counter Flow - Single Pass Effectiveness
        elif Arrangement is HX.CF:
            # test = c * math.exp(-Ntu * (1 - c))
            if c != 1 and c * math.exp(-Ntu * (1 - c)) != 1:
                eff = (1 - math.exp(-Ntu * (1 - c))) / (
                    1 - c * math.exp(-Ntu * (1 - c))
                )
            else:
                eff = Ntu / (1 + Ntu)
        # Parallel Flow - Single Pass Effectiveness
        elif Arrangement is HX.PF:
            eff = (1 - math.exp(-Ntu * (1 + c))) / (1 + c)
        # Cross Flow - Both Streams Unmixed Effectiveness
        elif Arrangement is HX.CrFUU:
            if Rows is None or Cmin_Phase is None:
                eff = CrossflowUnmixedEff1(Ntu, c)
            else:
                eff = CrossflowUnmixedEff2(Ntu, c, Rows, Cmin_Phase)
        # Cross Flow - Both Streams Mixed Effectiveness
        elif Arrangement is HX.CrFMM:
            eff = (
                1 / (1 - math.exp(-Ntu)) + c / (1 - math.exp(-Ntu * c)) - 1 / Ntu
            ) ** -1
        # Cross Flow - Stream Cmax Unmixed Effectiveness
        elif Arrangement is HX.CrFMUmax:
            eff = 1 - math.exp(-1 / c * (1 - math.exp(-Ntu * c)))
        # Cross Flow - Stream Cmin Unmixed Effectiveness
        elif Arrangement is HX.CrFMUmin:
            eff = 1 / c * (1 - math.exp(-c * (1 - math.exp(-Ntu))))
        # Shell and Tube - One Shell Pass; 2,4,6, etc., Tube Passes Effectiveness
        elif Arrangement is HX.ShellTube:
            d = (1 + c**2) ** 0.5
            eff = 2 / ((1 + c) + d * Coth(Ntu * d / 2))
        # Condensing or Evaporating of One Fluid
        elif Arrangement is HX.CondEvap:
            eff = 1 - math.exp(-Ntu)
        else:
            eff = HX_Eff(HX.CF, Ntu, c, 1)
    else:
        eff = 0

    if Passes > 1:
        Eff_p = eff
        return MultiPassEff(Eff_p, c, Passes)
    else:
        return eff


def HX_NTU(Arrangement, eff, c, Passes=None):
    """Compute NTU for a target effectiveness and exchanger arrangement.

    ``Arrangement`` is a ``HeatExchangerTypes`` member or its string value; an
    unknown arrangement returns ``-1``.
    """
    if Passes is None:
        Passes = 1
    Arrangement = _arrangement(Arrangement)

    if Passes > 1:
        Eff_p = MultiPassNTU(eff, c, Passes)
        eff = Eff_p

    if eff > 0 and eff < 1:
        # c = 0: the phase-change limit shared by every arrangement.
        if c == 0 and Arrangement is not None:
            Ntu = -math.log(1 - eff)
        # Counter Flow - Single Pass Effectiveness
        elif Arrangement is HX.CF:
            if c != 1:
                Ntu = 1 / (1 - c) * math.log((1 - eff * c) / (1 - eff))
            else:
                Ntu = eff / (1 - eff)
        # Parallel Flow - Single Pass Effectiveness
        elif Arrangement is HX.PF:
            Ntu = -math.log(1 - eff * (1 + c)) / (1 + c)
        # Cross Flow - Both Streams Unmixed NTU
        elif Arrangement is HX.CrFUU:
            Ntu = HX_NTU_Numerical(Arrangement, eff, c)
        # Cross Flow - Both Streams Mixed NTU
        elif Arrangement is HX.CrFMM:
            Ntu = HX_NTU_Numerical(Arrangement, eff, c)
        # Cross Flow - Stream Cmax Unmixed NTU
        elif Arrangement is HX.CrFMUmax:
            Ntu = -1 / c * math.log(1 + c * math.log(1 - eff))
        # Cross Flow - Stream Cmin Unmixed NTU
        elif Arrangement is HX.CrFMUmin:
            Ntu = -math.log(1 + 1 / c * math.log(1 - eff * c))
        # Shell and Tube - One Shell Pass; 2,4,6, etc., Tube Passes NTU
        elif Arrangement is HX.ShellTube:
            D1 = 1 + c - (1 + c**2) ** 0.5
            D2 = 1 + c + (1 + c**2) ** 0.5
            Ntu = (1 + c**2) ** -0.5 * math.log((2 - eff * D1) / (2 - eff * D2))
        # Condensing or Evaporating of One Fluid
        elif Arrangement is HX.CondEvap:
            Ntu = -math.log(1 - eff)
        else:
            Ntu = -1
    else:
        Ntu = 0

    return Ntu * Passes


def _checked_effectiveness(eff: float) -> float:
    """Return ``eff`` or ``inf`` at the limit; reject effectiveness above one."""
    if eff > 1.0 + 1e-9:
        raise ValueError(
            f"Effectiveness {eff:.6g} exceeds 1: the process outlet is beyond the "
            "utility temperature, so no exchanger can deliver this duty."
        )
    return eff


def CalcAreaUE(Arrangement, U, C_p, T_p1, T_p2, T_u1, T_u2, Passes):
    """Estimate the exchanger ``area * U`` product from duty and temperatures.

    An isothermal utility (``T_u1 == T_u2``, condensing or evaporating) has
    infinite capacity, so the process stream is Cmin and ``c = 0``. A duty that
    needs unit effectiveness (a zero terminal approach) needs unbounded area and
    returns ``inf``. Heat must flow from hot to cold: a process stream moving away
    from the utility temperature, a utility changing in the same direction as the
    process, or a duty needing more than unit effectiveness raises ``ValueError``.
    """
    Q = C_p * abs(T_p1 - T_p2)
    process_change = T_p2 - T_p1
    if process_change * (T_u1 - T_p1) < 0 or process_change * (T_u2 - T_u1) > 0:
        raise ValueError(
            "Infeasible exchange: the process stream must move toward the utility "
            "inlet temperature and the utility must change in the opposite direction."
        )
    if T_u1 == T_u2:
        C_min, c = C_p, 0.0
    else:
        C_u = Q / abs(T_u1 - T_u2)
        C_min, c = (C_p, C_p / C_u) if C_p < C_u else (C_u, C_u / C_p)
    eff = _checked_effectiveness(Q / C_min / abs(T_p1 - T_u1))
    if eff >= 1.0 - 1e-9:
        return math.inf
    Ntu = HX_NTU(Arrangement, eff, c, Passes)
    return Ntu * C_min / U


def eNTU_slope_Numerical(Arrangement, Ntu, c, Passes):
    """Compute a finite-difference effectiveness slope with respect to NTU."""
    dx = 1e-6
    if Ntu > 0:
        return (
            HX_Eff(Arrangement, Ntu + dx, c, Passes)
            - HX_Eff(Arrangement, Ntu, c, Passes)
        ) / dx


def Coth(R):
    """Convenience wrapper for the hyperbolic cotangent function."""
    return (math.exp(2 * R) + 1) / (math.exp(2 * R) - 1)


def MultiPassEff(eff, c, Passes):
    """Convert single-pass effectiveness into equivalent multi-pass effectiveness."""
    if c != 1:
        return (((1 - eff * c) / (1 - eff)) ** Passes - 1) / (
            ((1 - eff * c) / (1 - eff)) ** Passes - c
        )
    else:
        return Passes * eff / (1 + eff * (Passes - 1))


def MultiPassNTU(Eff_p, c, Passes):
    """Convert multi-pass effectiveness back to an equivalent single-pass value."""
    if c != 1:
        return (((1 - Eff_p * c) / (1 - Eff_p)) ** (1 / Passes) - 1) / (
            ((1 - Eff_p * c) / (1 - Eff_p)) ** (1 / Passes) - c
        )
    else:
        return Eff_p / (Passes - Eff_p * (Passes - 1))


def CrossflowUnmixedEff1(Ntu, c):
    """Series approximation for cross-flow effectiveness with unmixed streams."""
    Sum_Pn = 0
    for i in range(1, 21):
        Pn = 0
        for j in range(1, i):
            Pn = Pn + c**i / math.factorial(i + 1) * (i - j + 1) / math.factorial(
                j
            ) * Ntu ** (i + j)
        Sum_Pn = Sum_Pn + Pn
    return 1 - math.exp(-Ntu) - math.exp(-(1 + c) * Ntu) * Sum_Pn


def CrossflowUnmixedEff2(Ntu, c, Rows, Cmin_fluid):
    """Lookup-derived correlations for cross-flow exchangers with finite rows."""
    if isinstance(Rows, bool) or not isinstance(Rows, Integral) or Rows < 1:
        raise ValueError("Rows must be a positive integer.")

    # ESDU 86018
    if Cmin_fluid == "Air":
        if Rows == 1:
            eff = 1 / c * (1 - math.exp(-c * (1 - math.exp(-Ntu))))
        elif Rows == 2:
            k = 1 - math.exp(-Ntu / 2)
            eff = 1 / c * (1 - math.exp(-2 * k * c) * (1 + c * k**2))
        elif Rows == 3:
            k = 1 - math.exp(-Ntu / 3)
            eff = (
                1
                / c
                * (
                    1
                    - math.exp(-3 * k * c)
                    * (1 + c * k**2 * (3 - k) + (3 * c**2 * k**4) / 2)
                )
            )
        elif Rows == 4:
            k = 1 - math.exp(-Ntu / 4)
            eff = (
                1
                / c
                * (
                    1
                    - math.exp(-4 * k * c)
                    * (
                        1
                        + c * k**2 * (6 - 4 * k + k**2)
                        + 4 * c**2 * k**4 * (2 - k)
                        + (8 * c**3 * k**6) / 3
                    )
                )
            )
        else:
            eff = CrossflowUnmixedEff1(Ntu, c)
    else:
        if Rows == 1:
            eff = 1 - math.exp(-1 / c * (1 - math.exp(-Ntu * c)))
        elif Rows == 2:
            k = 1 - math.exp(-Ntu * c / 2)
            eff = 1 - math.exp(-2 * k / c) * (1 + (k**2) / c)
        elif Rows == 3:
            k = 1 - math.exp(-Ntu * c / 3)
            eff = 1 - math.exp(-3 * k / c) * (
                1 + k**2 * (3 - k) / c + (3 * k**4) / (2 * c**2)
            )
        elif Rows == 4:
            k = 1 - math.exp(-Ntu * c / 4)
            eff = 1 - math.exp(-4 * k / c) * (
                1
                + k**2 * (6 - 4 * k + k**2) / c
                + 4 * k**4 * (2 - k) / c**2
                + (8 * k**6) / (3 * c**3)
            )
        else:
            eff = CrossflowUnmixedEff1(Ntu, c)

    return eff


def HX_NTU_Numerical(Arrangement, eff, c):
    """Solve for NTU numerically when closed-form expressions are unavailable."""
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
            raise ValueError("Solution does not converge")

    return NTU3
