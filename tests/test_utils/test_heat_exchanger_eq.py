import math

import pytest

from OpenPinch.lib import *
from OpenPinch.utils.heat_exchanger import *


"""Test cases for the find_LMTD function."""

def test_lmtd_typical_counterflow():
    """Basic counter-current case with distinct ΔT1 and ΔT2."""
    T_hot_in, T_hot_out = 150, 50
    T_cold_in, T_cold_out = 30, 80
    result = compute_LMTD_from_ts(T_hot_in, T_hot_out, T_cold_in, T_cold_out)
    dT1 = T_hot_in - T_cold_out
    dT2 = T_hot_out - T_cold_in
    expected = (dT1 - dT2) / math.log(dT1 / dT2)
    assert math.isclose(result, expected, rel_tol=1e-6)


def test_lmtd_equal_temperature_difference_returns_deltaT():
    """Return arithmetic mean if ΔT1 == ΔT2."""
    T_hot_in = 100
    T_hot_out = 80
    T_cold_in = 40
    T_cold_out = 60
    result = compute_LMTD_from_ts(T_hot_in, T_hot_out, T_cold_in, T_cold_out)
    expected = 40  # ΔT1 = ΔT2 = 40
    assert math.isclose(result, expected, rel_tol=1e-6)


def test_lmtd_one_deltaT_zero_returns_half_sum():
    """If one ΔT is zero, fall back to arithmetic mean."""
    T_hot_in, T_hot_out = 150, 50
    T_cold_in, T_cold_out = 30, 30
    result = compute_LMTD_from_ts(
        T_hot_in, T_hot_out, T_cold_in, T_cold_out
    )  # Cold fluid at constant temperature (phase change)
    dT1 = T_hot_in - T_cold_out
    dT2 = T_hot_out - T_cold_in
    expected = (dT1 - dT2) / math.log(dT1 / dT2)
    assert math.isclose(result, expected, rel_tol=1e-6)


def test_lmtd_negative_temperature_difference_raises_error():
    """Raise error if either ΔT1 or ΔT2 < 0."""
    with pytest.raises(ValueError, match="must heat up"):
        compute_LMTD_from_ts(100, 80, 90, 70)  # cold fluid cooling


def test_lmtd_hot_fluid_heats_up_invalid():
    with pytest.raises(ValueError, match="Hot fluid must cool down"):
        compute_LMTD_from_ts(90, 100, 40, 80)


def test_lmtd_cold_fluid_cools_down_invalid():
    with pytest.raises(ValueError, match="Cold fluid must heat up"):
        compute_LMTD_from_ts(150, 100, 80, 60)


def test_Coth():
    assert round(Coth(1), 3) == round((math.exp(2) + 1) / (math.exp(2) - 1), 3)


@pytest.mark.parametrize(
    "arrangement, Ntu, c, expected_range",
    [
        (HX.CF.value, 2.0, 0.5, (0.75, 0.8)),
        (HX.PF.value, 2.0, 0.5, (0.6, 0.65)),
        (HX.CrFMM, 2.0, 0.5, (0.6, 0.7)),
        (HX.CrFMUmax, 2.0, 0.5, (0.7, 0.75)),
        (HX.CrFMUmin, 2.0, 0.5, (0.65, 0.75)),
        (HX.CondEvap, 2.0, 0.5, (0.86, 0.88)),
    ],
)
def test_HX_Eff_variants(arrangement, Ntu, c, expected_range):
    eff = HX_Eff(arrangement, Ntu, c)
    assert expected_range[0] < eff < expected_range[1]


def test_HX_Eff_crossflow_unmixed1():
    eff = HX_Eff(HX.CrFUU, 2.0, 0.5)
    assert 0.8 < eff < 0.81


def test_HX_Eff_crossflow_unmixed2_air():
    eff = HX_Eff(HX.CrFUU, 2.0, 0.5, Rows=2, Cmin_Phase="Air")
    assert 0.72 < eff < 0.73


def test_HX_Eff_crossflow_unmixed2_steam():
    eff = HX_Eff(HX.CrFUU, 2.0, 0.5, Rows=2, Cmin_Phase="Steam")
    assert 0.72 < eff < 0.73


def test_HX_Eff_multi_pass():
    eff = HX_Eff(HX.CF.value, 4.0, 0.5, Passes=2)
    assert 0.85 < eff < 0.95


@pytest.mark.parametrize(
    "arrangement, eff, c",
    [
        (HX.CF.value, 0.75, 0.5),
        (HX.PF.value, 0.6, 0.5),
        (HX.CrFUU, 0.6, 0.5),
        (HX.CrFMUmax, 0.8, 0.5),
        (HX.CrFMUmin, 0.7, 0.5),
        (HX.CondEvap, 0.86, 0.5),
    ],
)
def test_HX_NTU_coverage(arrangement, eff, c):
    Ntu = HX_NTU(arrangement, eff, c)
    assert Ntu > 0


def test_CalcAreaUE():
    A = CalcAreaUE(
        HX.CF.value, U=100, C_p=500, T_p1=150, T_p2=100, T_u1=50, T_u2=90, Passes=1
    )
    assert A > 0


def test_eNTU_slope_Numerical():
    slope = eNTU_slope_Numerical(HX.CF.value, Ntu=1.0, c=0.5, Passes=1)
    assert 0.1 < slope < 1.0


def test_MultiPassEff():
    eff = MultiPassEff(0.7, 0.5, 2)
    assert 0.799 < eff < 0.881


def test_MultiPassNTU():
    ntu = MultiPassNTU(0.8, 0.5, 2)
    assert ntu > 0.5


def test_CrossflowUnmixedEff1():
    eff = CrossflowUnmixedEff1(2.0, 0.5)
    assert 0.80 < eff < 0.81


def test_CrossflowUnmixedEff2():
    eff = CrossflowUnmixedEff2(2.0, 0.5, Rows=2, Cmin_fluid="Steam")
    assert 0.72 < eff < 0.73


def test_HX_NTU_Numerical():
    ntu = HX_NTU_Numerical(HX.CrFUU, 0.65, 0.5)
    assert 1.1 < ntu < 1.2
