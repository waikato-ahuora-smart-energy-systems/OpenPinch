import math
import pytest
from src.utils.heat_exchanger_eq import *
from src.lib import HeatExchangerTypes as HX

def test_Coth():
    assert round(Coth(1), 3) == round((math.exp(2) + 1)/(math.exp(2) - 1), 3)

@pytest.mark.parametrize("arrangement, Ntu, c, expected_range", [
    (HX.CF.value, 2.0, 0.5, (0.75, 0.8)),
    (HX.PF.value, 2.0, 0.5, (0.6, 0.65)),
    (HX.CrFMM, 2.0, 0.5, (0.6, 0.7)),
    (HX.CrFMUmax, 2.0, 0.5, (0.7, 0.75)),
    (HX.CrFMUmin, 2.0, 0.5, (0.65, 0.75)),
    (HX.CondEvap, 2.0, 0.5, (0.86, 0.88)),
])
def test_HX_Eff_variants(arrangement, Ntu, c, expected_range):
    eff = HX_Eff(arrangement, Ntu, c)
    assert expected_range[0] < eff < expected_range[1]

def test_HX_Eff_crossflow_unmixed1():
    eff = HX_Eff(HX.CrFUU, 2.0, 0.5)
    assert 0.8 < eff < 0.81

def test_HX_Eff_crossflow_unmixed2_air():
    eff = HX_Eff(HX.CrFUU, 2.0, 0.5, Rows=2, Cmin_Phase='Air')
    assert 0.72 < eff < 0.73

def test_HX_Eff_crossflow_unmixed2_steam():
    eff = HX_Eff(HX.CrFUU, 2.0, 0.5, Rows=2, Cmin_Phase='Steam')
    assert 0.72 < eff < 0.73

def test_HX_Eff_multi_pass():
    eff = HX_Eff(HX.CF.value, 4.0, 0.5, Passes=2)
    assert 0.85 < eff < 0.95

@pytest.mark.parametrize("arrangement, eff, c", [
    (HX.CF.value, 0.75, 0.5),
    (HX.PF.value, 0.6, 0.5),
    (HX.CrFUU, 0.6, 0.5),
    (HX.CrFMUmax, 0.8, 0.5),
    (HX.CrFMUmin, 0.7, 0.5),
    (HX.CondEvap, 0.86, 0.5),
])
def test_HX_NTU_coverage(arrangement, eff, c):
    Ntu = HX_NTU(arrangement, eff, c)
    assert Ntu > 0

def test_CalcAreaUE():
    A = CalcAreaUE(HX.CF.value, U=100, C_p=500, T_p1=150, T_p2=100, T_u1=50, T_u2=90, Passes=1)
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
