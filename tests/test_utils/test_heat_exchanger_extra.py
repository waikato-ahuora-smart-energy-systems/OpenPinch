"""Additional coverage tests for heat-exchanger helpers."""

import math

import pytest

import OpenPinch.utils.heat_exchanger as hx
from OpenPinch.lib.enums import HeatExchangerTypes as HX


def test_compute_lmtd_from_dts_rejects_non_positive_values():
    with pytest.raises(ValueError, match="Invalid temperature differences"):
        hx.compute_LMTD_from_dts([10.0, -1.0], [5.0, 2.0])


def test_hx_eff_counterflow_c_equal_one_branch():
    eff = hx.HX_Eff(HX.CF.value, Ntu=2.0, c=1.0)
    assert eff == pytest.approx(2.0 / 3.0)


def test_hx_eff_shell_and_tube_and_unknown_arrangement_branches():
    shell = hx.HX_Eff(HX.ShellTube.value, Ntu=2.0, c=0.7)
    fallback = hx.HX_Eff("unknown", Ntu=2.0, c=0.7)
    zero = hx.HX_Eff(HX.CF.value, Ntu=0.0, c=0.5)
    assert shell > 0.0
    assert fallback > 0.0
    assert zero == 0.0


def test_hx_ntu_additional_branches():
    multipass = hx.HX_NTU(HX.CF.value, eff=0.7, c=0.4, Passes=2)
    c_equal_one = hx.HX_NTU(HX.CF.value, eff=0.7, c=1.0)
    mixed = hx.HX_NTU(HX.CrFMM, eff=0.6, c=0.5)
    shell = hx.HX_NTU(HX.ShellTube.value, eff=0.5, c=0.7)
    unknown = hx.HX_NTU("unknown", eff=0.5, c=0.7)
    low_eff = hx.HX_NTU(HX.CF.value, eff=0.0, c=0.7)

    assert multipass > 0.0
    assert c_equal_one == pytest.approx(0.7 / (1 - 0.7))
    assert mixed > 0.0
    assert shell > 0.0
    assert unknown == -1
    assert low_eff == 0


def test_calc_area_ue_else_branch():
    # Chooses else branch: C_p >= C_u
    area_ue = hx.CalcAreaUE(
        Arrangement=HX.CF.value,
        U=100.0,
        C_p=500.0,
        T_p1=120.0,
        T_p2=100.0,
        T_u1=20.0,
        T_u2=80.0,
        Passes=1,
    )
    assert area_ue > 0.0


def test_multipass_helpers_c_equal_one_branches():
    eff = hx.MultiPassEff(eff=0.5, c=1.0, Passes=3)
    ntu = hx.MultiPassNTU(Eff_p=0.5, c=1.0, Passes=3)
    assert eff == pytest.approx(3 * 0.5 / (1 + 0.5 * (3 - 1)))
    assert ntu == pytest.approx(0.5 / (3 - 0.5 * (3 - 1)))


@pytest.mark.parametrize(
    "rows,cmin_fluid",
    [
        (1, "Air"),
        (3, "Air"),
        (4, "Air"),
        (5, "Air"),
        (1, "Water"),
        (3, "Water"),
        (4, "Water"),
        (5, "Water"),
    ],
)
def test_crossflow_unmixed_eff2_all_row_branches(rows, cmin_fluid):
    eff = hx.CrossflowUnmixedEff2(Ntu=2.0, c=0.5, Rows=rows, Cmin_fluid=cmin_fluid)
    assert math.isfinite(eff)
    assert eff >= 0.0


def test_hx_ntu_numerical_raises_when_not_converging(monkeypatch):
    monkeypatch.setattr(
        hx,
        "HX_Eff",
        lambda Arrangement, Ntu, c, Passes=None: Ntu**2 + 10.0,
    )
    with pytest.raises(ValueError, match="does not converge"):
        hx.HX_NTU_Numerical(HX.CrFUU, eff=0.5, c=0.5)
