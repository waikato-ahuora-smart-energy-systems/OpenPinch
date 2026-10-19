"""Regression tests for exergy targeting analysis routines."""

import math

import pytest

from OpenPinch.analysis.exergy.service import (
    compute_exergetic_temperature,
)

"""Test cases for the compute_exergetic_temperature function."""


def test_exergetic_temp_celsius_input():
    """Basic test with Celsius input."""
    result = compute_exergetic_temperature(100, T_ref_in_C=15, units_of_T="C")
    assert isinstance(result, float)
    assert result > 0


def test_exergetic_temp_kelvin_input():
    """Basic test with Kelvin input."""
    result = compute_exergetic_temperature(373.15, T_ref_in_C=15, units_of_T="K")
    assert isinstance(result, float)
    assert result > 0


def test_exergetic_temp_at_reference_returns_zero():
    """When T = T_ref, should return 0 exergetic temp."""
    result = compute_exergetic_temperature(15, T_ref_in_C=15, units_of_T="C")
    assert math.isclose(result, 0.0, abs_tol=1e-9)


def test_exergetic_temp_invalid_unit_raises():
    """Should raise for invalid temperature units."""
    with pytest.raises(ValueError, match="units must be either 'C' or 'K'"):
        compute_exergetic_temperature(100, units_of_T="F")


def test_exergetic_temp_below_absolute_zero_raises():
    """Should raise when temperature is below 0 K."""
    with pytest.raises(ValueError, match="Absolute temperature must be > 0 K"):
        compute_exergetic_temperature(-300, units_of_T="C")


def _balanced_curve_exergy(T_celsius, H, T0_kelvin):
    """Independent exergy of a piecewise-linear composite: sum dH (1 - T0/T_lm)."""
    import numpy as np

    T = np.asarray(T_celsius, dtype=float) + 273.15
    H = np.asarray(H, dtype=float)
    dH = H[:-1] - H[1:]
    T_u, T_l = T[:-1], T[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(
            np.abs(T_u - T_l) > 1e-9,
            T0_kelvin * np.log(T_u / T_l) / (T_u - T_l),
            T0_kelvin / T_u,
        )
    return float((dH * (1.0 - factor)).sum())


@pytest.mark.parametrize(
    "case, previous_ete",
    [
        ("basic_pinch.json", 1.0051),
        ("pulp_mill.json", 4.5867),
        ("chocolate_factory.json", 1.0888),
        ("Four-stream-Yee-and-Grossmann-1990-1.json", 0.2879),
    ],
)
def test_ete_is_the_exergy_transfer_effectiveness_of_the_balanced_curves(
    case, previous_ete
):
    # Audit M9: ETE was the above-pinch deficit exergy over the below-pinch
    # surplus exergy (458 % on pulp_mill). It is now the exergy gained by the
    # cold Balanced Composite Curve over the exergy released by the hot one,
    # relative to ENV_TEMPERATURE with utilities included, so it is bounded
    # by 100 % and its shortfall is the exergy destroyed in heat transfer.
    import warnings

    from OpenPinch import PinchProblem
    from OpenPinch.domain.enums import ProblemTableLabel as L

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        problem = PinchProblem(case, project_name="M9")
        problem.target.direct_heat_integration()
        internal = problem._master_zone.targets
    target = next(
        t for t in internal.values() if getattr(t, "type", None) == "Direct Integration"
    )
    from OpenPinch.analysis.exergy.service import apply_exergy_targeting

    apply_exergy_targeting(target)

    pt = target.pt_real
    T0 = target.config.environment.temperature + 273.15
    released = _balanced_curve_exergy(pt[L.T], pt[L.H_HOT] + pt[L.H_HOT_UT], T0)
    gained = _balanced_curve_exergy(pt[L.T], pt[L.H_COLD] + pt[L.H_COLD_UT], T0)
    assert target.exergy_released == pytest.approx(released, rel=1e-9)
    assert target.ETE == pytest.approx(gained / released, rel=1e-9)
    assert 0.0 < target.ETE <= 1.0
    assert target.ETE != pytest.approx(previous_ete, abs=1e-3)
