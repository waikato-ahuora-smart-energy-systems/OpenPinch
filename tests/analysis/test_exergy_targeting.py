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


def _exergy_changes(T_celsius, H, T0_kelvin, steps=2000):
    """Independent (gain, loss) of exergy along a piecewise-linear T-H curve.

    Each interval is cut into ``steps`` slices and dX = dH (1 - T0/T) is
    summed, keeping increases and decreases apart, so a curve crossing
    ambient is not netted out.
    """
    import numpy as np

    T = np.asarray(T_celsius, dtype=float) + 273.15
    H = np.asarray(H, dtype=float)
    gain = loss = 0.0
    for t_u, t_l, h_u, h_l in zip(T[:-1], T[1:], H[:-1], H[1:]):
        fractions = (np.arange(steps) + 0.5) / steps
        t_mid = t_u + (t_l - t_u) * fractions
        dX = (h_u - h_l) / steps * (1.0 - T0_kelvin / t_mid)
        gain += float(dX[dX > 0].sum())
        loss += float(-dX[dX < 0].sum())
    return gain, loss


def _balanced_curve_oracle(T_celsius, H_hot, H_cold, T0_kelvin):
    """Return (exergy released, ETE) of hot and cold balanced composites.

    Above ambient the hot curve gives up exergy and the cold curve gains it;
    below ambient the roles swap (cooling a hot stream raises its exergy).
    """
    hot_up, hot_down = _exergy_changes(T_celsius, H_hot, T0_kelvin)
    cold_up, cold_down = _exergy_changes(T_celsius, H_cold, T0_kelvin)
    released = hot_up + cold_down
    gained = cold_up + hot_down
    return released, gained / released


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
    released, ete = _balanced_curve_oracle(
        pt[L.T], pt[L.H_HOT] + pt[L.H_HOT_UT], pt[L.H_COLD] + pt[L.H_COLD_UT], T0
    )
    assert target.exergy_released == pytest.approx(released, rel=1e-5)
    assert target.ETE == pytest.approx(ete, rel=1e-5)
    assert 0.0 < target.ETE <= 1.0
    assert target.ETE != pytest.approx(previous_ete, abs=1e-3)


def test_ete_of_a_sub_ambient_problem_counts_exergy_sources_below_ambient():
    # Codex review on PR #191: below ENV_TEMPERATURE both balanced composites
    # change exergy with the opposite sign, so a plain signed sum gave a
    # negative "released" exergy and ETE 0 for a refrigeration problem.
    import warnings

    from OpenPinch import PinchProblem
    from OpenPinch.analysis.exergy.service import apply_exergy_targeting
    from OpenPinch.domain.enums import ProblemTableLabel as L

    data = {
        "streams": [
            {
                "zone": "Cold",
                "name": "H1",
                "t_supply": -5.0,
                "t_target": -35.0,
                "heat_flow": 300.0,
                "htc": 1.0,
            },
            {
                "zone": "Cold",
                "name": "C1",
                "t_supply": -45.0,
                "t_target": -15.0,
                "heat_flow": 240.0,
                "htc": 1.0,
            },
        ],
        "utilities": [],
        "options": {"ENV_TEMPERATURE": 15.0},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        problem = PinchProblem(data, project_name="Cold")
        problem.target.direct_heat_integration()
    target = next(
        t
        for t in problem._master_zone.targets.values()
        if getattr(t, "type", None) == "Direct Integration"
    )

    apply_exergy_targeting(target)

    pt = target.pt_real
    T0 = target.config.environment.temperature + 273.15
    released, ete = _balanced_curve_oracle(
        pt[L.T], pt[L.H_HOT] + pt[L.H_HOT_UT], pt[L.H_COLD] + pt[L.H_COLD_UT], T0
    )
    assert target.exergy_released == pytest.approx(released, rel=1e-5)
    assert target.ETE == pytest.approx(ete, rel=1e-5)
    assert 0.0 < target.ETE <= 1.0
