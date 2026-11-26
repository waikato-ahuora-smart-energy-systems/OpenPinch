"""The rating model's rounded heat curve on a segment far narrower than the
rounding.

A water vapour compression condenser desuperheats over tens of kelvin and
then condenses over 0.01 K: 177.1 kW behind a boundary that the solver's
expressions round over 0.05 K.
"""

from __future__ import annotations

import numpy as np
import pytest

from OpenPinch.analysis.heat_exchanger_networks.models.thermal_profiles import (
    ThermalProfile,
)


def condenser() -> ThermalProfile:
    ends = (497.9, 469.45, 447.91, 447.9)
    duties = (5.7, 4.67, 177.1)
    return ThermalProfile(
        side="hot",
        identities=("V.S1", "V.S2", "V.S3"),
        t_in=ends[:-1],
        t_out=ends[1:],
        cp=tuple(
            duty / (hot - cold) for duty, hot, cold in zip(duties, ends, ends[1:])
        ),
        htc=(1.0, 1.0, 1.0),
        dt_cont=(1.0, 1.0, 1.0),
        price=(0.0, 0.0, 0.0),
    )


def solver_value(temperature: float):
    """A value the profile treats as a solver expression, so it is rounded."""
    return np.array([temperature])


def test_the_rounded_heat_at_the_target_is_the_stream_duty():
    profile = condenser()

    assert profile.heat(solver_value(profile.target))[0] == pytest.approx(
        187.47, rel=1e-9
    )


def test_the_rounded_heat_at_the_supply_is_zero():
    profile = condenser()

    assert profile.heat(solver_value(profile.supply))[0] == pytest.approx(0.0, abs=1e-9)


def test_the_rounding_beside_a_narrow_segment_leaves_its_neighbour_alone():
    profile = condenser()
    one_kelvin_above_condensing = 448.91

    assert profile.heat(solver_value(one_kelvin_above_condensing))[0] == pytest.approx(
        profile.heat(one_kelvin_above_condensing), abs=1e-3
    )


def test_a_segment_of_no_width_does_not_take_the_rounding_away():
    from OpenPinch.analysis.heat_exchanger_networks.models.thermal_profiles import (
        SMOOTHING_K,
    )

    profile = ThermalProfile(
        side="hot",
        identities=("a", "b", "c"),
        t_in=(400.0, 380.0, 380.0),
        t_out=(380.0, 380.0, 350.0),
        cp=(10.0, 0.0, 20.0),
        htc=(1.0, 1.0, 1.0),
        dt_cont=(1.0, 1.0, 1.0),
        price=(0.0, 0.0, 0.0),
    )

    assert profile.rounding_widths == (SMOOTHING_K, SMOOTHING_K)


class _Recorder:
    """Stands in for the model: keeps each approach constraint's left side."""

    def __init__(self) -> None:
        self.approaches: list = []

    def _var(self, name, value, lower, upper):
        return 0.0

    def _equal(self, lhs, rhs) -> None:
        pass

    def _at_least(self, lhs, rhs) -> None:
        self.approaches.append(lhs)


def test_a_boundary_outside_a_match_within_a_narrow_segment_is_held_at_its_end():
    from OpenPinch.analysis.heat_exchanger_networks.models.fixed_structure import (
        _interior_approaches,
    )

    hot = condenser()
    cold = ThermalProfile.single("cold", "C", 400.0, 440.0, 20.0, 1.0)
    inlet, outlet = 447.905, 447.9
    model = _Recorder()

    _interior_approaches(
        model,
        "match",
        (hot, 1.0, solver_value(inlet), solver_value(outlet)),
        (cold, 1.0, solver_value(400.0), solver_value(404.0)),
        solver_value(88.55),
        1.0,
    )

    # With the partner at zero the left side is the boundary's point.
    for point in model.approaches:
        assert outlet - 1e-4 <= point[0] <= inlet + 1e-4


def test_a_cold_boundary_outside_a_match_within_a_narrow_segment_is_held_too():
    from OpenPinch.analysis.heat_exchanger_networks.models.fixed_structure import (
        _interior_approaches,
    )

    hot = ThermalProfile.single("hot", "H", 360.0, 340.0, 10.0, 1.0)
    evaporator = ThermalProfile(
        side="cold",
        identities=("E.S1", "E.S2"),
        t_in=(300.0, 300.01),
        t_out=(300.01, 320.0),
        cp=(100.0 / 0.01, 5.0 / 19.99),
        htc=(1.0, 1.0),
        dt_cont=(1.0, 1.0),
        price=(0.0, 0.0),
    )
    inlet, outlet = 300.0, 300.005
    model = _Recorder()

    _interior_approaches(
        model,
        "match",
        (hot, 1.0, solver_value(360.0), solver_value(355.0)),
        (evaporator, 1.0, solver_value(inlet), solver_value(outlet)),
        solver_value(50.0),
        1.0,
    )

    # With the partner at zero the left side is minus the boundary's point.
    (approach,) = model.approaches
    assert inlet - 1e-4 <= -approach[0] <= outlet + 1e-4


def test_a_cooler_within_a_narrow_segment_holds_an_outside_boundary_at_its_end():
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.models.fixed_structure import (
        COOLER,
        _linear_utility_interior,
    )

    inlet, outlet, duty = 447.905, 447.9, 88.55
    model = _Recorder()

    _linear_utility_interior(
        model,
        SimpleNamespace(side=COOLER),
        condenser(),
        solver_value(inlet),
        solver_value(outlet),
        0.0,
        0.0,
        duty,
        1.0,
    )

    # With the utility at zero the left side is the duty times the point.
    for approach in model.approaches:
        assert outlet - 1e-4 <= approach[0] / duty <= inlet + 1e-4


def test_a_heater_within_a_narrow_segment_holds_an_outside_boundary_at_its_end():
    from types import SimpleNamespace

    from OpenPinch.analysis.heat_exchanger_networks.models.fixed_structure import (
        HEATER,
        _linear_utility_interior,
    )

    evaporator = ThermalProfile(
        side="cold",
        identities=("E.S1", "E.S2"),
        t_in=(300.0, 300.01),
        t_out=(300.01, 320.0),
        cp=(100.0 / 0.01, 5.0 / 19.99),
        htc=(1.0, 1.0),
        dt_cont=(1.0, 1.0),
        price=(0.0, 0.0),
    )
    inlet, outlet, duty = 300.0, 300.005, 50.0
    model = _Recorder()

    _linear_utility_interior(
        model,
        SimpleNamespace(side=HEATER),
        evaporator,
        solver_value(inlet),
        solver_value(outlet),
        0.0,
        0.0,
        duty,
        1.0,
    )

    # With the utility at zero the left side is minus the duty times the point.
    (approach,) = model.approaches
    assert inlet - 1e-4 <= -approach[0] / duty <= outlet + 1e-4
