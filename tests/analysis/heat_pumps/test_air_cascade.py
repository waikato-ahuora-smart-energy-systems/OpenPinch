from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch import PinchProblem
from OpenPinch.analysis.heat_pumps.common._shared.air import (
    cascade_with_air,
    utility_bands,
)
from OpenPinch.analysis.heat_pumps.common.shared import _cascade_air_duties
from OpenPinch.analysis.heat_pumps.targeting._carnot_common import (
    parse_carnot_state_variables,
)

from .helpers import _sc, _stream


def test_air_takes_surplus_rejected_above_its_sink_temperature():
    # 100 kW flows down from 150 C to 50 C, then 60 kW more is released
    # below 30 C (cascade bottom 160 kW). Air at a 40 C sink can take only
    # the 100 kW crossing 40 C; the 60 kW released below stays external.
    T = np.array([150.0, 50.0, 30.0, 10.0])
    H = np.array([0.0, 100.0, 100.0, 160.0])

    out = cascade_with_air(T, H, T_air_source=0.0, T_air_sink=40.0)

    assert out.Q_air_sink == pytest.approx(100.0)
    assert out.Q_ext_bottom == pytest.approx(60.0)
    assert out.Q_ext_top == pytest.approx(0.0)


def test_air_supplies_heat_needed_below_its_source_temperature():
    # Pinched cascade: 80 kW enters at the top, 50 kW is absorbed 100-20 C,
    # 30 kW is absorbed 20-0 C (pinch at 0 C), 10 kW is released below.
    # Air at a 5 C source can meet only the 7.5 kW needed between 5 and 0 C.
    T = np.array([100.0, 20.0, 0.0, -20.0])
    H = np.array([80.0, 30.0, 0.0, 10.0])

    out = cascade_with_air(T, H, T_air_source=5.0, T_air_sink=200.0)

    assert out.Q_air_source == pytest.approx(7.5)
    assert out.Q_ext_top == pytest.approx(72.5)
    assert out.Q_ext_bottom == pytest.approx(10.0)


def test_air_cannot_supply_heat_needed_above_its_source_temperature():
    # All 80 kW is needed between 100 C and 60 C; the flow is zero below.
    T = np.array([100.0, 60.0, 10.0, 0.0])
    H = np.array([80.0, 0.0, 0.0, 0.0])

    out = cascade_with_air(T, H, T_air_source=5.0, T_air_sink=200.0)

    assert out.Q_air_source == pytest.approx(0.0)
    assert out.Q_ext_top == pytest.approx(80.0)


def test_unused_air_creates_no_demand():
    T = np.array([100.0, 50.0])
    H = np.array([0.0, 0.0])

    out = cascade_with_air(T, H, T_air_source=10.0, T_air_sink=20.0)

    assert (out.Q_ext_top, out.Q_ext_bottom, out.Q_air_source, out.Q_air_sink) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_air_levels_between_rows_use_the_interpolated_flow():
    # Heat is released evenly from 100 C down to 0 C (100 kW at the bottom).
    # An air sink at 50 C takes the 50 kW released above 50 C.
    T = np.array([100.0, 0.0])
    H = np.array([0.0, 100.0])

    out = cascade_with_air(T, H, T_air_source=-50.0, T_air_sink=50.0)

    assert out.Q_air_sink == pytest.approx(50.0)
    assert out.Q_ext_bottom == pytest.approx(50.0)


def test_air_never_replaces_heat_that_only_passes_through():
    # No pinch: 5 kW enters at the top and 1 kW leaves at the bottom, all
    # between 120 and 60 C. Air at 20 C is below every stream, so it can
    # neither supply nor take any of it.
    T = np.array([120.0, 60.0])
    H = np.array([5.0, 1.0])

    out = cascade_with_air(T, H, T_air_source=20.0, T_air_sink=20.0)

    assert out.Q_air_source == pytest.approx(0.0)
    assert out.Q_air_sink == pytest.approx(0.0)
    assert (out.Q_ext_top, out.Q_ext_bottom) == pytest.approx((5.0, 1.0))


def test_cooling_water_takes_nothing_when_colder_air_covers_the_heat():
    # 100 kW is released between 50 C and 0 C. Air at a 10 C sink takes the
    # 80 kW released above it, so cooling water at 30 C has nothing left and
    # the 20 kW released below 10 C needs refrigeration.
    T = np.array([100.0, 50.0, 0.0])
    H = np.array([0.0, 0.0, 100.0])

    out = cascade_with_air(
        T, H, T_air_source=-50.0, T_air_sink=10.0, T_cooling_water=30.0
    )

    assert out.Q_air_sink == pytest.approx(80.0)
    assert out.Q_cooling_water == pytest.approx(0.0)
    assert out.Q_ext_bottom == pytest.approx(20.0)


def test_cooling_water_takes_heat_below_warmer_air():
    # On a hot day air only takes heat above 40 C (20 kW); cooling water at
    # 30 C takes the next 20 kW and 60 kW is left for refrigeration.
    T = np.array([100.0, 50.0, 0.0])
    H = np.array([0.0, 0.0, 100.0])

    out = cascade_with_air(
        T, H, T_air_source=-50.0, T_air_sink=40.0, T_cooling_water=30.0
    )

    assert out.Q_air_sink == pytest.approx(20.0)
    assert out.Q_cooling_water == pytest.approx(20.0)
    assert out.Q_ext_bottom == pytest.approx(60.0)


def test_air_takes_a_utility_band_that_sits_on_its_sink_temperature():
    # A utility level is a 0.1 K band, so one at the air limit ends just below it.
    T = np.array([20.0, 19.9])
    H = np.array([0.0, 2000.0])

    out = cascade_with_air(
        T, H, T_air_source=0.0, T_air_sink=20.0, utility_bands=[(20.0, 19.9)]
    )

    assert out.Q_air_sink == pytest.approx(2000.0)
    assert out.Q_ext_bottom == pytest.approx(0.0)


def test_air_does_not_take_a_band_further_below_its_sink_temperature():
    T = np.array([19.5, 19.4])
    H = np.array([0.0, 2000.0])

    out = cascade_with_air(
        T, H, T_air_source=0.0, T_air_sink=20.0, utility_bands=[(19.5, 19.4)]
    )

    assert out.Q_air_sink == pytest.approx(0.0)
    assert out.Q_ext_bottom == pytest.approx(2000.0)


def test_air_does_not_take_ordinary_heat_just_below_its_sink_temperature():
    # Without a utility band this heat is really below the limit.
    T = np.array([20.0, 19.95])
    H = np.array([0.0, 2000.0])

    out = cascade_with_air(T, H, T_air_source=0.0, T_air_sink=20.0)

    assert out.Q_air_sink == pytest.approx(0.0)
    assert out.Q_ext_bottom == pytest.approx(2000.0)


def test_cooling_water_takes_a_utility_band_that_sits_on_its_temperature():
    T = np.array([25.0, 24.9])
    H = np.array([0.0, 500.0])
    bands = [(25.0, 24.9)]

    out = cascade_with_air(
        T,
        H,
        T_air_source=0.0,
        T_air_sink=10.0,
        T_cooling_water=25.0,
        utility_bands=bands,
    )

    assert out.Q_air_sink == pytest.approx(500.0)

    out = cascade_with_air(
        T,
        H,
        T_air_source=0.0,
        T_air_sink=30.0,
        T_cooling_water=25.0,
        utility_bands=bands,
    )

    assert out.Q_air_sink == pytest.approx(0.0)
    assert out.Q_cooling_water == pytest.approx(500.0)


def test_air_supplies_a_utility_band_that_sits_on_its_source_temperature():
    # The same band at the air source: it needs heat just above the limit.
    T = np.array([10.1, 10.0])
    H = np.array([300.0, 0.0])

    out = cascade_with_air(
        T, H, T_air_source=10.0, T_air_sink=200.0, utility_bands=[(10.1, 10.0)]
    )

    assert out.Q_air_source == pytest.approx(300.0)
    assert out.Q_ext_top == pytest.approx(0.0)


def test_only_narrow_steps_of_a_utility_background_count_as_bands():
    T = np.array([60.0, 59.9, 25.0, 20.0, 19.9])

    assert utility_bands(T) == [(60.0, 59.9), (20.0, 19.9)]


@pytest.mark.parametrize("is_utility, expected_air", [(True, 2000.0), (False, 0.0)])
def test_air_duties_treat_only_a_utility_background_as_bands(is_utility, expected_air):
    args = SimpleNamespace(period_idx=0, background_is_utility=is_utility)
    hot = _sc(_stream("Bg", 20.0, 19.9, 2000.0, is_process_stream=False))

    out = _cascade_air_duties(
        hot,
        _sc(),
        args,
        T_air_source=0.0,
        T_air_sink=20.0,
        T_background=np.array([20.0, 19.9]),
    )

    assert out.Q_air_sink == pytest.approx(expected_air)


def test_disabled_air_takes_nothing_and_leaves_cooling_water_in_place():
    # As above, air at a 40 C sink would take 100 kW. Without air, cooling
    # water at 20 C takes the 130 kW crossing 20 C and 30 kW stays external.
    T = np.array([150.0, 50.0, 30.0, 10.0])
    H = np.array([0.0, 100.0, 100.0, 160.0])

    out = cascade_with_air(
        T, H, T_air_source=200.0, T_air_sink=40.0, T_cooling_water=20.0, air=False
    )

    assert (out.Q_air_source, out.Q_air_sink) == (0.0, 0.0)
    assert out.Q_cooling_water == pytest.approx(130.0)
    assert out.Q_ext_bottom == pytest.approx(30.0)
    assert out.Q_ext_top == pytest.approx(0.0)


@pytest.mark.parametrize("enabled, expected_air", [(True, 2000.0), (False, 0.0)])
def test_air_duties_follow_the_ambient_air_option(enabled, expected_air):
    args = SimpleNamespace(period_idx=0, ambient_air_enabled=enabled)
    hot = _sc(_stream("Bg", 30.0, 25.0, 2000.0))

    out = _cascade_air_duties(hot, _sc(), args, T_air_source=0.0, T_air_sink=20.0)

    assert out.Q_air_sink == pytest.approx(expected_air)
    assert out.Q_ext_bottom == pytest.approx(2000.0 - expected_air)


def test_carnot_search_offers_no_air_availability_when_air_is_off():
    # The heat pump's evaporators may draw on air below ambient; without air
    # only the background's own heat is available, as if no level were ambient.
    def parse(z_amb_hot, enabled):
        args = SimpleNamespace(
            n_cond=1,
            n_evap=1,
            is_heat_pumping=True,
            Q_heat_max=100.0,
            Q_cool_max=80.0,
            T_cold=np.array([120.0, 80.0]),
            H_cold=np.array([100.0, 0.0]),
            T_hot=np.array([60.0, 30.0, 10.0]),
            H_hot=np.array([0.0, -50.0, -80.0]),
            z_amb_hot=z_amb_hot,
            z_amb_cold=np.zeros(2),
            ambient_air_enabled=enabled,
        )
        # The evaporator sits at 15 C, inside the ambient levels below 30 C.
        return parse_carnot_state_variables(np.array([0.5, 0.1, 1.0]), args)

    ambient_below_30 = np.array([0.0, 0.0, -1.0])
    with_air = parse(ambient_below_30, True)
    without_air = parse(ambient_below_30, False)
    no_ambient_levels = parse(np.zeros(3), True)

    assert no_ambient_levels.Q_cool_available.sum() == pytest.approx(72.5)
    assert with_air.Q_cool_available.sum() == pytest.approx(207.5)
    np.testing.assert_allclose(
        without_air.Q_cool_available, no_ambient_levels.Q_cool_available
    )


@pytest.mark.parametrize("enabled", [True, False])
def test_refrigeration_target_uses_air_only_when_the_option_allows(enabled):
    # With ambient at 60 C, air is a sink for heat the refrigerator rejects.
    problem = PinchProblem(
        "OpenPinch/tutorials/sample_cases/basic_pinch.json", project_name="Air"
    )
    problem.update_options(
        {"HPR_AMBIENT_AIR_ENABLED": enabled, "ENV_TEMPERATURE": 60.0}
    )

    target = problem.target.carnot_refrigeration(
        condensers=1,
        evaporators=1,
        maximum_restarts=1,
        maximum_iterations=20,
        maximum_evaluations=50,
    )

    assert target.hpr_success
    assert target.hpr_ambient_hot == 0.0
    if enabled:
        assert target.hpr_ambient_cold > 0.0
    else:
        assert target.hpr_ambient_cold == 0.0
