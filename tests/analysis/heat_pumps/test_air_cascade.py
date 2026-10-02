import numpy as np
import pytest

from OpenPinch.analysis.heat_pumps.common._shared.air import cascade_with_air


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
