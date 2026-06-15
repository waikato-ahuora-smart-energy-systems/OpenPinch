import numpy as np
import pytest

from OpenPinch.services.heat_pump_integration.common.encoding import (
    allocate_stage_duties,
    decode_duty_splits,
    encode_base_and_duty_splits,
    encode_duty_splits,
    map_Q_amb_to_x,
    map_T_arr_to_x_arr,
    map_x_arr_to_T_arr,
    map_x_to_Q_amb,
    require_stage_duty_allocation,
)


def test_map_x_to_T_returns_expected_descending_temperatures():
    x = np.array([0.1, 0.2, 0.3])
    T_0 = 200.0
    T_1 = 100.0

    result_T = map_x_arr_to_T_arr(x, T_0, T_1)
    np.testing.assert_allclose(result_T, np.array([190.0, 172.0, 150.4]))

    result_x = map_T_arr_to_x_arr(result_T, T_0, T_1)
    np.testing.assert_allclose(result_x, x)


def test_map_x_to_T_output_is_monotonically_descending():
    x = np.array([0.4, 0.2, 0.1, 0.3])
    T_0 = 180.0
    T_1 = 60.0

    result = map_x_arr_to_T_arr(x, T_0, T_1)

    assert result.size == x.size
    assert np.all(np.diff(result) <= 0.0)


@pytest.mark.parametrize(
    ("Q_amb_hot", "Q_amb_cold"),
    [(0.0, 400.0), (150.0, 0.0), (0.0, 0.0)],
)
def test_ambient_mapping_round_trips_with_bounded_x(Q_amb_hot, Q_amb_cold):
    scale = 200.0

    x_amb = map_Q_amb_to_x(Q_amb_hot, Q_amb_cold, scale)
    mapped_hot, mapped_cold = map_x_to_Q_amb(x_amb, scale)

    assert abs(x_amb) < 1.0
    assert mapped_hot == pytest.approx(Q_amb_hot)
    assert mapped_cold == pytest.approx(Q_amb_cold)


def test_duty_split_decode_handles_zero_base():
    out = decode_duty_splits(np.array([0.4, 0.5, 1.0]), 0.0)

    np.testing.assert_allclose(out, np.zeros(3))


def test_duty_split_encode_decode_round_trip_for_exhausted_base():
    Q_request = np.array([40.0, 30.0, 30.0])
    Q_base = 100.0

    x_split = encode_duty_splits(Q_request, Q_base)

    np.testing.assert_allclose(decode_duty_splits(x_split, Q_base), Q_request)
    assert np.all((x_split >= 0.0) & (x_split <= 1.0))


def test_duty_split_encode_clips_oversized_seed_duties():
    x_split = encode_duty_splits(np.array([80.0, 80.0]), 100.0)

    np.testing.assert_allclose(x_split, np.array([0.8, 1.0]))
    np.testing.assert_allclose(
        decode_duty_splits(x_split, 100.0),
        np.array([80.0, 20.0]),
    )


def test_encode_base_and_duty_splits_sanitises_seed_and_limits_base():
    Q_base, x_base, x_split = encode_base_and_duty_splits(
        np.array([80.0, np.nan, -5.0, 80.0]),
        100.0,
    )

    assert Q_base == pytest.approx(100.0)
    assert x_base == pytest.approx(1.0)
    np.testing.assert_allclose(
        decode_duty_splits(x_split, Q_base),
        np.array([50.0, 0.0, 0.0, 50.0]),
    )


def test_allocate_stage_duties_clips_to_availability_and_reports_excess():
    allocation = allocate_stage_duties(
        100.0,
        np.array([0.5, 1.0]),
        np.array([40.0, 60.0]),
    )

    np.testing.assert_allclose(allocation.Q_request, np.array([50.0, 50.0]))
    np.testing.assert_allclose(allocation.Q_model, np.array([40.0, 50.0]))
    np.testing.assert_allclose(allocation.Q_excess, np.array([10.0, 0.0]))


def test_require_stage_duty_allocation_reports_missing_split_contract():
    with pytest.raises(ValueError, match="Q_heat_base requires x_heat_split"):
        require_stage_duty_allocation(
            Q_base=100.0,
            x_split=None,
            Q_available=np.array([100.0]),
            duty_name="heat",
        )
