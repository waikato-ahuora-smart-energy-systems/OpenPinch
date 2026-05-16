import numpy as np
import pytest

from OpenPinch.services.heat_pump_integration.common.encoding import (
    map_Q_amb_to_x,
    map_T_arr_to_x_arr,
    map_x_to_Q_amb,
    map_x_arr_to_T_arr,
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
