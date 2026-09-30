import numpy as np
import pytest

from OpenPinch.analysis.heat_pumps.common.encoding import (
    DutyAllocationRequest,
    StageDutyRequest,
    allocate_stage_duties,
    decode_available_fractions,
    encode_available_fractions,
    limit_available_duty,
    map_DT_arr_to_x_arr,
    map_Q_amb_to_x,
    map_Q_arr_to_x_arr,
    map_T_arr_to_x_arr,
    map_x_arr_to_DT_arr,
    map_x_arr_to_Q_arr,
    map_x_arr_to_T_arr,
    map_x_to_Q_amb,
    require_stage_duty_allocation,
)
from OpenPinch.contracts.hpr import HPRParsedState


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


def test_duty_fractions_act_on_their_own_stage_only():
    Q_available = np.array([40.0, 0.0, 60.0])

    base = decode_available_fractions(np.array([0.5, 0.5, 0.5]), Q_available)
    moved = decode_available_fractions(np.array([0.9, 0.5, 0.5]), Q_available)

    np.testing.assert_allclose(base, [20.0, 0.0, 30.0])
    # Changing one fraction changes only that stage's duty.
    np.testing.assert_allclose(moved - base, [16.0, 0.0, 0.0])
    # Fractions are clipped to [0, 1], so no stage exceeds its availability.
    np.testing.assert_allclose(
        decode_available_fractions(np.array([1.5, -1.0, 1.0]), Q_available),
        [40.0, 0.0, 60.0],
    )


def test_distinct_fractions_give_distinct_duties_where_duty_is_available():
    Q_available = np.array([40.0, 60.0])
    a = decode_available_fractions(np.array([0.2, 0.7]), Q_available)
    b = decode_available_fractions(np.array([0.7, 0.2]), Q_available)

    assert not np.allclose(a, b)


def test_encode_available_fractions_round_trips_and_sanitises_seeds():
    Q_available = np.array([50.0, 0.0, 40.0, 30.0])
    Q_seed = np.array([25.0, 10.0, np.nan, 90.0])

    x = encode_available_fractions(Q_seed, Q_available)

    # Nothing available -> 0; non-finite -> 0; above availability -> 1.
    np.testing.assert_allclose(x, [0.5, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        decode_available_fractions(x, Q_available),
        np.minimum(np.nan_to_num(Q_seed), Q_available),
    )
    with pytest.raises(ValueError, match="same shape"):
        encode_available_fractions(np.array([1.0]), Q_available)


def test_limit_available_duty_scales_stages_to_the_capacity():
    np.testing.assert_allclose(
        limit_available_duty(np.array([60.0, 90.0]), 100.0), [40.0, 60.0]
    )
    np.testing.assert_allclose(
        limit_available_duty(np.array([30.0, 20.0]), 100.0), [30.0, 20.0]
    )
    np.testing.assert_allclose(
        limit_available_duty(np.array([30.0, -5.0]), 0.0), [0.0, 0.0]
    )


def test_allocate_stage_duties_decodes_fractions_of_availability():
    allocation = allocate_stage_duties(np.array([0.5, 1.0]), np.array([40.0, 60.0]))

    np.testing.assert_allclose(allocation.Q_model, [20.0, 60.0])
    assert allocation.Q_base == pytest.approx(80.0)

    with pytest.raises(ValueError, match="same length"):
        allocate_stage_duties(np.array([0.5, 1.0]), np.array([40.0]))


def test_linear_mapping_helpers_cover_zero_and_scalar_edges():
    np.testing.assert_allclose(
        map_x_arr_to_DT_arr(
            np.array([0.5, 1.0]),
            np.array([100.0, 50.0]),
            0.0,
        ),
        np.array([50.0, 50.0]),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        np.testing.assert_allclose(
            map_DT_arr_to_x_arr(
                np.array([20.0, 10.0]),
                np.array([40.0, 0.0]),
                0.0,
            ),
            np.array([0.5, 0.0]),
        )
    np.testing.assert_allclose(
        map_x_arr_to_Q_arr(np.array([0.25, 0.5]), 200.0),
        np.array([50.0, 100.0]),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        np.testing.assert_allclose(
            map_Q_arr_to_x_arr(np.array([50.0, 100.0]), 0.0),
            np.array([0.0, 0.0]),
        )
    assert map_x_to_Q_amb(0.5, 0.0) == (0.0, 0.0)
    assert map_Q_amb_to_x(10.0, 20.0, 0.0) == 0.0


def test_require_stage_duty_allocation_reports_missing_split_contract():
    allocation = require_stage_duty_allocation(
        x_split=np.array([0.5, 1.0]),
        Q_available=np.array([100.0, 25.0]),
        duty_name="heat",
    )

    np.testing.assert_allclose(allocation.Q_model, np.array([50.0, 25.0]))
    with pytest.raises(ValueError, match="Q_heat_base requires x_heat_split"):
        require_stage_duty_allocation(
            x_split=None,
            Q_available=np.array([100.0]),
            duty_name="heat",
        )


def test_duty_allocation_request_collects_parsed_state_fields():
    state = HPRParsedState(
        Q_heat_base=100.0,
        x_heat_split=np.array([0.5, 1.0]),
        Q_heat_available=np.array([100.0, 25.0]),
        Q_cool_available=np.array([40.0]),
    )

    request = DutyAllocationRequest.from_state(state)

    assert request.heat.Q_base == 100.0
    assert request.heat.x_split is state.x_heat_split
    assert request.heat.Q_available is state.Q_heat_available
    assert request.cool.Q_base is None
    assert request.cool.x_split is None
    assert request.cool.Q_available is state.Q_cool_available
    np.testing.assert_allclose(
        request.heat.allocate("heat").Q_model, np.array([50.0, 25.0])
    )
    assert DutyAllocationRequest().heat == StageDutyRequest()
    with pytest.raises(ValueError, match="Q_cool_base requires x_cool_split"):
        request.cool.allocate("cool")
