from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.analysis.heat_pump_and_refrigeration import shared as hp_shared
from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.lib.enums import PT

from ..helpers import _base_args, _sc, _stream


def test_compute_entropic_average_temperature_in_K_constant_temperature():
    result = hp_shared._compute_entropic_mean_temperature(
        np.array([60.0, 60.0, 60.0]),
        np.array([100.0, 150.0, 200.0]),
    )
    np.testing.assert_allclose(result, 60.0 + 273.15)


def test_compute_entropic_average_temperature_in_K_zero_net_duty_uses_arithmetic_mean():
    result = hp_shared._compute_entropic_mean_temperature(
        np.array([40.0, 60.0, 80.0]),
        np.zeros(3),
    )
    np.testing.assert_allclose(result, 60.0 + 273.15)


def test_prepare_latent_hp_profile_merges_hot_segments_and_sums_duty():
    T_out, Q_out = hp_shared._get_carnot_hpr_cycle_cascade_profile(
        [150.0, 149.7, 130.0], [10.0, 5.0, 20.0], dT_phase_change=1.0, is_hot=True
    )
    np.testing.assert_allclose(T_out, np.array([150.0, 130.0]))
    np.testing.assert_allclose(Q_out, np.array([15.0, 20.0]))


def test_prepare_latent_hp_profile_merges_hot_segments_and_sums_duty_consecutively():
    T_out, Q_out = hp_shared._get_carnot_hpr_cycle_cascade_profile(
        [150.0, 149.7, 149.01, 130.0],
        [10.0, 2.0, 3.0, 20.0],
        dT_phase_change=1.0,
        is_hot=True,
    )
    np.testing.assert_allclose(T_out, np.array([150.0, 130.0]))
    np.testing.assert_allclose(Q_out, np.array([15.0, 20.0]))


def test_prepare_latent_hp_profile_merges_cold_segments_with_lower_temperature():
    T_out, Q_out = hp_shared._get_carnot_hpr_cycle_cascade_profile(
        [60.0, 45.0, 30.4, 30.0],
        [12.5, 10.0, 7.5, 5.0],
        dT_phase_change=1.0,
        is_hot=False,
    )
    np.testing.assert_allclose(T_out, np.array([60.0, 45.0, 30.0]))
    np.testing.assert_allclose(Q_out, np.array([12.5, 10.0, 12.5]))


def test_prepare_latent_hp_profile_merges_cold_segments_with_complex_temperature():
    T_out, Q_out = hp_shared._get_carnot_hpr_cycle_cascade_profile(
        [31.5, 30.8, 30.4, 30.0],
        [12.5, 10.0, 7.5, 5.0],
        dT_phase_change=1.0,
        is_hot=False,
    )
    np.testing.assert_allclose(T_out, np.array([31.5, 30.0]))
    np.testing.assert_allclose(Q_out, np.array([12.5, 22.5]))


def test_prepare_latent_hp_profile_keeps_unique_sequences_unchanged():
    T_out, Q_out = hp_shared._get_carnot_hpr_cycle_cascade_profile(
        [180.0, 160.0, 140.0], [8.0, 6.0, 4.0], dT_phase_change=0.5, is_hot=True
    )
    np.testing.assert_allclose(T_out, np.array([180.0, 160.0, 140.0]))
    np.testing.assert_allclose(Q_out, np.array([8.0, 6.0, 4.0]))


def test_prepare_latent_hp_profile_handles_empty_input():
    T_out, Q_out = hp_shared._get_carnot_hpr_cycle_cascade_profile(
        [], [], dT_phase_change=1.0, is_hot=True
    )
    assert len(T_out) == 0
    assert len(Q_out) == 0


def test_validate_vapour_hp_refrigerant_ls_defaults_to_water_and_matches_length():
    refrigerants = hp_shared._validate_vapour_hp_refrigerant_ls(
        3, SimpleNamespace(refrigerant_ls=[])
    )
    assert refrigerants == ["water", "water", "water"]


def test_validate_vapour_hp_refrigerant_ls_preserves_length_when_provided():
    refrigerants = hp_shared._validate_vapour_hp_refrigerant_ls(
        2,
        SimpleNamespace(refrigerant_ls=["Water", "Ammonia"], do_refrigerant_sort=True),
    )
    assert len(refrigerants) == 2
    assert refrigerants[0] == "Water"


def test_validate_vapour_hp_refrigerant_ls_extends_to_match_n_cond():
    refrigerants = hp_shared._validate_vapour_hp_refrigerant_ls(
        4,
        SimpleNamespace(refrigerant_ls=["Ammonia", "Water"], do_refrigerant_sort=True),
    )
    assert refrigerants[:2] == ["Water", "Ammonia"]
    assert refrigerants[2:] == ["Ammonia", "Ammonia"]


def test_misc_heat_pump_helpers_and_stream_builders():
    with pytest.raises(ValueError, match="Infeasible temperature interval"):
        hp_shared._create_stream_collection_of_background_profile(
            T_vals=np.array([100.0, 100.0]),
            H_vals=np.array([0.0, 10.0]),
        )

    hot = hp_shared._create_stream_collection_of_background_profile(
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_vals=np.array([0.0, -30.0, 20.0]),
    )
    cold = hp_shared._create_stream_collection_of_background_profile(
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_vals=np.array([0.0, 30.0, -20.0]),
    )
    assert isinstance(hot, StreamCollection)
    assert isinstance(cold, StreamCollection)

    q_vals = hp_shared._get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_hpr=np.array([100.0, 60.0]),
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_vals=np.array([100.0, 50.0, 0.0]),
        is_cond=False,
    )
    assert q_vals.shape == (2,)

    t_avg = hp_shared._compute_entropic_mean_temperature(
        np.array([300.0, 310.0]), np.array([10.0, 5.0]), input_T_units="K"
    )
    assert t_avg > 0.0

    refs = hp_shared._validate_vapour_hp_refrigerant_ls(
        1, _base_args(refrigerant_ls=["R134A", "R600"], do_refrigerant_sort=False)
    )
    assert refs == ["R134A"]

    streams = hp_shared._build_latent_streams(
        T_ls=np.array([110.0, 109.2, 80.0]),
        dT_phase_change=1.0,
        Q_ls=np.array([20.0, 10.0, 30.0]),
        is_hot=True,
        prefix="HP",
    )
    assert len(streams) >= 2

    class _FakeCycle:
        def build_stream_collection(self, **kwargs):
            return _sc(
                _stream("C", 90.0, 70.0, 10.0, is_process_stream=False)
            )

    agg = hp_shared._build_simulated_hpr_streams(
        [_FakeCycle()], include_cond=True, include_evap=True, dtcont_hp=5.0
    )
    assert len(agg) >= 1

    amb0 = hp_shared._get_ambient_air_stream(0.0, 0.0, _base_args())
    amb_pos = hp_shared._get_ambient_air_stream(10.0, 0.0, _base_args())
    amb_neg = hp_shared._get_ambient_air_stream(0.0, 10.0, _base_args())
    assert len(amb0) == 0
    assert len(amb_pos) == 1
    assert len(amb_neg) == 1

    assert hp_shared._calc_obj(
        10.0, 5.0, 0.0, 100.0, heat_to_power_ratio=2.0, penalty=1.0
    ) == pytest.approx(0.21)
    assert hp_shared._calc_Q_amb(50.0, 30.0, 5.0) == pytest.approx(25.0)


def test_get_heat_pump_cascade_helper(monkeypatch):
    hot = _sc(_stream("H", 120.0, 110.0, 5.0, is_process_stream=False))
    cold = _sc(_stream("C", 70.0, 80.0, 5.0, is_process_stream=False))

    monkeypatch.setattr(
        hp_shared,
        "create_problem_table_with_t_int",
        lambda streams, is_shifted: ProblemTable(
            {
                PT.T.value: [120.0, 80.0],
                PT.H_HOT_UT.value: [0.0, 0.0],
                PT.H_COLD_UT.value: [0.0, 0.0],
            }
        ),
    )
    monkeypatch.setattr(
        hp_shared,
        "get_utility_heat_cascade",
        lambda *args, **kwargs: {
            PT.H_HOT_UT.value: np.array([5.0, 0.0]),
            PT.H_COLD_UT.value: np.array([3.0, 0.0]),
        },
    )

    out = hp_shared._get_hpr_cascade(hot, cold)
    assert len(out) == 3
