from types import SimpleNamespace

import numpy as np
import pytest

import OpenPinch.analysis.heat_pumps.common._shared.ambient_preallocation as hp_ambient
import OpenPinch.analysis.heat_pumps.common._shared.plotting as hp_plotting
import OpenPinch.analysis.heat_pumps.common._shared.streams as hp_streams
import OpenPinch.analysis.heat_pumps.common.preprocessing as hp_preprocessing
import OpenPinch.analysis.heat_pumps.common.shared as hp_shared
from OpenPinch.analysis.heat_pumps.common.postprocessing import (
    _compute_utility_cost,
)
from OpenPinch.contracts.hpr import (
    HPRBackendResult,
    HPRParsedState,
    HPRThermoArtifacts,
    SimulatedHPRAnnualizedCostAccounting,
)
from OpenPinch.domain.enums import ProblemTableLabel
from OpenPinch.domain.problem_table import ProblemTable
from OpenPinch.domain.stream_collection import StreamCollection
from OpenPinch.domain.value import Value

from .helpers import _base_args, _sc, _stream


def test_compute_entropic_average_temperature_in_K_constant_temperature():
    result = hp_shared.compute_entropic_mean_temperature(
        np.array([60.0, 60.0, 60.0]),
        np.array([100.0, 150.0, 200.0]),
    )
    np.testing.assert_allclose(result, 60.0 + 273.15)


def test_compute_entropic_average_temperature_in_K_zero_net_duty_uses_arithmetic_mean():
    result = hp_shared.compute_entropic_mean_temperature(
        np.array([40.0, 60.0, 80.0]),
        np.zeros(3),
    )
    np.testing.assert_allclose(result, 60.0 + 273.15)


def test_carnot_helpers_return_scalars_and_arrays_for_feasible_and_infeasible_lifts():
    assert hp_shared.calc_carnot_heat_pump_cop(350.0, 300.0, 0.5) == pytest.approx(4.0)
    cop = hp_shared.calc_carnot_heat_pump_cop(
        np.array([350.0, 300.0]), np.array([300.0, 300.0]), 0.5
    )
    np.testing.assert_allclose(cop, np.array([4.0, np.inf]))

    assert hp_shared.calc_carnot_heat_engine_eta(350.0, 300.0, 0.5) == pytest.approx(
        (1.0 - 300.0 / 350.0) * 0.5
    )
    eta = hp_shared.calc_carnot_heat_engine_eta(
        np.array([350.0, 300.0]), np.array([300.0, 300.0]), 0.5
    )
    np.testing.assert_allclose(eta, np.array([(1.0 - 300.0 / 350.0) * 0.5, 0.0]))


def test_prepare_latent_hp_profile_merges_hot_segments_and_sums_duty():
    T_out, Q_out = hp_streams._get_carnot_hpr_cycle_cascade_profile(
        [150.0, 149.7, 130.0], [10.0, 5.0, 20.0], dT_phase_change=1.0, is_hot=True
    )
    np.testing.assert_allclose(T_out, np.array([150.0, 130.0]))
    np.testing.assert_allclose(Q_out, np.array([15.0, 20.0]))


def test_prepare_latent_hp_profile_merges_hot_segments_and_sums_duty_consecutively():
    T_out, Q_out = hp_streams._get_carnot_hpr_cycle_cascade_profile(
        [150.0, 149.7, 149.01, 130.0],
        [10.0, 2.0, 3.0, 20.0],
        dT_phase_change=1.0,
        is_hot=True,
    )
    np.testing.assert_allclose(T_out, np.array([150.0, 130.0]))
    np.testing.assert_allclose(Q_out, np.array([15.0, 20.0]))


def test_prepare_latent_hp_profile_merges_cold_segments_with_lower_temperature():
    T_out, Q_out = hp_streams._get_carnot_hpr_cycle_cascade_profile(
        [60.0, 45.0, 30.4, 30.0],
        [12.5, 10.0, 7.5, 5.0],
        dT_phase_change=1.0,
        is_hot=False,
    )
    np.testing.assert_allclose(T_out, np.array([60.0, 45.0, 30.0]))
    np.testing.assert_allclose(Q_out, np.array([12.5, 10.0, 12.5]))


def test_prepare_latent_hp_profile_merges_cold_segments_with_complex_temperature():
    T_out, Q_out = hp_streams._get_carnot_hpr_cycle_cascade_profile(
        [31.5, 30.8, 30.4, 30.0],
        [12.5, 10.0, 7.5, 5.0],
        dT_phase_change=1.0,
        is_hot=False,
    )
    np.testing.assert_allclose(T_out, np.array([31.5, 30.0]))
    np.testing.assert_allclose(Q_out, np.array([12.5, 22.5]))


def test_prepare_latent_hp_profile_keeps_unique_sequences_unchanged():
    T_out, Q_out = hp_streams._get_carnot_hpr_cycle_cascade_profile(
        [180.0, 160.0, 140.0], [8.0, 6.0, 4.0], dT_phase_change=0.5, is_hot=True
    )
    np.testing.assert_allclose(T_out, np.array([180.0, 160.0, 140.0]))
    np.testing.assert_allclose(Q_out, np.array([8.0, 6.0, 4.0]))


def test_prepare_latent_hp_profile_handles_empty_input():
    T_out, Q_out = hp_streams._get_carnot_hpr_cycle_cascade_profile(
        [], [], dT_phase_change=1.0, is_hot=True
    )
    assert len(T_out) == 0
    assert len(Q_out) == 0


def test_validate_vapour_hp_refrigerant_ls_defaults_to_water_and_matches_length():
    refrigerants = hp_shared.validate_vapour_hp_refrigerant_ls(
        3, SimpleNamespace(refrigerant_ls=[])
    )
    assert refrigerants == ["water", "water", "water"]


def test_validate_vapour_hp_refrigerant_ls_preserves_length_when_provided():
    refrigerants = hp_shared.validate_vapour_hp_refrigerant_ls(
        2,
        SimpleNamespace(refrigerant_ls=["Water", "Ammonia"], do_refrigerant_sort=True),
    )
    assert len(refrigerants) == 2
    assert refrigerants[0] == "Water"


def test_validate_vapour_hp_refrigerant_ls_extends_to_match_n_cond():
    refrigerants = hp_shared.validate_vapour_hp_refrigerant_ls(
        4,
        SimpleNamespace(refrigerant_ls=["Ammonia", "Water"], do_refrigerant_sort=True),
    )
    assert refrigerants[:2] == ["Water", "Ammonia"]
    assert refrigerants[2:] == ["Ammonia", "Ammonia"]


def test_misc_heat_pump_helpers_and_stream_builders():
    with pytest.raises(ValueError, match="Infeasible temperature interval"):
        hp_preprocessing._create_stream_collection_of_background_profile(
            T_vals=np.array([100.0, 100.0]),
            H_vals=np.array([0.0, 10.0]),
        )

    hot = hp_preprocessing._create_stream_collection_of_background_profile(
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_vals=np.array([0.0, -30.0, 20.0]),
    )
    cold = hp_preprocessing._create_stream_collection_of_background_profile(
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_vals=np.array([0.0, 30.0, -20.0]),
    )
    assert isinstance(hot, StreamCollection)
    assert isinstance(cold, StreamCollection)

    q_vals = hp_streams.get_Q_vals_at_T_hpr_from_bckgrd_profile(
        T_hpr=np.array([100.0, 60.0]),
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_vals=np.array([100.0, 50.0, 0.0]),
        is_cond=False,
    )
    assert q_vals.shape == (2,)

    t_avg = hp_shared.compute_entropic_mean_temperature(
        np.array([300.0, 310.0]), np.array([10.0, 5.0]), input_T_units="K"
    )
    assert t_avg > 0.0

    refs = hp_shared.validate_vapour_hp_refrigerant_ls(
        1, _base_args(refrigerant_ls=["R134A", "R600"], do_refrigerant_sort=False)
    )
    assert refs == ["R134A"]

    streams = hp_streams._build_latent_streams(
        T_ls=np.array([110.0, 109.2, 80.0]),
        dT_phase_change=1.0,
        Q_ls=np.array([20.0, 10.0, 30.0]),
        is_hot=True,
        prefix="HP",
    )
    assert len(streams) >= 2

    amb0 = hp_streams.get_ambient_air_stream(0.0, 0.0, _base_args())
    amb_pos = hp_streams.get_ambient_air_stream(10.0, 0.0, _base_args())
    amb_neg = hp_streams.get_ambient_air_stream(0.0, 10.0, _base_args())
    assert len(amb0) == 0
    assert len(amb_pos) == 1
    assert len(amb_neg) == 1


def test_ambient_preallocation_helpers_cover_zero_and_out_of_range_edges():
    T_hot = np.array([120.0, 80.0, 40.0])
    H_hot = np.array([0.0, -20.0, -40.0])
    np.testing.assert_allclose(
        hp_ambient._remove_direct_ambient_sink_from_hot_profile(
            T_hot,
            H_hot,
            duty=0.0,
            T_amb=80.0,
        ),
        (T_hot, H_hot),
    )
    np.testing.assert_allclose(
        hp_ambient._remove_direct_ambient_sink_from_hot_profile(
            T_hot,
            np.array([40.0, 20.0, 0.0]),
            duty=5.0,
            T_amb=80.0,
        ),
        (T_hot, np.array([40.0, 20.0, 0.0])),
    )

    T_cold = np.array([120.0, 80.0, 40.0])
    H_cold = np.array([0.0, -5.0, -10.0])
    np.testing.assert_allclose(
        hp_ambient._remove_direct_ambient_source_from_cold_profile(
            T_cold,
            H_cold,
            duty=5.0,
            T_amb=80.0,
        ),
        (T_cold, H_cold),
    )

    T_inserted, H_inserted = hp_ambient._insert_profile_temperatures(
        T_hot,
        H_hot,
        200.0,
        80.0,
    )
    np.testing.assert_allclose(T_inserted, T_hot)
    np.testing.assert_allclose(H_inserted, H_hot)


def test_compute_utility_cost_skips_utilities_without_cost_data():
    no_cost = _stream("free", 120.0, 80.0, 10.0, is_process_stream=False)
    priced = _stream("priced", 120.0, 80.0, 10.0, is_process_stream=False)
    priced.set_value_attr("utility_cost", [3.5], update_derived=False)

    assert _compute_utility_cost(
        _sc(no_cost), _sc(priced), period_idx=0
    ) == pytest.approx(3.5)
    assert _compute_utility_cost(_sc(no_cost), _sc(), period_idx=0) == 0.0
    assert (
        _compute_utility_cost([SimpleNamespace(utility_cost=None)], [], period_idx=0)
        == 0.0
    )


def test_simulated_hpr_annualized_costs_use_value_units_and_duty_based_hx_cost():
    streams = _sc(
        _stream("cond", 120.0, 100.0, 20.0, is_process_stream=False),
        _stream("evap", 40.0, 60.0, 10.0, is_process_stream=False),
    )
    args = _base_args(
        ele_price=100.0,
        annual_op_time=1000.0,
        heat_to_power_ratio=0.5,
        cold_to_power_ratio=0.25,
        hpr_comp_fixed_cost=100.0,
        hpr_comp_variable_cost=10.0,
        hpr_comp_cost_exp=1.0,
        hpr_hx_fixed_cost=50.0,
        hpr_hx_variable_cost=2.0,
        hpr_hx_cost_exp=1.0,
        discount_rate=0.1,
        serv_life=10.0,
    )

    costs = hp_shared.calc_simulated_hpr_annualized_costs(
        work=10.0,
        work_arr=np.array([6.0, 4.0]),
        Q_ext_heat=8.0,
        Q_ext_cold=4.0,
        hpr_streams=streams,
        hx_units=len(streams.get_hot_utility_streams())
        + len(streams.get_cold_utility_streams()),
        penalty_power_equivalent=2.0,
        args=args,
    )

    assert isinstance(costs, SimulatedHPRAnnualizedCostAccounting)
    assert isinstance(costs.hpr_operating_cost, Value)
    assert costs.hpr_operating_cost.unit == "$/y"
    assert costs.hpr_operating_cost.value == pytest.approx(1500.0)
    assert costs.hpr_compressor_capital_cost.unit == "$"
    assert costs.hpr_compressor_capital_cost.value == pytest.approx(300.0)
    assert costs.hpr_heat_exchanger_capital_cost.unit == "$"
    assert costs.hpr_heat_exchanger_capital_cost.value == pytest.approx(160.0)
    assert costs.hpr_capital_cost.value == pytest.approx(460.0)
    assert costs.hpr_total_annualized_cost.unit == "$/y"
    assert costs.feasibility_penalty.unit == "$/y"
    assert costs.feasibility_penalty.value == pytest.approx(200.0)


def test_cycle_penalty_ignores_missing_and_negative_terms():
    args = _base_args(eta_penalty=0.001, rho_penalty=10.0)

    assert hp_shared._cycle_penalty(args=args) == 0.0
    assert (
        hp_shared._cycle_penalty(
            args=args,
            cycle_penalty_terms=[-1.0, 0.0],
        )
        == 0.0
    )


def test_cycle_penalty_scores_only_cycle_terms():
    args = _base_args(eta_penalty=0.0, rho_penalty=2.0)

    penalty = hp_shared._cycle_penalty(
        args=args,
        cycle_penalty_terms=[3.0, -10.0, 4.0],
    )

    assert penalty == pytest.approx(50.0)


def test_direct_ambient_sink_preallocation_reduces_background_hot_profile():
    args = _base_args()

    ambient = hp_ambient.preallocate_direct_ambient_duties(
        args=args,
        Q_amb_hot=0.0,
        Q_amb_cold=50.0,
    )

    assert ambient.Q_amb_cold_direct == pytest.approx(50.0)
    assert ambient.Q_amb_cold_residual == pytest.approx(0.0)
    np.testing.assert_allclose(
        ambient.T_hot_residual,
        np.array([140.0, 90.0, 71.25, 40.0]),
    )
    np.testing.assert_allclose(
        ambient.H_hot_residual,
        np.array([0.0, -80.0, -110.0, -110.0]),
    )
    np.testing.assert_allclose(ambient.T_cold_residual, args.T_cold)
    np.testing.assert_allclose(ambient.H_cold_residual, args.H_cold)


def test_direct_ambient_source_preallocation_reduces_background_cold_profile():
    args = _base_args(T_env=90.0)

    ambient = hp_ambient.preallocate_direct_ambient_duties(
        args=args,
        Q_amb_hot=50.0,
        Q_amb_cold=0.0,
    )

    assert ambient.Q_amb_hot_direct == pytest.approx(50.0)
    assert ambient.Q_amb_hot_residual == pytest.approx(0.0)
    np.testing.assert_allclose(
        ambient.T_cold_residual,
        np.array([130.0, 90.0, 80.0, 55.0, 30.0]),
    )
    np.testing.assert_allclose(
        ambient.H_cold_residual,
        np.array([150.0, 70.0, 50.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(ambient.T_hot_residual, args.T_hot)
    np.testing.assert_allclose(ambient.H_hot_residual, args.H_hot)


def test_direct_ambient_preallocation_keeps_excess_as_residual():
    args = _base_args()

    ambient = hp_ambient.preallocate_direct_ambient_duties(
        args=args,
        Q_amb_hot=0.0,
        Q_amb_cold=200.0,
    )

    assert ambient.Q_amb_cold_direct == pytest.approx(160.0)
    assert ambient.Q_amb_cold_residual == pytest.approx(40.0)
    np.testing.assert_allclose(ambient.T_hot_residual, args.T_hot)
    np.testing.assert_allclose(ambient.H_hot_residual, np.zeros(3))


def test_direct_ambient_preallocation_zero_duty_leaves_profiles_unchanged():
    args = _base_args()

    ambient = hp_ambient.preallocate_direct_ambient_duties(
        args=args,
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
    )

    assert ambient.Q_amb_hot_direct == pytest.approx(0.0)
    assert ambient.Q_amb_cold_direct == pytest.approx(0.0)
    np.testing.assert_allclose(ambient.T_hot_residual, args.T_hot)
    np.testing.assert_allclose(ambient.H_hot_residual, args.H_hot)
    np.testing.assert_allclose(ambient.T_cold_residual, args.T_cold)
    np.testing.assert_allclose(ambient.H_cold_residual, args.H_cold)


def test_vapour_evaluator_keeps_background_profiles_out_of_one_point_match():
    args = _base_args(
        T_hot=np.array([100.0, 50.0]),
        H_hot=np.array([0.0, -100.0]),
        T_cold=np.array([95.0, 45.0]),
        H_cold=np.array([100.0, 0.0]),
        z_amb_hot=np.zeros(2),
        z_amb_cold=np.zeros(2),
        Q_heat_max=100.0,
        Q_cool_max=100.0,
    )

    result = hp_shared.evaluate_vapour_hpr_result(
        args=args,
        state=HPRParsedState(
            Q_amb_hot=0.0,
            Q_amb_cold=0.0,
        ),
        work=0.0,
        work_arr=np.array([]),
        Q_heat=np.array([]),
        Q_cool=np.array([]),
        cop_h=1.0,
        hpr_streams=StreamCollection(),
    )

    assert result.Q_ext_heat == pytest.approx(100.0)
    assert result.Q_ext_cold == pytest.approx(100.0)


def test_vapour_evaluator_penalises_hpr_self_match_without_one_point_cascade():
    args = _base_args(
        T_hot=np.array([100.0, 50.0]),
        H_hot=np.array([0.0, 0.0]),
        T_cold=np.array([95.0, 45.0]),
        H_cold=np.array([0.0, 0.0]),
        z_amb_hot=np.zeros(2),
        z_amb_cold=np.zeros(2),
        Q_heat_max=0.0,
        Q_cool_max=0.0,
    )
    hpr_streams = _sc(
        _stream("cond", 120.0, 100.0, 100.0, is_process_stream=False),
        _stream("evap", 40.0, 60.0, 100.0, is_process_stream=False),
    )

    result = hp_shared.evaluate_vapour_hpr_result(
        args=args,
        state=HPRParsedState(
            Q_amb_hot=0.0,
            Q_amb_cold=0.0,
        ),
        work=0.0,
        work_arr=np.array([]),
        Q_heat=np.array([100.0]),
        Q_cool=np.array([100.0]),
        cop_h=1.0,
        hpr_streams=hpr_streams,
    )

    assert result.Q_ext == pytest.approx(0.0)
    assert result.feasibility_penalty > 0.0


def test_vapour_evaluator_counts_direct_ambient_sink_once():
    args = _base_args(
        T_hot=np.array([100.0, 50.0]),
        H_hot=np.array([0.0, -100.0]),
        T_cold=np.array([95.0, 45.0]),
        H_cold=np.array([0.0, 0.0]),
        z_amb_hot=np.zeros(2),
        z_amb_cold=np.zeros(2),
        Q_heat_max=0.0,
        Q_cool_max=100.0,
        T_env=20.0,
    )

    result = hp_shared.evaluate_vapour_hpr_result(
        args=args,
        state=HPRParsedState(
            Q_amb_hot=0.0,
            Q_amb_cold=60.0,
        ),
        work=0.0,
        work_arr=np.array([]),
        Q_heat=np.array([]),
        Q_cool=np.array([]),
        cop_h=1.0,
        hpr_streams=StreamCollection(),
    )

    assert result.Q_amb_cold == pytest.approx(60.0)
    assert result.Q_ext_cold == pytest.approx(40.0)


def test_carnot_debug_plot_uses_unmodified_background_profiles(monkeypatch):
    args = _base_args()

    def fake_plot(T_hot, H_hot, T_cold, H_cold, *_args, **_kwargs):
        assert len(T_hot) == len(H_hot)
        assert len(T_cold) == len(H_cold)
        np.testing.assert_allclose(T_hot, args.T_hot)
        np.testing.assert_allclose(T_cold, args.T_cold)
        return "figure"

    monkeypatch.setattr(hp_plotting, "plot_multi_hp_profiles_from_results", fake_plot)

    result = hp_shared.evaluate_carnot_hpr_result(
        args=args,
        state=HPRParsedState(
            Q_amb_hot=0.0,
            Q_amb_cold=0.0,
            T_cond=np.array([100.0]),
            T_evap=np.array([60.0]),
        ),
        w_net=10.0,
        w_hpr=np.array([10.0]),
        Q_cond_total=np.array([40.0]),
        Q_evap_total=np.array([30.0]),
        debug=True,
    )

    assert result.artifacts.debug_figure == "figure"


def test_vapour_debug_plot_uses_residual_profiles_after_direct_ambient(monkeypatch):
    args = _base_args()
    seen = {}

    def fake_plot(T_hot, H_hot, T_cold, H_cold, *_args, **_kwargs):
        seen["T_hot"] = T_hot
        assert len(T_hot) == len(H_hot)
        assert len(T_cold) == len(H_cold)
        return "figure"

    monkeypatch.setattr(hp_plotting, "plot_multi_hp_profiles_from_results", fake_plot)

    result = hp_shared.evaluate_vapour_hpr_result(
        args=args,
        state=HPRParsedState(
            Q_amb_hot=0.0,
            Q_amb_cold=50.0,
        ),
        work=0.0,
        work_arr=np.array([]),
        Q_heat=np.array([]),
        Q_cool=np.array([]),
        cop_h=1.0,
        hpr_streams=StreamCollection(),
        debug=True,
    )

    assert result.artifacts.debug_figure == "figure"
    assert len(seen["T_hot"]) > len(args.T_hot)


def test_vapour_evaluator_handles_empty_evaporator_side():
    args = _base_args(
        T_hot=np.array([100.0, 50.0]),
        H_hot=np.array([0.0, 0.0]),
        T_cold=np.array([95.0, 45.0]),
        H_cold=np.array([100.0, 0.0]),
        z_amb_hot=np.zeros(2),
        z_amb_cold=np.zeros(2),
        bckgrd_hot_streams=StreamCollection(),
        bckgrd_cold_streams=_sc(
            _stream("sink", 45.0, 95.0, 100.0, is_process_stream=False)
        ),
        Q_heat_max=100.0,
        Q_cool_max=0.0,
    )

    result = hp_shared.evaluate_vapour_hpr_result(
        args=args,
        state=HPRParsedState(
            Q_amb_hot=0.0,
            Q_amb_cold=0.0,
        ),
        work=0.0,
        work_arr=np.array([]),
        Q_heat=np.array([]),
        Q_cool=np.array([]),
        cop_h=1.0,
        hpr_streams=StreamCollection(),
    )

    assert result.Q_ext_heat == pytest.approx(100.0)
    assert result.Q_ext_cold == pytest.approx(0.0)
    assert result.feasibility_penalty == pytest.approx(0.0)


def test_get_heat_pump_cascade_helper(monkeypatch):
    hot = _sc(_stream("H", 120.0, 110.0, 5.0, is_process_stream=False))
    cold = _sc(_stream("C", 70.0, 80.0, 5.0, is_process_stream=False))

    monkeypatch.setattr(
        hp_plotting,
        "create_problem_table_with_t_int",
        lambda streams, is_shifted, period_idx: ProblemTable(
            {
                ProblemTableLabel.T: [120.0, 80.0],
                ProblemTableLabel.H_HOT_UT: [0.0, 0.0],
                ProblemTableLabel.H_COLD_UT: [0.0, 0.0],
            }
        ),
    )
    monkeypatch.setattr(
        hp_plotting,
        "get_utility_heat_cascade",
        lambda *args, **kwargs: {
            "T_col": np.array([120.0, 80.0]),
            "updates": {
                ProblemTableLabel.H_HOT_UT: np.array([5.0, 0.0]),
                ProblemTableLabel.H_COLD_UT: np.array([3.0, 0.0]),
            },
        },
    )

    out = hp_plotting._get_hpr_cascade(hot, cold)
    assert len(out) == 3


def test_hpr_backend_result_derives_hot_and_cold_stream_views_from_combined_streams():
    hot_stream = _stream("cond", 120.0, 100.0, 10.0, is_process_stream=False)
    cold_stream = _stream("evap", 20.0, 40.0, 8.0, is_process_stream=False)
    hpr_streams = _sc(hot_stream, cold_stream)

    res = HPRBackendResult(
        obj=1.0,
        utility_tot=2.0,
        w_net=1.0,
        Q_ext_heat=0.5,
        Q_ext_cold=0.25,
        Q_amb_hot=0.0,
        Q_amb_cold=0.0,
        artifacts=HPRThermoArtifacts(hpr_streams=hpr_streams),
    )

    assert res.hpr_streams is hpr_streams
    assert len(res.hpr_hot_streams) == 1
    assert len(res.hpr_cold_streams) == 1
