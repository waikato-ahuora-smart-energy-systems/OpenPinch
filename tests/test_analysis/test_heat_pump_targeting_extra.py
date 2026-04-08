"""Additional branch coverage tests for heat pump targeting internals."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from OpenPinch.analysis import heat_pump_targeting as hp
from OpenPinch.classes.problem_table import ProblemTable
from OpenPinch.classes.stream import Stream
from OpenPinch.classes.stream_collection import StreamCollection
from OpenPinch.lib.config import Configuration
from OpenPinch.lib.enums import HeatPumpType, ProblemTableLabel as PT


def _sc(*streams):
    coll = StreamCollection()
    for s in streams:
        coll.add(s)
    return coll


def _base_args(**overrides):
    args = {
        "Q_hp_target": 200.0,
        "Q_amb_max": 20.0,
        "dt_range_max": 110.0,
        "T_hot": np.array([140.0, 90.0, 40.0]),
        "H_hot": np.array([0.0, -80.0, -160.0]),
        "T_cold": np.array([130.0, 80.0, 30.0]),
        "H_cold": np.array([200.0, 100.0, 0.0]),
        "n_cond": 2,
        "n_evap": 2,
        "eta_comp": 0.75,
        "eta_exp": 0.7,
        "dtcont_hp": 5.0,
        "dt_hp_ihx": 3.0,
        "dt_cascade_hx": 2.0,
        "dt_phase_change": 1.0,
        "price_ratio": 1.0,
        "is_direct_integration": True,
        "is_heat_pumping": True,
        "max_multi_start": 2,
        "T_env": 20.0,
        "dt_env_cont": 5.0,
        "eta_hp_carnot": 0.6,
        "eta_he_carnot": 0.4,
        "refrigerant_ls": ["R134A", "R134A", "R134A"],
        "do_refrigerant_sort": False,
        "initialise_simulated_hp": True,
        "allow_integrated_expander": True,
        "dT_subcool": None,
        "dT_superheat": None,
        "net_hot_streams": _sc(
            Stream(name="H", t_supply=120.0, t_target=80.0, heat_flow=50.0)
        ),
        "net_cold_streams": _sc(
            Stream(name="C", t_supply=30.0, t_target=60.0, heat_flow=40.0)
        ),
        "bb_minimiser": "shgo",
        "eta_penalty": 0.001,
        "rho_penalty": 10.0,
        "debug": False,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def _pt_with_hnet(h0, h1):
    return ProblemTable({PT.T.value: [120.0, 60.0], PT.H_NET.value: [h0, h1]})


def _patch_output_model_validate(monkeypatch):
    monkeypatch.setattr(
        hp.HeatPumpTargetOutputs,
        "model_validate",
        classmethod(lambda cls, v: v),
    )


def test_get_heat_pump_targets_dispatch_and_invalid_handler(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1)
    zone_cfg = Configuration()

    monkeypatch.setattr(hp, "_prepare_heat_pump_target_inputs", lambda **_kwargs: args)
    monkeypatch.setattr(
        hp, "_get_ambient_air_stream", lambda _q, _a: StreamCollection()
    )
    _patch_output_model_validate(monkeypatch)

    def _handler(_args):
        return SimpleNamespace(Q_amb=5.0)

    monkeypatch.setitem(
        hp._HP_PLACEMENT_HANDLERS, HeatPumpType.MultiSimpleVapourComp.value, _handler
    )
    out = hp.get_heat_pump_targets(
        Q_hp_target=100.0,
        T_vals=np.array([120.0, 80.0]),
        H_hot=np.array([0.0, -50.0]),
        H_cold=np.array([50.0, 0.0]),
        zone_config=zone_cfg,
        is_direct_integration=True,
        is_heat_pumping=True,
    )
    assert hasattr(out, "amb_stream")

    monkeypatch.setitem(
        hp._HP_PLACEMENT_HANDLERS, HeatPumpType.MultiSimpleVapourComp.value, None
    )
    with pytest.raises(ValueError, match="No valid heat pump targeting type selected"):
        hp.get_heat_pump_targets(
            Q_hp_target=100.0,
            T_vals=np.array([120.0, 80.0]),
            H_hot=np.array([0.0, -50.0]),
            H_cold=np.array([50.0, 0.0]),
            zone_config=zone_cfg,
            is_direct_integration=True,
            is_heat_pumping=True,
        )


@pytest.mark.parametrize(
    "q_amb",
    [20.0, -20.0, 0.0],
)
def test_calc_heat_pump_cascade_branches(monkeypatch, q_amb):
    pt = ProblemTable(
        {
            PT.T.value: [120.0, 60.0],
            PT.H_NET_A.value: [1.0, 1.0],
            PT.H_NET_HOT.value: [2.0, 2.0],
            PT.H_NET_COLD.value: [3.0, 3.0],
        }
    )

    monkeypatch.setattr(
        hp,
        "create_problem_table_with_t_int",
        lambda streams, is_shifted=True: ProblemTable({PT.T.value: [120.0, 60.0]}),
    )
    monkeypatch.setattr(
        hp,
        "get_utility_heat_cascade",
        lambda **_kwargs: {
            PT.H_NET_UT.value: np.array([0.0, 0.0]),
            PT.H_HOT_UT.value: np.array([0.0, 0.0]),
            PT.H_COLD_UT.value: np.array([0.0, 0.0]),
        },
    )
    monkeypatch.setattr(
        hp, "get_process_heat_cascade", lambda **_kwargs: _pt_with_hnet(4.0, -1.0)
    )

    amb_stream = StreamCollection()
    if abs(q_amb) > 0:
        amb_stream.add(
            Stream(
                name="AIR",
                t_supply=20.0,
                t_target=19.0,
                heat_flow=abs(q_amb),
                is_process_stream=True,
            )
        )

    res = SimpleNamespace(
        hp_hot_streams=StreamCollection(),
        hp_cold_streams=StreamCollection(),
        Q_amb=q_amb,
        amb_stream=amb_stream,
    )
    out = hp.calc_heat_pump_cascade(
        pt, res, is_T_vals_shifted=True, is_direct_integration=True
    )
    assert isinstance(out, ProblemTable)


def test_hot_profile_trimming_and_balance_sign_handling():
    T = np.array([150.0, 120.0, 90.0, 60.0])
    H = np.array([-300.0, -250.0, -100.0, 0.0])
    T_out, H_out = hp._get_H_col_till_target_Q(200.0, T.copy(), H.copy(), is_cold=False)
    assert T_out.shape == H_out.shape

    T_hot, H_hot, T_cold, H_cold, q_amb = (
        hp._balance_hot_and_cold_heat_loads_with_ambient_air(
            T_hot=np.array([140.0, 100.0, 60.0]),
            H_hot=np.array([0.0, 120.0, 200.0]),
            T_cold=np.array([140.0, 100.0, 60.0]),
            H_cold=np.array([0.0, -50.0, -100.0]),
            dtcont=5.0,
            T_env=15.0,
            dT_env_cont=5.0,
            dt_phase_change=2.0,
            is_heat_pumping=True,
        )
    )
    assert T_hot.shape == H_hot.shape
    assert T_cold.shape == H_cold.shape
    assert q_amb >= 0.0


def test_multi_temp_carnot_optimiser_success_and_failure(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1)
    _patch_output_model_validate(monkeypatch)
    monkeypatch.setattr(hp, "multiminima", lambda **_kwargs: np.array([[0.2, 0.6]]))

    monkeypatch.setattr(
        hp,
        "_compute_multi_temperature_carnot_hp_opt_obj",
        lambda x, args, debug=False: {
            "obj": 0.1,
            "utility_tot": 1.0,
            "work_hp": 0.5,
            "Q_ext": 0.0,
            "Q_amb": 0.0,
            "cop": 3.0,
            "opt_success": True,
            "T_cond": np.array([100.0]),
            "Q_cond": np.array([50.0]),
            "T_evap": np.array([60.0]),
            "Q_evap": np.array([40.0]),
        },
    )
    monkeypatch.setattr(
        hp,
        "_get_carnot_hp_streams",
        lambda *_args, **_kwargs: {
            "hp_hot_streams": StreamCollection(),
            "hp_cold_streams": StreamCollection(),
        },
    )
    out = hp._optimise_multi_temperature_carnot_heat_pump_placement(args)
    assert out["opt_success"] is True

    monkeypatch.setattr(
        hp,
        "_compute_multi_temperature_carnot_hp_opt_obj",
        lambda x, args, debug=False: {"opt_success": False},
    )
    with pytest.raises(ValueError, match="failed to return an optimal result"):
        hp._optimise_multi_temperature_carnot_heat_pump_placement(args)


def test_multi_temp_carnot_objective_debug_branch(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1)
    x = np.array([0.2, 0.7])
    called = {"plot": 0}
    monkeypatch.setattr(
        hp,
        "plot_multi_hp_profiles_from_results",
        lambda *a, **k: called.__setitem__("plot", called["plot"] + 1),
    )
    out = hp._compute_multi_temperature_carnot_hp_opt_obj(x, args, debug=True)
    assert out["opt_success"] is True
    assert called["plot"] == 1


def test_cascade_optimiser_and_compute_branches(monkeypatch):
    args = _base_args(n_cond=2, n_evap=2, initialise_simulated_hp=True)
    _patch_output_model_validate(monkeypatch)

    monkeypatch.setattr(
        hp,
        "_optimise_multi_temperature_carnot_heat_pump_placement",
        lambda _args: SimpleNamespace(
            T_cond=np.array([100.0, 80.0]),
            Q_cond=np.array([120.0, 80.0]),
            T_evap=np.array([70.0, 50.0]),
            Q_evap=np.array([90.0, 60.0]),
        ),
    )
    monkeypatch.setattr(
        hp, "_validate_vapour_hp_refrigerant_ls", lambda n, a: ["R134A"] * n
    )
    monkeypatch.setattr(
        hp, "_get_bounds_for_cascade_hp_opt", lambda _args: [(0.0, 1.0)] * 9
    )
    monkeypatch.setattr(
        hp, "_get_x0_for_cascade_hp_opt", lambda **_kwargs: np.array([[0.5] * 9])
    )
    monkeypatch.setattr(hp, "multiminima", lambda **_kwargs: np.array([[0.5] * 9]))
    monkeypatch.setattr(
        hp,
        "_compute_cascade_hp_system_performance",
        lambda x, args, debug=False: {
            "obj": 0.2,
            "utility_tot": 1.0,
            "work_hp": np.array([1.0, 2.0]),
            "Q_ext": 0.0,
            "Q_amb": 0.0,
            "cop": 2.0,
            "hp_hot_streams": StreamCollection(),
            "hp_cold_streams": StreamCollection(),
        },
    )

    out = hp._optimise_cascade_heat_pump_placement(args)
    assert out["opt_success"] is True

    monkeypatch.setattr(hp, "multiminima", lambda **_kwargs: np.array([]))
    with pytest.raises(ValueError, match="failed"):
        hp._optimise_cascade_heat_pump_placement(args)


def test_cascade_x0_bounds_and_parse(monkeypatch):
    args = _base_args(n_cond=2, n_evap=2)
    bnds = [(0.0, 0.2)] * 9
    x0 = hp._get_x0_for_cascade_hp_opt(
        T_cond=np.array([120.0, 90.0]),
        Q_heat=np.array([120.0, 80.0]),
        T_evap=np.array([70.0, 50.0]),
        Q_cool=np.array([90.0, 60.0]),
        args=args,
        bnds=bnds,
    )
    assert x0.shape == (1, 9)
    assert np.all(x0[0] >= 0.0)

    with pytest.raises(ValueError, match="Bounds size must match x0 size"):
        hp._get_x0_for_cascade_hp_opt(
            T_cond=np.array([120.0, 90.0]),
            Q_heat=np.array([120.0, 80.0]),
            T_evap=np.array([70.0, 50.0]),
            Q_cool=np.array([90.0, 60.0]),
            args=args,
            bnds=[(0.0, 1.0)],
        )

    monkeypatch.setattr(
        hp, "PropsSI", lambda prop, *_args: 420.0 if prop == "Tmin" else 422.0
    )
    b = hp._get_bounds_for_cascade_hp_opt(args)
    assert len(b) == 9

    T_cond, dT_sc, Q_heat, T_evap, Q_cool = hp._parse_cascade_hp_state_variables(
        np.array([0.1] * 9), args
    )
    assert T_cond.shape == (2,)
    assert dT_sc.shape == (2,)
    assert T_evap.shape == (2,)
    assert Q_cool.shape[0] == 2


def test_compute_cascade_hp_system_performance_unsolved_and_solved(monkeypatch):
    args = _base_args(n_cond=2, n_evap=2)
    x = np.array([0.2] * 9)

    class _FakeCascadeUnsolved:
        solved = False
        work = 50.0

        def solve(self, **_kwargs):
            return None

    monkeypatch.setattr(hp, "CascadeHeatPumpCycle", _FakeCascadeUnsolved)
    out_unsolved = hp._compute_cascade_hp_system_performance(x, args)
    assert "obj" in out_unsolved

    class _FakeCascadeSolved:
        solved = True
        work = 40.0
        work_arr = np.array([10.0, 30.0])
        Q_cool = 70.0
        Q_heat_arr = np.array([120.0, 80.0])
        Q_cool_arr = np.array([50.0, 20.0])
        penalty = 1.0

        def solve(self, **_kwargs):
            return None

        def build_stream_collection(self, **_kwargs):
            return _sc(
                Stream(
                    name="HP",
                    t_supply=100.0,
                    t_target=90.0,
                    heat_flow=10.0,
                    is_process_stream=False,
                )
            )

    monkeypatch.setattr(hp, "CascadeHeatPumpCycle", _FakeCascadeSolved)
    seq = iter([_pt_with_hnet(5.0, -1.0), _pt_with_hnet(6.0, -2.0)])
    monkeypatch.setattr(hp, "get_process_heat_cascade", lambda **_kwargs: next(seq))
    calls = {"plot": 0}
    monkeypatch.setattr(
        hp,
        "plot_multi_hp_profiles_from_results",
        lambda *a, **k: calls.__setitem__("plot", calls["plot"] + 1),
    )
    out = hp._compute_cascade_hp_system_performance(x, args, debug=True)
    assert "hp_hot_streams" in out
    assert calls["plot"] == 1


def test_multi_simple_carnot_and_multi_simple_simulated_paths(monkeypatch):
    args = _base_args(n_cond=2, n_evap=2, initialise_simulated_hp=True)
    _patch_output_model_validate(monkeypatch)

    monkeypatch.setattr(
        hp, "multiminima", lambda **_kwargs: np.array([[0.4, 0.4, 0.5, 0.5]])
    )
    monkeypatch.setattr(
        hp,
        "_get_carnot_hp_streams",
        lambda *_args, **_kwargs: {
            "hp_hot_streams": StreamCollection(),
            "hp_cold_streams": StreamCollection(),
        },
    )
    out_carnot = hp._optimise_multi_simple_carnot_heat_pump_placement(args)
    assert out_carnot["opt_success"] is True

    monkeypatch.setattr(
        hp,
        "_optimise_multi_simple_carnot_heat_pump_placement",
        lambda _args: SimpleNamespace(
            T_cond=np.array([120.0, 100.0]),
            Q_cond=np.array([120.0, 80.0]),
            T_evap=np.array([70.0, 50.0]),
        ),
    )
    monkeypatch.setattr(
        hp, "_validate_vapour_hp_refrigerant_ls", lambda n, a: ["R134A"] * n
    )
    monkeypatch.setattr(
        hp, "_get_bounds_for_multi_single_hp_opt", lambda _args: [(0.0, 1.0)] * 10
    )
    monkeypatch.setattr(
        hp, "_get_x0_for_multi_single_hp_opt", lambda **_kwargs: np.array([[0.2] * 10])
    )
    monkeypatch.setattr(hp, "multiminima", lambda **_kwargs: np.array([[0.2] * 10]))
    monkeypatch.setattr(
        hp,
        "_compute_multi_simple_hp_system_performance",
        lambda x, args, debug=False: {
            "obj": 0.1,
            "utility_tot": 1.0,
            "work_hp": np.array([1.0, 2.0]),
            "Q_ext": 0.0,
            "Q_amb": 0.0,
            "cop": 2.5,
            "hp_hot_streams": StreamCollection(),
            "hp_cold_streams": StreamCollection(),
        },
    )
    out_sim = hp._optimise_multi_simple_heat_pump_placement(args)
    assert out_sim["opt_success"] is True

    monkeypatch.setattr(hp, "multiminima", lambda **_kwargs: np.array([]))
    with pytest.raises(ValueError, match="failed"):
        hp._optimise_multi_simple_heat_pump_placement(args)


def test_multi_single_x0_bounds_parse_and_performance(monkeypatch):
    args = _base_args(n_cond=2, n_evap=2)
    bnds = [(0.0, 0.8)] * 10
    x0 = hp._get_x0_for_multi_single_hp_opt(
        T_cond=np.array([120.0, 100.0]),
        Q_cond=np.array([120.0, 80.0]),
        T_evap=np.array([70.0, 50.0]),
        args=args,
        bnds=bnds,
    )
    assert x0.shape == (1, 10)

    with pytest.raises(ValueError, match="Bounds size must match x0 size"):
        hp._get_x0_for_multi_single_hp_opt(
            T_cond=np.array([120.0, 100.0]),
            Q_cond=np.array([120.0, 80.0]),
            T_evap=np.array([70.0, 50.0]),
            args=args,
            bnds=[(0.0, 1.0)],
        )

    monkeypatch.setattr(
        hp, "PropsSI", lambda prop, *_args: 420.0 if prop == "Tmin" else 422.0
    )
    b = hp._get_bounds_for_multi_single_hp_opt(args)
    assert len(b) == 10

    x_bad = np.array([0.9] * 10)
    assert hp._constrain_min_temperature_lift(x_bad, args) <= 0.0
    out_bad = hp._compute_multi_simple_hp_system_performance(x_bad, args)
    assert np.isinf(out_bad["obj"])

    class _FakeMultiSimple:
        work = 30.0
        work_arr = np.array([10.0, 20.0])
        Q_cool = 60.0
        Q_heat_arr = np.array([120.0, 80.0])
        Q_cool_arr = np.array([40.0, 20.0])
        penalty = 1.0

        def solve(self, **_kwargs):
            return None

        def build_stream_collection(self, **_kwargs):
            return _sc(
                Stream(
                    name="HP",
                    t_supply=100.0,
                    t_target=90.0,
                    heat_flow=10.0,
                    is_process_stream=False,
                )
            )

    monkeypatch.setattr(hp, "MultiSimpleHeatPumpCycle", _FakeMultiSimple)
    seq = iter([_pt_with_hnet(5.0, -1.0), _pt_with_hnet(6.0, -2.0)])
    monkeypatch.setattr(hp, "get_process_heat_cascade", lambda **_kwargs: next(seq))
    out = hp._compute_multi_simple_hp_system_performance(
        np.array([0.2] * 10), args, debug=False
    )
    assert "hp_model" in out


def test_brayton_paths_and_helpers(monkeypatch):
    args = _base_args(n_cond=1, n_evap=1)
    _patch_output_model_validate(monkeypatch)

    class _OptRes:
        success = True
        x = np.array([0.1, 0.2, 0.3, 0.9])
        message = "ok"

    monkeypatch.setattr(hp, "minimize", lambda **_kwargs: _OptRes())
    monkeypatch.setattr(
        hp,
        "_compute_brayton_hp_system_performance",
        lambda x, args: {
            "obj": 0.1,
            "utility_tot": 1.0,
            "work_hp": 0.5,
            "Q_ext": 0.0,
            "Q_amb": 0.0,
            "cop": 2.0,
            "hp_hot_streams": StreamCollection(),
            "hp_cold_streams": StreamCollection(),
        },
    )
    out = hp._optimise_brayton_heat_pump_placement(args)
    assert out["opt_success"] is True

    T_co, dT_c, dT_gc, q_h = hp._parse_brayton_hp_state_variables(
        np.array([0.1, 0.2, 0.3, 0.9]), args
    )
    assert len(T_co) == len(dT_c) == len(dT_gc) == len(q_h) == 1

    class _FakeBraytonCycle:
        def __init__(self):
            self.work_net = 15.0
            self.Q_cool = 40.0
            self.cycle_states = [{}, {}, {}, {"T": 50.0}]

        def solve(self, **_kwargs):
            return None

    monkeypatch.setattr(hp, "SimpleBraytonHeatPumpCycle", _FakeBraytonCycle)
    hp_list = hp._create_brayton_hp_list(
        T_comp_out=np.array([120.0]),
        dT_gc=np.array([20.0]),
        Q_gc=np.array([100.0]),
        dT_comp=np.array([10.0]),
        args=args,
    )
    assert len(hp_list) == 1

    class _FakeHPObj:
        def __init__(self):
            self.work_net = 12.0
            self.Q_cool = 35.0
            self.cycle_states = [{}, {}, {}, {"T": 42.0}]

    monkeypatch.setattr(hp, "_create_brayton_hp_list", lambda **_kwargs: [_FakeHPObj()])
    monkeypatch.setattr(
        hp,
        "_build_simulated_hps_streams",
        lambda hp_list, **_kwargs: _sc(
            Stream(
                name="S",
                t_supply=100.0,
                t_target=90.0,
                heat_flow=10.0,
                is_process_stream=False,
            )
        ),
    )
    seq = iter([_pt_with_hnet(8.0, -1.0), _pt_with_hnet(7.0, -2.0)])
    monkeypatch.setattr(hp, "get_process_heat_cascade", lambda **_kwargs: next(seq))
    out_perf = hp._compute_brayton_hp_system_performance(
        np.array([0.1, 0.2, 0.3, 0.9]), args
    )
    assert "cop" in out_perf


def test_misc_heat_pump_helpers_and_stream_builders(monkeypatch):
    with pytest.raises(ValueError, match="Infeasible temperature interval"):
        hp._create_net_hot_and_cold_stream_collections_for_background_profile(
            T_vals=np.array([100.0, 100.0]),
            H_vals=np.array([0.0, 10.0]),
        )

    hot, cold = hp._create_net_hot_and_cold_stream_collections_for_background_profile(
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_vals=np.array([0.0, -30.0, 20.0]),
    )
    assert isinstance(hot, StreamCollection)
    assert isinstance(cold, StreamCollection)

    q_vals = hp._get_Q_vals_from_T_vals(
        T_hp=np.array([100.0, 60.0]),
        T_vals=np.array([120.0, 80.0, 40.0]),
        H_vals=np.array([100.0, 50.0, 0.0]),
        is_cond=False,
    )
    assert q_vals.shape == (2,)

    t_avg = hp._compute_entropic_average_temperature_in_K(
        np.array([300.0, 310.0]), np.array([10.0, 5.0]), T_units="K"
    )
    assert t_avg > 0.0

    args = _base_args(refrigerant_ls=["R134A", "R600"], do_refrigerant_sort=False)
    refs = hp._validate_vapour_hp_refrigerant_ls(1, args)
    assert refs == ["R134A"]

    streams = hp._build_latent_streams(
        T_ls=np.array([110.0, 109.2, 80.0]),
        dT_phase_change=1.0,
        Q_ls=np.array([20.0, 10.0, 30.0]),
        is_hot=True,
        prefix="HP",
    )
    assert len(streams) >= 2

    class _FakeCycle:
        def build_stream_collection(self, **_kwargs):
            return _sc(
                Stream(
                    name="C",
                    t_supply=90.0,
                    t_target=70.0,
                    heat_flow=10.0,
                    is_process_stream=False,
                )
            )

    agg = hp._build_simulated_hps_streams(
        [_FakeCycle()], include_cond=True, include_evap=True, dtcont_hp=5.0
    )
    assert len(agg) >= 1

    amb0 = hp._get_ambient_air_stream(0.0, _base_args())
    amb_pos = hp._get_ambient_air_stream(10.0, _base_args())
    amb_neg = hp._get_ambient_air_stream(-10.0, _base_args())
    assert len(amb0) == 0
    assert len(amb_pos) == 1
    assert len(amb_neg) == 1

    assert hp._calc_obj(
        10.0, 5.0, 100.0, price_ratio=2.0, penalty=1.0
    ) == pytest.approx(0.135)
    assert hp._calc_Q_amb(50.0, 30.0, 5.0) == pytest.approx(25.0)
